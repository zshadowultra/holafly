"""Load the FlyWire connectivity table without confusing release 783 with a node count.

The public :func:`load_connectome` function returns a plain dictionary with
``W``, ``ids``, ``n``, and ``id_to_index``.  ``W`` is a dense NumPy array for
small graphs (including a 783-node graph when the input has that size) and a
``scipy.sparse.csr_matrix`` for the full graph.  A dense 138,639-square graph
would require roughly 154 GB for float64 values, so the automatic storage
choice is deliberately conservative.

The cell-type sidecar is optional.  If a matching
``consolidated_cell_types.csv.gz`` file is available below the configured
FlyWire checkout, exact ``R1-6``/``R7``/``R8`` matches are used to apply the
histamine photoreceptor sign correction.  If it is absent or unreadable, the
loader keeps the signed edge values from the parquet file rather than guessing
at cell types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

try:
    # Use the repository's source-of-truth labels when the module is imported
    # as part of fly-master.  The fallback keeps direct execution from
    # connectome/loader.py working when the parent package is not on sys.path.
    from fixes.histamine import PHOTORECEPTOR_TYPES as _PHOTORECEPTOR_TYPES
except ModuleNotFoundError:
    _PHOTORECEPTOR_TYPES = frozenset({"R1-6", "R7", "R8"})


# The signed product is authoritative; Connectivity and Excitatory do not
# need to be materialized merely to reproduce it.
_REQUIRED_COLUMNS = (
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Excitatory x Connectivity",
)
_SIGNED_VALUE_COLUMN = "Excitatory x Connectivity"
_DEFAULT_CHUNK_ROWS = 1_000_000
# 2,048^2 float64 values are about 32 MiB.  This is large enough for the
# requested 783-node dense case while preventing an accidental full-graph
# allocation.
_DEFAULT_DENSE_LIMIT = 2_048
_MAX_DENSE_BYTES = 256 * 1024 * 1024
_CELL_TYPE_ROOT = Path("/home/hatch/workspace/fly-zoo/fruit-fly")
_CELL_TYPE_FILENAME = "consolidated_cell_types.csv.gz"
_MISSING = object()


def _missing_mask(values: Any) -> np.ndarray:
    """Return a one-dimensional missing-value mask for an ID-like array."""

    try:
        mask = np.asarray(pd.isna(values))
    except (TypeError, ValueError):
        # IDs in the supported parquet schema are scalar values.  This
        # fallback keeps unusual object arrays from crashing the optional
        # missing-ID path.
        return np.zeros(len(values), dtype=bool)
    if mask.ndim == 0:
        return np.full(len(values), bool(mask), dtype=bool)
    return mask.astype(bool).reshape(-1)


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar values to ordinary Python/hashable values."""

    if isinstance(value, np.generic):
        value = value.item()
    # Pandas promotes an integer ID column to float when a null is present.
    # Restore integral IDs so the public list and mapping remain ID-like.
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _canonical_id(value: Any) -> str:
    """Make a stable comparison key without changing the returned ID value."""

    value = _python_scalar(value)
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    return str(value)


def _mapping_key(value: Any) -> Any:
    """Return the original scalar when hashable, otherwise a string key."""

    value = _python_scalar(value)
    try:
        hash(value)
    except TypeError:
        return _canonical_id(value)
    return value


def _validate_indices(series: pd.Series, column: str) -> np.ndarray:
    """Validate and return a non-negative integral source-index array."""

    values = series.to_numpy(copy=False)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if _missing_mask(values).any():
        raise ValueError(f"{column} contains missing source indices")

    try:
        numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(
            copy=False
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{column} must contain integer source indices") from exc

    limits = np.iinfo(np.int64)
    if np.issubdtype(numeric.dtype, np.integer):
        if np.issubdtype(numeric.dtype, np.unsignedinteger):
            if (numeric > limits.max).any():
                raise ValueError(
                    f"{column} contains source indices outside int64"
                )
        elif (numeric < limits.min).any() or (numeric > limits.max).any():
            raise ValueError(
                f"{column} contains source indices outside int64"
            )
        return numeric.astype(np.int64, copy=False)

    numeric = np.asarray(numeric, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{column} contains non-finite source indices")
    rounded = np.rint(numeric)
    if not np.equal(numeric, rounded).all():
        raise ValueError(f"{column} contains non-integral source indices")
    if (rounded < 0).any():
        raise ValueError(f"{column} contains negative source indices")
    if (rounded < limits.min).any() or (rounded > limits.max).any():
        raise ValueError(f"{column} contains source indices outside int64")
    return rounded.astype(np.int64)


def _validate_values(series: pd.Series) -> np.ndarray:
    """Convert the signed edge column to finite float64 values."""

    values = series.to_numpy(copy=False)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    if _missing_mask(values).any():
        raise ValueError(f"{_SIGNED_VALUE_COLUMN} contains missing values")
    try:
        numeric = np.asarray(
            pd.to_numeric(pd.Series(values), errors="raise"),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{_SIGNED_VALUE_COLUMN} must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{_SIGNED_VALUE_COLUMN} contains non-finite values")
    return numeric


def _compact_indices(indices: np.ndarray) -> np.ndarray:
    """Use int32 coordinates for the normal connectome without truncating."""

    if indices.size == 0:
        return indices.astype(np.int32, copy=False)
    if (
        int(indices.min()) >= np.iinfo(np.int32).min
        and int(indices.max()) <= np.iinfo(np.int32).max
    ):
        return indices.astype(np.int32, copy=False)
    return indices


def _iter_parquet_chunks(
    path: Any, columns: list[str], chunksize: int
) -> Iterator[pd.DataFrame]:
    """Yield column-limited parquet chunks through pandas/pyarrow.

    Some pandas releases do not expose ``chunksize`` for the pyarrow engine.
    In that case use pyarrow's row-group iterator as a lazy fallback instead
    of materializing the full 15-million-row table in pandas.
    """

    try:
        reader = pd.read_parquet(
            path,
            columns=columns,
            engine="pyarrow",
            chunksize=chunksize,
        )
    except (TypeError, ValueError) as exc:
        if "chunksize" not in str(exc):
            raise
        try:
            import pyarrow.parquet as parquet
        except ImportError as import_exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "chunked parquet loading requires pandas' pyarrow support"
            ) from import_exc
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=chunksize, columns=columns
        ):
            yield batch.to_pandas()
        return

    if isinstance(reader, pd.DataFrame):
        # This branch is useful for small test doubles; the normal pyarrow
        # reader is iterable and is closed below.
        yield reader
        return
    try:
        yield from reader
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()


def _bind_ids(
    indices: np.ndarray,
    id_values: Any,
    ids_by_index: dict[int, Any],
    index_by_id: dict[str, int],
) -> None:
    """Bind source indices to IDs and reject contradictory mappings.

    Missing IDs are intentionally not bound.  Their source-index slots remain
    in the result as ``None`` and any edges using those slots stay aligned with
    the source indices; this avoids a KeyError without inventing an ID.
    """

    values = np.asarray(id_values)
    present = ~_missing_mask(values)
    if not present.any():
        return

    # Removing repeated pairs first keeps the Python-level validation small
    # even when a parquet chunk contains millions of rows.
    pairs = pd.DataFrame(
        {"source_index": indices[present], "id": values[present]}
    ).drop_duplicates()
    for source_index, raw_id in pairs[["source_index", "id"]].itertuples(
        index=False, name=None
    ):
        index = int(source_index)
        neuron_id = _python_scalar(raw_id)
        key = _canonical_id(neuron_id)

        previous = ids_by_index.get(index, _MISSING)
        if previous is not _MISSING and _canonical_id(previous) != key:
            raise ValueError(
                "source index maps to multiple IDs: "
                f"index={index}, IDs={previous!r} and {neuron_id!r}"
            )
        if key in index_by_id and index_by_id[key] != index:
            raise ValueError(
                "ID maps to multiple source indices: "
                f"ID={neuron_id!r}, indices={index_by_id[key]} and {index}"
            )
        ids_by_index[index] = neuron_id
        index_by_id[key] = index


def _discover_cell_type_file() -> Path | None:
    """Find the optional consolidated cell-type sidecar, if present."""

    if not _CELL_TYPE_ROOT.exists():
        return None
    try:
        matches = sorted(_CELL_TYPE_ROOT.rglob(_CELL_TYPE_FILENAME))
    except OSError:
        return None
    return matches[0] if matches else None


def _load_photoreceptor_ids(path: Path | None) -> set[str]:
    """Read exact photoreceptor labels from an optional gzip CSV sidecar."""

    if path is None or not path.is_file():
        # The sidecar is optional.  In particular, do not infer cell types
        # from missing IDs or silently disable the supplied signed values.
        return set()

    try:
        header = pd.read_csv(path, compression="gzip", nrows=0)
        id_column = "root_id" if "root_id" in header.columns else None
        if id_column is None:
            for candidate in ("cell_id", "id"):
                if candidate in header.columns:
                    id_column = candidate
                    break
        type_column = (
            "primary_type"
            if "primary_type" in header.columns
            else "cell_type"
            if "cell_type" in header.columns
            else None
        )
        if id_column is None or type_column is None:
            return set()

        table = pd.read_csv(
            path,
            compression="gzip",
            usecols=[id_column, type_column],
            dtype={id_column: "string", type_column: "string"},
        )
        ids = table[id_column].to_numpy(dtype=object)
        types = table[type_column].to_numpy(dtype=object)
        present = ~_missing_mask(ids) & ~_missing_mask(types)
        return {
            _canonical_id(ids[i])
            for i in np.flatnonzero(present)
            if str(types[i]) in _PHOTORECEPTOR_TYPES
        }
    except Exception:
        # The correction is an optional annotation layer.  A missing,
        # malformed, or temporarily unreadable sidecar must not make the
        # connectivity parquet unusable.
        return set()


def _apply_histamine_correction(
    values: np.ndarray,
    presynaptic_indices: np.ndarray,
    photoreceptor_rows: np.ndarray,
) -> None:
    """Apply the exact R1-6/R7/R8 row-level sign fix in place."""

    if not photoreceptor_rows.any() or values.size == 0:
        return
    rows = photoreceptor_rows[presynaptic_indices]
    if rows.any():
        # This is the sparse-safe equivalent of
        # fixes.histamine.correct_photoreceptor_sign and the source's
        # sign[photo_mask[pre]] = -1 before edge aggregation: photoreceptor
        # output is always inhibitory, while its edge magnitude is retained.
        values[rows] = -np.abs(values[rows])


def _build_csr(
    row_blocks: list[np.ndarray],
    column_blocks: list[np.ndarray],
    value_blocks: list[np.ndarray],
    n: int,
    photoreceptor_rows: np.ndarray,
) -> sparse.csr_matrix:
    """Build one CSR matrix, summing duplicate coordinates with scipy."""

    if not row_blocks or n == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    if len(row_blocks) == 1:
        rows = row_blocks[0]
        columns = column_blocks[0]
        values = value_blocks[0]
    else:
        rows = np.concatenate(row_blocks)
        columns = np.concatenate(column_blocks)
        values = np.concatenate(value_blocks)

    # Keep the raw signed values until this point so the sidecar correction is
    # applied per connection, before duplicate coordinates are aggregated.
    _apply_histamine_correction(values, rows, photoreceptor_rows)
    graph = sparse.coo_matrix(
        (values, (rows, columns)), shape=(n, n), dtype=np.float64
    ).tocsr()
    # ``tocsr`` normally performs this already; calling both methods makes the
    # duplicate-coordinate aggregation contract explicit and harmless.
    graph.sum_duplicates()
    graph.eliminate_zeros()
    return graph


def to_sparse(value: Any) -> sparse.csr_matrix:
    """Return ``value`` as a SciPy CSR matrix.

    ``value`` may be either a matrix or the dictionary returned by
    :func:`load_connectome`.  The helper never densifies a matrix, so it is
    safe to use on the full connectome.
    """

    if isinstance(value, Mapping):
        if "W" not in value:
            raise KeyError("connectome result has no 'W' entry")
        value = value["W"]
    if isinstance(value, sparse.csr_matrix):
        return value
    if sparse.issparse(value):
        return sparse.csr_matrix(value)
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("to_sparse expects a two-dimensional matrix")
    return sparse.csr_matrix(array)


def load_connectome(
    path: Any,
    *,
    dense: bool | None = None,
    expected_n: int | None = None,
    cell_types_path: Any = None,
    apply_photoreceptor_fix: bool = True,
    dense_limit: int = _DEFAULT_DENSE_LIMIT,
    chunksize: int = _DEFAULT_CHUNK_ROWS,
) -> dict[str, Any]:
    """Load a FlyWire connectivity parquet file.

    Parameters
    ----------
    path : path-like
        Parquet file containing the documented connectivity columns, including
        the authoritative signed ``Excitatory x Connectivity`` column.
        The signed edge value is read *only* from
        ``Excitatory x Connectivity`` and is placed at
        ``W[presynaptic_index, postsynaptic_index]``.
    dense : bool or None, optional
        ``None`` selects dense storage for at most ``dense_limit`` nodes and
        CSR storage for larger graphs.  ``True`` requests dense storage but
        refuses an unsafe allocation.  ``False`` always returns CSR.
    expected_n : int, optional
        If supplied, the source-index range must describe exactly this many
        slots.  A mismatch is an error; this argument never truncates or
        reshapes the data.
    cell_types_path : path-like, optional
        Explicit consolidated cell-type CSV.  When omitted, the loader looks
        below ``/home/hatch/workspace/fly-zoo/fruit-fly``.
    apply_photoreceptor_fix : bool, optional
        Apply the exact ``R1-6``/``R7``/``R8`` histamine correction when the
        sidecar is available.  Missing sidecars are skipped gracefully.
    dense_limit : int, optional
        Maximum node count for an automatic or explicitly requested dense
        matrix.  Dense output is also capped at 256 MiB regardless of this
        value, so the full 138,639-node graph remains CSR.
    chunksize : int, optional
        Number of parquet rows read at a time.

    Returns
    -------
    dict
        A plain dictionary containing ``W``, an index-aligned ``ids`` list,
        ``n``, and an ``id_to_index`` mapping.  Missing IDs occupy ``None``
        slots in ``ids`` while their source-index edges remain usable.
    """

    if not isinstance(chunksize, (int, np.integer)) or int(chunksize) <= 0:
        raise ValueError("chunksize must be a positive integer")
    if not isinstance(dense_limit, (int, np.integer)) or int(dense_limit) < 0:
        raise ValueError("dense_limit must be a non-negative integer")
    chunksize = int(chunksize)
    dense_limit = int(dense_limit)
    if expected_n is not None:
        if not isinstance(expected_n, (int, np.integer)) or int(expected_n) < 0:
            raise ValueError("expected_n must be a non-negative integer")
        expected_n = int(expected_n)

    if apply_photoreceptor_fix:
        sidecar = (
            Path(cell_types_path)
            if cell_types_path is not None
            else _discover_cell_type_file()
        )
        photoreceptor_ids = _load_photoreceptor_ids(sidecar)
    else:
        photoreceptor_ids = set()

    ids_by_index: dict[int, Any] = {}
    index_by_id: dict[str, int] = {}
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    max_source_index = -1

    for chunk in _iter_parquet_chunks(path, list(_REQUIRED_COLUMNS), chunksize):
        if len(chunk) == 0:
            continue
        presynaptic_indices = _validate_indices(
            chunk["Presynaptic_Index"], "Presynaptic_Index"
        )
        postsynaptic_indices = _validate_indices(
            chunk["Postsynaptic_Index"], "Postsynaptic_Index"
        )
        if presynaptic_indices.size:
            max_source_index = max(
                max_source_index, int(presynaptic_indices.max())
            )
        if postsynaptic_indices.size:
            max_source_index = max(
                max_source_index, int(postsynaptic_indices.max())
            )

        _bind_ids(
            presynaptic_indices,
            chunk["Presynaptic_ID"].to_numpy(copy=False),
            ids_by_index,
            index_by_id,
        )
        _bind_ids(
            postsynaptic_indices,
            chunk["Postsynaptic_ID"].to_numpy(copy=False),
            ids_by_index,
            index_by_id,
        )

        values = _validate_values(chunk[_SIGNED_VALUE_COLUMN])
        if values.size:
            row_blocks.append(_compact_indices(presynaptic_indices))
            column_blocks.append(_compact_indices(postsynaptic_indices))
            value_blocks.append(values)

    # ``783`` names the FlyWire release, not the number of rows/neurons in
    # this export.  Derive n from the source indices; never crop or modulo
    # the real connectome to force a 783-by-783 matrix.
    n = max_source_index + 1
    if expected_n is not None and n != expected_n:
        raise ValueError(
            f"source indices imply n={n}, not expected_n={expected_n}; "
            "refusing to truncate or reshape the connectome"
        )

    ids: list[Any] = [None] * n
    for index, neuron_id in ids_by_index.items():
        if 0 <= index < n:
            ids[index] = neuron_id

    id_to_index: dict[Any, int] = {}
    for index, neuron_id in enumerate(ids):
        if neuron_id is not None:
            id_to_index[_mapping_key(neuron_id)] = index

    photoreceptor_rows = np.zeros(n, dtype=bool)
    if photoreceptor_ids:
        for index, neuron_id in enumerate(ids):
            if neuron_id is not None and _canonical_id(neuron_id) in photoreceptor_ids:
                photoreceptor_rows[index] = True

    graph = _build_csr(
        row_blocks, column_blocks, value_blocks, n, photoreceptor_rows
    )

    # Release coordinate blocks before a possible dense conversion.  In
    # particular, never call toarray() for a full-sized connectome.
    del row_blocks, column_blocks, value_blocks
    dense_bytes = n * n * np.dtype(np.float64).itemsize
    if dense is None:
        use_dense = n <= dense_limit and dense_bytes <= _MAX_DENSE_BYTES
    elif dense:
        if n > dense_limit or dense_bytes > _MAX_DENSE_BYTES:
            raise ValueError(
                f"refusing dense W for {n} nodes "
                f"({dense_bytes} bytes; limit={dense_limit} nodes/"
                f"{_MAX_DENSE_BYTES} bytes); use dense=False for the full "
                "connectome"
            )
        use_dense = True
    else:
        use_dense = False

    W: np.ndarray | sparse.csr_matrix
    if use_dense:
        W = graph.toarray()
    else:
        W = graph

    return {"W": W, "ids": ids, "n": n, "id_to_index": id_to_index}


def _smoke(
    path: Any = "/home/hatch/workspace/fly-brain/data/2025_Connectivity_783.parquet",
) -> None:
    """Run a sparse-only smoke test against the real v783 parquet export."""

    result = load_connectome(path)
    graph = to_sparse(result)
    n = int(result["n"])
    shape = graph.shape
    possible = n * n
    sparsity = (graph.nnz / possible) if possible else 0.0
    total_weight = float(graph.sum())

    # Compute incident absolute weight as sparse row/column reductions.  This
    # deliberately avoids W.toarray() and remains safe for the full graph.
    absolute_graph = graph.copy()
    absolute_graph.data = np.abs(absolute_graph.data)
    row_weight = np.asarray(absolute_graph.sum(axis=1)).reshape(-1)
    column_weight = np.asarray(absolute_graph.sum(axis=0)).reshape(-1)
    incident_weight = row_weight + column_weight
    candidates = [
        index
        for index, neuron_id in enumerate(result["ids"])
        if neuron_id is not None
    ]
    top = sorted(
        candidates,
        key=lambda index: (
            -float(incident_weight[index]),
            str(result["ids"][index]),
        ),
    )[:3]

    print(f"shape: {shape}")
    print(f"sparsity: {sparsity:.6%} ({int(graph.nnz):,} nonzero entries)")
    print(f"total weight: {total_weight:.12g}")
    print(
        "top 3 connected neuron IDs (incident absolute weight):",
        [(result["ids"][i], float(incident_weight[i])) for i in top],
    )


if __name__ == "__main__":
    _smoke()
