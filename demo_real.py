"""Condition a small sampled circuit from the real FlyWire v783 wiring.

The parquet used here contains root IDs and signed connectivity, but its
cell-type sidecar is not part of the loader result.  When no type/name
information is available, the selection below is deliberately described as a
*topology proxy*, not as a claim of biological identity:

* MBON-like targets have a high fraction of input from a mid-degree source
  population, while their own out-degree is bounded to avoid selecting the
  whole-brain hubs.
* KC-like sources project to those targets and have the characteristic
  moderate out-degree band.
* DAN-like sources are recurrent (high in-degree), selectively project to the
  sampled targets, and are split into two teaching groups by the first
  principal component of their DAN-to-target profiles.

All KC->MBON and DAN->MBON values used below are still read from the real CSR
connectome.  The proxy labels only tell the existing compartment/plasticity
APIs which sampled source group is the PAM-like or PPL1-like channel.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse

from connectome.loader import load_connectome, to_sparse
from plasticity.compartments import build_compartment_map
from plasticity.mushroom_body import (
    DopaminergicPlasticity,
    MBConfig,
    MushroomBody,
    PlasticityConfig,
)


DATA_PATH = "/home/hatch/workspace/fly-brain/data/2025_Connectivity_783.parquet"
MAX_KC = 96
MAX_DAN = 32
MAX_MBON_POOL = 64
SEED = 783

KC_PATTERN = re.compile(r"^(?:KC|Kenyon)", re.IGNORECASE)
DAN_PATTERN = re.compile(r"^(?:PAM\d|PPL1\d|DAN)", re.IGNORECASE)
MBON_PATTERN = re.compile(r"^MBON", re.IGNORECASE)


# The loader currently returns only the four documented keys, but accepting a
# few conventional metadata spellings makes the demo useful if the loader is
# extended later.
def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _type_array(result: Mapping[str, Any], n: int, ids: list[Any]) -> tuple[np.ndarray | None, str | None]:
    """Find an optional per-neuron type array without importing pandas."""

    def direct_length_array(value: Any) -> np.ndarray | None:
        try:
            array = np.asarray(value, dtype=object)
        except (TypeError, ValueError):
            return None
        if array.ndim == 1 and array.size == n:
            return array
        return None

    def from_object(value: Any) -> np.ndarray | None:
        if value is None:
            return None

        # A DataFrame-like object is handled through its small public surface;
        # the demo itself remains NumPy/SciPy-only.
        columns = getattr(value, "columns", None)
        if columns is not None:
            for name in ("primary_type", "cell_type", "type", "label", "name"):
                try:
                    if name in columns:
                        array = direct_length_array(value[name].to_numpy())
                        if array is not None:
                            return array
                except (AttributeError, TypeError, ValueError):
                    pass

        if isinstance(value, Mapping):
            # First support {source_index: type} and {root_id: type}.
            aligned: list[Any] = [None] * n
            found = 0
            for index, neuron_id in enumerate(ids):
                item = None
                try:
                    if index in value:
                        item = value[index]
                    elif neuron_id in value:
                        item = value[neuron_id]
                except (TypeError, KeyError):
                    item = None
                if item is not None:
                    aligned[index] = item
                    found += 1
            if found >= max(4, n // 100):
                return np.asarray(aligned, dtype=object)

            # Then support metadata dictionaries such as
            # {"primary_type": [...]} without assuming pandas.
            for name in ("primary_type", "cell_type", "type", "labels"):
                if name in value:
                    array = from_object(value[name])
                    if array is not None:
                        return array
            return None

        return direct_length_array(value)

    for key in (
        "cell_types",
        "cell_type",
        "types",
        "type",
        "annotations",
        "metadata",
        "neuron_types",
    ):
        if key not in result:
            continue
        array = from_object(result[key])
        if array is not None:
            return array, key
    return None, None


def _sample_indices(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=np.int64))
    if values.size <= limit:
        return values
    chosen = rng.choice(values, size=limit, replace=False)
    return np.sort(chosen.astype(np.int64))


def _top_indices(score: np.ndarray, eligible: np.ndarray, limit: int) -> np.ndarray:
    candidates = np.flatnonzero(eligible & np.isfinite(score))
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    order = candidates[np.argsort(score[candidates])[::-1]]
    return order[:limit].astype(np.int64)


def _submatrix(W: Any, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Extract a small dense block without ever densifying the full graph."""

    if sparse.issparse(W):
        return np.asarray(W[np.asarray(rows, dtype=np.int64), :][:, np.asarray(columns, dtype=np.int64)].toarray())
    return np.asarray(W)[np.ix_(rows, columns)]


def _make_compartment_map(
    W: Any,
    dan_indices: np.ndarray,
    mbon_indices: np.ndarray,
    dan_groups: list[str],
) -> tuple[Any, np.ndarray]:
    """Build the API's type-level DAN->MBON map from real sampled edges."""

    dan_indices = np.asarray(dan_indices, dtype=np.int64)
    mbon_indices = np.asarray(mbon_indices, dtype=np.int64)
    dan_values = np.abs(_submatrix(W, dan_indices, mbon_indices).astype(float, copy=False))
    edges: list[tuple[str, str, float]] = []
    for dan_row, group in enumerate(dan_groups):
        # Unique labels preserve per-neuron columns while the regular
        # expressions preserve the two API teaching channels.
        pre_type = f"{group}-sample-{dan_row:02d}"
        for mbon_col, value in enumerate(dan_values[dan_row]):
            if value > 0.0:
                edges.append((pre_type, f"MBON-sample-{mbon_col:02d}", float(value)))
    if not edges:
        raise RuntimeError("selected DAN-like sources have no sampled DAN->MBON edges")

    cmap = build_compartment_map(
        edges,
        dan_clusters={"PAM": r"^PAM-", "PPL1": r"^PPL1-"},
        mbon_pattern=r"^MBON-",
    )
    # build_compartment_map retains only columns with at least one DAN edge.
    # Reorder the real neuron indices into exactly that map order.
    name_to_index = {
        f"MBON-sample-{column:02d}": int(mbon_indices[column])
        for column in range(len(mbon_indices))
    }
    ordered = np.asarray(
        [name_to_index[name] for name in cmap.mbon_types if name in name_to_index],
        dtype=np.int64,
    )
    if ordered.size != len(cmap.mbon_types):
        raise RuntimeError("compartment map lost a sampled MBON column")
    return cmap, ordered


def _usable_map(cmap: Any) -> bool:
    valence = np.asarray(cmap.valence, dtype=float)
    gates = [np.asarray(gate, dtype=float) for gate in cmap.gates.values()]
    return bool(
        np.isfinite(valence).all()
        and np.any(valence > 0.0)
        and np.any(valence < 0.0)
        and gates
        and all(np.isfinite(gate).all() and np.any(np.abs(gate) > 0.0) for gate in gates)
    )


def _profile_groups(profiles: np.ndarray, variant: int) -> np.ndarray:
    """Split DAN-like rows into two balanced, topology-derived groups."""

    profiles = np.asarray(profiles, dtype=float)
    row_mass = profiles.sum(axis=1)
    if not np.any(row_mass > 0.0):
        return np.zeros(profiles.shape[0], dtype=np.int64)

    normalized = profiles / np.maximum(row_mass[:, None], 1e-12)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    if variant in (0, 1):
        try:
            _, singular, right = np.linalg.svd(centered, full_matrices=False)
            component = min(variant, right.shape[0] - 1)
            if singular[component] > 1e-10:
                score = centered @ right[component]
            else:
                score = row_mass
        except np.linalg.LinAlgError:
            score = row_mass
    elif variant == 2:
        # A deterministic fallback for nearly identical profiles.
        score = np.argmax(normalized, axis=1).astype(float) + row_mass / max(row_mass.max(), 1e-12)
    else:
        score = row_mass

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(score, kind="stable")
    labels = np.ones(profiles.shape[0], dtype=np.int64)
    midpoint = max(1, profiles.shape[0] // 2)
    labels[order[:midpoint]] = 0
    return labels


def _metadata_selection(
    type_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    labels = np.asarray([_text(value) for value in type_values], dtype=object)
    kc = np.flatnonzero(np.asarray([KC_PATTERN.search(label) is not None for label in labels], dtype=bool))
    mbon = np.flatnonzero(np.asarray([MBON_PATTERN.search(label) is not None for label in labels], dtype=bool))
    pam = np.flatnonzero(np.asarray([re.match(r"^PAM\d", label, re.IGNORECASE) is not None for label in labels], dtype=bool))
    ppl1 = np.flatnonzero(np.asarray([re.match(r"^PPL1\d", label, re.IGNORECASE) is not None for label in labels], dtype=bool))
    if kc.size < 2 or mbon.size < 2 or pam.size == 0 or ppl1.size == 0:
        return None

    kc = _sample_indices(kc, MAX_KC, rng)
    mbon = _sample_indices(mbon, MAX_MBON_POOL, rng)
    per_group = min(MAX_DAN // 2, pam.size, ppl1.size)
    dan = np.sort(
        np.concatenate(
            (
                _sample_indices(pam, per_group, rng),
                _sample_indices(ppl1, per_group, rng),
            )
        )
    )
    dan = _sample_indices(dan, min(MAX_DAN, dan.size), rng)
    labels_dan = [
        "PAM"
        if re.match(r"^PAM\d", _text(type_values[index]), re.IGNORECASE)
        else "PPL1"
        for index in dan
    ]
    return kc, dan, mbon, labels_dan


def _heuristic_selection(
    W: Any,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], Any, np.ndarray, dict[str, int]]:
    """Select a small MB/KC/DAN topology proxy from the sparse graph."""

    binary = W.copy()
    binary.data = np.ones_like(binary.data, dtype=float)
    out_degree = np.asarray(binary.sum(axis=1)).ravel()
    in_degree = np.asarray(binary.sum(axis=0)).ravel()

    mid_sources = (out_degree >= 20) & (out_degree <= 150)
    high_sources = out_degree > 150
    mid_inputs = np.asarray(binary[mid_sources, :].sum(axis=0)).ravel()
    high_inputs = np.asarray(binary[high_sources, :].sum(axis=0)).ravel()
    mid_fraction = mid_inputs / (in_degree + 1.0)

    # The high-input-support term avoids tiny, purely feed-forward nodes while
    # keeping the score from being dominated by global hubs.
    m_eligible = (
        (in_degree >= 100)
        & (out_degree >= 50)
        & (out_degree < 1200)
        & (mid_inputs >= 20)
        & (high_inputs >= 3)
    )
    m_score = mid_fraction + 1e-3 * np.log1p(high_inputs)
    m_pool = _top_indices(m_score, m_eligible, MAX_MBON_POOL)
    if m_pool.size < 16:
        relaxed = (in_degree >= 30) & (out_degree >= 20) & (out_degree < 1600) & (mid_inputs >= 5)
        m_pool = _top_indices(m_score, relaxed, MAX_MBON_POOL)
    if m_pool.size < 4:
        raise RuntimeError("could not find MBON-like targets in the sparse graph")

    target_graph = W[:, m_pool]
    target_count = np.asarray(binary[:, m_pool].sum(axis=1)).ravel()
    positive_graph = target_graph.copy()
    positive_graph.data = (positive_graph.data > 0.0).astype(float)
    positive_count = np.asarray(positive_graph.sum(axis=1)).ravel()

    excluded = np.zeros(n, dtype=bool)
    excluded[m_pool] = True
    kc_eligible = (
        (target_count > 0)
        & (out_degree >= 20)
        & (out_degree <= 180)
        & (in_degree < 350)
        & (positive_count > 0)
        & ~excluded
    )
    kc_score = target_count * np.exp(-np.abs(out_degree - 90.0) / 60.0)
    kc_score *= 0.5 + positive_count / np.maximum(target_count, 1.0)
    kc = _top_indices(kc_score, kc_eligible, MAX_KC)
    if kc.size < 8:
        kc = _top_indices(
            target_count,
            (target_count > 0) & ~excluded,
            min(MAX_KC, int(np.count_nonzero(target_count > 0))),
        )
    if kc.size < 4:
        raise RuntimeError("could not find KC-like sources in the sparse graph")

    excluded[kc] = True
    dan_score = (target_count / (out_degree + 1.0)) * np.log1p(in_degree + 1.0)
    dan_eligible = (
        (target_count > 0)
        & (out_degree >= 50)
        & (in_degree >= 120)
        & ~excluded
    )
    dan_pool = _top_indices(dan_score, dan_eligible, 96)
    if dan_pool.size < 8:
        dan_pool = _top_indices(
            dan_score,
            (target_count > 0) & ~excluded,
            min(96, int(np.count_nonzero(target_count > 0))),
        )
    if dan_pool.size < 4:
        raise RuntimeError("could not find DAN-like sources in the sparse graph")

    # Try several deterministic partitions.  The first principal component is
    # normally best; the fallbacks make the demo robust to a degenerate graph.
    best: tuple[Any, np.ndarray, np.ndarray, list[str]] | None = None
    best_score = -1
    for pool_size in (min(96, dan_pool.size), min(64, dan_pool.size), min(32, dan_pool.size)):
        pool = dan_pool[:pool_size]
        profiles = np.abs(_submatrix(W, pool, m_pool))
        for variant in range(4):
            groups_all = _profile_groups(profiles, variant)
            chosen: list[int] = []
            chosen_groups: list[str] = []
            per_group = MAX_DAN // 2
            for group_value, group_name in ((0, "PAM"), (1, "PPL1")):
                members = np.flatnonzero(groups_all == group_value)
                members = members[np.argsort(dan_score[pool[members]])[::-1]]
                take = members[:per_group]
                chosen.extend(pool[take].tolist())
                chosen_groups.extend([group_name] * len(take))
            if not chosen:
                continue
            chosen_array = np.asarray(chosen, dtype=np.int64)
            # Preserve score order while keeping group labels paired.
            order = np.argsort(chosen_array)
            chosen_array = chosen_array[order]
            chosen_groups = [chosen_groups[i] for i in order]
            if chosen_array.size < min(MAX_DAN, 8):
                continue
            try:
                cmap, ordered_mbon = _make_compartment_map(
                    W, chosen_array, m_pool, chosen_groups
                )
            except RuntimeError:
                continue
            score = int(np.count_nonzero(cmap.valence > 0)) + int(
                np.count_nonzero(cmap.valence < 0)
            )
            if score > best_score:
                best_score = score
                best = (cmap, chosen_array, ordered_mbon, chosen_groups)
            if _usable_map(cmap):
                return kc, chosen_array, chosen_groups, cmap, ordered_mbon, m_pool, {
                    "m_pool": int(m_pool.size),
                    "candidate_dan": int(dan_pool.size),
                }

    if best is None:
        raise RuntimeError("could not build a two-compartment DAN->MBON map")
    cmap, chosen_array, ordered_mbon, chosen_groups = best
    return kc, chosen_array, chosen_groups, cmap, ordered_mbon, m_pool, {
        "m_pool": int(m_pool.size),
        "candidate_dan": int(dan_pool.size),
    }


def _prepare_learning_weights(
    W: Any,
    kc_indices: np.ndarray,
    mbon_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    raw = _submatrix(W, kc_indices, mbon_indices).astype(float, copy=False)
    positive = np.maximum(raw, 0.0)
    if np.count_nonzero(positive) < max(4, int(0.05 * raw.size)):
        # A topology proxy can include a non-cholinergic source.  Preserve the
        # real edge magnitudes rather than inventing a synthetic projection.
        positive = np.abs(raw)
    if not np.any(positive > 0.0):
        raise RuntimeError("sampled KC->MBON block has no finite positive weights")
    scale = float(np.mean(positive[positive > 0.0]))
    weights = positive / scale
    strength = positive.sum(axis=1)
    if not np.any(strength > 0.0):
        strength = np.abs(raw).sum(axis=1)
    return raw, positive, scale, strength


def _run_conditioning(
    W: Any,
    kc_indices: np.ndarray,
    mbon_indices: np.ndarray,
    cmap: Any,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[float]]:
    raw, positive, scale, strength = _prepare_learning_weights(
        W, kc_indices, mbon_indices
    )
    weights = positive / scale
    nonzero_rows = np.flatnonzero(strength > 0.0)
    if nonzero_rows.size == 0:
        raise RuntimeError("no sampled KC has an odor drive")
    active_count = min(nonzero_rows.size, max(8, int(np.ceil(0.25 * len(kc_indices)))))
    active = nonzero_rows[np.argsort(strength[nonzero_rows])[-active_count:]]
    odor_drive = np.zeros(len(kc_indices), dtype=float)
    odor_drive[active] = strength[active] / max(float(strength[active].max()), 1e-12)

    mb = MushroomBody(weights, cmap, cfg=MBConfig(apl_enabled=False))
    plasticity = DopaminergicPlasticity(
        mb,
        PlasticityConfig(lr=0.12, tau_eligibility=30.0, recovery=0.0),
    )
    before = mb.W_kc_mbon.copy()
    trial_moves: list[float] = []
    for _ in range(5):
        plasticity.reset()
        kc_activity = mb.encode(odor_drive)
        for _ in range(3):
            plasticity.observe(kc_activity)
        trial_moves.append(float(plasticity.teach(reward=1.0, punishment=0.0)))

    after = mb.W_kc_mbon.copy()
    return mb, before, after, raw, positive, scale, trial_moves


def main() -> None:
    started = time.perf_counter()
    print(f"Loading real FlyWire wiring: {DATA_PATH}")
    result = load_connectome(DATA_PATH)
    keys = sorted(str(key) for key in result)
    print("Loader keys at runtime:", ", ".join(keys))

    W = to_sparse(result)
    n = int(result["n"])
    ids = result.get("ids", [None] * n)
    print(f"Connectome: n={n:,}, shape={W.shape}, sparse_nnz={getattr(W, 'nnz', 0):,}")

    type_values, type_key = _type_array(result, n, ids)
    rng = np.random.default_rng(SEED)
    pattern_sources: list[tuple[str, np.ndarray]] = []
    if type_values is not None:
        pattern_sources.append((f"loader field {type_key}", type_values))
    pattern_sources.append(("neuron IDs/names", np.asarray(ids, dtype=object)))

    metadata_selection = None
    pattern_source = "none"
    for source_name, source_values in pattern_sources:
        labels = [_text(value) for value in source_values]
        pattern_counts = {
            "KC": sum(KC_PATTERN.search(label) is not None for label in labels),
            "DAN": sum(DAN_PATTERN.search(label) is not None for label in labels),
            "MBON": sum(MBON_PATTERN.search(label) is not None for label in labels),
        }
        print(f"Pattern source {source_name}: {pattern_counts}")
        candidate_selection = _metadata_selection(source_values, rng)
        if candidate_selection is not None:
            metadata_selection = candidate_selection
            pattern_source = source_name
            break

    if metadata_selection is not None:
        kc_indices, dan_indices, mbon_pool, dan_groups = metadata_selection
        try:
            cmap, mbon_indices = _make_compartment_map(
                W, dan_indices, mbon_pool, dan_groups
            )
            if not _usable_map(cmap):
                raise RuntimeError("metadata map did not yield both valences")
        except RuntimeError:
            metadata_selection = None

    if metadata_selection is not None:
        selection_mode = f"cell-type/name patterns ({pattern_source})"
        candidate_mbon = int(mbon_pool.size)
        candidate_dan = int(dan_indices.size)
        details = {"m_pool": candidate_mbon, "candidate_dan": candidate_dan}
    else:
        selection_mode = "connectivity heuristic (no usable cell types)"
        (
            kc_indices,
            dan_indices,
            dan_groups,
            cmap,
            mbon_indices,
            mbon_pool,
            details,
        ) = _heuristic_selection(W, n)
        candidate_mbon = int(mbon_pool.size)
        candidate_dan = int(dan_indices.size)

    kc_indices = np.asarray(kc_indices, dtype=np.int64)
    dan_indices = np.asarray(dan_indices, dtype=np.int64)
    mbon_indices = np.asarray(mbon_indices, dtype=np.int64)
    subcircuit_size = int(kc_indices.size + dan_indices.size + mbon_indices.size)
    if subcircuit_size > 240:
        raise RuntimeError(f"sampled subcircuit is too large: {subcircuit_size}")
    if not (len(kc_indices) and len(dan_indices) and len(mbon_indices)):
        raise RuntimeError("one of the sampled compartments is empty")

    print(f"Selection mode: {selection_mode}")
    if selection_mode.startswith("connectivity"):
        print(
            "Heuristic: mid-degree input hubs -> moderate out-degree sources -> "
            "recurrent selective DAN-like sources; DAN groups split by target-profile PC1"
        )
    print(
        f"Candidate pools: KC<={MAX_KC}, DAN<={MAX_DAN}, MBON<={candidate_mbon}; "
        f"DAN candidates={candidate_dan}"
    )
    print(f"Selection details: {details}")
    print(f"Subcircuit size: {subcircuit_size} neurons")
    print(
        "Neuron counts by type (sampled labels): "
        f"KC={len(kc_indices)}, DAN={len(dan_indices)}, MBON={len(mbon_indices)}"
    )
    print(
        "DAN teaching groups: "
        f"PAM-like={sum(group == 'PAM' for group in dan_groups)}, "
        f"PPL1-like={sum(group == 'PPL1' for group in dan_groups)}"
    )

    (
        mb,
        before,
        after,
        raw,
        positive,
        scale,
        trial_moves,
    ) = _run_conditioning(W, kc_indices, mbon_indices, cmap)

    valence = np.asarray(cmap.valence, dtype=float)
    supported = positive.sum(axis=0) > 0.0
    approach = (valence > 0.0) & supported
    avoidance = (valence < 0.0) & supported
    if not approach.any():
        approach = valence > 0.0
    if not avoidance.any():
        avoidance = valence < 0.0
    if not approach.any() or not avoidance.any():
        raise RuntimeError("compartment map did not retain both learning polarities")

    delta = after - before
    real_after = after * scale
    real_delta = delta * scale
    approach_change = float(np.mean(real_delta[:, approach]))
    avoidance_change = float(np.mean(real_delta[:, avoidance]))
    approach_before = float(np.mean(positive[:, approach]))
    approach_after = float(np.mean(real_after[:, approach]))
    avoidance_before = float(np.mean(positive[:, avoidance]))
    avoidance_after = float(np.mean(real_after[:, avoidance]))
    if not np.isfinite(after).all():
        raise RuntimeError("plasticity produced non-finite weights")

    print(
        "Sampled root IDs: "
        f"KC={[_text(ids[i]) for i in kc_indices[:3]]}, "
        f"DAN={[_text(ids[i]) for i in dan_indices[:3]]}, "
        f"MBON={[_text(ids[i]) for i in mbon_indices[:3]]}"
    )
    print(
        f"KC->MBON block: shape={positive.shape}, raw_nnz={np.count_nonzero(raw)}, "
        f"positive_fraction={np.count_nonzero(positive) / max(np.count_nonzero(raw), 1):.3f}, "
        f"normalization={scale:.6g}"
    )
    active_odor_rows = min(
        int(np.count_nonzero(positive.sum(axis=1) > 0.0)),
        max(8, int(np.ceil(0.25 * len(kc_indices)))),
    )
    print(
        f"Odor drive: normalized top-{active_odor_rows} real KC projection rows; "
        "five paired reward trials"
    )
    print("Weight change before/after (real-edge units):")
    print(
        f"  all mean: {positive.mean():+.6f} -> {float(np.mean(real_after)):+.6f} "
        f"(delta={float(np.mean(real_delta)):+.6f})"
    )
    print(
        f"  approach/positive-valence mean: {approach_before:+.6f} -> {approach_after:+.6f} "
        f"(delta={approach_change:+.6f})"
    )
    print(
        f"  avoidance/negative-valence mean: {avoidance_before:+.6f} -> {avoidance_after:+.6f} "
        f"(delta={avoidance_change:+.6f})"
    )
    for trial, move in enumerate(trial_moves, start=1):
        print(f"  reward trial {trial}: absolute weight movement={move * scale:.6f}")

    if not (approach_change > 0.0 and avoidance_change < 0.0):
        raise RuntimeError(
            "reward did not move approach weights up and avoidance weights down: "
            f"{approach_change:+.6g}, {avoidance_change:+.6g}"
        )

    elapsed = time.perf_counter() - started
    print(f"Elapsed: {elapsed:.2f} seconds")
    if elapsed >= 180.0:
        print("WARNING: runtime exceeded three minutes")
    print("DONE")
    print(f"Loaded the real v783 sparse wiring and sampled {subcircuit_size} neurons.")
    print(f"Five reward trials ran through DopaminergicPlasticity on {positive.shape[0]} sampled KCs.")
    print(f"Approach weights increased by {approach_change:+.6f}; avoidance weights changed by {avoidance_change:+.6f}.")


if __name__ == "__main__":
    main()
