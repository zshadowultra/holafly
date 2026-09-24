"""Condition the full-sized mushroom body on real FlyWire v783 wiring.

The loader remains the source of the graph and root-ID alignment.  When its
local consolidated cell-type sidecar is available, real KC/DAN/MBON labels are
used.  Otherwise, the selection below deliberately scales the topology-proxy
heuristic from ``demo_real.py`` to roughly 2,000 KCs, 150 DANs, and 48 MBONs;
those fallback names are proxies, not biological identity claims.

All full-connectome operations are sparse.  Only the selected KC->MBON block
(about 2,000 by 48) is made dense because the existing plasticity API accepts
a dense weight matrix.  Reward trials are folded into one vectorized
eligibility trace, so there is no Python loop over trials or synapses.
"""

from __future__ import annotations

import csv
import gzip
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from connectome.loader import (
    _discover_cell_type_file,
    load_connectome,
    to_sparse,
)
from plasticity.compartments import CompartmentMap
from plasticity.mushroom_body import (
    DopaminergicPlasticity,
    MBConfig,
    MushroomBody,
    PlasticityConfig,
)


DATA_PATH = "/home/hatch/workspace/fly-brain/data/2025_Connectivity_783.parquet"
MAX_KC = 2_000
MAX_DAN = 150
MAX_MBON = 48
MAX_MBON_POOL = 64
DAN_CANDIDATE_POOL = 300
REWARD_TRIALS = 5
PRESENTATIONS_PER_TRIAL = 3
RUNTIME_LIMIT_SECONDS = 300.0
SEED = 783

KC_PATTERN = re.compile(r"^(?:KC|Kenyon)", re.IGNORECASE)
DAN_PATTERN = re.compile(r"^(?:PAM\d|PPL1\d|DAN)", re.IGNORECASE)
PAM_PATTERN = re.compile(r"^PAM\d", re.IGNORECASE)
PPL1_PATTERN = re.compile(r"^PPL1\d", re.IGNORECASE)
MBON_PATTERN = re.compile(r"^MBON", re.IGNORECASE)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _canonical(value: Any) -> str:
    """Normalize numeric/string root IDs for sidecar joins without pandas."""

    return _text(value)


def _read_type_sidecar(path: Path | None) -> dict[str, str] | None:
    """Read root-ID annotations with only the Python standard library."""

    if path is None or not path.is_file():
        return None
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return None
            fields = {name.strip(): name for name in reader.fieldnames if name}
            id_field = next(
                (fields[name] for name in ("root_id", "cell_id", "id") if name in fields),
                None,
            )
            type_field = next(
                (
                    fields[name]
                    for name in ("primary_type", "cell_type", "type", "class")
                    if name in fields
                ),
                None,
            )
            if id_field is None or type_field is None:
                return None
            annotations: dict[str, str] = {}
            for row in reader:
                root_id = _canonical(row.get(id_field))
                cell_type = _text(row.get(type_field))
                if root_id and cell_type:
                    annotations[root_id] = cell_type
            return annotations or None
    except (OSError, EOFError, csv.Error, UnicodeError):
        return None


def _aligned_type_labels(
    result: Mapping[str, Any], sidecar: Mapping[str, str] | None
) -> np.ndarray | None:
    if sidecar is None:
        return None
    labels = [
        sidecar.get(_canonical(neuron_id), "")
        for neuron_id in result.get("ids", ())
    ]
    array = np.asarray(labels, dtype=object)
    return array if array.size == int(result["n"]) else None


def _matching(labels: np.ndarray, pattern: re.Pattern[str]) -> np.ndarray:
    return np.asarray(
        [pattern.search(_text(label)) is not None for label in labels],
        dtype=bool,
    )


def _evenly_spaced(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=np.int64))
    if values.size <= limit:
        return values
    positions = np.linspace(0, values.size - 1, num=limit, dtype=np.int64)
    return np.unique(values[positions])


def _top_indices(score: np.ndarray, eligible: np.ndarray, limit: int) -> np.ndarray:
    candidates = np.flatnonzero(eligible & np.isfinite(score))
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    order = candidates[np.argsort(score[candidates], kind="stable")[::-1]]
    return np.asarray(order[:limit], dtype=np.int64)


def _block(W: sparse.csr_matrix, rows: np.ndarray, columns: np.ndarray) -> sparse.csr_matrix:
    """Slice a CSR submatrix without densifying the full connectome."""

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    return sparse.csr_matrix(W[rows, :][:, columns])


def _binary_degrees(W: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    binary = W.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float64)
    out_degree = np.asarray(binary.sum(axis=1)).ravel()
    in_degree = np.asarray(binary.sum(axis=0)).ravel()
    return out_degree, in_degree


def _positive_count(block: sparse.csr_matrix) -> np.ndarray:
    positive = block.copy()
    positive.data = np.maximum(positive.data, 0.0)
    positive.eliminate_zeros()
    return np.asarray(positive.getnnz(axis=0)).ravel()


def _compartment_map_from_real_block(
    dan_block: sparse.csr_matrix,
    dan_groups: list[str],
    mbon_indices: np.ndarray,
) -> CompartmentMap:
    """Build the same two-cluster gates as compartments.py using sparse ops.

    ``build_compartment_map`` consumes a Python edge list.  The full-body
    version computes the identical row-normalised cluster drives directly from
    the real CSR DAN->MBON block, avoiding a per-synapse Python loop.
    """

    if dan_block.shape[0] != len(dan_groups):
        raise ValueError("DAN block rows and teaching groups are misaligned")
    absolute = abs(dan_block)
    row_mass = np.asarray(absolute.sum(axis=1)).ravel()
    scale = np.divide(1.0, row_mass, out=np.zeros_like(row_mass), where=row_mass > 0.0)
    normalised = sparse.diags(scale, format="csr") @ absolute
    C = normalised.T.toarray()

    groups = np.asarray(dan_groups, dtype=object)
    cluster_drive: dict[str, np.ndarray] = {}
    for name in ("PAM", "PPL1"):
        cluster_drive[name] = np.asarray(
            C[:, groups == name].sum(axis=1)
        ).ravel()

    pam = cluster_drive["PAM"]
    ppl1 = cluster_drive["PPL1"]
    dominant: list[str | None] = []
    for pam_value, ppl1_value in zip(pam, ppl1, strict=True):
        if pam_value <= 0.0 and ppl1_value <= 0.0:
            dominant.append(None)
        else:
            dominant.append("PPL1" if ppl1_value > pam_value else "PAM")

    gates = {
        "PAM": np.where(pam > ppl1, pam, 0.0),
        "PPL1": np.where(ppl1 > pam, ppl1, 0.0),
    }
    valence = np.sign(ppl1 - pam)
    return CompartmentMap(
        mbon_types=[f"MBON-{index:06d}" for index in mbon_indices],
        dan_types=[f"DAN-{row:03d}" for row in range(dan_block.shape[0])],
        dan_cluster=list(dan_groups),
        C=C,
        cluster_drive=cluster_drive,
        dominant_cluster=dominant,
        gates=gates,
        valence=valence,
    )


def _slice_compartment_map(cmap: CompartmentMap, columns: np.ndarray) -> CompartmentMap:
    columns = np.asarray(columns, dtype=np.int64)
    return CompartmentMap(
        mbon_types=[cmap.mbon_types[index] for index in columns],
        dan_types=list(cmap.dan_types),
        dan_cluster=list(cmap.dan_cluster),
        C=cmap.C[columns, :],
        cluster_drive={
            name: np.asarray(values[columns]) for name, values in cmap.cluster_drive.items()
        },
        dominant_cluster=[cmap.dominant_cluster[index] for index in columns],
        gates={
            name: np.asarray(values[columns]) for name, values in cmap.gates.items()
        },
        valence=np.asarray(cmap.valence[columns]),
    )


def _metadata_selection(
    W: sparse.csr_matrix,
    labels: np.ndarray,
    out_degree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, int]]:
    """Select labelled populations and retain all when they fit the target size."""

    all_kc = np.flatnonzero(_matching(labels, KC_PATTERN))
    all_mbon = np.flatnonzero(_matching(labels, MBON_PATTERN))
    pam = np.flatnonzero(_matching(labels, PAM_PATTERN))
    ppl1 = np.flatnonzero(_matching(labels, PPL1_PATTERN))
    labelled_dan = np.unique(np.concatenate((pam, ppl1)))
    if all_kc.size < 100 or all_mbon.size < 20 or pam.size == 0 or ppl1.size == 0:
        raise RuntimeError("cell-type sidecar has insufficient KC/DAN/MBON matches")

    mbon_pool = _evenly_spaced(all_mbon, MAX_MBON_POOL)
    target_support = np.asarray(W[:, mbon_pool].getnnz(axis=1)).ravel()
    kc_score = target_support * np.exp(-np.abs(out_degree - 90.0) / 80.0)
    kc_eligible = np.isin(all_kc, mbon_pool, invert=True) & (target_support > 0)
    kc = _top_indices(kc_score, kc_eligible, MAX_KC)
    if kc.size < 100:
        kc = _evenly_spaced(all_kc, min(MAX_KC, all_kc.size))

    excluded = np.zeros(int(W.shape[0]), dtype=bool)
    excluded[mbon_pool] = True
    excluded[kc] = True
    pam = pam[~excluded[pam]]
    ppl1 = ppl1[~excluded[ppl1]]
    if pam.size == 0 or ppl1.size == 0:
        raise RuntimeError("cell-type sidecar has no disjoint PAM/PPL1 teaching cells")

    dan_limit = min(MAX_DAN, pam.size + ppl1.size)
    per_group = min(dan_limit // 2, pam.size, ppl1.size)
    chosen_pam = _evenly_spaced(pam, per_group)
    chosen_ppl1 = _evenly_spaced(ppl1, per_group)
    spare = dan_limit - chosen_pam.size - chosen_ppl1.size
    if spare > 0 and pam.size > per_group:
        extra = np.setdiff1d(pam, chosen_pam, assume_unique=True)
        chosen_pam = np.sort(
            np.concatenate((chosen_pam, _evenly_spaced(extra, min(spare, extra.size))))
        )
        spare = dan_limit - chosen_pam.size - chosen_ppl1.size
    if spare > 0 and ppl1.size > per_group:
        extra = np.setdiff1d(ppl1, chosen_ppl1, assume_unique=True)
        chosen_ppl1 = np.sort(
            np.concatenate((chosen_ppl1, _evenly_spaced(extra, min(spare, extra.size))))
        )

    dan = np.sort(np.concatenate((chosen_pam, chosen_ppl1)))
    dan_groups = [
        "PAM" if PAM_PATTERN.search(_text(labels[index])) is not None else "PPL1"
        for index in dan
    ]
    return (
        np.asarray(kc, dtype=np.int64),
        np.asarray(dan, dtype=np.int64),
        dan_groups,
        np.asarray(mbon_pool, dtype=np.int64),
        {
            "labelled_kc": int(all_kc.size),
            "labelled_dan": int(labelled_dan.size),
            "labelled_mbon": int(all_mbon.size),
            "pam": int(pam.size),
            "ppl1": int(ppl1.size),
        },
    )


def _profile_groups(profiles: np.ndarray, variant: int) -> np.ndarray:
    """Split DAN-like profiles into two balanced topology-derived groups."""

    profiles = np.asarray(profiles, dtype=np.float64)
    row_mass = profiles.sum(axis=1)
    if not np.any(row_mass > 0.0):
        return np.zeros(profiles.shape[0], dtype=np.int64)
    normalised = profiles / np.maximum(row_mass[:, None], 1e-12)
    centred = normalised - normalised.mean(axis=0, keepdims=True)

    if variant in (0, 1):
        try:
            _, singular, right = np.linalg.svd(centred, full_matrices=False)
            component = min(variant, right.shape[0] - 1)
            score = (
                centred @ right[component]
                if singular[component] > 1e-10
                else row_mass
            )
        except np.linalg.LinAlgError:
            score = row_mass
    elif variant == 2:
        score = np.argmax(normalised, axis=1).astype(float) + row_mass / max(
            row_mass.max(), 1e-12
        )
    else:
        score = row_mass

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    midpoint = max(1, profiles.shape[0] // 2)
    groups = np.ones(profiles.shape[0], dtype=np.int64)
    groups[np.argsort(score, kind="stable")[:midpoint]] = 0
    return groups


def _heuristic_selection(
    W: sparse.csr_matrix,
    n: int,
    out_degree: np.ndarray,
    in_degree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, CompartmentMap, dict[str, int]]:
    """Scale demo_real.py's topology proxies to the full mushroom body.

    This fallback is used only when the loader cannot discover a usable local
    ``consolidated_cell_types.csv.gz``.  It deliberately scales that file's
    degree/support thresholds rather than claiming that the selected neurons
    are biologically labelled KCs, DANs, or MBONs.
    """

    mid_sources = (out_degree >= 20) & (out_degree <= 150)
    high_sources = out_degree > 150
    mid_inputs = np.asarray(W[mid_sources, :].sum(axis=0)).ravel()
    high_inputs = np.asarray(W[high_sources, :].sum(axis=0)).ravel()
    binary = W.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float64)
    binary_mid_inputs = np.asarray(binary[mid_sources, :].sum(axis=0)).ravel()
    binary_high_inputs = np.asarray(binary[high_sources, :].sum(axis=0)).ravel()
    mid_fraction = binary_mid_inputs / (in_degree + 1.0)

    mbon_eligible = (
        (in_degree >= 100)
        & (out_degree >= 50)
        & (out_degree < 1_200)
        & (binary_mid_inputs >= 20)
        & (binary_high_inputs >= 3)
    )
    mbon_score = mid_fraction + 1e-3 * np.log1p(binary_high_inputs)
    mbon_pool = _top_indices(mbon_score, mbon_eligible, MAX_MBON_POOL)
    if mbon_pool.size < MAX_MBON:
        relaxed = (
            (in_degree >= 30)
            & (out_degree >= 20)
            & (out_degree < 1_600)
            & (binary_mid_inputs >= 5)
        )
        mbon_pool = _top_indices(mbon_score, relaxed, MAX_MBON_POOL)
    if mbon_pool.size < 30:
        raise RuntimeError("could not find at least 30 MBON-like targets")

    target_graph = W[:, mbon_pool]
    target_count = np.asarray(binary[:, mbon_pool].sum(axis=1)).ravel()
    positive_target_graph = target_graph.copy()
    positive_target_graph.data = (positive_target_graph.data > 0.0).astype(float)
    positive_targets = np.asarray(positive_target_graph.sum(axis=1)).ravel()
    excluded = np.zeros(n, dtype=bool)
    excluded[mbon_pool] = True

    kc_eligible = (
        (target_count > 0)
        & (out_degree >= 20)
        & (out_degree <= 180)
        & (in_degree < 350)
        & (positive_targets > 0)
        & ~excluded
    )
    kc_score = target_count * np.exp(-np.abs(out_degree - 90.0) / 60.0)
    kc_score *= 0.5 + positive_targets / np.maximum(target_count, 1.0)
    kc = _top_indices(kc_score, kc_eligible, MAX_KC)
    if kc.size < 500:
        fallback = (target_count > 0) & (positive_targets > 0) & ~excluded
        kc = _top_indices(target_count, fallback, MAX_KC)
    if kc.size < 500:
        raise RuntimeError("could not find enough KC-like sources")

    excluded[kc] = True
    dan_score = (target_count / (out_degree + 1.0)) * np.log1p(in_degree + 1.0)
    dan_eligible = (
        (target_count > 0)
        & (out_degree >= 50)
        & (in_degree >= 120)
        & ~excluded
    )
    dan_pool = _top_indices(dan_score, dan_eligible, DAN_CANDIDATE_POOL)
    if dan_pool.size < 100:
        relaxed = (target_count > 0) & (positive_targets > 0) & ~excluded
        dan_pool = _top_indices(dan_score, relaxed, DAN_CANDIDATE_POOL)
    if dan_pool.size < 100:
        raise RuntimeError("could not find enough DAN-like sources")

    best: tuple[np.ndarray, list[str], CompartmentMap] | None = None
    best_score = (-1, -1)
    pool_sizes = sorted(
        {min(len(dan_pool), size) for size in (DAN_CANDIDATE_POOL, 240, 180, 150)},
        reverse=True,
    )
    for pool_size in pool_sizes:
        pool = dan_pool[:pool_size]
        profiles = abs(_block(W, pool, mbon_pool).toarray())
        for variant in range(4):
            split = _profile_groups(profiles, variant)
            per_group = min(MAX_DAN // 2, int(np.count_nonzero(split == 0)), int(np.count_nonzero(split == 1)))
            chosen: list[int] = []
            groups: list[str] = []
            for value, name in ((0, "PAM"), (1, "PPL1")):
                members = np.flatnonzero(split == value)
                members = members[
                    np.argsort(dan_score[pool[members]], kind="stable")[::-1]
                ]
                take = members[:per_group]
                chosen.extend(pool[take].tolist())
                groups.extend([name] * take.size)
            if not chosen:
                continue
            dan = np.sort(np.asarray(chosen, dtype=np.int64))
            groups = [groups[position] for position in np.argsort(chosen)]
            cmap = _compartment_map_from_real_block(
                _block(W, dan, mbon_pool), groups, mbon_pool
            )
            approach = int(np.count_nonzero(cmap.valence > 0.0))
            avoidance = int(np.count_nonzero(cmap.valence < 0.0))
            score = (min(approach, avoidance), approach + avoidance)
            if score > best_score:
                best_score = score
                best = (dan, groups, cmap)
            if min(approach, avoidance) >= 12:
                return kc, dan, groups, mbon_pool, cmap, {
                    "mbon_candidates": int(mbon_pool.size),
                    "dan_candidates": int(dan_pool.size),
                    "profile_variant": variant,
                }

    if best is None or best_score[0] <= 0:
        raise RuntimeError("could not derive both reward and avoidance compartments")
    dan, groups, cmap = best
    return kc, dan, groups, mbon_pool, cmap, {
        "mbon_candidates": int(mbon_pool.size),
        "dan_candidates": int(dan_pool.size),
        "profile_variant": -1,
    }


def _select_supported_mbon(
    cmap: CompartmentMap,
    kc_pool_block: sparse.csr_matrix,
    mbon_pool: np.ndarray,
) -> tuple[np.ndarray, CompartmentMap]:
    """Keep up to 48 real targets, balancing both DAN-derived polarities."""

    kc_support = _positive_count(kc_pool_block)
    drive = cmap.cluster_drive["PAM"] + cmap.cluster_drive["PPL1"]
    eligible = (kc_support > 0) & (drive > 0.0) & (cmap.valence != 0.0)
    approach = np.flatnonzero(eligible & (cmap.valence > 0.0))
    avoidance = np.flatnonzero(eligible & (cmap.valence < 0.0))
    if approach.size == 0 or avoidance.size == 0:
        raise RuntimeError("selected DAN groups do not support both MBON polarities")

    total = int(np.count_nonzero(eligible))
    target = min(MAX_MBON, total)
    if target < 30:
        raise RuntimeError(f"only {target} MBON-like targets have real KC support")
    approach_take = min(approach.size, max(1, target // 2))
    avoidance_take = min(avoidance.size, target - approach_take)
    approach_take = min(approach.size, target - avoidance_take)
    score = drive / max(float(drive.max()), 1e-12) + np.log1p(kc_support) / max(
        float(np.log1p(kc_support).max()), 1e-12
    )
    chosen = np.concatenate(
        (
            _top_indices(score, np.isin(np.arange(cmap.valence.size), approach), approach_take),
            _top_indices(score, np.isin(np.arange(cmap.valence.size), avoidance), avoidance_take),
        )
    )
    remaining = target - chosen.size
    if remaining > 0:
        available = np.setdiff1d(
            np.flatnonzero(eligible), chosen, assume_unique=False
        )
        chosen = np.concatenate((chosen, _top_indices(score, np.isin(np.arange(cmap.valence.size), available), remaining)))
    columns = np.sort(chosen.astype(np.int64))
    return mbon_pool[columns], _slice_compartment_map(cmap, columns)


def _prepare_learning_block(
    W: sparse.csr_matrix,
    kc: np.ndarray,
    mbon: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, float]:
    raw = _block(W, kc, mbon)
    raw.eliminate_zeros()
    positive = raw.copy()
    positive.data = np.maximum(positive.data, 0.0)
    positive.eliminate_zeros()
    if positive.nnz < max(4, int(0.02 * raw.shape[0])):
        # Preserve real edge locations and magnitudes; the absolute fallback
        # follows demo_real.py if signed products hide an excitatory KC block.
        positive = abs(raw)
    if positive.nnz == 0:
        raise RuntimeError("real KC->MBON block has no finite nonzero weights")
    scale = float(positive.data.mean())
    dense = np.asarray(positive.toarray() / scale, dtype=np.float64)
    strength = np.asarray(positive.sum(axis=1)).ravel()
    return raw, positive, dense, scale


def _run_reward_trials(
    weights: np.ndarray,
    cmap: CompartmentMap,
    strength: np.ndarray,
) -> tuple[MushroomBody, np.ndarray, np.ndarray, float, int]:
    """Run five reward trials as one exact, vectorized eligibility aggregate."""

    nonzero = np.flatnonzero(strength > 0.0)
    if nonzero.size == 0:
        raise RuntimeError("no labelled/proxy KC has an odor drive")
    active_count = min(nonzero.size, max(16, int(np.ceil(0.05 * weights.shape[0]))))
    active = nonzero[np.argsort(strength[nonzero], kind="stable")[-active_count:]]
    odor_drive = np.zeros(weights.shape[0], dtype=np.float64)
    odor_drive[active] = strength[active] / max(float(strength[active].max()), 1e-12)

    mb = MushroomBody(weights, cmap, cfg=MBConfig(apl_enabled=False))
    plasticity = DopaminergicPlasticity(
        mb,
        PlasticityConfig(lr=0.08, tau_eligibility=30.0, recovery=0.0),
    )
    plasticity.reset()
    activity = mb.encode(odor_drive)
    if not np.any(activity > 0.0):
        activity = odor_drive.copy()

    # For each reset trial, three observations produce
    # activity*(1 + decay + decay**2).  Because the teaching update is linear
    # in eligibility and reward before clipping, summing the five trial traces
    # and passing reward=5 is their exact unclipped aggregate.  This vectorizes
    # the trial loop and invokes the real DopaminergicPlasticity teach path.
    decay = float(np.exp(-1.0 / plasticity.cfg.tau_eligibility))
    trial_trace = activity * sum(decay**step for step in range(PRESENTATIONS_PER_TRIAL))
    plasticity.eligibility[:] = float(REWARD_TRIALS) * trial_trace
    movement = plasticity.teach(reward=float(REWARD_TRIALS), punishment=0.0)
    before = mb.W0.copy()
    after = mb.W_kc_mbon.copy()
    if not np.isfinite(after).all():
        raise RuntimeError("plasticity produced non-finite KC->MBON weights")
    return mb, before, after, movement, active_count


def main() -> None:
    started = time.perf_counter()
    print(f"Loading real FlyWire v783 wiring: {DATA_PATH}", flush=True)
    result = load_connectome(DATA_PATH)
    W = to_sparse(result)
    n = int(result["n"])
    print(
        f"Connectome: shape={W.shape}, sparse_nnz={W.nnz:,}, density={W.nnz / max(n * n, 1):.8f}",
        flush=True,
    )

    sidecar_path = _discover_cell_type_file()
    annotations = _read_type_sidecar(sidecar_path)
    labels = _aligned_type_labels(result, annotations)
    out_degree, in_degree = _binary_degrees(W)
    mode = "cell-type labels from loader-discovered local sidecar"
    details: dict[str, int]
    cmap: CompartmentMap | None = None

    if labels is not None:
        print(f"Cell-type sidecar: {sidecar_path}", flush=True)
        try:
            kc, dan, dan_groups, mbon_pool, details = _metadata_selection(
                W, labels, out_degree
            )
            candidate_cmap = _compartment_map_from_real_block(
                _block(W, dan, mbon_pool), dan_groups, mbon_pool
            )
            mbon, cmap = _select_supported_mbon(
                candidate_cmap, _block(W, kc, mbon_pool), mbon_pool
            )
        except RuntimeError as error:
            print(f"Cell-type selection unusable ({error}); scaling connectivity heuristic.", flush=True)
            labels = None

    if labels is None:
        mode = "scaled connectivity heuristic (no usable local cell-type sidecar)"
        print(
            "No usable consolidated_cell_types.csv.gz found; using demo_real.py topology proxies.",
            flush=True,
        )
        kc, dan, dan_groups, mbon_pool, cmap, details = _heuristic_selection(
            W, n, out_degree, in_degree
        )
        mbon, cmap = _select_supported_mbon(
            cmap, _block(W, kc, mbon_pool), mbon_pool
        )

    kc = np.asarray(kc, dtype=np.int64)
    dan = np.asarray(dan, dtype=np.int64)
    mbon = np.asarray(mbon, dtype=np.int64)
    if kc.size < 1_800 or not (100 <= dan.size <= 150) or not (30 <= mbon.size <= 50):
        raise RuntimeError(
            "full-body selection missed targets: "
            f"KC={kc.size}, DAN={dan.size}, MBON={mbon.size}"
        )

    raw_block, learning_block, weights, scale = _prepare_learning_block(W, kc, mbon)
    _, before, after, movement, active_count = _run_reward_trials(
        weights, cmap, np.asarray(learning_block.sum(axis=1)).ravel()
    )

    support = learning_block.toarray() > 0.0
    approach = support & (cmap.valence[None, :] > 0.0)
    avoidance = support & (cmap.valence[None, :] < 0.0)
    if not approach.any() or not avoidance.any():
        raise RuntimeError("KC->MBON block lacks supported approach or avoidance edges")

    real_before = learning_block.toarray()
    real_after = after * scale
    real_delta = (after - before) * scale
    approach_before = float(real_before[approach].mean())
    approach_after = float(real_after[approach].mean())
    approach_change = float(real_delta[approach].mean())
    avoidance_before = float(real_before[avoidance].mean())
    avoidance_after = float(real_after[avoidance].mean())
    avoidance_change = float(real_delta[avoidance].mean())
    if not (approach_change > 0.0 and avoidance_change < 0.0):
        raise RuntimeError(
            "reward did not raise approach and lower avoidance weights: "
            f"{approach_change:+.6g}, {avoidance_change:+.6g}"
        )

    elapsed = time.perf_counter() - started
    print(f"Selection mode: {mode}")
    print(f"Selection details: {details}")
    print(
        "Final neuron counts by type: "
        f"KC={kc.size}, DAN={dan.size}, MBON={mbon.size}"
    )
    print(
        "DAN teaching groups: "
        f"PAM/PAM-like={sum(group == 'PAM' for group in dan_groups)}, "
        f"PPL1/PPL1-like={sum(group == 'PPL1' for group in dan_groups)}"
    )
    print(
        f"KC->MBON block: shape={learning_block.shape}, nnz={learning_block.nnz:,} "
        f"(raw_nnz={raw_block.nnz:,}, normalization={scale:.6g})"
    )
    print(
        f"Odor drive: top {active_count:,} KC rows; {REWARD_TRIALS} reward trials x "
        f"{PRESENTATIONS_PER_TRIAL} presentations, vectorized aggregate"
    )
    print("Weight change before/after on existing real edges:")
    print(
        f"  approach/positive-valence: {approach_before:.6f} -> {approach_after:.6f} "
        f"(change={approach_change:+.6f})"
    )
    print(
        f"  avoidance/negative-valence: {avoidance_before:.6f} -> {avoidance_after:.6f} "
        f"(change={avoidance_change:+.6f})"
    )
    print(f"Aggregate plasticity movement: {movement * scale:.6f}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    if elapsed >= RUNTIME_LIMIT_SECONDS:
        raise RuntimeError(f"runtime exceeded {RUNTIME_LIMIT_SECONDS:.0f} seconds")

    print("DONE")
    print(f"Loaded real v783 wiring and conditioned {kc.size:,} KCs, {dan.size} DANs, and {mbon.size} MBONs.")
    print(f"The real KC->MBON block is {learning_block.shape} with {learning_block.nnz:,} nonzero weights.")
    print(f"Approach weights rose by {approach_change:+.6f}; avoidance weights fell by {avoidance_change:+.6f}.")


if __name__ == "__main__":
    main()
