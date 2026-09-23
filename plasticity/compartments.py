"""DAN -> MBON compartment mapping, built from a connectivity edge list.

Both source repos derive the mushroom-body compartment structure from real
DAN->MBON connectivity instead of hardcoding the anatomical compartment table:

* flytris (flytris/sim/mushroom_body.py): "We do not hardcode the Aso
  compartment table; the connectome states it directly." PAM/PPL1 gates are
  the DAN->MBON weight matrix masked to compartments each cluster dominates.
* FlyBrain (fly/dopamina.py): "Una DAN hace sinapsis en las dendritas de las
  MBON de su compartimento, asi que es un buen indicador de que MBON modula."
  Builds C = (n_mbon, n_dan), rows normalised to sum 1.

Input is a plain type-level edge list: (pre_type, post_type, weight).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default DAN cluster matchers, as regexes on cell type.
# From flytris/flytris/connectome/build.py:
#     "dan_pam":  r"^PAM\d",
#     "dan_ppl1": r"^PPL1\d",
DEFAULT_DAN_CLUSTERS: dict[str, str] = {
    "PAM": r"^PAM\d",
    "PPL1": r"^PPL1\d",
}

import re as _re


@dataclass
class CompartmentMap:
    """DAN->MBON compartment structure derived from connectivity."""

    mbon_types: list[str]                      # MBON type per row
    dan_types: list[str]                       # DAN type per column
    dan_cluster: list[str]                     # cluster name per DAN column
    C: np.ndarray                              # (n_mbon, n_dan) DAN->MBON, rows sum to 1 or 0
    cluster_drive: dict[str, np.ndarray] = field(default_factory=dict)
    #   cluster name -> (n_mbon,) total DAN weight of that cluster per MBON
    dominant_cluster: list[str | None] = field(default_factory=list)
    #   per-MBON cluster with the largest drive (None if no DAN input)
    gates: dict[str, np.ndarray] = field(default_factory=dict)
    #   cluster name -> (n_mbon,) dominance-masked gate, flytris style:
    #   gate_c = drive_c where drive_c strictly exceeds every other cluster's
    valence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #   per-MBON approach(+1)/avoid(-1) polarity from PAM vs PPL1 dominance


def _match(types: list[str], pattern: str) -> list[bool]:
    rx = _re.compile(pattern)
    return [rx.search(t) is not None for t in types]


def build_compartment_map(
    edges: list[tuple[str, str, float]],
    dan_clusters: dict[str, str] | None = None,
    mbon_pattern: str = r"^MBON",
    min_weight: float = 0.0,
) -> CompartmentMap:
    """Build the DAN->MBON compartment map from a type-level edge list.

    Parameters
    ----------
    edges:
        (pre_type, post_type, weight) tuples. Only edges whose pre type
        matches a DAN cluster and whose post type matches ``mbon_pattern``
        are kept.
    dan_clusters:
        {cluster_name: regex} mapping, e.g. {"PAM": r"^PAM\\d",
        "PPL1": r"^PPL1\\d"}. Defaults to DEFAULT_DAN_CLUSTERS.
    mbon_pattern:
        regex selecting MBON types.
    min_weight:
        edges below this weight are ignored (FlyBrain uses UMERAL = 5 on
        raw synapse counts).

    Returns
    -------
    CompartmentMap with per-MBON dominant cluster, dominance-masked gates
    and PAM/PPL1-derived valence.
    """
    dan_clusters = dan_clusters or DEFAULT_DAN_CLUSTERS

    dan_types: list[str] = []
    dan_of: dict[str, str] = {}          # dan type -> cluster name
    for pre, post, w in edges:
        for cluster, pat in dan_clusters.items():
            if _re.search(pat, pre):
                if pre not in dan_of:
                    dan_of[pre] = cluster
                    dan_types.append(pre)
                break

    mbon_types: list[str] = []
    mbon_ix: dict[str, int] = {}
    for pre, post, w in edges:
        if pre in dan_of and _re.search(mbon_pattern, post) and w >= min_weight:
            if post not in mbon_ix:
                mbon_ix[post] = len(mbon_types)
                mbon_types.append(post)

    n_mbon, n_dan = len(mbon_types), len(dan_types)
    Craw = np.zeros((n_mbon, n_dan))
    for pre, post, w in edges:
        if pre in dan_of and post in mbon_ix and w >= min_weight:
            Craw[mbon_ix[post], dan_types.index(pre)] += w

    # Row-normalised compartment matrix (FlyBrain `compartimentos`):
    # C[i, j] = share of MBON i's DAN input arriving from DAN j.
    row_sum = Craw.sum(axis=1, keepdims=True)
    C = Craw / np.maximum(row_sum, 1e-9)

    # Column-normalised DAN->MBON matrix (flytris: unsigned, column-normalised
    # so MBONs with very different in-degrees start on comparable footing).
    # Cluster drives and gates are computed on this, exactly as in
    # flytris MushroomBody (W_dan_mbon, _compartment_gates).
    col_sum = Craw.sum(axis=0, keepdims=True)
    Ccol = Craw / np.maximum(col_sum, 1e-9)

    cluster_names = list(dan_clusters.keys())
    dan_cluster = [dan_of[t] for t in dan_types]
    cluster_drive: dict[str, np.ndarray] = {}
    for c in cluster_names:
        cols = [j for j, cl in enumerate(dan_cluster) if cl == c]
        cluster_drive[c] = Ccol[:, cols].sum(axis=1) if cols else np.zeros(n_mbon)

    # Dominant cluster per MBON.
    dominant: list[str | None] = []
    for i in range(n_mbon):
        drives = {c: cluster_drive[c][i] for c in cluster_names}
        best = max(drives, key=drives.get)
        dominant.append(best if drives[best] > 0 else None)

    # Dominance-masked gates (flytris `_compartment_gates`, generalised to
    # any number of clusters): each cluster may only depress the MBONs it
    # dominates. Without this, driving a whole cluster depresses nearly
    # every MBON and the memory erases itself.
    gates: dict[str, np.ndarray] = {}
    for c in cluster_names:
        others = np.maximum.reduce(
            [cluster_drive[o] for o in cluster_names if o != c]
        ) if len(cluster_names) > 1 else np.zeros(n_mbon)
        d = cluster_drive[c]
        gates[c] = np.where(d > others, d, 0.0)

    # MBON valence from PAM vs PPL1 dominance (flytris `_infer_mbon_valence`):
    # punishment (PPL1) depresses approach-promoting MBONs, reward (PAM)
    # depresses avoidance-promoting ones, so a PPL1-dominated MBON is
    # approach-promoting (+1) and a PAM-dominated one avoidance-promoting (-1).
    valence = np.zeros(n_mbon)
    if "PAM" in cluster_drive and "PPL1" in cluster_drive:
        pam, ppl1 = cluster_drive["PAM"], cluster_drive["PPL1"]
        total = pam + ppl1
        nz = total > 0
        valence[nz] = np.sign(ppl1[nz] - pam[nz])

    return CompartmentMap(
        mbon_types=mbon_types,
        dan_types=dan_types,
        dan_cluster=dan_cluster,
        C=C,
        cluster_drive=cluster_drive,
        dominant_cluster=dominant,
        gates=gates,
        valence=valence,
    )
