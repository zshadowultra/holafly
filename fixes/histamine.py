"""Histamine photoreceptor correction for the fly-brain master build.

Port of rembish/fruit-fly's compile-time fix (``fruitfly/data.py``,
``prepare()``), quoted verbatim from the source:

    # Photoreceptors release histamine, which is INHIBITORY on their targets
    # (the basis of the ON/OFF pathway split). The FlyWire NT classifier has no
    # histamine class and mislabels ~74% of photoreceptor outputs as excitatory,
    # so their outgoing sign is forced negative during compilation.
    PHOTORECEPTOR_TYPES = {"R1-6", "R7", "R8"}

    # histamine fix: photoreceptor outputs are always inhibitory
    photo_mask = np.isin(cls["cell_type"], list(PHOTORECEPTOR_TYPES))
    flipped = photo_mask[pre] & (sign > 0)
    sign[photo_mask[pre]] = -1.0

Photoreceptor identification in the source is an EXACT ``cell_type`` string
match against ``{"R1-6", "R7", "R8"}`` (values from FlyWire Codex
``consolidated_cell_types.csv.gz`` -> ``primary_type``). The fix is applied
per connection row BEFORE per-(pre, post) aggregation; ``sign`` on each row
comes from the ``NT_SIGN`` table below applied to the row's ``nt_type``.

VERACITY NOTE (from the survey): the "~74%" misclassification figure and the
"1.97% of connectome by synapse weight is monoaminergic / octopamine 0.28%"
figures appear only as assertions in the source repo's README/comments, with
no citation and no measurement script in the repo. Do not quote them as
measured facts. The fix itself (force negative sign) is the portable part.
"""

from __future__ import annotations

import numpy as np

# Exact photoreceptor cell_type strings, verbatim from fruitfly/data.py.
PHOTORECEPTOR_TYPES = {"R1-6", "R7", "R8"}

# Neurotransmitter sign convention, verbatim from fruitfly/data.py
# (Shiu et al. 2024): acetylcholine excitatory; GABA and glutamate
# inhibitory (GluCl channels in fly); monoamines treated as excitatory.
# NOTE: the monoamine-as-excitatory entries are the documented placeholder
# the source repo itself flags for replacement by real neuromodulatory gain
# (see fly-master/neuromod/).
NT_SIGN = {
    "ACH": +1.0,
    "GABA": -1.0,
    "GLUT": -1.0,
    "DA": +1.0,
    "OCT": +1.0,
    "SER": +1.0,
}


def nt_sign(nt_types) -> np.ndarray:
    """Map neurotransmitter labels to signs via NT_SIGN.

    Unknown nt_type defaults to +1.0, exactly as in the source
    (``NT_SIGN.get(t, +1.0)``).
    """
    nt_types = np.asarray(nt_types, dtype=str)
    return np.array([NT_SIGN.get(t, +1.0) for t in nt_types], dtype=float)


def correct_photoreceptor_sign(types, weights):
    """Force inhibitory outgoing sign for photoreceptor neurons.

    This is the row-level source fix (``sign[photo_mask[pre]] = -1.0``)
    applied to an aggregated weight matrix: for every presynaptic neuron
    whose ``cell_type`` is exactly one of ``PHOTORECEPTOR_TYPES``
    (``"R1-6"``, ``"R7"``, ``"R8"``), all of its outgoing weights keep
    their magnitudes but are forced to non-positive sign.

    Parameters
    ----------
    types : array-like of str, shape (n_pre,)
        ``cell_type`` per presynaptic neuron. Photoreceptors are
        identified by EXACT string match against PHOTORECEPTOR_TYPES --
        the same identification logic as the source.
    weights : array-like of float, shape (n_pre, n_post)
        Signed weight matrix (e.g. sign * synapse count per pair).

    Returns
    -------
    corrected : np.ndarray, shape (n_pre, n_post)
        Copy of ``weights`` with photoreceptor rows forced to ``-abs``.
    n_flipped : int
        Number of individual weights that were positive and got flipped
        (the source's ``flipped.sum()`` counter).
    """
    types = np.asarray(types, dtype=str)
    weights = np.asarray(weights, dtype=float)
    if types.shape[0] != weights.shape[0]:
        raise ValueError(
            f"types has {types.shape[0]} entries but weights has "
            f"{weights.shape[0]} presynaptic rows"
        )
    photo = np.isin(types, list(PHOTORECEPTOR_TYPES))
    corrected = weights.copy()
    photo_rows = corrected[photo]
    n_flipped = int((photo_rows > 0).sum())
    corrected[photo] = -np.abs(photo_rows)
    return corrected, n_flipped


def _smoke() -> None:
    # exact-match identification: only the three canonical types count
    types = np.array(["R1-6", "R7", "R8", "L1", "Mi1", "R1-6 ", "r7"])
    w = np.array(
        [
            [+5.0, +3.0, -1.0],   # R1-6: positives must flip, negative kept
            [+2.0, 0.0, +4.0],    # R7
            [-7.0, -2.0, -9.0],   # R8: already inhibitory, untouched
            [+5.0, +3.0, -1.0],   # L1: not a photoreceptor, untouched
            [+1.0, +1.0, +1.0],   # Mi1: untouched
            [+9.0, +9.0, +9.0],   # "R1-6 " (trailing space): NOT matched
            [+8.0, +8.0, +8.0],   # "r7" (wrong case): NOT matched
        ]
    )
    corrected, n_flipped = correct_photoreceptor_sign(types, w)

    # magnitudes preserved, signs forced non-positive on photoreceptor rows
    assert np.all(corrected[:3] <= 0.0)
    assert np.allclose(np.abs(corrected[:3]), np.abs(w[:3]))
    # non-photoreceptors byte-identical
    assert np.array_equal(corrected[3:], w[3:])
    # flipped counter: 2 (R1-6) + 2 (R7) + 0 (R8) = 4
    assert n_flipped == 4, n_flipped
    # input not mutated
    assert w[0, 0] == 5.0

    # no photoreceptors present: identity
    c2, n2 = correct_photoreceptor_sign(["L1", "L2"], np.ones((2, 2)))
    assert n2 == 0 and np.array_equal(c2, np.ones((2, 2)))

    # nt_sign table incl. unknown default
    s = nt_sign(["ACH", "GABA", "GLUT", "DA", "OCT", "SER", "???"])
    assert list(s) == [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], s

    # shape mismatch raises
    try:
        correct_photoreceptor_sign(["R7"], np.ones((2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")

    print("histamine.py smoke test: OK")


if __name__ == "__main__":
    _smoke()
