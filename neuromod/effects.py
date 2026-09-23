"""Neuromodulatory excitability effects.

Port of ``NeuromodulatorEffects::Apply`` from
``mechabrain/core/neuromodulator_effects.h`` (fwmc). Applied after
``ModulatorState.step`` and before the neuron dynamics step: tonic currents
added to the external current array, plus an octopamine gain on the synaptic
current array.

Units are pA-equivalent added to ``i_ext`` / multiplicative on ``i_syn``.
"""

import numpy as np

# Exact values from neuromodulator_effects.h
MIN_CONCENTRATION = 0.01   # effects below this are skipped
OA_IEXT_PER_UNIT = 3.0     # OA: depolarizing, arousal (Roeder 2005; Claridge-Chang 2009)
OA_ISYN_GAIN = 0.3         # OA boosts evoked PSP amplitude 20-40% (Farooqui 2004)
SER_IEXT_PER_UNIT = -2.0   # 5HT: net hyperpolarizing (Pooryasin & Bhatt 2021)
DA_IEXT_PER_UNIT = 1.5     # DA: mild depolarizing, D1-like (Kim 2007; Berry 2012)


def apply_effects(i_ext, i_syn, dopamine, serotonin, octopamine):
    """Apply tonic neuromodulatory effects in place.

    Parameters
    ----------
    i_ext, i_syn : ndarray float, shape (n_neurons,)
        External and synaptic current arrays; modified in place.
    dopamine, serotonin, octopamine : ndarray float, shape (n_neurons,)
        Per-neuron concentrations from :class:`ModulatorState`.
    """
    oa = np.asarray(octopamine, dtype=np.float64)
    ser = np.asarray(serotonin, dtype=np.float64)
    da = np.asarray(dopamine, dtype=np.float64)

    oa_m = oa > MIN_CONCENTRATION
    if np.any(oa_m):
        i_ext[oa_m] += oa[oa_m] * OA_IEXT_PER_UNIT
        i_syn[oa_m] *= 1.0 + oa[oa_m] * OA_ISYN_GAIN

    ser_m = ser > MIN_CONCENTRATION
    if np.any(ser_m):
        i_ext[ser_m] += ser[ser_m] * SER_IEXT_PER_UNIT

    da_m = da > MIN_CONCENTRATION
    if np.any(da_m):
        i_ext[da_m] += da[da_m] * DA_IEXT_PER_UNIT


if __name__ == "__main__":
    n = 3
    i_ext = np.zeros(n)
    i_syn = np.ones(n) * 10.0
    da = np.array([0.0, 0.5, 0.0])
    ser = np.array([0.0, 0.0, 0.4])
    oa = np.array([1.0, 0.0, 0.0])

    apply_effects(i_ext, i_syn, da, ser, oa)

    assert i_ext[0] == 1.0 * 3.0, i_ext            # OA depolarizing
    assert i_syn[0] == 10.0 * (1 + 1.0 * 0.3), i_syn  # OA synaptic gain
    assert i_ext[1] == 0.5 * 1.5, i_ext            # DA mild depolarizing
    assert i_ext[2] == 0.4 * -2.0, i_ext          # 5HT hyperpolarizing
    assert i_syn[1] == 10.0 and i_syn[2] == 10.0  # no OA -> no synaptic gain

    # below-threshold concentrations are ignored
    i2 = np.zeros(n)
    s2 = np.ones(n)
    apply_effects(i2, s2, np.full(n, 0.005), np.full(n, 0.005), np.full(n, 0.005))
    assert np.all(i2 == 0.0) and np.all(s2 == 1.0)
    print("effects.py smoke test OK")
