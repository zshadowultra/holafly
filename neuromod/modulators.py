"""Per-neuron neuromodulator concentration dynamics.

Port of ``NeuromodulatorUpdate()`` / ``NeuromodulatorParams`` from
``mechabrain/core/stdp.h`` (fwmc). Cheap per-neuron model: modulators are
released onto postsynaptic targets along existing chemical synapses on
presynaptic spikes, decay exponentially, and are clamped to [0, 1].

Cell-type labels (from ``mechabrain/core/cell_type_defs.h``):
    DAN_PPL1 = 5        dopaminergic, punishment (PPL1 cluster)
    DAN_PAM = 6         dopaminergic, reward (PAM cluster)
    SEROTONERGIC = 13   serotonin releasing
    OCTOPAMINERGIC = 14 octopamine releasing (Tdc2+)

These are *functional* labels, NOT connectome annotations. For FlyWire data
they must be mapped from FlyWire cell-type annotations (DAN clusters, Tdc2+
OA neurons, 5HT clusters) before use -- garbage labels -> garbage modulation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Cell-type enum (mechabrain/core/cell_type_defs.h)
# ---------------------------------------------------------------------------
DAN_PPL1 = 5
DAN_PAM = 6
SEROTONERGIC = 13
OCTOPAMINERGIC = 14

_CELL_TYPE_DOC = {
    DAN_PPL1: "dopaminergic, punishment-coding (PPL1 cluster)",
    DAN_PAM: "dopaminergic, reward-coding (PAM cluster)",
    SEROTONERGIC: "serotonin releasing",
    OCTOPAMINERGIC: "octopamine releasing (Tdc2+)",
}


class NeuromodulatorParams:
    """Exact values from fwmc's ``NeuromodulatorParams``."""

    # dopamine: release per spike, exponential decay rate (tau ~ 1 s)
    da_release = 0.2
    da_decay = 0.001          # per ms
    da_autoreceptor = 0.5     # 50% of release feeds back onto the DAN itself
    # serotonin (tau ~ 2 s)
    serotonin_release = 0.15
    serotonin_decay = 0.0005  # per ms
    # octopamine (tau ~ 1 s)
    oa_release = 0.1
    oa_decay = 0.001           # per ms
    # hard bounds
    conc_min = 0.0
    conc_max = 1.0


class ModulatorState:
    """Per-neuron dopamine / serotonin / octopamine concentrations.

    Parameters
    ----------
    n_neurons : int
    params : NeuromodulatorParams, optional

    Arrays ``dopamine``, ``serotonin``, ``octopamine`` are float64, shape
    (n_neurons,), always within [conc_min, conc_max].
    """

    def __init__(self, n_neurons, params=None):
        self.n = int(n_neurons)
        self.p = params or NeuromodulatorParams()
        self.dopamine = np.zeros(self.n, dtype=np.float64)
        self.serotonin = np.zeros(self.n, dtype=np.float64)
        self.octopamine = np.zeros(self.n, dtype=np.float64)

    # -- core update ------------------------------------------------------
    def step(self, dt_ms, spiked, cell_types, out_row_ptr, out_col_idx):
        """Advance one timestep.

        Parameters
        ----------
        dt_ms : float
            Timestep in milliseconds.
        spiked : array_like bool, shape (n_neurons,)
            Which neurons spiked this step.
        cell_types : array_like int, shape (n_neurons,)
            Functional cell-type label per neuron (see enum above).
        out_row_ptr, out_col_idx : CSR of outgoing chemical synapses.
            ``out_col_idx[out_row_ptr[i]:out_row_ptr[i+1]]`` are the
            postsynaptic targets of neuron ``i``.
        """
        p = self.p
        # 1. exponential decay: conc *= max(0, 1 - decay_rate * dt)
        self.dopamine *= max(0.0, 1.0 - p.da_decay * dt_ms)
        self.serotonin *= max(0.0, 1.0 - p.serotonin_decay * dt_ms)
        self.octopamine *= max(0.0, 1.0 - p.oa_decay * dt_ms)

        # 2. spike-triggered release, keyed on presynaptic cell type
        spk = np.nonzero(np.asarray(spiked, dtype=bool))[0]
        ct = np.asarray(cell_types)
        for i in spk:
            targets = out_col_idx[out_row_ptr[i]:out_row_ptr[i + 1]]
            c = ct[i]
            if c == DAN_PPL1 or c == DAN_PAM:
                if targets.size:
                    new = self.dopamine[targets] + p.da_release
                    np.minimum(new, p.conc_max, out=new)
                    self.dopamine[targets] = new
                # autoreceptor feedback onto the DAN itself
                self.dopamine[i] = min(
                    p.conc_max, self.dopamine[i] + p.da_release * p.da_autoreceptor
                )
            elif c == SEROTONERGIC:
                if targets.size:
                    new = self.serotonin[targets] + p.serotonin_release
                    np.minimum(new, p.conc_max, out=new)
                    self.serotonin[targets] = new
            elif c == OCTOPAMINERGIC:
                if targets.size:
                    new = self.octopamine[targets] + p.oa_release
                    np.minimum(new, p.conc_max, out=new)
                    self.octopamine[targets] = new

        # 3. hard clamp (decay cannot exceed bounds, release is min()'d,
        #    but keep this as a safety invariant)
        np.clip(self.dopamine, p.conc_min, p.conc_max, out=self.dopamine)
        np.clip(self.serotonin, p.conc_min, p.conc_max, out=self.serotonin)
        np.clip(self.octopamine, p.conc_min, p.conc_max, out=self.octopamine)

    def describe_cell_types(self):
        """Human-readable doc for the expected label enum."""
        return dict(_CELL_TYPE_DOC)


if __name__ == "__main__":
    # Smoke test: 4 neurons, neuron 0 is a PAM DAN driving 1 and 2.
    n = 4
    out_row_ptr = np.array([0, 2, 2, 2, 2], dtype=np.int64)
    out_col_idx = np.array([1, 2], dtype=np.int64)
    cell_types = np.array([DAN_PAM, 0, 0, 0], dtype=np.int64)

    ms = ModulatorState(n)
    spiked = np.zeros(n, dtype=bool)
    spiked[0] = True
    ms.step(0.1, spiked, cell_types, out_row_ptr, out_col_idx)

    assert ms.dopamine[1] == 0.2 and ms.dopamine[2] == 0.2, ms.dopamine
    assert ms.dopamine[0] == 0.2 * 0.5, ms.dopamine  # autoreceptor
    assert ms.dopamine[3] == 0.0
    assert np.all(ms.serotonin == 0.0) and np.all(ms.octopamine == 0.0)

    # decay: after 100 ms at 0.001/ms the concentration must have fallen
    before = ms.dopamine[1]
    ms.step(100.0, np.zeros(n, dtype=bool), cell_types, out_row_ptr, out_col_idx)
    assert abs(ms.dopamine[1] - before * 0.9) < 1e-12, ms.dopamine
    # long dt drives it to the max(0, ...) floor, never negative
    ms.step(100000.0, np.zeros(n, dtype=bool), cell_types, out_row_ptr, out_col_idx)
    assert np.all(ms.dopamine >= 0.0)

    # clamp: hammer release, must never exceed 1
    for _ in range(50):
        ms.step(0.1, spiked, cell_types, out_row_ptr, out_col_idx)
    assert np.all(ms.dopamine <= 1.0)

    # serotonergic + octopaminergic paths
    ms2 = ModulatorState(n)
    out_row_ptr2 = np.array([0, 1, 2, 2, 2], dtype=np.int64)
    out_col_idx2 = np.array([2, 3], dtype=np.int64)
    ct2 = np.array([SEROTONERGIC, OCTOPAMINERGIC, 0, 0], dtype=np.int64)
    sp2 = np.zeros(n, dtype=bool)
    sp2[0] = sp2[1] = True
    ms2.step(0.1, sp2, ct2, out_row_ptr2, out_col_idx2)
    assert ms2.serotonin[2] == 0.15 and ms2.octopamine[3] == 0.1
    print("modulators.py smoke test OK")
