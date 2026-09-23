"""Short-term plasticity (Tsodyks-Markram) per synapse.

Port of ``mechabrain/core/short_term_plasticity.h`` (fwmc).

Per-synapse state: ``u`` (utilization / residual calcium), ``x`` (available
vesicle resources). On a presynaptic spike::

    u += U_se * (1 - u)          (clamped to [0, 1])
    ux = u * x
    x  = max(0, x - ux)

and the spike is transmitted with effective weight ``w * u * x``.

Per step (all synapses)::

    u += (U_se - u) * (1 - exp(-dt / tau_f))
    x += (1 - x)    * (1 - exp(-dt / tau_d))

Drosophila presets are grounded in NMJ electrophysiology
(Kittel 2006 / Hallermann 2010).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Neurotransmitter sign for propagation: i_syn[post] += Sign(nt) * w * ...
# GABA and glutamate are inhibitory; glutamate is inhibitory in the fly via
# GluCl. Everything else (incl. DA/5HT/OA) is excitatory here.
#
# nt codes kDA=3, k5HT=4, kOA=5 are from mechabrain/core/synapse_table.h.
# ACH/GABA/GLUTAMATE/HISTAMINE codes are port-defined for the sign function.
# ---------------------------------------------------------------------------
NT_ACETYLCHOLINE = 0
NT_GABA = 1
NT_GLUTAMATE = 2
NT_DOPAMINE = 3     # kDA in mechabrain
NT_SEROTONIN = 4    # k5HT in mechabrain
NT_OCTOPAMINE = 5   # kOA in mechabrain
NT_HISTAMINE = 6
NT_UNKNOWN = 7

_INHIBITORY_NT = frozenset({NT_GABA, NT_GLUTAMATE})


def nt_sign(nt_type):
    """+1 for excitatory, -1 for inhibitory neurotransmitter types."""
    return -1.0 if int(nt_type) in _INHIBITORY_NT else 1.0


# ---------------------------------------------------------------------------
# Drosophila presets: (U_se, tau_d_ms, tau_f_ms)
# ---------------------------------------------------------------------------
PRESET_FACILITATING = (0.15, 40.0, 200.0)   # sparse KC synapses
PRESET_DEPRESSING = (0.50, 40.0, 30.0)      # NMJ, KC->MBON
PRESET_COMBINED = (0.30, 40.0, 100.0)


class ShortTermPlasticity:
    """Tsodyks-Markram state for a fixed synapse population.

    Parameters
    ----------
    n_synapses : int
    U_se, tau_d_ms, tau_f_ms : float
        Release probability parameter and recovery/facilitation time constants.
        Use one of the PRESET_* tuples.
    """

    def __init__(self, n_synapses, U_se, tau_d_ms, tau_f_ms):
        self.n_syn = int(n_synapses)
        self.U = float(U_se)
        self.tau_d = float(tau_d_ms)
        self.tau_f = float(tau_f_ms)
        self.u = np.full(self.n_syn, self.U, dtype=np.float64)
        self.x = np.ones(self.n_syn, dtype=np.float64)

    # -- spike-triggered --------------------------------------------------
    def on_pre_spike(self, syn_idx):
        """Apply the presynaptic-spike update; return the u*x factor.

        The caller transmits ``weight * factor`` (times nt sign / scaling).
        """
        syn_idx = np.asarray(syn_idx, dtype=np.int64)
        if syn_idx.size == 0:
            return np.zeros(0, dtype=np.float64)
        u = self.u[syn_idx]
        u = u + self.U * (1.0 - u)
        np.clip(u, 0.0, 1.0, out=u)
        x = self.x[syn_idx]
        ux = u * x
        x = np.maximum(0.0, x - ux)
        self.u[syn_idx] = u
        self.x[syn_idx] = x
        return ux

    def effective_weight(self, weight, syn_idx):
        """Current ``w * u * x`` for the given synapses (no state change)."""
        syn_idx = np.asarray(syn_idx, dtype=np.int64)
        return np.asarray(weight, dtype=np.float64)[syn_idx] * self.u[syn_idx] * self.x[syn_idx]

    # -- recovery ----------------------------------------------------------
    def recover(self, dt_ms):
        """Per-step relaxation of u and x toward rest (all synapses)."""
        self.u += (self.U - self.u) * (1.0 - np.exp(-dt_ms / self.tau_f))
        self.x += (1.0 - self.x) * (1.0 - np.exp(-dt_ms / self.tau_d))


if __name__ == "__main__":
    # depressing synapse: repeated spikes at short intervals deplete x
    stp = ShortTermPlasticity(1, *PRESET_DEPRESSING)
    f1 = stp.on_pre_spike(np.array([0]))[0]
    f2 = stp.on_pre_spike(np.array([0]))[0]
    assert f2 < f1, (f1, f2)  # depletion -> smaller second factor
    assert 0.0 <= stp.u[0] <= 1.0 and 0.0 <= stp.x[0] <= 1.0

    # facilitating synapse: u grows with repeated spikes
    stp2 = ShortTermPlasticity(1, *PRESET_FACILITATING)
    u_before = stp2.u[0]
    stp2.on_pre_spike(np.array([0]))
    assert stp2.u[0] > u_before  # facilitation

    # recovery restores x and u toward rest over long dt
    stp.recover(1000.0)
    assert abs(stp.x[0] - 1.0) < 1e-3 and abs(stp.u[0] - stp.U) < 1e-3

    # effective weight helper
    stp3 = ShortTermPlasticity(2, *PRESET_COMBINED)
    w = np.array([2.0, 3.0])
    ew = stp3.effective_weight(w, np.array([0, 1]))
    assert abs(ew[0] - 2.0 * stp3.U * 1.0) < 1e-12

    # nt sign: GABA/glutamate inhibitory, DA/5HT/OA excitatory
    assert nt_sign(NT_GABA) == -1.0
    assert nt_sign(NT_GLUTAMATE) == -1.0
    assert nt_sign(NT_ACETYLCHOLINE) == 1.0
    assert nt_sign(NT_DOPAMINE) == 1.0
    assert nt_sign(NT_SEROTONIN) == 1.0
    assert nt_sign(NT_OCTOPAMINE) == 1.0
    assert nt_sign(NT_HISTAMINE) == 1.0
    print("stp.py smoke test OK")
