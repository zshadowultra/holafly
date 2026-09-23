"""Dopamine-gated STDP, direct and eligibility-trace (three-factor) modes.

Port of ``mechabrain/core/stdp.h`` (fwmc): ``STDPUpdate`` (direct pair-based)
and ``EligibilityTraceUpdate`` (Izhikevich 2007 three-factor rule).

Depression-dominant (fly mushroom body is LTD-biased, Hige et al. 2015):
a_minus > a_plus.
"""

import numpy as np


class STDPParams:
    """Exact values from fwmc's ``STDPParams``."""

    a_plus = 0.005            # LTP amplitude (pre-before-post)
    a_minus = 0.015           # LTD amplitude (post-before-pre)
    tau_plus = 15.0           # ms
    tau_minus = 25.0          # ms
    w_min = 0.0
    w_max = 10.0
    da_scale = 5.0            # dopamine gating strength
    tau_eligibility_ms = 1000.0  # eligibility trace decay ~1 s
    window_factor = 5.0       # spike pairs beyond tau*5 are ignored
    trace_zero_eps = 1e-7     # traces below this are zeroed


def _build_csr(keys, n, n_syn):
    """CSR over synapse indices keyed by neuron id.

    Returns (indptr, syn_ids) with syn_ids[k] = original synapse index.
    """
    keys = np.asarray(keys, dtype=np.int64)
    counts = np.bincount(keys, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    order = np.argsort(keys, kind="stable")
    return indptr, order


class GatedSTDPDirect:
    """Direct pair-based dopamine-gated STDP.

    On each post-spike: for every incoming synapse, pair with the source's
    last pre-spike time (pre-before-post -> LTP). On each pre-spike: for every
    outgoing synapse, pair with the target's last post-spike time
    (post-before-pre -> LTD). If ``dopamine_gated``,
    ``dw *= (1 + da_scale * dopamine[post])``.
    Weights are clamped to [w_min, w_max].

    Only rows where pre/post spiked are visited: O(spikes), not O(synapses).
    """

    def __init__(self, n_neurons, pre, post, params=None):
        self.n = int(n_neurons)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.n_syn = self.pre.shape[0]
        self.p = params or STDPParams()
        self.out_ptr, self.out_syn = _build_csr(self.pre, self.n, self.n_syn)
        self.in_ptr, self.in_syn = _build_csr(self.post, self.n, self.n_syn)
        self.last_pre = np.full(self.n, -np.inf)
        self.last_post = np.full(self.n, -np.inf)

    def step(self, t_ms, pre_spiked, post_spiked, weights, dopamine,
             dopamine_gated=True):
        p = self.p
        pre_spiked = np.asarray(pre_spiked, dtype=bool)
        post_spiked = np.asarray(post_spiked, dtype=bool)
        dopamine = np.asarray(dopamine, dtype=np.float64)

        # pre spikes -> LTD on outgoing synapses (post-before-pre):
        # pair this pre spike with each target's last post spike
        for i in np.nonzero(pre_spiked)[0]:
            s0, s1 = self.out_ptr[i], self.out_ptr[i + 1]
            if s0 == s1:
                self.last_pre[i] = t_ms
                continue
            syn = self.out_syn[s0:s1]
            j = self.post[syn]
            dt = t_ms - self.last_post[j]
            win = p.tau_minus * p.window_factor
            valid = (dt > 0.0) & (dt < win)
            dw = np.zeros(syn.shape[0])
            dw[valid] = -p.a_minus * np.exp(-dt[valid] / p.tau_minus)
            if dopamine_gated:
                dw = dw * (1.0 + p.da_scale * dopamine[j])
            weights[syn] = np.clip(weights[syn] + dw, p.w_min, p.w_max)
            self.last_pre[i] = t_ms

        # post spikes -> LTP on incoming synapses (pre-before-post):
        # pair this post spike with each source's last pre spike
        for j in np.nonzero(post_spiked)[0]:
            s0, s1 = self.in_ptr[j], self.in_ptr[j + 1]
            if s0 == s1:
                self.last_post[j] = t_ms
                continue
            syn = self.in_syn[s0:s1]
            i = self.pre[syn]
            dt = t_ms - self.last_pre[i]
            win = p.tau_plus * p.window_factor
            valid = (dt > 0.0) & (dt < win)
            dw = np.zeros(syn.shape[0])
            dw[valid] = p.a_plus * np.exp(-dt[valid] / p.tau_plus)
            if dopamine_gated:
                dw = dw * (1.0 + p.da_scale * dopamine[j])
            weights[syn] = np.clip(weights[syn] + dw, p.w_min, p.w_max)
            self.last_post[j] = t_ms


class GatedSTDEligibility:
    """Three-factor dopamine-gated STDP via eligibility traces (Izhikevich 2007).

    Spike pairs accumulate raw (ungated) ``dw`` into a per-synapse eligibility
    trace instead of changing the weight. Every step::

        dw = trace * dopamine[post] * da_scale * dt_ms
        weight = clamp(weight + dw, w_min, w_max)
        trace *= exp(-dt_ms / tau_eligibility_ms);  zero if |trace| < 1e-7
    """

    def __init__(self, n_neurons, pre, post, params=None):
        self.n = int(n_neurons)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.n_syn = self.pre.shape[0]
        self.p = params or STDPParams()
        self.out_ptr, self.out_syn = _build_csr(self.pre, self.n, self.n_syn)
        self.in_ptr, self.in_syn = _build_csr(self.post, self.n, self.n_syn)
        self.last_pre = np.full(self.n, -np.inf)
        self.last_post = np.full(self.n, -np.inf)
        self.trace = np.zeros(self.n_syn, dtype=np.float64)

    def on_spikes(self, t_ms, pre_spiked, post_spiked):
        """Accumulate raw pair dw into eligibility traces (no weight change)."""
        p = self.p
        pre_spiked = np.asarray(pre_spiked, dtype=bool)
        post_spiked = np.asarray(post_spiked, dtype=bool)

        # pre spikes: pair with last post spike -> post-before-pre -> LTD trace
        for i in np.nonzero(pre_spiked)[0]:
            s0, s1 = self.out_ptr[i], self.out_ptr[i + 1]
            if s0 == s1:
                self.last_pre[i] = t_ms
                continue
            syn = self.out_syn[s0:s1]
            j = self.post[syn]
            dt = t_ms - self.last_post[j]
            win = p.tau_minus * p.window_factor
            valid = (dt > 0.0) & (dt < win)
            self.trace[syn[valid]] -= p.a_minus * np.exp(-dt[valid] / p.tau_minus)
            self.last_pre[i] = t_ms

        # post spikes: pair with last pre spike -> pre-before-post -> LTP trace
        for j in np.nonzero(post_spiked)[0]:
            s0, s1 = self.in_ptr[j], self.in_ptr[j + 1]
            if s0 == s1:
                self.last_post[j] = t_ms
                continue
            syn = self.in_syn[s0:s1]
            i = self.pre[syn]
            dt = t_ms - self.last_pre[i]
            win = p.tau_plus * p.window_factor
            valid = (dt > 0.0) & (dt < win)
            self.trace[syn[valid]] += p.a_plus * np.exp(-dt[valid] / p.tau_plus)
            self.last_post[j] = t_ms

    def step(self, dt_ms, weights, dopamine):
        """Apply dopamine-gated trace update and decay traces (every step)."""
        p = self.p
        dopamine = np.asarray(dopamine, dtype=np.float64)
        dw = self.trace * dopamine[self.post] * p.da_scale * dt_ms
        weights[:] = np.clip(weights + dw, p.w_min, p.w_max)
        self.trace *= np.exp(-dt_ms / p.tau_eligibility_ms)
        self.trace[np.abs(self.trace) < p.trace_zero_eps] = 0.0


if __name__ == "__main__":
    # --- direct mode -----------------------------------------------------
    # one synapse 0 -> 1; pre spikes at t=0, post at t=10 ms, dopamine=1.0
    n = 2
    pre = np.array([0])
    post = np.array([1])
    w = np.array([1.0])
    da = np.array([0.0, 1.0])

    stdp = GatedSTDPDirect(n, pre, post)
    pre_spk = np.array([True, False])
    post_spk = np.array([False, True])
    stdp.step(0.0, pre_spk, np.array([False, False]), w, da)
    stdp.step(10.0, np.array([False, False]), post_spk, w, da)
    expect = 1.0 + 0.005 * np.exp(-10.0 / 15.0) * (1 + 5.0 * 1.0)
    assert abs(w[0] - expect) < 1e-12, (w[0], expect)

    # post-before-pre -> depression, depression-dominant
    w2 = np.array([1.0])
    stdp2 = GatedSTDPDirect(n, pre, post)
    stdp2.step(0.0, np.array([False, False]), post_spk, w2, da)   # post first
    stdp2.step(10.0, pre_spk, np.array([False, False]), w2, da)   # then pre
    expect2 = 1.0 - 0.015 * np.exp(-10.0 / 25.0) * (1 + 5.0 * 1.0)
    assert abs(w2[0] - expect2) < 1e-12, (w2[0], expect2)
    assert (1.0 - w2[0]) > (w[0] - 1.0)  # LTD-biased at equal |dt| params

    # ungated mode: no dopamine amplification
    w3 = np.array([1.0])
    stdp3 = GatedSTDPDirect(n, pre, post)
    stdp3.step(0.0, pre_spk, np.array([False, False]), w3, da, dopamine_gated=False)
    stdp3.step(10.0, np.array([False, False]), post_spk, w3, da, dopamine_gated=False)
    assert abs(w3[0] - (1.0 + 0.005 * np.exp(-10.0 / 15.0))) < 1e-12

    # clamping
    w4 = np.array([9.999])
    stdp4 = GatedSTDPDirect(n, pre, post)
    stdp4.step(0.0, pre_spk, np.array([False, False]), w4, da)
    stdp4.step(1.0, np.array([False, False]), post_spk, w4, da)
    assert w4[0] <= 10.0

    # --- eligibility mode ------------------------------------------------
    w5 = np.array([1.0])
    elig = GatedSTDEligibility(n, pre, post)
    elig.on_spikes(0.0, pre_spk, np.array([False, False]))
    elig.on_spikes(10.0, np.array([False, False]), post_spk)
    raw = 0.005 * np.exp(-10.0 / 15.0)
    assert abs(elig.trace[0] - raw) < 1e-12
    # no dopamine -> no weight change (trace still decays every step)
    w5b = w5.copy()
    elig.step(0.1, w5b, np.array([0.0, 0.0]))
    assert w5b[0] == 1.0
    # dopamine present -> weight moves; trace decayed once in the zero-DA step
    elig.step(0.1, w5, da)
    expect5 = 1.0 + raw * np.exp(-0.1 / 1000.0) * 1.0 * 5.0 * 0.1
    assert abs(w5[0] - expect5) < 1e-12, (w5[0], expect5)
    # trace decays toward zero
    for _ in range(20000):
        elig.step(1.0, w5, np.zeros(2))
    assert elig.trace[0] == 0.0  # zeroed below eps
    print("gated_stdp.py smoke test OK")
