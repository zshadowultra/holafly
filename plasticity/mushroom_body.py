"""Mushroom-body dopamine-gated plasticity, ported from flytris and FlyBrain.

Two learning rules, both driven by real DAN->MBON compartment structure
(see compartments.py):

1. DopaminergicPlasticity (flytris, flytris/sim/plasticity.py)
   Bidirectional delta rule on KC->MBON weights:
       dw[kc_i -> mbon_j] = lr * elig_i * error * valence_j * gate_c(j)
   error = max(reward,0) - max(punishment,0); gates are the DAN cluster
   drives masked to the compartments each cluster dominates. Better than
   expected potentiates KC synapses onto approach MBONs and depresses those
   onto avoidance MBONs (and vice versa). lr is a FRACTION of a typical
   baseline synapse (effective_lr = 0.02 * mean(W0[W0>0])).

2. SpikingPlasticity (FlyBrain, fly/dopamina.py)
   Depression-only, multiplicative, tick-based:
       traza = max(traza * exp(-tick_ms/1000), kc_spiked)
       da    = max(dan_spikes - base - 2.0, 0)      # spikes above 2 s mean
       llega = C @ da                               # dopamine per MBON
       W    *= exp(-0.05 * outer(traza, llega)); floor at 0.1 * w0

Both need sparse KC codes. MushroomBody.encode implements flytris's damped
APL fixed-point loop (60 iterations, damping 0.3, gain calibrated to 5%
sparsity), with an optional FlyBrain-style fixed negative KC bias
(SESGO_KC = -3.0) as a cheaper sparsity stand-in.

Pure numpy. No torch, no connectome access: weights are passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compartments import CompartmentMap


# ---------------------------------------------------------------------------
# Mushroom body: sparse KC code + valence readout
# ---------------------------------------------------------------------------

@dataclass
class MBConfig:
    target_sparsity: float = 0.05  # fraction of KCs active per scene (Lin et al. 2014)
    apl_iterations: int = 60       # feedback loop is solved iteratively
    apl_damping: float = 0.3       # undamped, the loop oscillates instead of settling
    apl_enabled: bool = True       # ablation switch; False -> dense relu code
    kc_bias: float = 0.0           # FlyBrain SESGO_KC = -3.0: fixed negative KC
                                   # current, measured to give 5.2% @ 0.82 Hz


class MushroomBody:
    """Connectome-constrained mushroom body with an ablatable APL loop."""

    def __init__(
        self,
        W_kc_mbon: np.ndarray,
        cmap: CompartmentMap,
        w_kc_apl: np.ndarray | None = None,
        w_apl_kc: np.ndarray | None = None,
        cfg: MBConfig | None = None,
    ) -> None:
        self.cfg = cfg or MBConfig()
        self.W_kc_mbon = np.asarray(W_kc_mbon, dtype=float)  # (n_kc, n_mbon)
        self.W0 = self.W_kc_mbon.copy()                      # baseline for recovery
        self.cmap = cmap
        self.n_kc, self.n_mbon = self.W_kc_mbon.shape
        self.mbon_valence = np.asarray(cmap.valence, dtype=float)
        # flytris _compartment_gates: dominance-masked cluster drives.
        self.pam_gate = np.asarray(cmap.gates.get("PAM", np.zeros(self.n_mbon)))
        self.ppl1_gate = np.asarray(cmap.gates.get("PPL1", np.zeros(self.n_mbon)))

        # APL loop weights. flytris: w_kc_apl column-normalised, w_apl_kc
        # absolute and rescaled by its mean; inhibition profile is the
        # per-KC mean of w_apl_kc (APL is a single cell: fixed output profile,
        # only the pooled magnitude varies).
        if w_kc_apl is None:
            w_kc_apl = np.full((self.n_kc, 1), 1.0 / self.n_kc)
        if w_apl_kc is None:
            w_apl_kc = np.ones((1, self.n_kc))
        self.w_kc_apl = np.asarray(w_kc_apl, dtype=float)
        w_ak = np.asarray(w_apl_kc, dtype=float)
        w_ak = w_ak / max(w_ak.mean(), 1e-9)
        self._apl_profile = w_ak.mean(axis=0)  # (n_kc,)
        self._apl_gain = 1.0

    # -- encoding ----------------------------------------------------------
    def encode(self, drive: np.ndarray) -> np.ndarray:
        """Excitatory KC drive -> sparse KC activity via APL feedback.

        The loop is subtractive: APL pools total KC activity and inhibits all
        KCs, so only the strongest-driven few percent stay above zero. Damped
        fixed-point iteration (apl_damping=0.3); undamped it is bistable and
        oscillates.
        """
        drive = np.asarray(drive, dtype=float)
        one_d = drive.ndim == 1
        if one_d:
            drive = drive[None, :]
        if self.cfg.kc_bias:
            drive = drive + self.cfg.kc_bias
        if not self.cfg.apl_enabled:
            out = np.maximum(drive, 0.0)
        else:
            kc = np.maximum(drive, 0.0)
            a = self.cfg.apl_damping
            for _ in range(self.cfg.apl_iterations):
                apl = kc @ self.w_kc_apl                       # (batch, n_apl)
                inhib = (apl.sum(axis=1, keepdims=True)
                         * self._apl_gain) * self._apl_profile
                kc = (1.0 - a) * kc + a * np.maximum(drive - inhib, 0.0)
            out = kc
        return out[0] if one_d else out

    def calibrate_apl(self, drives: np.ndarray, tol: float = 0.005) -> float:
        """Fit APL gain so the KC code hits the target sparsity (bisection)."""
        lo, hi = 1e-4, 1e6
        for _ in range(60):
            self._apl_gain = float(np.sqrt(lo * hi))
            frac = (self.encode(drives) > 1e-6).mean()
            if abs(frac - self.cfg.target_sparsity) < tol:
                break
            if frac > self.cfg.target_sparsity:
                lo = self._apl_gain   # too many active -> more inhibition
            else:
                hi = self._apl_gain
        return self._apl_gain

    def sparsity(self, drive: np.ndarray) -> float:
        return float((self.encode(drive) > 1e-6).mean())

    # -- readout -------------------------------------------------------------
    def readout(self, kc: np.ndarray) -> np.ndarray:
        """KC activity -> MBON activity."""
        return np.maximum(kc @ self.W_kc_mbon, 0.0)

    def valence(self, kc: np.ndarray) -> float:
        """Net approach(+) / avoid(-) signal driving steering."""
        return float(self.readout(kc) @ self.mbon_valence / self.n_mbon)


# ---------------------------------------------------------------------------
# Rule 1: flytris bidirectional dopamine-gated plasticity
# ---------------------------------------------------------------------------

@dataclass
class PlasticityConfig:
    # Depression per coincidence event, as a FRACTION of a typical baseline
    # KC->MBON synapse. Small on purpose: lr=1.0 oscillated instead of
    # converging (offline r swinging 0.56-0.97); lr<=0.03 sat at r=0.998.
    lr: float = 0.02
    tau_eligibility: float = 30.0  # steps; ~how far credit reaches back
    recovery: float = 1e-3         # homeostatic drift toward baseline
    weight_floor: float = 0.0      # KC->MBON is cholinergic; never flip sign
    weight_ceiling: float = 3.0    # multiples of the baseline weight
    pam_enabled: bool = True       # ablation switch (reward pathway)
    ppl1_enabled: bool = True      # ablation switch (punishment pathway)


class DopaminergicPlasticity:
    """Bidirectional KC->MBON learning rule, gated by DAN compartment drive.

    A depression-only rule cannot work here: weights clamp at zero and ~1000
    teaching events wipe the active synapses. The dopaminergic timing window
    in the fly is bidirectional, so the update follows the gradient of valence
    and weights settle around baseline instead of collapsing.
    """

    def __init__(self, mb: MushroomBody, cfg: PlasticityConfig | None = None) -> None:
        self.mb = mb
        self.cfg = cfg or PlasticityConfig()
        self.eligibility = np.zeros(mb.n_kc)
        self._decay = float(np.exp(-1.0 / self.cfg.tau_eligibility))
        nz = mb.W0[mb.W0 > 0]
        self._w_scale = float(nz.mean()) if nz.size else 1.0
        self.effective_lr = self.cfg.lr * self._w_scale
        self._channel_traces: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.eligibility[:] = 0.0
        self._channel_traces.clear()

    def observe(self, kc: np.ndarray, channel: int | None = None) -> None:
        """Record that these KCs were active, for later credit assignment."""
        kc = np.asarray(kc, dtype=float)
        if kc.ndim > 1:
            kc = kc.mean(axis=0)
        self.eligibility = self.eligibility * self._decay + kc
        if channel is not None:
            for ch in self._channel_traces:
                self._channel_traces[ch] *= self._decay
            if channel not in self._channel_traces:
                self._channel_traces[channel] = np.zeros_like(self.eligibility)
            self._channel_traces[channel] += kc

    def dan_activity(self, reward: float, punishment: float) -> dict[str, float]:
        """DAN activity: PAM carries reward, PPL1 punishment."""
        pam = max(reward, 0.0) if self.cfg.pam_enabled else 0.0
        ppl1 = max(punishment, 0.0) if self.cfg.ppl1_enabled else 0.0
        return {"PAM": pam, "PPL1": ppl1}

    def compartment_drive(self, reward: float, punishment: float) -> np.ndarray:
        """Dopaminergic drive reaching each MBON compartment.

        Gated so reward only reaches avoidance compartments and punishment
        only approach compartments -- without that specificity both signals
        depress every synapse and learning erases itself.
        """
        drive = np.zeros(self.mb.n_mbon)
        if self.cfg.pam_enabled:
            drive = drive + max(reward, 0.0) * self.mb.pam_gate
        if self.cfg.ppl1_enabled:
            drive = drive + max(punishment, 0.0) * self.mb.ppl1_gate
        return drive

    def teach(self, reward: float, punishment: float,
              readout: np.ndarray | None = None) -> float:
        """Apply one dopaminergic teaching event. Returns total weight change.

        `readout` is the direction in MBON space the update moves along;
        defaults to the approach/avoid valence vector. The button policy
        passes the descending-neuron readout instead, with per-channel
        eligibility traces routing credit along the readout of the channel
        actually driven.
        """
        error = 0.0
        if self.cfg.pam_enabled:
            error += max(reward, 0.0)
        if self.cfg.ppl1_enabled:
            error -= max(punishment, 0.0)
        if error == 0.0:
            self._recover()
            return 0.0

        compartment = self.mb.pam_gate + self.mb.ppl1_gate

        if self._channel_traces and readout is not None:
            delta = np.zeros_like(self.mb.W_kc_mbon)
            for ch, trace in self._channel_traces.items():
                gate = compartment * readout[:, ch]
                delta += np.outer(trace, gate)
            delta *= self.effective_lr * error
        else:
            direction = self.mb.mbon_valence if readout is None else np.asarray(readout)
            gate = compartment * direction
            delta = self.effective_lr * error * np.outer(self.eligibility, gate)

        self.mb.W_kc_mbon += delta
        np.clip(self.mb.W_kc_mbon, self.cfg.weight_floor, None,
                out=self.mb.W_kc_mbon)
        np.minimum(self.mb.W_kc_mbon,
                   self.cfg.weight_ceiling * self.mb.W0 + 1e-6,
                   out=self.mb.W_kc_mbon)
        self._recover()
        return float(np.abs(delta).sum())

    def _recover(self) -> None:
        """Slow homeostatic drift back toward the connectome baseline."""
        if self.cfg.recovery > 0:
            self.mb.W_kc_mbon += self.cfg.recovery * (self.mb.W0 - self.mb.W_kc_mbon)

    def weight_change(self) -> float:
        """Mean absolute deviation from the unlearned baseline weights."""
        return float(np.abs(self.mb.W_kc_mbon - self.mb.W0).mean())


# ---------------------------------------------------------------------------
# Rule 2: FlyBrain spiking, depression-only plasticity
# ---------------------------------------------------------------------------

@dataclass
class FlyBrainConfig:
    traza_ms: float = 1000.0  # eligibility window: KC stays eligible 1 s after spiking
    eta: float = 0.05         # depression per full-punishment tick on an eligible synapse
    piso: float = 0.1         # synapse floor: 10% of original weight
    margen: float = 2.0       # DAN spikes/tick above running mean still counted as noise
    base_tau_ticks: float = 120.0  # running-mean window for DAN rate (~2 s)
    tick_ms: float = 16.67    # ms per simulation tick
    # DAN drive currents (external current injected into DAN neurons in the
    # LIF net; pulse sizing derived from MARGEN in fly/ataque.py):
    dan_pulse_current: float = 10.0  # DOPAMINA: PAM/PPL1 drive on hit/success
    pulse_ticks: int = 6             # PULSO: 100 ms pulse; ~halves eligible synapses
    pain_current: float = 3.0        # DOLOR: PPL1 current on wall touch


class SpikingPlasticity:
    """FlyBrain's tick-based, depression-only KC->MBON rule.

    "En la mosca las dos deprimen, y lo que cambia es en qué compartimento"
    (in the fly both depress; what changes is in which compartment).
    Multiplicative exponential updates, floored at 10% of the original
    weight. DANs fire spontaneously at rest, so dopamine is measured as
    spikes above each DAN's own running mean, minus MARGEN.
    """

    def __init__(self, W_kc_mbon: np.ndarray, C: np.ndarray,
                 cfg: FlyBrainConfig | None = None) -> None:
        self.cfg = cfg or FlyBrainConfig()
        self.W = np.asarray(W_kc_mbon, dtype=float)  # (n_kc, n_mbon)
        self.w0 = self.W.copy()
        self.C = np.asarray(C, dtype=float)          # (n_mbon, n_dan), rows sum to 1
        self.n_kc, self.n_mbon = self.W.shape
        self.n_dan = self.C.shape[1]
        self.traza = np.zeros(self.n_kc)             # eligibility trace
        self.base = np.zeros(self.n_dan)             # DAN running-mean spike rate
        self._decay = float(np.exp(-self.cfg.tick_ms / self.cfg.traza_ms))

    def reset(self) -> None:
        self.traza[:] = 0.0
        self.base[:] = 0.0

    def tick(self, kc_spikes: np.ndarray, dan_spikes: np.ndarray) -> float:
        """One simulation tick. Returns total weight change.

        kc_spikes: (n_kc,) bool/float spike raster for this tick.
        dan_spikes: (n_dan,) spike counts of each DAN neuron this tick.
        """
        kc_spikes = np.asarray(kc_spikes, dtype=float)
        dan_spikes = np.asarray(dan_spikes, dtype=float)
        # Eligibility: KC stays eligible TRAZA_MS after spiking (max-decay).
        self.traza = np.maximum(self.traza * self._decay, kc_spikes > 0)
        # Dopamine: spikes above the DAN's own running mean, minus noise margin.
        da = np.maximum(dan_spikes - self.base - self.cfg.margen, 0.0)
        self.base += (dan_spikes - self.base) / self.cfg.base_tau_ticks
        llega = self.C @ da                          # dopamine arriving per MBON
        cambio = self.cfg.eta * np.outer(self.traza, llega)
        moved = 0.0
        if cambio.any():
            before = self.W.copy()
            self.W = np.maximum(self.W * np.exp(-cambio),
                                self.cfg.piso * self.w0)
            moved = float(np.abs(self.W - before).sum())
        return moved

    def weight_fraction(self) -> float:
        """Mean KC->MBON weight as a fraction of the original."""
        return float((self.W / np.maximum(self.w0, 1e-12)).mean())


# ---------------------------------------------------------------------------
# FlyBrain valence balance -> binary decision ("la balanza")
# ---------------------------------------------------------------------------

class ValenceBalance:
    """Turns the MBON valence vote into a binary behavior.

    balance = sum over the window of ((mbon_spikes - resting) @ vote).
    The lunge threshold is 3 sigma of the readout's OWN noise at rest --
    not a knob for how much it attacks, but where its noise ends.
    """

    def __init__(self, vote: np.ndarray, window: int = 6) -> None:
        self.vote = np.asarray(vote, dtype=float)
        self.window = window
        self.reposo: np.ndarray | None = None
        self.umbral: float | None = None
        self._hist: list[float] = []

    def calibrate(self, rest_rasters: np.ndarray) -> float:
        """Measure per-MBON resting spike counts and the 3-sigma threshold."""
        R = np.asarray(rest_rasters, dtype=float)
        self.reposo = R.mean(axis=0)
        bal = (R - self.reposo) @ self.vote
        self.umbral = 3.0 * float(bal.std())
        return self.umbral

    def decide(self, mbon_spikes: np.ndarray) -> tuple[float, bool]:
        """Returns (balance, lunge?)."""
        assert self.reposo is not None and self.umbral is not None, \
            "call calibrate() first"
        b = float((np.asarray(mbon_spikes, dtype=float) - self.reposo) @ self.vote)
        self._hist.append(b)
        self._hist = self._hist[-self.window:]
        total = float(sum(self._hist))
        return total, total > self.umbral


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _build_toy(n_kc: int = 20, heterogeneous_apl: bool = False):
    """Tiny KC -> DAN -> MBON circuit: n_kc KCs, 2 MBONs, 2 DANs."""
    from .compartments import build_compartment_map
    rng = np.random.default_rng(7)
    edges = [("PAM01", "MBON-avoid", 10.0), ("PPL101", "MBON-approach", 9.0)]
    cmap = build_compartment_map(edges)
    W = 0.5 + rng.random((n_kc, 2))          # baseline KC->MBON weights
    kw = {}
    if heterogeneous_apl:
        # Biologically realistic: heavy-tailed PN drive needs a non-uniform
        # APL inhibition profile to reach 5% sparsity (uniform toys rail the
        # bisection at the gain ceiling).
        prof = rng.lognormal(mean=0.0, sigma=0.8, size=n_kc)
        kw["w_apl_kc"] = prof[None, :] / prof.mean()
    mb = MushroomBody(W, cmap, **kw)
    return mb, cmap


def _smoke() -> None:
    print("== compartments ==")
    mb, cmap = _build_toy()
    assert list(cmap.valence) == [-1.0, 1.0], cmap.valence
    print("valence (avoid, approach):", cmap.valence)

    print("== APL sparsity ==")
    mb_apl, _ = _build_toy(n_kc=400, heterogeneous_apl=True)
    rng = np.random.default_rng(1)
    drives = rng.lognormal(mean=-0.5, sigma=1.2, size=(40, 400))
    dense = (np.maximum(drives, 0.0) > 1e-6).mean()
    gain = mb_apl.calibrate_apl(drives)
    s = mb_apl.sparsity(drives)
    print(f"dense={dense:.3f} apl_gain={gain:.3f} sparsity={s:.3f}")
    assert s < dense / 3, "APL loop must sparsify the code"
    assert 0.02 < s < 0.10, s

    print("== flytris rule: reward association ==")
    mb_r, _ = _build_toy()
    plast = DopaminergicPlasticity(mb_r)
    odor = np.zeros(20); odor[:5] = 1.0          # odor A drives KCs 0-4
    kc = mb_r.encode(odor)
    for _ in range(10):
        plast.observe(kc)                       # odor presentation...
    plast.teach(reward=1.0, punishment=0.0)     # ...then one reward (PAM)
    dW = mb_r.W_kc_mbon - mb_r.W0
    d_approach = dW[:5, 1].mean()   # eligible KCs -> approach MBON
    d_avoid = dW[:5, 0].mean()      # eligible KCs -> avoidance MBON
    print(f"dW approach={d_approach:+.4f} avoid={d_avoid:+.4f}")
    assert 0 < d_approach < 1.0, "reward must potentiate KC->approach (graded)"
    assert -1.0 < d_avoid < 0, "reward must depress KC->avoid (graded)"
    assert (mb_r.W_kc_mbon >= 0).all(), "weights must never flip sign"
    # non-eligible KCs untouched
    assert np.abs(dW[5:, :]).max() < 1e-9
    # homeostatic recovery drifts back toward baseline
    for _ in range(200):
        plast.teach(0.0, 0.0)
    assert plast.weight_change() < np.abs(dW).mean(), "recovery must shrink deviation"

    print("== flytris rule: punishment association ==")
    mb_p, _ = _build_toy()
    plast_p = DopaminergicPlasticity(mb_p)
    for _ in range(10):
        plast_p.observe(kc)
    plast_p.teach(reward=0.0, punishment=1.0)  # PPL1: punishment
    dW = mb_p.W_kc_mbon - mb_p.W0
    print(f"dW approach={dW[:5,1].mean():+.4f} avoid={dW[:5,0].mean():+.4f}")
    assert -1.0 < dW[:5, 1].mean() < 0, "punishment must depress KC->approach"
    assert 0 < dW[:5, 0].mean() < 1.0, "punishment must potentiate KC->avoid"

    print("== FlyBrain rule: multiplicative depression ==")
    mb_f, cmap_f = _build_toy()
    sp = SpikingPlasticity(mb_f.W_kc_mbon, cmap_f.C)
    kc_sp = np.zeros(20); kc_sp[:5] = 1.0
    for _ in range(20):                       # 20 ticks: KCs fire, PAM bursts
        sp.tick(kc_sp, np.array([5.0, 0.0]))  # PAM 5 spikes/tick > MARGEN
    frac = sp.weight_fraction()
    print(f"weight fraction of original: {frac:.3f}")
    assert frac < 1.0, "eligible synapses must depress"
    assert (sp.W >= 0.1 * sp.w0 - 1e-12).all(), "PISO floor violated"
    # non-eligible KCs (5:) barely touched
    assert sp.W[5:, :].mean() > 0.99 * sp.w0[5:, :].mean()

    print("== valence balance ==")
    vb = ValenceBalance(np.array([-1.0, 1.0]))
    rest = rng.poisson(3.0, size=(90, 2)).astype(float)
    umbral = vb.calibrate(rest)
    bal, lunge = vb.decide(np.array([3.0, 3.0]))
    print(f"umbral={umbral:.2f} rest balance={bal:.2f} lunge={lunge}")
    assert not lunge
    bal2, lunge2 = vb.decide(np.array([0.0, 30.0]))  # approach MBON bursts
    print(f"approach burst balance={bal2:.2f} lunge={lunge2}")
    assert lunge2, "strong approach signal must cross the 3-sigma threshold"

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    _smoke()
