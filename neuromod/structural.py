"""Structural plasticity: synaptic pruning and sprouting.

This is a NumPy port of fwmc's ``StructuralPlasticity`` implementation.  The
ported defaults are a strict pruning threshold of 0.05, a sprouting probability
of 0.001 per ordered active-neuron pair, and a cap of 100 outgoing synapses per
neuron.  The requested Python scheduling default is every 5000 steps.

A dense square weight matrix represents the directed graph: ``weights[pre,
post] == 0`` denotes no live connection.  fwmc treats two neurons as correlated
when both have a nonzero spike flag in the same update; it does not compute a
Pearson coefficient or retain a spike-history window.
"""

import numpy as np


DEFAULT_PRUNE_THRESHOLD = 0.05
DEFAULT_SPROUT_RATE = 0.001
DEFAULT_UPDATE_INTERVAL = 5000
DEFAULT_MAX_SYNAPSES_PER_NEURON = 100


class StructuralPlasticity:
    """Prune weak synapses and sprout connections between co-firing neurons.

    Parameters
    ----------
    prune_threshold : float
        Prune a live synapse when its absolute weight is strictly below this
        value. The fwmc default is 0.05.
    sprout_rate : float
        Probability of accepting each ordered pair of simultaneously active
        neurons. The fwmc default is 0.001.
    update_interval : int
        Run structural updates when ``step % update_interval == 0``. Values less
        than or equal to zero disable updates, matching fwmc.
    max_synapses_per_neuron : int
        Cap on live outgoing synapses per presynaptic neuron. The fwmc default
        is 100.
    rng : numpy.random.Generator, optional
        Random source used for sprouting. A fresh generator is created when this
        is omitted.

    Notes
    -----
    fwmc stores synapses in CSR form and can therefore represent parallel
    connections to the same ordered neuron pair. A dense matrix has only one
    value per pair, so this port preserves the sampling and cap order but cannot
    preserve parallel-edge multiplicity. Following fwmc, an accepted sample of
    an existing pair resets that matrix entry to the threshold and is included
    in the returned sprout count.
    """

    def __init__(
        self,
        prune_threshold=DEFAULT_PRUNE_THRESHOLD,
        sprout_rate=DEFAULT_SPROUT_RATE,
        update_interval=DEFAULT_UPDATE_INTERVAL,
        max_synapses_per_neuron=DEFAULT_MAX_SYNAPSES_PER_NEURON,
        rng=None,
    ):
        self.prune_threshold = float(prune_threshold)
        self.sprout_rate = float(sprout_rate)
        self.update_interval = int(update_interval)
        self.max_synapses_per_neuron = int(max_synapses_per_neuron)
        self.rng = np.random.default_rng() if rng is None else rng

        if np.isnan(self.prune_threshold) or self.prune_threshold < 0.0:
            raise ValueError("prune_threshold must be a non-negative number")
        if not 0.0 <= self.sprout_rate <= 1.0:
            raise ValueError("sprout_rate must be between 0 and 1")
        if self.max_synapses_per_neuron < 0:
            raise ValueError("max_synapses_per_neuron must be non-negative")

    @staticmethod
    def _validate_weights(weights):
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a numpy.ndarray")
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("weights must be a square two-dimensional array")
        if not np.issubdtype(weights.dtype, np.floating):
            raise TypeError("weights must have a floating-point dtype")
        if not weights.flags.writeable:
            raise ValueError("weights must be writable")

    def prune(self, weights):
        """Zero weak live weights in place and return the number pruned.

        The comparison is ``weight != 0 and abs(weight) < threshold``. Thus a
        weight exactly equal to the pruning threshold survives, as in fwmc.
        """

        self._validate_weights(weights)
        weak = (weights != 0.0) & (np.abs(weights) < self.prune_threshold)
        pruned = int(np.count_nonzero(weak))
        weights[weak] = 0.0
        return pruned

    def sprout(self, weights, spiked):
        """Add probabilistic directed connections and return their count.

        ``spiked`` is a neuron-length array. Nonzero entries identify the
        neurons active during this update. All ordered pairs of distinct active
        neurons are sampled independently, and accepted synapses are assigned
        ``prune_threshold``.
        """

        self._validate_weights(weights)
        n_neurons = weights.shape[0]
        active_mask = np.asarray(spiked, dtype=bool)
        if active_mask.shape != (n_neurons,):
            raise ValueError("spiked must have one entry per neuron")

        active = np.flatnonzero(active_mask)
        if active.size < 2:
            return 0

        # fwmc excludes dead synapses from the per-presynaptic-neuron degree.
        out_degree = np.count_nonzero(weights != 0.0, axis=1)
        sprouted = 0

        for i, pre in enumerate(active):
            pre = int(pre)
            if out_degree[pre] >= self.max_synapses_per_neuron:
                continue

            for j, post in enumerate(active):
                if i == j:
                    continue

                # Keep fwmc's draw/cap order: sample first, then check the cap.
                if float(self.rng.random()) > self.sprout_rate:
                    continue
                if out_degree[pre] >= self.max_synapses_per_neuron:
                    break

                post = int(post)
                weights[pre, post] = self.prune_threshold
                out_degree[pre] += 1
                sprouted += 1

        return sprouted

    # Names matching fwmc's C++ methods.
    prune_weak = prune
    sprout_new = sprout

    def update(self, weights, spiked, step):
        """Run a scheduled prune-then-sprout update.

        Returns ``None`` away from an update boundary, or ``(pruned, sprouted)``
        when the structural update runs.
        """

        if self.update_interval <= 0:
            return None
        if int(step) % self.update_interval != 0:
            return None

        pruned = self.prune(weights)
        sprouted = self.sprout(weights, spiked)
        return pruned, sprouted


def _smoke_test():
    weights = np.array(
        [
            [0.01, 0.20, 0.00, 0.00],
            [0.00, -0.04, 0.30, 0.00],
            [0.00, 0.00, 0.40, 0.00],
            [0.00, 0.00, 0.00, 0.50],
        ],
        dtype=np.float64,
    )
    spiked = np.array([True, True, True, False])
    plasticity = StructuralPlasticity(
        sprout_rate=1.0,
        rng=np.random.default_rng(42),
    )

    # The default interval is 5000; no work occurs before its boundary.
    assert plasticity.update_interval == 5000
    assert plasticity.update(weights, spiked, step=4999) is None

    # Three active neurons yield 3 * 2 = six directed sprout attempts.
    # Two weak edges are pruned before those sprouts are added.
    assert plasticity.update(weights, spiked, step=5000) == (2, 6)
    assert weights[0, 0] == 0.0
    assert weights[1, 1] == 0.0
    assert weights[2, 2] == 0.40

    active_pair = spiked[:, None] & spiked[None, :]
    off_diagonal = ~np.eye(weights.shape[0], dtype=bool)
    sprouted_pairs = active_pair & off_diagonal
    assert np.all(weights[sprouted_pairs] == 0.05)
    assert not np.any(weights[3, :3])
    assert not np.any(weights[:3, 3])

    print("structural.py smoke test OK")
    print("DONE")
    print("Ported fwmc strict absolute-weight pruning with the 0.05 default.")
    print("Ported probabilistic directed sprouting between co-firing neurons.")
    print("Added prune-before-sprout updates every 5000 steps by default.")


if __name__ == "__main__":
    _smoke_test()
