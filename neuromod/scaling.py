"""Homeostatic synaptic scaling (Turrigiano 2008).

Port of ``SynapticScaling`` from ``mechabrain/core/stdp.h`` (fwmc).

Applied periodically (every ~100 ms in fwmc; the caller controls cadence --
call :meth:`SynapticScaling.apply` on that schedule). Per neuron::

    scale = clamp((target_rate / actual_rate) ** alpha, 0.5, 2.0)

and all *incoming* weights of that neuron are multiplied by ``scale``.
"""

import numpy as np


class SynapticScaling:
    """Turrigiano-style multiplicative homeostatic scaling.

    Parameters
    ----------
    target_rate : float
        Target firing rate in Hz (fwmc default 5.0).
    alpha : float
        Scaling exponent (fwmc default 0.1).
    min_scale, max_scale : float
        Clamp bounds on the per-neuron scale factor (0.5, 2.0).
    """

    def __init__(self, target_rate=5.0, alpha=0.1, min_scale=0.5, max_scale=2.0):
        self.target_rate = float(target_rate)
        self.alpha = float(alpha)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def scale_factors(self, actual_rates):
        """Per-neuron scale factors for the given firing rates (Hz)."""
        actual = np.asarray(actual_rates, dtype=np.float64)
        ratio = self.target_rate / np.maximum(actual, 1e-9)
        return np.clip(ratio ** self.alpha, self.min_scale, self.max_scale)

    def apply(self, weights, post, actual_rates):
        """Multiply each synapse's weight by its postsynaptic scale factor.

        Parameters
        ----------
        weights : ndarray float, shape (n_synapses,)
            Modified in place.
        post : ndarray int, shape (n_synapses,)
            Postsynaptic neuron index per synapse.
        actual_rates : ndarray float, shape (n_neurons,)
            Measured firing rate per neuron (Hz).

        Returns
        -------
        scale : ndarray float, shape (n_neurons,)
            The per-neuron factors that were applied.
        """
        scale = self.scale_factors(actual_rates)
        weights[:] = weights * scale[np.asarray(post, dtype=np.int64)]
        return scale


if __name__ == "__main__":
    sc = SynapticScaling(target_rate=5.0, alpha=0.1)

    # neuron firing below target gets upscaled, above target downscaled
    rates = np.array([1.0, 5.0, 25.0])
    s = sc.scale_factors(rates)
    assert s[0] > 1.0 and abs(s[1] - 1.0) < 1e-12 and s[2] < 1.0, s
    assert np.all(s >= 0.5) and np.all(s <= 2.0)

    # silent neuron -> ratio -> inf -> clamped to max_scale
    s0 = sc.scale_factors(np.array([0.0]))
    assert s0[0] == 2.0

    # apply() scales incoming weights per postsynaptic neuron
    weights = np.array([1.0, 2.0, 4.0])
    post = np.array([0, 1, 0])  # synapses 0,2 -> neuron 0; synapse 1 -> neuron 1
    out = sc.apply(weights, post, rates)
    assert abs(weights[0] - 1.0 * out[0]) < 1e-12
    assert abs(weights[2] - 4.0 * out[0]) < 1e-12
    assert abs(weights[1] - 2.0 * out[1]) < 1e-12
    print("scaling.py smoke test OK")
