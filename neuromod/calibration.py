"""NumPy port of fwmc's supervised synaptic calibration.

The algorithm is transcribed from
``fly-zoo/fwmc/src/bridge/calibrator.h``.  The C++ code calls it a
perturbation-based, gradient-free method, but does not differentiate through
its neuron model or run paired perturbations inside ``Calibrator``.  Instead,
when a pre-synaptic neuron spikes, it attributes the post-synaptic prediction
error to each active synapse, averages that surrogate gradient over a batch,
and applies momentum SGD with L2 weight decay.

A dense target vector uses ``-1`` for neurons without biological observations,
mirroring the source's missing ``BioReading`` entries.  Observed values may be
binary spike labels or spike probabilities in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Calibrator:
    """Optimize synaptic weights from target-versus-actual spike patterns.

    Parameters
    ----------
    pre_neurons, post_neurons:
        One integer neuron index per synapse.
    weights:
        Mutable one-dimensional synaptic weight vector.
    n_neurons:
        Number of neurons represented by spike vectors.  By default this is
        inferred from the largest endpoint index.
    learning_rate, momentum, weight_decay:
        Source defaults: ``0.001``, ``0.9``, and ``1e-5``.
    weight_clip:
        Symmetric source weight limit.  The C++ implementation clamps to
        ``[-20, 20]`` so inhibitory signs are preserved.
    """

    def __init__(
        self,
        pre_neurons: Sequence[int] | np.ndarray,
        post_neurons: Sequence[int] | np.ndarray,
        weights: Sequence[float] | np.ndarray,
        *,
        n_neurons: int | None = None,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        weight_clip: float = 20.0,
    ) -> None:
        self.pre_neurons = self._index_vector(pre_neurons, "pre_neurons")
        self.post_neurons = self._index_vector(post_neurons, "post_neurons")
        self.weights = np.asarray(weights, dtype=float).copy()

        if not (
            self.pre_neurons.size
            == self.post_neurons.size
            == self.weights.size
        ):
            raise ValueError(
                "pre_neurons, post_neurons, and weights must have equal length"
            )
        if self.weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        if weight_clip <= 0.0:
            raise ValueError("weight_clip must be positive")

        if n_neurons is None:
            largest_endpoint = int(
                max(
                    self.pre_neurons.max(initial=-1),
                    self.post_neurons.max(initial=-1),
                )
            )
            n_neurons = largest_endpoint + 1
        if n_neurons < 0:
            raise ValueError("n_neurons must be non-negative")
        if np.any(self.pre_neurons >= n_neurons) or np.any(
            self.post_neurons >= n_neurons
        ):
            raise ValueError("a synapse endpoint is outside n_neurons")

        self.n_neurons = int(n_neurons)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.weight_clip = float(weight_clip)

        self.weight_velocity = np.zeros_like(self.weights)
        self.error_accum = np.zeros_like(self.weights)
        self.n_samples = 0

    @staticmethod
    def _index_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values)
        if result.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.issubdtype(result.dtype, np.integer):
            raise TypeError(f"{name} must contain integers")
        result = result.astype(np.intp, copy=True)
        if np.any(result < 0):
            raise ValueError(f"{name} cannot contain negative indices")
        return result

    def _pattern_vector(self, values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values)
        if result.ndim != 1 or result.size != self.n_neurons:
            raise ValueError(f"{name} must have shape ({self.n_neurons},)")
        return result

    def accumulate_error(
        self,
        actual_spikes: Sequence[bool] | np.ndarray,
        target_spikes: Sequence[float] | np.ndarray,
    ) -> None:
        """Accumulate one timestep of pre-spike-gated prediction error.

        The signed source error is ``actual - target``.  It is added only to
        synapses whose pre-neuron spiked and whose post-neuron was observed.
        Duplicate pre/post edges each receive the full local error, matching
        the C++ edge loop.
        """

        actual = self._pattern_vector(actual_spikes, "actual_spikes") != 0
        target = self._pattern_vector(target_spikes, "target_spikes").astype(
            float, copy=False
        )
        observed = target >= 0.0

        if not np.any(observed):
            return

        active_synapses = actual[self.pre_neurons] & observed[self.post_neurons]
        active_indices = np.flatnonzero(active_synapses)
        if active_indices.size:
            post_indices = self.post_neurons[active_indices]
            signed_error = actual[post_indices].astype(float) - target[post_indices]
            np.add.at(self.error_accum, active_indices, signed_error)

        # The C++ implementation increments once per non-empty biological frame,
        # including frames where no pre-synaptic neuron happened to spike.
        self.n_samples += 1

    def apply_gradients(self) -> int:
        """Apply the accumulated batch with momentum SGD and return update count."""

        if self.n_samples == 0:
            return 0

        gradient = self.error_accum / float(self.n_samples)
        self.weight_velocity = (
            self.momentum * self.weight_velocity - self.learning_rate * gradient
        )
        self.weight_velocity -= self.weight_decay * self.weights
        updated_weights = np.clip(
            self.weights + self.weight_velocity,
            -self.weight_clip,
            self.weight_clip,
        )

        n_updated = int(np.count_nonzero(updated_weights != self.weights))
        self.weights = updated_weights
        self.error_accum.fill(0.0)
        self.n_samples = 0
        return n_updated

    def mean_error(
        self,
        actual_spikes: Sequence[bool] | np.ndarray,
        target_spikes: Sequence[float] | np.ndarray,
    ) -> float:
        """Return mean absolute error over observed target neurons."""

        actual = self._pattern_vector(actual_spikes, "actual_spikes") != 0
        target = self._pattern_vector(target_spikes, "target_spikes").astype(
            float, copy=False
        )
        observed = target >= 0.0
        if not np.any(observed):
            return 0.0
        return float(np.mean(np.abs(actual[observed].astype(float) - target[observed])))


def _smoke() -> None:
    """Optimize three weights with periodic, spike-triggered updates."""

    pre_neurons = np.array([0, 0, 0])
    post_neurons = np.array([1, 2, 3])
    weights = np.array([0.20, 0.50, 0.80])
    target = np.array([-1.0, 1.0, 0.0, 0.0])
    thresholds = np.array([0.50, 0.50, 0.50])
    calibrator = Calibrator(
        pre_neurons,
        post_neurons,
        weights,
        learning_rate=0.20,
        momentum=0.50,
    )

    def actual_pattern() -> np.ndarray:
        pattern = np.zeros(4, dtype=bool)
        pattern[0] = True
        pattern[1:] = calibrator.weights >= thresholds
        return pattern

    errors: list[float] = []
    calibration_interval = 2
    for step in range(1, 7):
        actual = actual_pattern()
        errors.append(calibrator.mean_error(actual, target))
        calibrator.accumulate_error(actual, target)
        if step % calibration_interval == 0:
            calibrator.apply_gradients()

    final_actual = actual_pattern()
    final_error = calibrator.mean_error(final_actual, target)
    errors.append(final_error)

    assert errors[0] > 0.0
    assert all(later <= earlier for earlier, later in zip(errors, errors[1:]))
    assert final_error < errors[0]
    assert np.array_equal(final_actual[1:], target[1:].astype(bool))

    print(f"error trace: {' -> '.join(f'{error:.3f}' for error in errors)}")
    print(f"optimized weights: {np.array2string(calibrator.weights, precision=3)}")
    print("DONE")
    print("Ported fwmc pre-spike-gated error accumulation exactly.")
    print("Applied averaged momentum SGD with L2 decay and signed clipping.")
    print(f"Mean prediction error decreased from {errors[0]:.3f} to {final_error:.3f}.")


if __name__ == "__main__":
    _smoke()
