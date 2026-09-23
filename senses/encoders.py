"""NumPy sensory encoders ported from ``flyverse-core``.

The source model keeps physical sensor quantities separate from neural state.  This
module preserves that boundary:

* vision: ``[UV, B, G, R]`` radiance per ommatidium -> photoreceptor contrast;
* smell: concentration per glomerulus and antenna -> ORN rate in Hz;
* taste: normalized sugar contact -> sweet-GRN rate in Hz;
* wind: normalized antennal deflection -> JO-E/JO-C rate in Hz.

The vision implementation is the photoreceptor stage from
``flyverse/optic.py``.  It projects each receptor's four-channel spectrum,
averages receptors by ``(ommatidium, receptor-family)``, applies the 10 ms
low-pass and 300 ms adaptation filters, and clips contrast to ``[-1, 2]``.
The connectome projection from those photoreceptor values to optic-lobe and
spiking cells is intentionally not recreated here because it requires the
source's sparse weights.

All arrays are NumPy arrays.  No torch, scipy, pandas, or source package is
needed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


# Source constants: retina.py, in [UV, B, G, R] channel order.
SPECTRAL_SENSITIVITY = {
    "R1-R6": (0.50, 0.70, 1.00, 0.05),
    "R7p": (1.00, 0.05, 0.00, 0.00),
    "R7y": (1.00, 0.15, 0.00, 0.00),
    "R7d": (1.00, 0.05, 0.00, 0.00),
    "R7_unclear": (1.00, 0.10, 0.00, 0.00),
    "R8p": (0.20, 1.00, 0.10, 0.00),
    "R8y": (0.05, 0.30, 1.00, 0.10),
    "R8d": (1.00, 0.05, 0.00, 0.00),
    "R8_unclear": (0.12, 0.65, 0.55, 0.05),
    "R7R8_unclear": (0.60, 0.40, 0.30, 0.02),
}

# Source optic.py family partition.
FAMILY_OF_TYPE = {
    "R1-R6": 0,
    "R7p": 1,
    "R7y": 2,
    "R7d": 1,
    "R7_unclear": 2,
    "R8p": 3,
    "R8y": 4,
    "R8d": 1,
    "R8_unclear": 4,
    "R7R8_unclear": 2,
}

DEFAULT_TAU_LP_MS = 10.0
DEFAULT_TAU_ADAPT_MS = 300.0
DEFAULT_EPS = 0.02
DEFAULT_CONTRAST_CLIP = 2.0

DEFAULT_SMELL_BASE_HZ = 1.0
DEFAULT_SMELL_MAX_HZ = 150.0
DEFAULT_SMELL_HALF_CONC = 0.5
DEFAULT_TASTE_MAX_HZ = 120.0
DEFAULT_WIND_BASE_HZ = 2.0
DEFAULT_WIND_MAX_HZ = 50.0


# ---------------------------------------------------------------------------
# Shared validation helpers


def _batch_size(batch: int) -> int:
    if isinstance(batch, (bool, np.bool_)) or not isinstance(batch, (int, np.integer)):
        raise ValueError("batch must be a positive integer")
    batch = int(batch)
    if batch < 1:
        raise ValueError("batch must be a positive integer")
    return batch


def _batch_values(value, batch: int, name: str) -> np.ndarray:
    """Match the source's scalar-or-(batch,) validation exactly."""
    batch = _batch_size(batch)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full((batch,), float(array), dtype=float)
    if array.shape != (batch,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and scalar or shape ({batch},)")
    return array


def _side_values(values, name: str) -> np.ndarray:
    """Coerce numeric signs or human-readable L/R/bilateral labels to -1/0/1."""
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    labels = {
        "l": 1,
        "left": 1,
        "r": -1,
        "right": -1,
        "b": 0,
        "both": 0,
        "bilateral": 0,
        "mean": 0,
    }
    result = np.empty(len(array), dtype=int)
    for index, value in enumerate(array):
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in labels:
                raise ValueError(f"{name}[{index}] is not a valid side")
            result[index] = labels[key]
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] is not numeric or a side label") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
        result[index] = int(np.sign(number))
    return result


def _glomerulus_name(value) -> str:
    # Source Smell.__init__ removes the literal ORN_ prefix from neuron type.
    return str(value).replace("ORN_", "")


def _nonnegative_finite(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


# ---------------------------------------------------------------------------
# Vision


class VisionEncoder:
    """Stateful port of the photoreceptor stage in ``optic.py``.

    Parameters
    ----------
    receptor_types:
        One source receptor type per photoreceptor, such as ``"R1-R6"``.
        If omitted, a compact one-R1-R6-receptor-per-ommatidium layout is used.
    receptor_columns:
        Ommatidium index for each photoreceptor.  If omitted, receptors are
        assigned one per ommatidium in order.
    n_columns:
        Number of ommatidia.  Inferred from ``receptor_columns`` when needed.
    sensitivities:
        Optional ``(n_receptors, 4)`` sensitivity matrix.  Supplying it makes
        the function independent of source receptor labels.
    families:
        Optional receptor-family indices 0 through 4.  They are inferred from
        receptor types when possible.
    """

    def __init__(
        self,
        receptor_types=None,
        receptor_columns=None,
        *,
        n_columns: int | None = None,
        sensitivities=None,
        families=None,
        tau_lp_ms: float = DEFAULT_TAU_LP_MS,
        tau_adapt_ms: float = DEFAULT_TAU_ADAPT_MS,
        eps: float = DEFAULT_EPS,
        contrast_clip: float = DEFAULT_CONTRAST_CLIP,
    ) -> None:
        self.tau_lp_ms = float(tau_lp_ms)
        self.tau_adapt_ms = float(tau_adapt_ms)
        self.eps = float(eps)
        self.contrast_clip = float(contrast_clip)
        if not np.isfinite(self.tau_lp_ms) or self.tau_lp_ms <= 0.0:
            raise ValueError("tau_lp_ms must be positive and finite")
        if not np.isfinite(self.tau_adapt_ms) or self.tau_adapt_ms <= 0.0:
            raise ValueError("tau_adapt_ms must be positive and finite")
        if not np.isfinite(self.eps) or self.eps < 0.0:
            raise ValueError("eps must be finite and nonnegative")
        if not np.isfinite(self.contrast_clip) or self.contrast_clip < 1.0:
            raise ValueError("contrast_clip must be finite and at least 1")

        # A 2-D first metadata argument is treated as a custom sensitivity matrix.
        if sensitivities is None and receptor_types is not None:
            candidate = np.asarray(receptor_types)
            if candidate.ndim == 2:
                sensitivities = candidate
                receptor_types = None

        self._lazy = (
            receptor_types is None
            and receptor_columns is None
            and sensitivities is None
            and families is None
            and n_columns is None
        )
        self.n_columns: int | None = None
        self.receptor_types = None
        self.receptor_columns = None
        self.sensitivities = None
        self.families = None
        self._rows = None
        self._has = None
        self._groups = None
        self._reset_state()

        if not self._lazy:
            self._configure(receptor_types, receptor_columns, n_columns, sensitivities, families)

    def _configure(self, receptor_types, receptor_columns, n_columns, sensitivities, families) -> None:
        type_names = None
        sensitivity_array = None

        if receptor_types is not None:
            type_array = np.asarray(receptor_types, dtype=object)
            if type_array.ndim == 0:
                type_array = type_array.reshape(1)
            if type_array.ndim != 1:
                raise ValueError("receptor_types must be one-dimensional")
            type_names = np.asarray([str(value) for value in type_array], dtype=object)
            n_receptors = len(type_names)
        elif sensitivities is not None:
            sensitivity_array = np.asarray(sensitivities, dtype=float)
            if sensitivity_array.ndim == 1:
                sensitivity_array = sensitivity_array[None, :]
            if sensitivity_array.ndim != 2 or sensitivity_array.shape[1] != 4:
                raise ValueError("sensitivities must have shape (n_receptors, 4)")
            n_receptors = sensitivity_array.shape[0]
        else:
            # A standalone default: one R1-R6 receptor at every ommatidium.
            if receptor_columns is not None:
                column_array = np.asarray(receptor_columns)
                if column_array.ndim == 0:
                    column_array = column_array.reshape(1)
                if column_array.ndim != 1:
                    raise ValueError("receptor_columns must be one-dimensional")
                n_receptors = len(column_array)
            elif n_columns is not None:
                n_columns = _batch_size(n_columns)
                n_receptors = n_columns
            else:
                raise ValueError("vision metadata is empty")
            type_names = np.asarray(["R1-R6"] * n_receptors, dtype=object)

        if sensitivity_array is None:
            if type_names is None:
                raise ValueError("receptor_types or sensitivities is required")
            try:
                sensitivity_array = np.asarray(
                    [SPECTRAL_SENSITIVITY[str(value)] for value in type_names],
                    dtype=float,
                )
            except KeyError as exc:
                raise ValueError(f"unknown photoreceptor type: {exc.args[0]}") from exc
        else:
            sensitivity_array = np.asarray(sensitivity_array, dtype=float)
            if sensitivity_array.ndim != 2 or sensitivity_array.shape[1] != 4:
                raise ValueError("sensitivities must have shape (n_receptors, 4)")
            if not np.isfinite(sensitivity_array).all():
                raise ValueError("sensitivities must be finite")
        if len(sensitivity_array) != n_receptors:
            raise ValueError("sensitivity rows and receptor metadata must align")
        if len(sensitivity_array) == 0:
            raise ValueError("at least one photoreceptor is required")

        if receptor_columns is None:
            column_array = np.arange(n_receptors, dtype=np.intp)
        else:
            column_array = np.asarray(receptor_columns)
            if column_array.ndim == 0:
                column_array = column_array.reshape(1)
            if column_array.ndim != 1 or len(column_array) != n_receptors:
                raise ValueError("receptor_columns must have one entry per receptor")
            try:
                column_array = column_array.astype(np.intp, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("receptor_columns must contain integer indices") from exc
        if np.any(column_array < 0):
            raise ValueError("receptor_columns must be nonnegative")

        if n_columns is None:
            n_columns = int(column_array.max()) + 1
        n_columns = _batch_size(n_columns)
        if np.any(column_array >= n_columns):
            raise ValueError("receptor_columns contains an index outside n_columns")

        if families is None:
            if type_names is not None:
                try:
                    family_array = np.asarray(
                        [FAMILY_OF_TYPE[str(value)] for value in type_names], dtype=int
                    )
                except KeyError as exc:
                    raise ValueError(f"unknown photoreceptor family: {exc.args[0]}") from exc
            else:
                family_array = np.zeros(n_receptors, dtype=int)
        else:
            family_array = np.asarray(families)
            if family_array.ndim == 0:
                family_array = family_array.reshape(1)
            if family_array.ndim != 1 or len(family_array) != n_receptors:
                raise ValueError("families must have one entry per receptor")
            try:
                family_float = family_array.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("families must contain integer family indices") from exc
            if not np.isfinite(family_float).all() or not np.equal(family_float, np.floor(family_float)).all():
                raise ValueError("families must contain finite integer indices")
            family_array = family_float.astype(int)
        if np.any((family_array < 0) | (family_array >= 5)):
            raise ValueError("families must be in 0..4")

        self.n_columns = n_columns
        self.receptor_types = type_names
        self.receptor_columns = column_array.copy()
        self.sensitivities = sensitivity_array.copy()
        self.families = family_array.copy()
        rows = self.receptor_columns * 5 + self.families
        self._rows = rows
        self._has = np.zeros(n_columns * 5, dtype=bool)
        self._has[rows] = True
        self._groups = tuple(
            (int(row), np.flatnonzero(rows == row)) for row in np.unique(rows)
        )
        self._lazy = False
        self._reset_state()

    def _reset_state(self) -> None:
        self._batch = None
        self._fresh = None
        self._i_lp = None
        self._i_mean = None
        self._contrast = None
        self.last = {}

    def reset(self) -> None:
        """Reset the temporal filters so the next frame starts at zero contrast."""
        self._reset_state()

    def _normalise_radiance(self, radiance) -> np.ndarray:
        array = np.asarray(radiance, dtype=float)
        if array.ndim == 2:
            array = array[None, :, :]
        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError("radiance must have shape (n_columns, 4) or (batch, n_columns, 4)")
        if self.n_columns is not None and array.shape[1] != self.n_columns:
            raise ValueError(
                f"radiance has {array.shape[1]} ommatidia; encoder expects {self.n_columns}"
            )
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("radiance must be finite and nonnegative")
        return array

    def _ensure_geometry(self, radiance: np.ndarray) -> None:
        if not self._lazy:
            return
        n_columns = int(radiance.shape[1])
        if n_columns < 1:
            raise ValueError("at least one ommatidium is required")
        self._configure(
            ["R1-R6"] * n_columns,
            np.arange(n_columns, dtype=np.intp),
            n_columns,
            None,
            None,
        )

    def _intensity(self, radiance: np.ndarray) -> np.ndarray:
        selected = radiance[:, self.receptor_columns, :]
        return np.einsum("bpc,pc->bp", selected, self.sensitivities, optimize=True)

    @property
    def n_receptors(self) -> int:
        if self.receptor_columns is None:
            return 0
        return int(len(self.receptor_columns))

    def encode(self, radiance, frame_ms: float = 10.0, *, intensity=None) -> np.ndarray:
        """Encode one radiance frame and return per-photoreceptor contrast.

        The first frame after construction or ``reset`` has zero contrast, as
        in the source's ``_fresh`` initialization.  ``intensity`` optionally
        supplies pre-projected ``(batch, n_receptors)`` values, matching the
        hook in ``OpticLobe.photoreceptor_activity``.
        """
        radiance_array = self._normalise_radiance(radiance)
        self._ensure_geometry(radiance_array)
        batch = int(radiance_array.shape[0])
        frame_ms = float(frame_ms)
        if not np.isfinite(frame_ms) or frame_ms <= 0.0:
            raise ValueError("frame_ms must be positive and finite")

        if intensity is None:
            intensity_array = self._intensity(radiance_array)
        else:
            intensity_array = np.asarray(intensity, dtype=float)
            if intensity_array.ndim == 1:
                intensity_array = intensity_array[None, :]
            if intensity_array.shape != (batch, self.n_receptors):
                raise ValueError(
                    f"intensity must have shape ({batch}, {self.n_receptors})"
                )
            if not np.isfinite(intensity_array).all():
                raise ValueError("intensity must be finite")

        if self._i_lp is not None and self._batch != batch:
            raise ValueError("a VisionEncoder cannot change batch size without reset")
        if self._i_lp is None:
            bins = self.n_columns * 5
            self._i_lp = np.zeros((batch, bins), dtype=float)
            self._i_mean = np.zeros((batch, bins), dtype=float)
            self._fresh = np.ones((batch,), dtype=bool)
            self._batch = batch

        current = np.zeros((batch, self.n_columns * 5), dtype=float)
        for row, columns in self._groups:
            current[:, row] = intensity_array[:, columns].mean(axis=1)

        # This is the same fresh-frame rule as optic.py: initialize both filters
        # to the current intensity before applying this frame's coefficients.
        lowpass_previous = np.where(self._fresh[:, None], current, self._i_lp)
        mean_previous = np.where(self._fresh[:, None], current, self._i_mean)
        a_lp = float(np.exp(-frame_ms / self.tau_lp_ms))
        a_adapt = float(np.exp(-frame_ms / self.tau_adapt_ms))
        self._i_lp = a_lp * lowpass_previous + (1.0 - a_lp) * current
        self._i_mean = a_adapt * mean_previous + (1.0 - a_adapt) * self._i_lp
        self._fresh[:] = False

        contrast = (self._i_lp - self._i_mean) / (self._i_mean + self.eps)
        contrast = np.clip(contrast, -1.0, self.contrast_clip)
        contrast[:, ~self._has] = 0.0
        self._contrast = contrast
        self.last = {
            "intensity": intensity_array.copy(),
            "ommatidium_intensity": current.copy(),
            "contrast": contrast.copy(),
        }
        return contrast[:, self._rows]

    def photoreceptor_activity(self, radiance, frame_ms: float = 10.0, *, intensity=None) -> np.ndarray:
        return self.encode(radiance, frame_ms, intensity=intensity)

    def photoreceptor_drive(self, radiance, frame_ms: float = 10.0, *, intensity=None) -> np.ndarray:
        return self.encode(radiance, frame_ms, intensity=intensity)


# ---------------------------------------------------------------------------
# Smell


def _mapping_keys(*values) -> list[str]:
    """Return stable, normalized mapping keys while preserving input order."""
    result = []
    seen = set()
    for mapping in values:
        if not isinstance(mapping, Mapping):
            raise TypeError("concentrations must be dictionaries keyed by glomerulus")
        for key in mapping:
            name = _glomerulus_name(key)
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def _orn_spec(orn_glomeruli, orn_sides):
    """Accept a name array, a name-to-side mapping, or (name, side) pairs."""
    if orn_glomeruli is None:
        return None, None
    if isinstance(orn_glomeruli, Mapping) and orn_sides is None:
        return list(orn_glomeruli.keys()), list(orn_glomeruli.values())
    if orn_sides is None:
        array = np.asarray(orn_glomeruli, dtype=object)
        if array.ndim == 2 and array.shape[1] == 2:
            return array[:, 0].tolist(), array[:, 1].tolist()
    return orn_glomeruli, orn_sides


class SmellEncoder:
    """Concentration-per-glomerulus-to-ORN-rate transducer.

    ``orn_sides`` is the source graph's signed laterality: 1 is left, -1 is
    right, and 0 is bilateral.  Use :func:`smell_sides_from_weights` when only
    aggregate left/right projection weights are available.
    """

    def __init__(
        self,
        orn_glomeruli,
        orn_sides=None,
        *,
        base_hz: float = DEFAULT_SMELL_BASE_HZ,
        max_hz: float = DEFAULT_SMELL_MAX_HZ,
        half_conc: float = DEFAULT_SMELL_HALF_CONC,
    ) -> None:
        names, sides = _orn_spec(orn_glomeruli, orn_sides)
        if names is None:
            raise ValueError("orn_glomeruli is required")
        if isinstance(names, str):
            names = [names]
        name_array = np.asarray([_glomerulus_name(value) for value in names], dtype=object)
        if name_array.ndim != 1:
            raise ValueError("orn_glomeruli must be one-dimensional")
        if sides is None:
            sides = np.zeros(len(name_array), dtype=int)
        self.glomeruli = name_array
        self.sides = _side_values(sides, "orn_sides")
        if len(self.sides) != len(self.glomeruli):
            raise ValueError("orn_sides must have one entry per ORN")
        self.base_hz = _nonnegative_finite(base_hz, "base_hz")
        self.max_hz = _nonnegative_finite(max_hz, "max_hz")
        self.half_conc = float(half_conc)
        if not np.isfinite(self.half_conc) or self.half_conc <= 0.0:
            raise ValueError("half_conc must be positive and finite")

    @property
    def n_orns(self) -> int:
        return int(len(self.glomeruli))

    def _expand(self, values, batch: int, name: str) -> np.ndarray:
        if not isinstance(values, Mapping):
            raise TypeError("concentrations must be dictionaries keyed by glomerulus")
        result = np.zeros((batch, self.n_orns), dtype=float)
        for key, value in values.items():
            vector = _batch_values(value, batch, f"{name}[{key!r}]")
            if np.any(vector < 0.0):
                raise ValueError("concentrations must be nonnegative")
            normalized = _glomerulus_name(key)
            result[:, self.glomeruli == normalized] = vector[:, None]
        return result

    def encode(self, cL: Mapping, cR: Mapping, batch: int = 1) -> np.ndarray:
        """Return ORN rates with shape ``(batch, n_orns)``."""
        batch = _batch_size(batch)
        left = self._expand(cL, batch, "left concentration")
        right = self._expand(cR, batch, "right concentration")
        concentration = np.where(
            self.sides[None, :] > 0,
            left,
            np.where(self.sides[None, :] < 0, right, 0.5 * (left + right)),
        )
        return self.base_hz + self.max_hz * concentration / (concentration + self.half_conc)

    def rates(self, cL: Mapping, cR: Mapping, batch: int = 1) -> np.ndarray:
        return self.encode(cL, cR, batch=batch)


def smell_sides_from_weights(left_weights, right_weights, side_threshold: float = 0.2) -> np.ndarray:
    """Apply the source smell laterality rule to aggregate ORN weights."""
    left = np.asarray(left_weights, dtype=float)
    right = np.asarray(right_weights, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left_weights and right_weights must be equal-length vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("weights must be finite")
    if not np.isfinite(side_threshold) or side_threshold < 0.0:
        raise ValueError("side_threshold must be finite and nonnegative")
    laterality = (left - right) / np.maximum(left + right, 1.0)
    return np.where(laterality > side_threshold, 1, np.where(laterality < -side_threshold, -1, 0))


# ---------------------------------------------------------------------------
# Taste


class TasteEncoder:
    """Sugar-contact-to-sweet-GRN-rate transducer.

    ``sweet_idx`` can be the source neuron-index array or simply a population
    size.  The source's shipped population is 165 cells; the standalone module
    does not load its CSV, so the caller supplies the retained count or IDs.
    """

    def __init__(self, sweet_idx=None, *, max_hz: float = DEFAULT_TASTE_MAX_HZ) -> None:
        if sweet_idx is None:
            n_grns = 1
        elif isinstance(sweet_idx, (int, np.integer)) and not isinstance(sweet_idx, (bool, np.bool_)):
            n_grns = int(sweet_idx)
        else:
            n_grns = len(sweet_idx)
        if isinstance(n_grns, (bool, np.bool_)) or not isinstance(n_grns, (int, np.integer)) or n_grns < 0:
            raise ValueError("sweet_idx must be a nonnegative population count or sequence")
        self.sweet_idx = None if sweet_idx is None else np.asarray(sweet_idx)
        self.n_grns = int(n_grns)
        self.max_hz = _nonnegative_finite(max_hz, "max_hz")

    def encode(self, sugar, batch: int = 1) -> np.ndarray:
        """Return GRN rates with shape ``(batch, n_grns)``."""
        contact = _batch_values(sugar, batch, "sugar contact")
        if np.any((contact < 0.0) | (contact > 1.0)):
            raise ValueError("sugar contact must be between 0 and 1")
        return np.repeat(contact[:, None] * self.max_hz, self.n_grns, axis=1)

    def rates(self, sugar, batch: int = 1) -> np.ndarray:
        return self.encode(sugar, batch=batch)


# ---------------------------------------------------------------------------
# Wind


class WindEncoder:
    """Antennal-deflection-to-JO-rate transducer.

    ``side_e`` and ``side_c`` are signed source-graph laterality arrays for
    JO-E and JO-C cells.  JO-C receives the inverted deflection, exactly as in
    ``flyverse.senses.Wind``.
    """

    def __init__(
        self,
        side_e,
        side_c=None,
        *,
        base_hz: float = DEFAULT_WIND_BASE_HZ,
        max_hz: float = DEFAULT_WIND_MAX_HZ,
    ) -> None:
        if side_e is None:
            if side_c is None:
                raise ValueError("side_e is required")
            side_e = side_c
        self.side_e = _side_values(side_e, "side_e")
        if side_c is None:
            self.side_c = self.side_e.copy()
        else:
            self.side_c = _side_values(side_c, "side_c")
            if len(self.side_c) != len(self.side_e):
                raise ValueError("side_e and side_c must have equal length")
        self.base_hz = _nonnegative_finite(base_hz, "base_hz")
        self.max_hz = _nonnegative_finite(max_hz, "max_hz")

    def encode(self, dL, dR, batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(JO-E rates, JO-C rates)``, each shape ``(batch, n)``."""
        batch = _batch_size(batch)
        left = _batch_values(dL, batch, "left deflection")[:, None]
        right = _batch_values(dR, batch, "right deflection")[:, None]

        def rate(deflection: np.ndarray) -> np.ndarray:
            return self.base_hz + self.max_hz * np.clip(deflection, 0.0, 1.0)

        e_side = self.side_e[None, :]
        c_side = self.side_c[None, :]
        r_e = np.where(
            e_side > 0,
            rate(left),
            np.where(e_side < 0, rate(right), rate(0.5 * (left + right))),
        )
        r_c = np.where(
            c_side > 0,
            rate(-left),
            np.where(c_side < 0, rate(-right), rate(-0.5 * (left + right))),
        )
        return r_e, r_c

    def rates(self, dL, dR, batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        return self.encode(dL, dR, batch=batch)


def wind_sides_from_weights(left_weights, right_weights) -> np.ndarray:
    """Apply the source wind rule: signed laterality without a threshold."""
    left = np.asarray(left_weights, dtype=float)
    right = np.asarray(right_weights, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left_weights and right_weights must be equal-length vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("weights must be finite")
    laterality = (left - right) / np.maximum(left + right, 1.0)
    return np.where(laterality > 0.0, 1, np.where(laterality < 0.0, -1, 0))


# ---------------------------------------------------------------------------
# Functional surface


def encode_vision(
    radiance,
    receptor_types=None,
    receptor_columns=None,
    *,
    n_columns: int | None = None,
    sensitivities=None,
    families=None,
    encoder: VisionEncoder | None = None,
    frame_ms: float = 10.0,
    tau_lp_ms: float = DEFAULT_TAU_LP_MS,
    tau_adapt_ms: float = DEFAULT_TAU_ADAPT_MS,
    eps: float = DEFAULT_EPS,
    contrast_clip: float = DEFAULT_CONTRAST_CLIP,
) -> np.ndarray:
    """One-shot vision encoding, or one step of an existing ``VisionEncoder``."""
    if encoder is not None:
        if any(value is not None for value in (receptor_types, receptor_columns, n_columns, sensitivities, families)):
            raise TypeError("metadata cannot be supplied with an existing encoder")
        return encoder.encode(radiance, frame_ms)
    if sensitivities is None and receptor_types is not None:
        candidate = np.asarray(receptor_types)
        if candidate.ndim == 2:
            sensitivities = candidate
            receptor_types = None
    return VisionEncoder(
        receptor_types,
        receptor_columns,
        n_columns=n_columns,
        sensitivities=sensitivities,
        families=families,
        tau_lp_ms=tau_lp_ms,
        tau_adapt_ms=tau_adapt_ms,
        eps=eps,
        contrast_clip=contrast_clip,
    ).encode(radiance, frame_ms)


def encode_smell(
    cL: Mapping,
    cR: Mapping,
    orn_glomeruli=None,
    orn_sides=None,
    *,
    batch: int = 1,
    base_hz: float = DEFAULT_SMELL_BASE_HZ,
    max_hz: float = DEFAULT_SMELL_MAX_HZ,
    half_conc: float = DEFAULT_SMELL_HALF_CONC,
) -> np.ndarray:
    """One-shot smell encoding.

    If ORN metadata is omitted, each unique normalized glomerulus key gets one
    bilateral ORN channel.  This is a convenient standalone fallback; a real
    connectome should pass its ORN glomerulus and laterality arrays.
    """
    names, sides = _orn_spec(orn_glomeruli, orn_sides)
    if names is None:
        names = _mapping_keys(cL, cR)
    return SmellEncoder(
        names,
        sides,
        base_hz=base_hz,
        max_hz=max_hz,
        half_conc=half_conc,
    ).encode(cL, cR, batch=batch)


def encode_taste(
    sugar,
    n_grns: int = 1,
    *,
    batch: int = 1,
    max_hz: float = DEFAULT_TASTE_MAX_HZ,
) -> np.ndarray:
    """One-shot sugar-contact encoding with shape ``(batch, n_grns)``."""
    return TasteEncoder(n_grns, max_hz=max_hz).encode(sugar, batch=batch)


def encode_wind(
    dL,
    dR,
    side_e=None,
    side_c=None,
    *,
    sides=None,
    jo_sides=None,
    batch: int = 1,
    base_hz: float = DEFAULT_WIND_BASE_HZ,
    max_hz: float = DEFAULT_WIND_MAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot wind encoding, returning ``(JO-E rates, JO-C rates)``.

    ``side_e`` and ``side_c`` are the explicit JO populations.  ``sides`` (or
    ``jo_sides``) is an alias for one shared side array, which is useful for a
    compact fixture.  With no metadata, one left-sided JO cell is used.
    """
    if sides is not None and jo_sides is not None:
        raise TypeError("use only one of sides and jo_sides")
    shared = sides if sides is not None else jo_sides
    if shared is not None:
        if isinstance(shared, Mapping):
            if side_e is None:
                side_e = shared.get("E", shared.get("JO-E"))
            if side_c is None:
                side_c = shared.get("C", shared.get("JO-C"))
        elif side_e is None:
            side_e = shared
    if side_e is None:
        side_e = np.array([1], dtype=int) if side_c is None else side_c
    if side_c is None:
        side_c = side_e
    return WindEncoder(
        side_e,
        side_c,
        base_hz=base_hz,
        max_hz=max_hz,
    ).encode(dL, dR, batch=batch)


# Source-style names are kept as aliases so callers can use either the layer
# terminology ("drive") or the optic terminology ("activity").
photoreceptor_activity = encode_vision
vision_drive = encode_vision
smell_drive = encode_smell
taste_drive = encode_taste
wind_drive = encode_wind

__all__ = [
    "SPECTRAL_SENSITIVITY",
    "FAMILY_OF_TYPE",
    "VisionEncoder",
    "SmellEncoder",
    "TasteEncoder",
    "WindEncoder",
    "smell_sides_from_weights",
    "wind_sides_from_weights",
    "encode_vision",
    "encode_smell",
    "encode_taste",
    "encode_wind",
    "photoreceptor_activity",
    "vision_drive",
    "smell_drive",
    "taste_drive",
    "wind_drive",
]


# ---------------------------------------------------------------------------
# Smoke test


def _smoke() -> None:
    """Encode one synthetic stimulus through all four sensory paths."""
    # Two frames exercise the fresh-frame and temporal-contrast behavior.
    vision = VisionEncoder(
        receptor_types=("R1-R6", "R7p", "R8y"),
        receptor_columns=(0, 0, 0),
        n_columns=1,
    )
    first = np.array([[0.10, 0.20, 0.30, 0.05]])
    second = np.array([[0.30, 0.20, 0.50, 0.05]])
    first_drive = vision.encode(first, frame_ms=10.0)
    second_drive = vision.encode(second, frame_ms=10.0)
    assert first_drive.shape == (1, 3)
    assert np.allclose(first_drive, 0.0)
    assert second_drive.shape == (1, 3) and np.isfinite(second_drive).all()

    smell = SmellEncoder(("DM1", "DM1", "DM1"), (1, -1, 0)).encode(
        {"DM1": 0.5}, {"DM1": 1.0}
    )
    np.testing.assert_allclose(smell, [[76.0, 101.0, 91.0]])

    taste = encode_taste(np.array([0.0, 0.25, 1.0]), n_grns=3, batch=3)
    np.testing.assert_allclose(
        taste,
        [[0.0, 0.0, 0.0], [30.0, 30.0, 30.0], [120.0, 120.0, 120.0]],
    )

    wind_e, wind_c = encode_wind(1.0, 0.0, sides=(1, -1, 0))
    np.testing.assert_allclose(wind_e, [[52.0, 2.0, 27.0]])
    np.testing.assert_allclose(wind_c, [[2.0, 2.0, 2.0]])

    print("DONE")
    print("Vision: spectral projection, family averaging, adaptation, and contrast clipping.")
    print("Smell, taste, and wind: source Hz laws with source-side selection and validation.")
    print("Smoke test: one synthetic stimulus produced finite drive for all four senses.")


if __name__ == "__main__":
    _smoke()
