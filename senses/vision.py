"""A small, NumPy-only visual pipeline for the fly simulation.

The pipeline is deliberately small, but keeps the useful biological ordering:

``16 x 16`` ommatidia -> temporal photoreceptor contrast -> histamine-corrected
ON/OFF drive -> center-surround filtering -> direction and looming readouts.

There are two important approximations:

* the photoreceptor stage reuses :class:`senses.encoders.VisionEncoder`, but
  the downstream circuits are hand-sized correlation and scale-change
  detectors rather than the fly's full optic-lobe connectome;
* the histamine fix is an edge-sign correction.  The loader applies it to
  connectome rows, while this module applies the same function to a one-target
  relay matrix and then interprets the inhibitory relay as ON/OFF polarity.
  It does not claim to model the complete histamine/connectin receptor biology.

``connectome.loader.load_connectome`` exposes ``W``, ``ids``, ``n``, and
``id_to_index``.  It does *not* return the sidecar-derived photoreceptor ID set;
:func:`photoreceptor_rows_from_connectome` accepts that optional annotation
and resolves it through the loader's public ``id_to_index`` mapping.

Run this file directly for the moving-bar and looming-flash checks.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

# Keep ``python senses/vision.py`` as convenient as ``python -m senses.vision``.
# The imported local modules themselves use only NumPy.
if __package__ in (None, ""):
    _REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, _REPOSITORY_ROOT)

import numpy as np

from fixes.histamine import PHOTORECEPTOR_TYPES, correct_photoreceptor_sign
from senses.encoders import VisionEncoder


DEFAULT_HEIGHT = 16
DEFAULT_WIDTH = 16
DEFAULT_FRAME_MS = 10.0

# Vectors are (dy, dx), with image rows increasing downward.  Keeping eight
# directions is a cheap approximation of the fly's many direction-tuned cells.
DIRECTION_VECTORS = np.asarray(
    [
        (0, 1),  # right
        (1, 1),  # down-right
        (1, 0),  # down
        (1, -1),  # down-left
        (0, -1),  # left
        (-1, -1),  # up-left
        (-1, 0),  # up
        (-1, 1),  # up-right
    ],
    dtype=int,
)
DIRECTION_NAMES = (
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
    "up",
    "up-right",
)

DEFAULT_LOOM_SCALES = (0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)

__all__ = [
    "DIRECTION_NAMES",
    "DIRECTION_VECTORS",
    "DEFAULT_LOOM_SCALES",
    "VisualPipeline",
    "center_surround",
    "direction_readout",
    "histamine_corrected_activity",
    "histamine_corrected_on_off",
    "looming_detector",
    "photoreceptor_rows_from_connectome",
]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_frame(frame, height: int, width: int) -> np.ndarray:
    """Return one non-negative ``[UV, B, G, R]`` frame as float64."""

    array = np.asarray(frame, dtype=float)
    if array.ndim == 2:
        # A luminance stimulus is useful for small simulations and is treated
        # as equal energy in all four channels.
        array = np.repeat(array[..., None], 4, axis=2)
    if array.ndim != 3 or array.shape != (height, width, 4):
        raise ValueError(
            "a frame must have shape "
            f"({height}, {width}, 4) or ({height}, {width}); "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("a frame must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError("radiance must be non-negative")
    return array


def _canonical_id(value) -> str:
    """Match the loader's stable string key for ordinary and numeric IDs."""

    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    return str(value)


def _requested_ids(photoreceptor_ids: Iterable) -> set[str]:
    if isinstance(photoreceptor_ids, (str, bytes, np.str_)):
        values = [photoreceptor_ids]
    else:
        values = list(photoreceptor_ids)
    return {_canonical_id(value) for value in values}


def photoreceptor_rows_from_connectome(
    connectome: Mapping,
    photoreceptor_ids: Iterable | None,
) -> np.ndarray:
    """Resolve receptor rows from the result returned by ``load_connectome``.

    The loader's public result contains an index-aligned ``ids`` list and an
    ``id_to_index`` mapping, but intentionally does not expose its optional
    sidecar's photoreceptor-ID set.  Pass that set (or another annotation) as
    ``photoreceptor_ids`` and this helper performs only the public lookup.

    The returned indices are sorted and de-duplicated.  An empty array means
    that no requested ID occurs in this connectome.
    """

    if not isinstance(connectome, Mapping):
        raise TypeError("connectome must be the mapping returned by load_connectome")
    if photoreceptor_ids is None:
        return np.empty(0, dtype=np.intp)
    requested = _requested_ids(photoreceptor_ids)
    if not requested:
        return np.empty(0, dtype=np.intp)

    rows: set[int] = set()
    id_to_index = connectome.get("id_to_index")
    if isinstance(id_to_index, Mapping):
        for raw_id, raw_index in id_to_index.items():
            if _canonical_id(raw_id) in requested:
                try:
                    rows.add(int(raw_index))
                except (TypeError, ValueError) as exc:
                    raise ValueError("id_to_index contains a non-integer index") from exc

    # ``ids`` is the loader's other public ID surface.  It is a useful fallback
    # for small test doubles that omit id_to_index.
    if not rows:
        ids = connectome.get("ids")
        if ids is not None:
            for index, raw_id in enumerate(ids):
                if raw_id is not None and _canonical_id(raw_id) in requested:
                    rows.add(index)

    n = connectome.get("n")
    if n is not None:
        n = int(n)
        rows = {index for index in rows if 0 <= index < n}
    return np.asarray(sorted(rows), dtype=np.intp)


def _histamine_type_array(
    receptor_types, count: int
) -> np.ndarray:
    """Map encoder labels to the exact strings used by the source histamine fix."""

    if receptor_types is None:
        values = np.full(count, "R1-6", dtype=object)
    else:
        values = np.asarray(receptor_types, dtype=object)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f"receptor_types must contain {count} labels")

    # VisionEncoder calls the major class R1-R6; the connectome sidecar calls
    # the same class R1-6.  Preserve the fix's exact-match requirement here.
    aliases = {"R1-R6": "R1-6", "R1-6": "R1-6", "R7": "R7", "R8": "R8"}
    canonical = []
    for value in values:
        name = str(value)
        name = aliases.get(name, name)
        if name not in PHOTORECEPTOR_TYPES:
            raise ValueError(
                "receptor_types must be canonical histamine labels "
                "R1-6, R7, or R8"
            )
        canonical.append(name)
    return np.asarray(canonical, dtype=object)


def _flat_activity(activity) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(activity, dtype=float)
    if array.ndim == 0 or array.size == 0:
        raise ValueError("activity must contain at least one photoreceptor")
    if not np.isfinite(array).all():
        raise ValueError("activity must be finite")
    shape = tuple(int(size) for size in array.shape)
    return array.reshape(-1), shape


def histamine_corrected_activity(
    activity,
    receptor_types=None,
) -> np.ndarray:
    """Apply the source histamine sign correction to a synthetic relay edge.

    ``correct_photoreceptor_sign`` expects a signed edge matrix, while a frame
    is an activity vector.  We therefore make the smallest useful edge matrix:
    one positive outgoing relay weight per receptor.  The source fix changes
    those weights to ``-1`` while retaining their magnitude.  The returned
    signed relay is consequently negative for increased light drive and
    positive for decreased light drive; downstream code can split that relay
    into ON and OFF responses.

    This is intentionally an explicit approximation, not a claim that every
    photoreceptor has exactly one synapse or that the complete optic lobe is a
    single relay.
    """

    flat, shape = _flat_activity(activity)
    types = _histamine_type_array(receptor_types, len(flat))
    relay_weights = np.ones((len(flat), 1), dtype=float)
    corrected_weights, _ = correct_photoreceptor_sign(types, relay_weights)
    relay = flat * corrected_weights[:, 0]
    return relay.reshape(shape)


def histamine_corrected_on_off(
    activity,
    receptor_types=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rectified ON and OFF maps after the histamine sign correction.

    A positive light contrast is treated as an ON event and a negative contrast
    as an OFF event.  The sign is obtained by using the inhibitory histamine
    relay, rather than hard-coding the opposite polarity in this function.
    """

    relay = histamine_corrected_activity(activity, receptor_types)
    on = np.maximum(-relay, 0.0)
    off = np.maximum(relay, 0.0)
    return on, off


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Edge-padded box mean implemented with NumPy slicing only."""

    if radius == 0:
        return image.astype(float, copy=True)
    padded = np.pad(image, radius, mode="edge")
    height, width = image.shape
    total = np.zeros((height, width), dtype=float)
    count = float((2 * radius + 1) ** 2)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            total += padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return total / count


def center_surround(
    image,
    *,
    center_radius: int = 1,
    surround_radius: int = 2,
) -> np.ndarray:
    """Apply a small center-surround (difference-of-boxes) filter.

    The two box means are not a full Laplacian pyramid or a faithful model of
    lamina neurons.  They are a compact, explainable spatial filter: a local
    bright/dark center is emphasized relative to its larger neighborhood.
    Edge padding keeps the grid size fixed.
    """

    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("image must be finite")
    center_radius = _positive_int(center_radius, "center_radius") - 1
    surround_radius = _positive_int(surround_radius, "surround_radius") - 1
    if surround_radius < center_radius:
        raise ValueError("surround_radius must be at least center_radius")
    center = _box_mean(array, center_radius)
    surround = _box_mean(array, surround_radius)
    return center - surround


def _shift_for_direction(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an image so it predicts motion by clipping at the border."""

    height, width = image.shape
    source_y = np.clip(np.arange(height) - dy, 0, height - 1)
    source_x = np.clip(np.arange(width) - dx, 0, width - 1)
    return image[np.ix_(source_y, source_x)]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - float(first.mean())
    second_centered = second - float(second.mean())
    denominator = float(
        np.sqrt(np.sum(first_centered * first_centered) * np.sum(second_centered * second_centered))
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _positive_centroid(image: np.ndarray) -> np.ndarray | None:
    positive = np.maximum(image, 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return None
    yy, xx = np.indices(image.shape, dtype=float)
    return np.asarray(
        [np.sum(xx * positive) / total, np.sum(yy * positive) / total],
        dtype=float,
    )


def direction_readout(
    previous,
    current,
    *,
    min_motion_energy: float = 1e-8,
) -> dict:
    """Estimate image-plane motion by shifted correlations and centroid flow.

    ``previous`` and ``current`` are typically center-surround-filtered ON/OFF
    maps.  A shift by ``(dy, dx)`` samples the previous frame one position in
    the opposite direction, so a bar moving right scores highest for ``right``.
    A small positive-signal centroid displacement breaks the ties that a
    perfectly uniform bar creates between axial and diagonal shifts.  This is
    still only a direction-tuned cell approximation, not a full
    elementary-motion circuit with separable ON and OFF subunits.
    """

    current_array = np.asarray(current, dtype=float)
    if current_array.ndim != 2 or current_array.size == 0:
        raise ValueError("current must be a non-empty two-dimensional array")
    if not np.isfinite(current_array).all():
        raise ValueError("current must be finite")

    if previous is None:
        previous_array = np.zeros_like(current_array)
    else:
        previous_array = np.asarray(previous, dtype=float)
        if previous_array.shape != current_array.shape:
            raise ValueError("previous and current must have the same shape")
        if not np.isfinite(previous_array).all():
            raise ValueError("previous must be finite")

    energy = float(np.mean(np.abs(current_array - previous_array)))
    reference_energy = float(
        np.mean(np.abs(current_array) + np.abs(previous_array))
    )
    relative_energy = energy / (reference_energy + 1e-12)
    correlation_scores = np.asarray(
        [
            _correlation(
                current_array,
                _shift_for_direction(previous_array, int(dy), int(dx)),
            )
            for dy, dx in DIRECTION_VECTORS
        ],
        dtype=float,
    )

    current_centroid = _positive_centroid(current_array)
    previous_centroid = _positive_centroid(previous_array)
    centroid_displacement = np.zeros(2, dtype=float)
    centroid_projections = np.zeros(len(DIRECTION_VECTORS), dtype=float)
    if current_centroid is not None and previous_centroid is not None:
        centroid_displacement = current_centroid - previous_centroid
        displacement_norm = float(np.linalg.norm(centroid_displacement))
        if displacement_norm > 1e-6:
            direction_units = np.column_stack(
                (DIRECTION_VECTORS[:, 1], DIRECTION_VECTORS[:, 0])
            ).astype(float)
            direction_units /= np.linalg.norm(direction_units, axis=1, keepdims=True)
            centroid_projections = direction_units @ centroid_displacement / displacement_norm
            centroid_weight = 0.8 * min(1.0, displacement_norm / 0.5)
            scores = (
                (1.0 - centroid_weight) * correlation_scores
                + centroid_weight * centroid_projections
            )
        else:
            scores = correlation_scores.copy()
    else:
        scores = correlation_scores.copy()

    if energy <= min_motion_energy or reference_energy <= min_motion_energy:
        return {
            "direction": (0.0, 0.0),  # (dx, dy), for caller convenience
            "direction_image": (0, 0),  # (dy, dx), matching array indexing
            "label": "stationary",
            "angle_degrees": 0.0,
            "scores": np.zeros_like(scores),
            "correlation_scores": np.zeros_like(correlation_scores),
            "centroid_displacement": (0.0, 0.0),  # (dx, dy)
            "best_score": 0.0,
            "motion_energy": energy,
            "motion_strength": relative_energy,
        }

    best = int(np.argmax(scores))
    dy, dx = DIRECTION_VECTORS[best]
    # Conventional image-plane angle: right is 0 degrees and up is positive.
    angle = float(np.degrees(np.arctan2(-int(dy), int(dx))))
    return {
        "direction": (float(dx), float(dy)),
        "direction_image": (int(dy), int(dx)),
        "label": DIRECTION_NAMES[best],
        "angle_degrees": angle,
        "scores": scores,
        "correlation_scores": correlation_scores,
        "centroid_displacement": (
            float(centroid_displacement[0]),
            float(centroid_displacement[1]),
        ),
        "best_score": float(scores[best]),
        "motion_energy": energy,
        "motion_strength": relative_energy,
    }


def _sample_bilinear(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2-D image with clipped coordinates."""

    height, width = image.shape
    y = np.clip(y, 0.0, height - 1.0)
    x = np.clip(x, 0.0, width - 1.0)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = y - y0
    wx = x - x0
    return (
        (1.0 - wy) * (1.0 - wx) * image[y0, x0]
        + (1.0 - wy) * wx * image[y0, x1]
        + wy * (1.0 - wx) * image[y1, x0]
        + wy * wx * image[y1, x1]
    )


def _scaled_about_center(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape
    center_y = (height - 1.0) / 2.0
    center_x = (width - 1.0) / 2.0
    y = (np.arange(height, dtype=float) - center_y) / scale + center_y
    x = (np.arange(width, dtype=float) - center_x) / scale + center_x
    return _sample_bilinear(
        image,
        y[:, None] * np.ones((1, width)),
        np.ones((height, 1)) * x[None, :],
    )


def _radial_spread(image: np.ndarray) -> float:
    positive = np.maximum(image, 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return 0.0
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=float)
    center_y = (height - 1.0) / 2.0
    center_x = (width - 1.0) / 2.0
    return float(
        np.sum(positive * ((yy - center_y) ** 2 + (xx - center_x) ** 2)) / total
    )


def looming_detector(
    previous,
    current,
    *,
    scales: Iterable[float] = DEFAULT_LOOM_SCALES,
    min_energy: float = 1e-5,
    min_correlation: float = 0.35,
    expansion_threshold: float = 0.04,
) -> dict:
    """Detect approximate looming by scale correlation and radial spread.

    If an object expands, sampling the previous frame at a scale greater than
    one should predict the current frame.  The radial-spread term catches a
    simpler expanding silhouette when the scale fit is noisy.  This is a
    stand-in for expansion-sensitive looming neurons; it does not model their
    receptive fields, adaptation, or the fly's complete multisensory logic.
    """

    current_array = np.asarray(current, dtype=float)
    if current_array.ndim != 2 or current_array.size == 0:
        raise ValueError("current must be a non-empty two-dimensional array")
    if not np.isfinite(current_array).all():
        raise ValueError("current must be finite")
    if previous is None:
        previous_array = np.zeros_like(current_array)
    else:
        previous_array = np.asarray(previous, dtype=float)
        if previous_array.shape != current_array.shape:
            raise ValueError("previous and current must have the same shape")
        if not np.isfinite(previous_array).all():
            raise ValueError("previous must be finite")

    scale_values = np.asarray(list(scales), dtype=float)
    if scale_values.ndim != 1 or scale_values.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional iterable")
    if not np.isfinite(scale_values).all() or np.any(scale_values <= 0.0):
        raise ValueError("scales must be positive and finite")
    scale_values = np.unique(scale_values)

    current_positive = np.maximum(current_array, 0.0)
    previous_positive = np.maximum(previous_array, 0.0)
    current_energy = float(current_positive.mean())
    previous_energy = float(previous_positive.mean())
    baseline_index = int(np.argmin(np.abs(scale_values - 1.0)))
    correlations = np.asarray(
        [
            _correlation(current_positive, _scaled_about_center(previous_positive, float(scale)))
            for scale in scale_values
        ],
        dtype=float,
    )

    spread_current = _radial_spread(current_positive)
    spread_previous = _radial_spread(previous_positive)
    spread_gain = max(0.0, spread_current - spread_previous)
    best_index = int(np.argmax(correlations))
    best_scale = float(scale_values[best_index])
    baseline_correlation = float(correlations[baseline_index])
    best_correlation = float(correlations[best_index])
    shape_gain = max(0.0, best_correlation - baseline_correlation)
    energy_factor = min(1.0, current_energy / max(min_energy * 5.0, 1e-3))
    scale_gain = min(1.0, max(0.0, best_scale - 1.0) / 0.75)
    spread_factor = min(1.0, spread_gain / 4.0)
    score = float(max(0.0, 0.65 * scale_gain + 0.35 * spread_factor) * energy_factor)

    active = (
        current_energy > min_energy
        and previous_energy > min_energy
        and best_scale > 1.05
        and best_correlation >= min_correlation
        and (shape_gain >= expansion_threshold or spread_gain >= 0.25)
    )
    return {
        "looming": bool(active and score > 0.0),
        "score": score,
        "best_scale": best_scale,
        "best_correlation": best_correlation,
        "baseline_correlation": baseline_correlation,
        "correlation_gain": shape_gain,
        "current_energy": current_energy,
        "previous_energy": previous_energy,
        "spread_current": spread_current,
        "spread_previous": spread_previous,
        "spread_gain": spread_gain,
        "correlations": correlations,
    }


class VisualPipeline:
    """A compact stateful visual pipeline for a rectangular receptor grid.

    ``radiance`` is a ``[UV, B, G, R]`` frame.  The encoder's temporal filters
    turn it into photoreceptor contrast, the histamine helper creates rectified
    ON/OFF maps, and the remaining stages produce a direction/looming readout.
    The grid is deliberately independent of the full FlyWire graph; use
    :func:`photoreceptor_rows_from_connectome` when connecting this code to
    loader output.
    """

    def __init__(
        self,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        *,
        frame_ms: float = DEFAULT_FRAME_MS,
        center_radius: int = 1,
        surround_radius: int = 2,
        loom_scales: Iterable[float] = DEFAULT_LOOM_SCALES,
        loom_min_energy: float = 1e-5,
    ) -> None:
        self.height = _positive_int(height, "height")
        self.width = _positive_int(width, "width")
        self.frame_ms = float(frame_ms)
        if not np.isfinite(self.frame_ms) or self.frame_ms <= 0.0:
            raise ValueError("frame_ms must be positive and finite")
        self.center_radius = center_radius
        self.surround_radius = surround_radius
        # Validate the filter radii now rather than on the first frame.
        center_surround(
            np.zeros((self.height, self.width), dtype=float),
            center_radius=center_radius,
            surround_radius=surround_radius,
        )
        self.loom_scales = tuple(float(scale) for scale in loom_scales)
        self.loom_min_energy = float(loom_min_energy)
        if not np.isfinite(self.loom_min_energy) or self.loom_min_energy < 0.0:
            raise ValueError("loom_min_energy must be finite and nonnegative")

        n_ommatidia = self.height * self.width
        # VisionEncoder's spectral label is R1-R6; the histamine fix's exact
        # FlyWire cell_type label is R1-6.  Keep both explicit at the seam.
        self.receptor_types = np.asarray(
            ["R1-R6"] * n_ommatidia,
            dtype=object,
        )
        self._histamine_types = np.asarray(
            ["R1-6"] * n_ommatidia,
            dtype=object,
        )
        self.encoder = VisionEncoder(
            receptor_types=self.receptor_types,
            receptor_columns=np.arange(n_ommatidia, dtype=np.intp),
            n_columns=n_ommatidia,
        )
        self.reset()

    @property
    def n_ommatidia(self) -> int:
        return self.height * self.width

    def reset(self) -> None:
        """Clear temporal, motion, and looming history."""

        self.encoder.reset()
        self._previous_motion_signal = None

    def process(self, radiance) -> dict:
        """Process one frame and return spatial arrays plus scalar readouts."""

        frame = _finite_frame(radiance, self.height, self.width)
        flat_radiance = frame.reshape(self.n_ommatidia, 4)
        contrast = self.encoder.encode(flat_radiance, self.frame_ms)[0]
        contrast_grid = contrast.reshape(self.height, self.width)
        on, off = histamine_corrected_on_off(
            contrast,
            receptor_types=self._histamine_types,
        )
        on = on.reshape(self.height, self.width)
        off = off.reshape(self.height, self.width)
        on_filtered = center_surround(
            on,
            center_radius=self.center_radius,
            surround_radius=self.surround_radius,
        )
        off_filtered = center_surround(
            off,
            center_radius=self.center_radius,
            surround_radius=self.surround_radius,
        )
        motion_signal = on_filtered + off_filtered

        direction = direction_readout(
            self._previous_motion_signal,
            motion_signal,
        )
        looming = looming_detector(
            self._previous_motion_signal,
            motion_signal,
            scales=self.loom_scales,
            min_energy=self.loom_min_energy,
        )
        self._previous_motion_signal = motion_signal.copy()

        return {
            "photoreceptor_contrast": contrast_grid.copy(),
            "histamine_relay": (off - on),
            "on": on,
            "off": off,
            "on_filtered": on_filtered,
            "off_filtered": off_filtered,
            "motion_signal": motion_signal,
            "direction": direction["direction"],
            "direction_image": direction["direction_image"],
            "direction_label": direction["label"],
            "direction_angle_degrees": direction["angle_degrees"],
            "direction_scores": direction["scores"],
            "motion_energy": direction["motion_energy"],
            "motion_strength": direction["motion_strength"],
            "looming": looming["looming"],
            "looming_score": looming["score"],
            "looming_scale": looming["best_scale"],
            "looming_correlation_gain": looming["correlation_gain"],
        }

    def step(self, radiance) -> dict:
        """Alias for :meth:`process` for simulation-loop callers."""

        return self.process(radiance)


def _constant_frame(height: int, width: int, value: float) -> np.ndarray:
    return np.full((height, width, 4), value, dtype=float)


def _bar_frame(height: int, width: int, center_x: float) -> np.ndarray:
    frame = _constant_frame(height, width, 0.15)
    yy, xx = np.indices((height, width))
    bar = np.abs(xx - center_x) <= 1.5
    frame[bar] = 0.90
    return frame


def _disk_frame(height: int, width: int, radius: float) -> np.ndarray:
    frame = _constant_frame(height, width, 0.15)
    yy, xx = np.indices((height, width))
    center = (height - 1) / 2.0
    disk = (yy - center) ** 2 + (xx - center) ** 2 <= radius * radius
    frame[disk] = 0.95
    return frame


def _self_test() -> None:
    """Run a moving-bar direction check and an expanding-flash check."""

    pipeline = VisualPipeline(16, 16)
    baseline = _constant_frame(16, 16, 0.15)
    for _ in range(5):
        pipeline.process(baseline)

    direction_labels = []
    for center_x in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
        result = pipeline.process(_bar_frame(16, 16, center_x))
        if result["motion_strength"] > 1e-4:
            direction_labels.append(result["direction_label"])
    rightward = direction_labels.count("right")
    assert rightward > len(direction_labels) / 2, direction_labels
    assert rightward > 0, direction_labels
    print(
        "moving bar: direction readout follows the bar "
        f"(rightward in {rightward}/{len(direction_labels)} moving frames)"
    )

    pipeline.reset()
    for _ in range(5):
        pipeline.process(baseline)
    looming_flags = []
    for radius in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
        result = pipeline.process(_disk_frame(16, 16, radius))
        looming_flags.append(bool(result["looming"]))
    assert any(looming_flags), looming_flags
    print(
        "looming flash: detector fires for the expanding stimulus "
        f"({sum(looming_flags)}/{len(looming_flags)} expansion frames)"
    )

    print("DONE")
    print("16x16 receptor grid with the encoder's temporal contrast stage.")
    print("Histamine sign correction feeds rectified ON/OFF center-surround maps.")
    print("Shifted correlations and scale/shape change approximate motion and looming.")


if __name__ == "__main__":
    _self_test()
