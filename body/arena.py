"""A small two-dimensional foraging world for the fly.

The arena contains one odor source.  :class:`Fly` converts approach and
avoidance drives to motor commands with ``body.motor`` and obtains its hunger
snapshot from ``metabolism.hunger.EnergyState``.  Positive approach drive steers
up the odor gradient; positive avoidance drive steers down it.  With no sensed
odor, hunger-dependent random turns make the fly search locally.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__:
    from .motor import mbon_valence_to_command
    from metabolism.hunger import EnergyState
else:  # Allow ``python body/arena.py`` from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from body.motor import mbon_valence_to_command
    from metabolism.hunger import EnergyState


DEFAULT_DT_S = 0.15
ARENA_UNITS_PER_METRE = 100.0
HUNGER_SPEED_GAIN = 0.75
SEARCH_NOISE_RAD = 0.35


def _finite_scalar(value, name: str) -> float:
    """Return one finite scalar."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array).item():
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _positions(values) -> np.ndarray:
    """Return positions with shape ``(2,)`` or ``(n, 2)``."""

    positions = np.asarray(values, dtype=float)
    if positions.ndim == 1 and positions.shape == (2,):
        pass
    elif positions.ndim == 2 and positions.shape[1] == 2:
        pass
    else:
        raise ValueError("positions must have shape (2,) or (n, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    return positions


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


class OdorSource:
    """An isotropic concentration field that decays away from its position."""

    def __init__(
        self,
        position=(80.0, 75.0),
        *,
        strength: float = 1.0,
        spread: float = 30.0,
        arrival_radius: float = 0.75,
    ) -> None:
        self.position = _positions(position).copy()
        self.strength = _finite_scalar(strength, "strength")
        self.spread = _finite_scalar(spread, "spread")
        self.arrival_radius = _finite_scalar(arrival_radius, "arrival_radius")
        if self.strength <= 0.0 or self.spread <= 0.0:
            raise ValueError("odor strength and spread must be positive")
        if self.arrival_radius < 0.0:
            raise ValueError("odor arrival radius must be nonnegative")

    def concentration(self, positions) -> np.ndarray | float:
        """Return odor concentration at one position or an array of positions."""

        points = _positions(positions)
        distances = np.linalg.norm(points - self.position, axis=-1)
        values = self.strength * np.exp(-distances / self.spread)
        return float(values) if points.ndim == 1 else values

    def gradient(self, positions) -> np.ndarray:
        """Return the concentration gradient, pointing toward the source."""

        points = _positions(positions)
        direction = self.position - points
        distances = np.linalg.norm(direction, axis=-1)
        unit = np.divide(
            direction,
            distances[..., np.newaxis],
            out=np.zeros_like(direction),
            where=distances[..., np.newaxis] > 0.0,
        )
        magnitude = (self.strength / self.spread) * np.exp(-distances / self.spread)
        return magnitude[..., np.newaxis] * unit


class Arena:
    """A bounded 2D plane with one odor source and a field query API."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        *,
        source_position=(80.0, 75.0),
        odor_strength: float = 1.0,
        odor_spread: float = 30.0,
        odor_threshold: float = 0.08,
        arrival_radius: float = 0.75,
    ) -> None:
        self.width = _finite_scalar(width, "width")
        self.height = _finite_scalar(height, "height")
        self.odor_threshold = _finite_scalar(odor_threshold, "odor_threshold")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("arena width and height must be positive")
        if self.odor_threshold < 0.0:
            raise ValueError("odor threshold must be nonnegative")
        self.source = OdorSource(
            source_position,
            strength=odor_strength,
            spread=odor_spread,
            arrival_radius=arrival_radius,
        )
        self.lower_bounds = np.array([0.0, 0.0])
        self.upper_bounds = np.array([self.width, self.height])

    def odor_concentration(self, positions) -> np.ndarray | float:
        """Return the odor field at one or more positions."""

        return self.source.concentration(positions)

    def odor_gradient(self, positions) -> np.ndarray:
        """Return the odor field gradient at one or more positions."""

        return self.source.gradient(positions)

    def contain(self, position) -> np.ndarray:
        """Clamp a position to the arena bounds."""

        return np.clip(_positions(position), self.lower_bounds, self.upper_bounds)


class Fly:
    """A heading-and-speed fly driven by odor and metabolic state."""

    _MOTOR_VOTE = np.array([-1.0, 1.0])

    def __init__(
        self,
        arena: Arena,
        position=(20.0, 30.0),
        heading: float = 0.0,
        *,
        seed: int = 0,
    ) -> None:
        self.arena = arena
        self.position = arena.contain(position).astype(float, copy=True)
        self.heading = _wrap_angle(_finite_scalar(heading, "heading"))
        self.speed = 0.0
        self.rng = np.random.default_rng(seed)
        self.closest_distance = self.distance_to_odor
        self.arrived = False

    @property
    def distance_to_odor(self) -> float:
        """Euclidean distance from the fly to the source position."""

        return float(np.linalg.norm(self.position - self.arena.source.position))

    @property
    def sensed_odor(self) -> float:
        """Concentration currently available to the fly."""

        return float(self.arena.odor_concentration(self.position))

    def step(
        self,
        approach_drive: float,
        avoidance_drive: float,
        hunger: float | EnergyState,
        *,
        dt_s: float = DEFAULT_DT_S,
    ) -> None:
        """Advance the fly once.

        ``approach_drive`` and ``avoidance_drive`` are passed through the
        existing valence motor map.  A positive net drive follows the odor
        gradient, while a negative net drive follows its reverse. ``hunger``
        may be a number in ``[0, 1]`` or an :class:`EnergyState`; when state is
        supplied, it is also advanced with the movement and feeding costs.
        """

        approach = _finite_scalar(approach_drive, "approach_drive")
        avoidance = _finite_scalar(avoidance_drive, "avoidance_drive")
        dt = _finite_scalar(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")

        metabolism = hunger if isinstance(hunger, EnergyState) else None
        hunger_level = metabolism.hunger if metabolism is not None else float(hunger)
        if not np.isfinite(hunger_level) or not 0.0 <= hunger_level <= 1.0:
            raise ValueError("hunger must be a finite value in [0, 1]")

        self.closest_distance = min(self.closest_distance, self.distance_to_odor)
        if self.distance_to_odor <= self.arena.source.arrival_radius:
            self.arrived = True
            self.speed = 0.0
            if metabolism is not None:
                metabolism.step(dt, speed=0.0, tasting=True)
            return

        # The first activity component carries PAM/avoidance and the second
        # PPL1/approach, matching the motor module's [-1, +1] vote ordering.
        command = mbon_valence_to_command(
            np.array([avoidance, approach]),
            self._MOTOR_VOTE,
        )
        net_drive = approach - avoidance
        gradient = self.arena.odor_gradient(self.position)

        if self.sensed_odor >= self.arena.odor_threshold and np.linalg.norm(gradient) > 0.0:
            if net_drive > 0.0:
                desired_direction = gradient
            elif net_drive < 0.0:
                desired_direction = -gradient
            else:
                desired_direction = None

            if desired_direction is not None:
                desired_heading = float(np.arctan2(desired_direction[1], desired_direction[0]))
                error = _wrap_angle(desired_heading - self.heading)
                max_turn = abs(command["yaw"]) * dt
                self.heading = _wrap_angle(
                    self.heading + np.clip(0.75 * error, -max_turn, max_turn)
                )
        else:
            # The odor is below threshold: hungry flies make increasingly wide
            # random heading corrections while searching for a plume.
            heading_noise = self.rng.normal(0.0, SEARCH_NOISE_RAD * hunger_level)
            self.heading = _wrap_angle(self.heading + heading_noise)

        # Motor output is in m/s; one arena unit is one centimetre.  The
        # absolute value lets avoidance use the command's locomotion magnitude
        # while its signed yaw selects the down-gradient steering branch.
        self.speed = (
            abs(float(command["speed"]))
            * ARENA_UNITS_PER_METRE
            * (1.0 + HUNGER_SPEED_GAIN * hunger_level)
        )
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position = self.arena.contain(self.position + self.speed * dt * direction)

        if metabolism is not None:
            metabolism.step(
                dt,
                speed=self.speed / ARENA_UNITS_PER_METRE,
                tasting=self.arrived or self.distance_to_odor <= self.arena.source.arrival_radius,
            )


def _self_test() -> None:
    """Run high-approach and high-avoidance episodes for 300 steps each."""

    arena = Arena()

    approach_energy = EnergyState(energy=0.8)
    seeker = Fly(arena, position=(20.0, 30.0), heading=0.0, seed=7)
    approach_start = seeker.distance_to_odor
    for _ in range(300):
        seeker.step(approach_drive=1.0, avoidance_drive=0.0, hunger=approach_energy)
    approach_end = seeker.distance_to_odor
    assert approach_end <= arena.source.arrival_radius, (
        approach_start,
        approach_end,
        arena.source.arrival_radius,
    )
    print(
        "Approach distance:",
        f"start={approach_start:.2f}, end={approach_end:.2f} units",
    )

    avoidance_energy = EnergyState(energy=0.8)
    avoider = Fly(arena, position=(50.0, 50.0), heading=np.pi / 4.0, seed=11)
    avoidance_start = avoider.distance_to_odor
    for _ in range(300):
        avoider.step(approach_drive=0.0, avoidance_drive=1.0, hunger=avoidance_energy)
    avoidance_end = avoider.distance_to_odor
    assert avoidance_end > avoidance_start, (avoidance_start, avoidance_end)
    print(
        "Avoidance distance:",
        f"start={avoidance_start:.2f}, end={avoidance_end:.2f} units",
    )

    print("DONE")
    print("A bounded 2D arena emits an analytic odor concentration gradient.")
    print("Motor valence commands steer toward odor for approach and away for avoidance.")
    print("EnergyState hunger increases movement speed and hungry no-odor search noise.")


if __name__ == "__main__":
    _self_test()
