"""Energy-state / hunger dynamics for the fly-brain master build.

Port of flyverse-core's ``body.Metabolism`` (``flyverse/body.py``), which
closes the foraging loop: an energy reserve drains with time (faster while
walking), refills while feeding; ``hunger = 1 - energy`` scales sensory and
foraging gain (starved flies are more responsive to food odours); satiety
ends a meal (a sated fly leaves the fruit and wanders).

NOTE on immortal-fruit-fly: its "starvation" mechanic is NOT a hunger model
and was NOT used here. In ``sim/flysim.py`` the sim does ``self.energy -= 1``
once per tick and sets ``self.alive = False`` when it hits zero -- a pure
countdown game timer with no replenishment dynamics, no satiety, no neural
coupling, and no gain modulation. We implement the flyverse-core dynamics
instead, which have real state dynamics driven by the animal's own behavior.
"""

from __future__ import annotations

import numpy as np


class EnergyState:
    """A self-contained energy reserve with hunger-driven sensory gain.

    Parameters are the flyverse-core Metabolism defaults (demo-scale: a full
    tank lasts ~4 min of walking, a meal takes ~15 s). Real Drosophila
    deplete reserves over ~1-2 days of starvation; rescale ``drain_per_s``
    (e.g. ``1/86400`` for a ~24 h tank at rest) for realistic timescales.
    """

    def __init__(
        self,
        energy: float = 0.6,
        drain_per_s: float = 1.0 / 300.0,   # resting drain (source)
        walk_drain_per_m: float = 0.15,      # extra drain per metre walked (source)
        feed_per_s: float = 1.0 / 15.0,      # refill rate while tasting food (source)
        satiety: float = 0.95,               # stop feeding above this (source)
        resume_below: float = 0.7,           # ...and only start a new meal below this (source, hysteresis)
        e_crit: float = 0.15,                # starvation threshold (source's "starving" cutoff)
        gain_min: float = 0.5,               # sensory-gain multiplier when fully sated
        gain_max: float = 2.0,               # sensory-gain multiplier when fully starved
    ):
        self.energy = float(np.clip(energy, 0.0, 1.0))
        self.drain_per_s = drain_per_s
        self.walk_drain_per_m = walk_drain_per_m
        self.feed_per_s = feed_per_s
        self.satiety = satiety
        self.resume_below = resume_below
        self.e_crit = e_crit
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.sated = False
        self.meals = 0
        self._feeding_prev = False

    # ------------------------------------------------------------------ state
    @property
    def hunger(self) -> float:
        """Hunger level in [0, 1]. Drives gain; not a manual knob."""
        return float(np.clip(1.0 - self.energy, 0.0, 1.0))

    @property
    def state(self) -> str:
        """Qualitative state, ported from the source's thresholds."""
        if self.sated:
            return "sated"
        if self.energy < self.e_crit:
            return "starving"
        if self.energy < 0.5:
            return "hungry"
        return "ok"

    @property
    def debilitated(self) -> bool:
        """True below the starvation threshold: the animal is failing."""
        return self.energy < self.e_crit

    # --------------------------------------------------------------- dynamics
    def step(self, dt_s: float, speed: float = 0.0, tasting: bool = False) -> bool:
        """Advance the reserve by ``dt_s`` seconds. Returns True if feeding.

        Energy drains every step (resting + movement cost). If ``tasting``
        (sugar GRNs active) and not sated, the animal feeds and the reserve
        refills. Satiety has hysteresis: feeding stops at ``satiety`` and
        does not resume until energy drops below ``resume_below`` -- the
        fly leaves the fruit and wanders, exactly as in the source.
        """
        self.energy -= self.drain_per_s * dt_s + self.walk_drain_per_m * abs(speed) * dt_s
        if self.sated and self.energy < self.resume_below:
            self.sated = False
        feeding = bool(tasting) and not self.sated
        if feeding:
            self.energy += self.feed_per_s * dt_s
            if not self._feeding_prev:
                self.meals += 1
            if self.energy >= self.satiety:
                self.sated = True
                feeding = False
        self._feeding_prev = feeding
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        return feeding

    def feed(self, amount: float) -> float:
        """Discrete feeding event: add ``amount`` to the reserve (clipped).

        Use for scripted food delivery (e.g. a food pellet appearing).
        Counts as a meal and applies the same satiety cutoff as ``step``.
        Returns the new energy level.
        """
        was_feeding = self._feeding_prev
        self.energy = float(np.clip(self.energy + amount, 0.0, 1.0))
        if not was_feeding and amount > 0:
            self.meals += 1
        if self.energy >= self.satiety:
            self.sated = True
        self._feeding_prev = False
        return self.energy

    # ------------------------------------------------------------------- gain
    def gain(self) -> float:
        """Sensory-gain multiplier for sugar-GRN / food-odour pathways.

        Mapping: ``gain = gain_min + (gain_max - gain_min) * hunger`` --
        a hungry fly (hunger -> 1) gets up to ``gain_max``x the sensory
        drive of a sated fly (``gain_min``x). This mirrors flyverse-core's
        ``AnemotaxisProgram`` hunger scaling (``0.1 + 0.9 * hunger`` applied
        to odour-gated upwind drive; see ``foraging_drive`` below), recast
        here as a multiplier on the sensory input itself.

        Starvation collapse: below ``e_crit`` the nervous system is failing,
        so gain is scaled by ``energy / e_crit`` and collapses toward zero
        as the reserve empties. A starving fly is hyper-responsive; a dying
        fly's responses degrade.
        """
        g = self.gain_min + (self.gain_max - self.gain_min) * self.hunger
        if self.energy < self.e_crit:
            g *= self.energy / self.e_crit
        return float(g)

    def foraging_drive(self) -> float:
        """Exact port of flyverse-core's hunger scaling for odour-gated
        upwind drive: ``0.1 + 0.9 * hunger`` (``hunger_gain_min = 0.1`` in
        ``AnemotaxisProgram``). Sated flies barely chase odours; starved
        flies chase at full strength."""
        return 0.1 + 0.9 * self.hunger


def _smoke() -> None:
    # dynamics: resting drain empties a full tank in ~300 s (source rate)
    m = EnergyState(energy=1.0)
    for _ in range(300):
        m.step(1.0)
    assert abs(m.energy - 0.0) < 1e-9, m.energy

    # walking drains faster than resting
    a = EnergyState(energy=1.0)
    b = EnergyState(energy=1.0)
    for _ in range(60):
        a.step(1.0, speed=0.0)
        b.step(1.0, speed=0.02)  # 2 cm/s walk
    assert b.energy < a.energy

    # feeding refills; satiety stops the meal with hysteresis
    m = EnergyState(energy=0.5)
    fed = False
    for _ in range(30):
        if m.step(1.0, tasting=True):
            fed = True
        if m.sated:
            break
    assert fed and m.sated and m.energy >= m.satiety, (m.energy, m.satiety)
    assert m.meals >= 1
    # still tasting, but sated: no feeding until reserve drops below resume_below
    assert m.step(1.0, tasting=True) is False
    m.energy = 0.6  # below resume_below
    assert m.step(1.0, tasting=True) is True

    # feed() discrete events
    m = EnergyState(energy=0.2)
    assert m.feed(0.5) == 0.7
    assert m.feed(10.0) == 1.0  # clipped

    # gain: hungry > sated; collapses below e_crit
    sated = EnergyState(energy=0.95)
    hungry = EnergyState(energy=0.2)
    assert hungry.gain() > sated.gain() > 0
    assert abs(hungry.gain() - (0.5 + 1.5 * 0.8)) < 1e-9
    dying = EnergyState(energy=0.0)
    assert dying.gain() == 0.0
    assert dying.debilitated and dying.state == "starving"
    mid = EnergyState(energy=0.075)  # half of e_crit
    assert mid.gain() < hungry.gain()

    # hunger property and foraging_drive bounds
    assert EnergyState(energy=1.0).hunger == 0.0
    assert EnergyState(energy=0.0).hunger == 1.0
    assert abs(EnergyState(energy=0.0).foraging_drive() - 1.0) < 1e-9
    assert abs(EnergyState(energy=1.0).foraging_drive() - 0.1) < 1e-9

    # states
    assert EnergyState(energy=0.8).state == "ok"
    assert EnergyState(energy=0.3).state == "hungry"
    print("hunger.py smoke test: OK")


if __name__ == "__main__":
    _smoke()
