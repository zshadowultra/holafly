"""Neural activity -> motor commands, ported from FlyBrain and flyverse-core.

Two complementary readouts live here:

* FlyBrain's mushroom-body valence vote becomes a signed approach/avoidance
  command.  The binary lunge path reuses
  ``plasticity.mushroom_body.ValenceBalance`` directly, including its six-tick
  history and self-calibrated 3-sigma threshold.
* flyverse-core's ``body.Locomotion`` law maps named descending-neuron and
  leg-motor-neuron rates (Hz) to walking speed (m/s) and yaw rate (rad/s).
  Its source gains and signs are preserved.  Optional quantile calibration only
  replaces rest-noise thresholds/deadbands; it does not fit the behavioural
  gains.

Only the standard library and NumPy are used.  The target file also works when
run directly as ``python body/motor.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

if not __package__:  # Allow `python body/motor.py` without requiring PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plasticity.mushroom_body import ValenceBalance


# flyverse-core/flyverse/motor.py: motor_groups(), exact named populations.
FLYVERSE_FWD_TYPES = ("DNp09", "DNa01", "DNa03", "DNb01", "DNa04")

# Public mapping order.  Mapping/object inputs are read by these source names;
# sequence inputs must have this order.
RATE_FIELDS = (
    "fwd_dn",
    "back_dn",
    "turn_L",
    "turn_R",
    "opto_L",
    "opto_R",
    "leg_L",
    "leg_R",
    "proboscis",
)

# flyverse-core/flyverse/body.py: Locomotion defaults.  These are physical
# gains, not dimensionless decoder weights.
DEFAULT_K_FWD = 0.02 / 40.0
DEFAULT_K_LEG = 0.02 / 60.0
DEFAULT_K_BACK = 0.02 / 40.0
DEFAULT_K_TURN = np.deg2rad(200.0) / 40.0
DEFAULT_K_LEG_TURN = np.deg2rad(100.0) / 30.0
DEFAULT_MAX_SPEED = 0.03
DEFAULT_MAX_YAW = np.deg2rad(400.0)
DEFAULT_BASELINE_SPEED = 0.008
DEFAULT_MDN_THRESHOLD_HZ = 15.0


# ---------------------------------------------------------------------------
# Shared validation


def _finite_vector(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: float, name: str, *, strictly: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or (strictly and result == 0.0):
        qualifier = "positive" if strictly else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _rate_columns(rates, *, samples: bool) -> np.ndarray:
    """Return an ``(n_samples, 9)`` rate matrix from sequence/mapping/object."""

    if isinstance(rates, Mapping) or all(hasattr(rates, name) for name in RATE_FIELDS):
        columns = [
            np.asarray(rates.get(name, 0.0) if isinstance(rates, Mapping)
                       else getattr(rates, name), dtype=float)
            for name in RATE_FIELDS
        ]
        sample_shape = next((column.shape for column in columns if column.ndim), None)
        if sample_shape is not None:
            columns = [
                np.full(sample_shape, column) if column.ndim == 0 else column
                for column in columns
            ]
        if any(column.shape != columns[0].shape for column in columns[1:]):
            raise ValueError("all motor-rate fields must have the same shape")
        result = np.stack(columns, axis=-1)
    else:
        result = np.asarray(rates, dtype=float)
        if result.ndim == 1:
            result = result[None, :]
        if result.ndim != 2 or result.shape[1] != len(RATE_FIELDS):
            raise ValueError(
                f"rate sequence must have shape (n_samples, {len(RATE_FIELDS)}) "
                f"in RATE_FIELDS order"
            )

    if result.ndim == 1:  # Mapping/object with scalar source fields.
        result = result[None, :]
    if not np.isfinite(result).all():
        raise ValueError("motor rates must be finite")
    if samples:
        if result.shape[0] == 0:
            raise ValueError("calibration rates must contain at least one sample")
        return result
    if result.shape[0] != 1:
        raise ValueError(
            "motor-rate command input must be scalar values or one sample; "
            "use RateMotorMap for repeated stateful steps"
        )
    return result[0]


# ---------------------------------------------------------------------------
# FlyBrain: MBON valence vote -> approach/avoidance command


def mbon_valence_vote(ppl1_drive, pam_drive) -> np.ndarray:
    """Build FlyBrain's per-MBON approach/avoidance vote.

    FlyBrain ``fly/ataque.py`` defines:

    ``(PPL1 input > 0) - (PAM input > 0)``

    so PPL1-only MBONs vote +1 (approach), PAM-only MBONs vote -1 (avoid), and
    MBONs receiving both signs vote 0 (ambiguous).  Inputs are broadcast if
    one is a scalar.
    """

    approach = np.asarray(ppl1_drive, dtype=float)
    avoid = np.asarray(pam_drive, dtype=float)
    if approach.ndim == 0:
        approach = np.full(avoid.shape, approach) if avoid.ndim else approach
    if avoid.ndim == 0:
        avoid = np.full(approach.shape, avoid)
    if approach.shape != avoid.shape:
        try:
            approach, avoid = np.broadcast_arrays(approach, avoid)
        except ValueError as exc:
            raise ValueError("PPL1 and PAM compartment drives must be broadcastable") from exc
    if not np.isfinite(approach).all() or not np.isfinite(avoid).all():
        raise ValueError("compartment drives must be finite")
    return (approach > 0.0).astype(float) - (avoid > 0.0).astype(float)


def mbon_valence(
    mbon_activity,
    vote,
    *,
    resting_activity=None,
) -> float:
    """Raw rest-subtracted MBON valence used by FlyBrain's ``balanza``.

    This is ``(activity - reposo) @ voto``.  It intentionally does not divide
    by neuron count: that exact sum is the quantity the existing
    ``ValenceBalance`` class thresholds.
    """

    activity = _finite_vector(mbon_activity, "mbon_activity")
    weights = _finite_vector(vote, "vote")
    if activity.shape != weights.shape:
        raise ValueError("mbon_activity and vote must have the same shape")
    if resting_activity is None:
        resting = np.zeros_like(activity)
    else:
        resting = np.asarray(resting_activity, dtype=float)
        if resting.ndim == 0:
            resting = np.full_like(activity, resting)
        if resting.shape != activity.shape or not np.isfinite(resting).all():
            raise ValueError("resting_activity must be finite with mbon_activity's shape")
    return float((activity - resting) @ weights)


def mbon_valence_to_command(
    mbon_activity,
    vote,
    *,
    resting_activity=None,
    balance_scale: float = 1.0,
    baseline_speed: float = DEFAULT_BASELINE_SPEED,
    speed_gain: float = 0.02,
    yaw_gain: float = np.deg2rad(200.0),
    min_speed: float = -DEFAULT_MAX_SPEED,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> dict[str, float]:
    """Map the MBON vote to a signed walking command.

    The raw balance is divided by total vote weight, then squashed to
    ``[-1, 1]``.  A positive (approach) value produces left yaw and faster
    forward walking; a negative (avoid) value produces right yaw and slower
    or reverse walking.  The sign convention is an explicit motor-readout
    convention, not an anatomical claim: FlyBrain's sign says approach versus
    avoid, while choosing where to turn remains mechanical.

    Units are ``speed`` in m/s and ``yaw`` in rad/s.  Defaults reuse
    flyverse-core's walking endpoints while the tanh keeps commands bounded.
    """

    balance_scale = _finite_nonnegative(balance_scale, "balance_scale", strictly=True)
    baseline_speed = float(baseline_speed)
    speed_gain = float(speed_gain)
    yaw_gain = float(yaw_gain)
    min_speed = float(min_speed)
    max_speed = float(max_speed)
    if not np.isfinite([baseline_speed, speed_gain, yaw_gain, min_speed, max_speed]).all():
        raise ValueError("motor command gains and limits must be finite")
    if max_speed <= min_speed:
        raise ValueError("max_speed must be greater than min_speed")

    raw = mbon_valence(
        mbon_activity, vote, resting_activity=resting_activity
    )
    weights = _finite_vector(vote, "vote")
    vote_weight = float(np.abs(weights).sum())
    normalised = raw / vote_weight if vote_weight > 0.0 else 0.0
    drive = float(np.tanh(normalised / balance_scale))
    return {
        "yaw": float(np.clip(yaw_gain * drive, -abs(yaw_gain), abs(yaw_gain))),
        "speed": float(np.clip(baseline_speed + speed_gain * drive, min_speed, max_speed)),
        "valence": drive,
    }


# ---------------------------------------------------------------------------
# FlyBrain: "la balanza" -> binary lunge


def calibrate_lunge_balance(
    vote,
    rest_rasters,
    *,
    window: int = 6,
) -> ValenceBalance:
    """Calibrate and return the existing ``ValenceBalance`` lunge decider.

    ``rest_rasters`` is a ``(time, n_mbon)`` rest raster.  The six-tick window
    and 3-sigma calibration are deliberately left to the imported class; this
    wrapper does not reimplement either rule.
    """

    weights = _finite_vector(vote, "vote")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    rest = np.asarray(rest_rasters, dtype=float)
    if rest.ndim != 2 or rest.shape[1] != weights.size or rest.shape[0] == 0:
        raise ValueError("rest_rasters must have shape (time, len(vote))")
    if not np.isfinite(rest).all():
        raise ValueError("rest_rasters must be finite")

    balance = ValenceBalance(weights, window=int(window))
    balance.calibrate(rest)
    return balance


def lunge_decision(
    mbon_spikes,
    balance: ValenceBalance,
) -> tuple[float, bool]:
    """Advance the existing six-tick balance by one MBON raster.

    Returns ``(balance, lunge)``.  The ``lunge`` flag is true only above the
    threshold measured by ``balance.calibrate(rest_rasters)``.
    """

    activity = _finite_vector(mbon_spikes, "mbon_spikes")
    if not isinstance(balance, ValenceBalance):
        raise TypeError("balance must be a plasticity.mushroom_body.ValenceBalance")
    if activity.shape != balance.vote.shape:
        raise ValueError("mbon_spikes must have the same shape as the balance vote")
    result = balance.decide(activity)
    return float(result[0]), bool(result[1])


# Source-style alias matching the noun used in fly/ataque.py.
balanza = lunge_decision


# ---------------------------------------------------------------------------
# flyverse-core: quantile-calibrated rate -> speed/yaw map


@dataclass(frozen=True)
class RateThresholds:
    """Rest-noise floors used by :class:`RateMotorMap`.

    Defaults reproduce flyverse-core exactly.  In particular, MDN contributes
    only above 15 Hz and the turning populations have no deadband.  A
    calibration replaces these with quantiles measured from rest-rate samples;
    source gains remain unchanged.
    """

    forward_threshold_hz: float = 0.0
    backward_threshold_hz: float = DEFAULT_MDN_THRESHOLD_HZ
    turn_center_hz: float = 0.0
    turn_threshold_hz: float = 0.0
    leg_center_hz: float = 0.0
    leg_threshold_hz: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "forward_threshold_hz",
            "backward_threshold_hz",
            "turn_threshold_hz",
            "leg_threshold_hz",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in ("turn_center_hz", "leg_center_hz"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")


def calibrate_rate_thresholds(
    rest_rates,
    *,
    quantile: float = 0.99,
) -> RateThresholds:
    """Calibrate motor deadbands from an empirical rest-rate distribution.

    ``rest_rates`` has shape ``(n_samples, 9)`` in ``RATE_FIELDS`` order, or
    may be a mapping/object whose values are sample vectors.  The requested
    quantile becomes the forward and backward activation floors and the
    left-right turning deadband width.  Median left-right differences remove a
    constant resting turn bias.  This is calibration of thresholds only; the
    physical gains are still the flyverse-core defaults.
    """

    q = float(quantile)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    rates = _rate_columns(rest_rates, samples=True)
    fwd, back, turn_l, turn_r, _, _, leg_l, leg_r, _ = rates.T

    turn_difference = turn_l - turn_r
    leg_difference = leg_l - leg_r
    turn_center = float(np.quantile(turn_difference, 0.5))
    leg_center = float(np.quantile(leg_difference, 0.5))
    return RateThresholds(
        forward_threshold_hz=float(np.quantile(fwd, q)),
        backward_threshold_hz=float(np.quantile(back, q)),
        turn_center_hz=turn_center,
        turn_threshold_hz=float(
            np.quantile(np.abs(turn_difference - turn_center), q)
        ),
        leg_center_hz=leg_center,
        leg_threshold_hz=float(
            np.quantile(np.abs(leg_difference - leg_center), q)
        ),
    )


def _deadband(value: float, center: float, width: float) -> float:
    """Signed value outside ``center +/- width``; zero inside the deadband."""

    centred = value - center
    return float(np.sign(centred) * max(abs(centred) - width, 0.0))


@dataclass
class RateMotorMap:
    """Stateful NumPy port of flyverse-core ``body.Locomotion.readout``.

    The source law is:

    * ``speed = baseline + k_fwd*fwd_dn + k_leg*mean(leg_L, leg_R)
      - k_back*max(MDN - 15 Hz, 0)``
    * ``yaw = k_turn*(DNa02_L - DNa02_R)
      + k_leg_turn*(legMN_L - legMN_R)``; positive is left
    * optional optomotor correction is high-pass filtered and disabled by
      default, matching the source.

    With default thresholds this is the literal source map.  Supplying
    :func:`calibrate_rate_thresholds` replaces only activation floors and
    left-right deadbands.  Repeated calls retain the optomotor high-pass
    state; call :meth:`reset` between episodes.
    """

    thresholds: RateThresholds | None = None
    k_fwd: float = DEFAULT_K_FWD
    k_leg: float = DEFAULT_K_LEG
    k_back: float = DEFAULT_K_BACK
    k_turn: float = DEFAULT_K_TURN
    k_leg_turn: float = DEFAULT_K_LEG_TURN
    k_opto: float = 0.0
    opto_hp_tau_s: float = 2.0
    max_speed: float = DEFAULT_MAX_SPEED
    max_yaw: float = DEFAULT_MAX_YAW
    baseline_speed: float = DEFAULT_BASELINE_SPEED
    proboscis_full_scale_hz: float = 30.0
    _opto_bias: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.thresholds = self.thresholds or RateThresholds()
        if not isinstance(self.thresholds, RateThresholds):
            raise TypeError("thresholds must be a RateThresholds instance")
        gains = (
            self.k_fwd,
            self.k_leg,
            self.k_back,
            self.k_turn,
            self.k_leg_turn,
            self.k_opto,
            self.opto_hp_tau_s,
            self.max_speed,
            self.max_yaw,
            self.baseline_speed,
            self.proboscis_full_scale_hz,
        )
        if not np.isfinite(gains).all():
            raise ValueError("rate-map gains and limits must be finite")
        if self.max_speed <= 0.0 or self.max_yaw <= 0.0:
            raise ValueError("max_speed and max_yaw must be positive")
        if self.opto_hp_tau_s <= 0.0 or self.proboscis_full_scale_hz <= 0.0:
            raise ValueError("time constants and proboscis scale must be positive")

    @classmethod
    def from_rest_rates(
        cls,
        rest_rates,
        *,
        quantile: float = 0.99,
        **kwargs,
    ) -> RateMotorMap:
        """Build a map with quantile-calibrated rest thresholds."""

        return cls(
            thresholds=calibrate_rate_thresholds(rest_rates, quantile=quantile),
            **kwargs,
        )

    def reset(self) -> None:
        """Clear the source's persistent optomotor bias estimate."""

        self._opto_bias = None

    def command(self, rates, *, dt_s: float = 0.01) -> dict[str, object]:
        """Map one snapshot of named neural rates to a body command."""

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and nonnegative")
        fwd, back, turn_l, turn_r, opto_l, opto_r, leg_l, leg_r, proboscis = (
            _rate_columns(rates, samples=False)
        )

        thresholds = self.thresholds
        forward = max(fwd - thresholds.forward_threshold_hz, 0.0)
        backward = max(back - thresholds.backward_threshold_hz, 0.0)
        turn = _deadband(
            turn_l - turn_r,
            thresholds.turn_center_hz,
            thresholds.turn_threshold_hz,
        )
        leg_turn = _deadband(
            leg_l - leg_r,
            thresholds.leg_center_hz,
            thresholds.leg_threshold_hz,
        )

        asymmetry = opto_l - opto_r
        decay = float(np.exp(-dt_s / self.opto_hp_tau_s))
        if self._opto_bias is None:
            self._opto_bias = asymmetry
        else:
            self._opto_bias = decay * self._opto_bias + (1.0 - decay) * asymmetry
        optomotor = asymmetry - self._opto_bias

        speed = (
            self.baseline_speed
            + self.k_fwd * forward
            + self.k_leg * 0.5 * (leg_l + leg_r)
            - self.k_back * backward
        )
        yaw = self.k_turn * turn + self.k_leg_turn * leg_turn + self.k_opto * optomotor
        return {
            "speed": float(np.clip(speed, -self.max_speed, self.max_speed)),
            "yaw": float(np.clip(yaw, -self.max_yaw, self.max_yaw)),
            "proboscis": float(
                np.clip(proboscis / self.proboscis_full_scale_hz, 0.0, 1.0)
            ),
            "rates": {
                "fwdDN": fwd,
                "MDN": back,
                "opto_L": opto_l,
                "opto_R": opto_r,
                "DNa02_L": turn_l,
                "DNa02_R": turn_r,
                "legMN_L": leg_l,
                "legMN_R": leg_r,
                "MN9": proboscis,
            },
        }

    def __call__(self, rates, *, dt_s: float = 0.01) -> dict[str, object]:
        return self.command(rates, dt_s=dt_s)


def rate_based_motor_map(
    rates,
    *,
    thresholds: RateThresholds | None = None,
    dt_s: float = 0.01,
) -> dict[str, object]:
    """One-shot convenience wrapper around :class:`RateMotorMap`."""

    return RateMotorMap(thresholds=thresholds).command(rates, dt_s=dt_s)


__all__ = [
    "FLYVERSE_FWD_TYPES",
    "RATE_FIELDS",
    "RateThresholds",
    "RateMotorMap",
    "mbon_valence_vote",
    "mbon_valence",
    "mbon_valence_to_command",
    "calibrate_lunge_balance",
    "lunge_decision",
    "balanza",
    "calibrate_rate_thresholds",
    "rate_based_motor_map",
]


# ---------------------------------------------------------------------------
# Smoke test


def _smoke() -> None:
    rng = np.random.default_rng(19)

    # Fake FlyBrain compartments: PAM -> avoidance, PPL1 -> approach.
    vote = mbon_valence_vote([0, 1, 0, 0], [1, 0, 0, 0])
    np.testing.assert_array_equal(vote, [-1.0, 1.0, 0.0, 0.0])
    rest_rasters = rng.poisson(2.0, size=(96, 4)).astype(float)
    resting = rest_rasters.mean(axis=0)
    balance = calibrate_lunge_balance(vote, rest_rasters)

    neutral = mbon_valence_to_command(resting, vote, resting_activity=resting)
    approach_activity = resting + np.array([-2.0, 10.0, 0.0, 0.0])
    approach = mbon_valence_to_command(
        approach_activity, vote, resting_activity=resting
    )
    assert neutral["speed"] == DEFAULT_BASELINE_SPEED
    assert approach["speed"] > neutral["speed"] and approach["yaw"] > 0.0

    for _ in range(6):
        rest_balance, rest_lunge = lunge_decision(resting, balance)
    assert not rest_lunge
    for _ in range(6):
        approach_balance, lunge = lunge_decision(approach_activity, balance)
    assert lunge

    # Fake flyverse rest distribution, then a strong left-turn/forward burst.
    rest_means = np.array([4.0, 10.0, 5.0, 5.0, 0.0, 0.0, 20.0, 20.0, 0.0])
    rest_stds = np.array([0.4, 1.0, 0.3, 0.3, 0.0, 0.0, 1.0, 1.0, 0.0])
    rest_rate_samples = rng.normal(rest_means, rest_stds, size=(512, len(RATE_FIELDS)))
    rest_rate_samples = np.maximum(rest_rate_samples, 0.0)
    thresholds = calibrate_rate_thresholds(rest_rate_samples, quantile=0.99)
    motor = RateMotorMap(thresholds=thresholds)

    active = np.array([40.0, 10.0, 35.0, 5.0, 0.0, 0.0, 25.0, 15.0, 15.0])
    active_command = motor(active)
    assert active_command["speed"] > DEFAULT_BASELINE_SPEED
    assert active_command["yaw"] > 0.0
    assert active_command["proboscis"] == 0.5

    reverse = motor(np.array([0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert reverse["speed"] < 0.0
    assert -motor.max_speed <= active_command["speed"] <= motor.max_speed
    assert -motor.max_yaw <= active_command["yaw"] <= motor.max_yaw

    print(
        "fake MBON approach command:",
        f"speed={approach['speed']:.4f} m/s, yaw={approach['yaw']:.3f} rad/s",
    )
    print(
        "la balanza:",
        f"3sigma={balance.umbral:.2f}, balance={approach_balance:.2f}, lunge={lunge}",
    )
    print(
        "quantile motor command:",
        f"speed={active_command['speed']:.4f} m/s, yaw={active_command['yaw']:.3f} rad/s",
    )
    print("DONE")
    print("MBON votes map rest-subtracted approach/avoidance to bounded speed and yaw commands.")
    print("La balanza reuses ValenceBalance and fires only above its self-calibrated 3σ threshold.")
    print("Flyverse rates retain the source motor gains while calibration sets rest-noise deadbands.")


if __name__ == "__main__":
    _smoke()
