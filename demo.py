"""End-to-end integration demo for the small fly-master stack.

The circuit is intentionally tiny and synthetic:

    odor -> ORN rates -> KC drive -> KC -> MBON -> motor command
                                      ^
                         PAM/PPL1 dopamine teaching signal

The requested ``body.motor`` module supplies the motor command.  A small
NumPy-only compatibility decoder is retained only as a fallback for partial
checkouts where that optional module is unavailable.
"""

from __future__ import annotations

import numpy as np

from metabolism.hunger import EnergyState
from neuromod.modulators import DAN_PAM, DAN_PPL1, ModulatorState
from plasticity.compartments import build_compartment_map
from plasticity.mushroom_body import (
    DopaminergicPlasticity,
    MBConfig,
    MushroomBody,
    PlasticityConfig,
)
from senses.encoders import SmellEncoder, TasteEncoder

try:
    from body import motor as _body_motor
except ImportError:
    # The requested body/motor.py is optional in this partial checkout.
    _body_motor = None


def _numpy_motor_readout(
    mbon_activity: np.ndarray,
    mbon_valence: np.ndarray,
    hunger_gain: float,
) -> dict[str, float]:
    """Fallback body readout: signed MBON activity becomes turn and speed.

    Positive turn means a left turn.  Approach output raises forward speed;
    avoidance output leaves a slower, oppositely signed command.
    """
    activity = np.asarray(mbon_activity, dtype=float)
    valence = np.asarray(mbon_valence, dtype=float)
    signed = activity * valence
    approach = float(np.maximum(signed, 0.0).sum())
    avoid = float(np.maximum(-signed, 0.0).sum())
    turn = (approach - avoid) * float(hunger_gain)
    speed = 0.01 + 0.02 * (approach + avoid) * float(hunger_gain)
    speed += 0.01 * max(turn, 0.0)
    return {"turn": float(turn), "speed": float(speed)}


def read_motor_commands(
    mbon_activity: np.ndarray,
    mbon_valence: np.ndarray,
    hunger_gain: float,
) -> dict[str, float]:
    """Read commands through body.motor, or the local NumPy fallback."""
    if _body_motor is not None:
        mbon_to_command = getattr(_body_motor, "mbon_valence_to_command", None)
        if mbon_to_command is not None:
            # Hunger increases the effective sensory drive reaching the body.
            result = mbon_to_command(
                np.asarray(mbon_activity, dtype=float) * float(hunger_gain),
                np.asarray(mbon_valence, dtype=float),
            )
            return {
                "turn": float(result["yaw"]),
                "speed": float(result["speed"]),
            }
    return _numpy_motor_readout(mbon_activity, mbon_valence, hunger_gain)


def make_dopamine_wiring(n_kc: int, n_dan: int, n_mbon: int):
    """Build the tiny DAN -> MBON CSR and a matching modulator state."""
    edges = [
        ("PAM01", "MBON-avoid", 10.0),
        ("PPL101", "MBON-approach", 9.0),
    ]
    cmap = build_compartment_map(edges)
    if cmap.mbon_types != ["MBON-avoid", "MBON-approach"]:
        raise RuntimeError(f"unexpected MBON order: {cmap.mbon_types}")

    dan_start = n_kc
    pam_index = dan_start
    ppl1_index = dan_start + 1
    mbon_start = dan_start + n_dan
    avoid_index = mbon_start
    approach_index = mbon_start + 1

    cell_types = np.zeros(n_kc + n_dan + n_mbon, dtype=np.int64)
    cell_types[pam_index] = DAN_PAM
    cell_types[ppl1_index] = DAN_PPL1

    targets = [[] for _ in range(len(cell_types))]
    targets[pam_index].append(avoid_index)
    targets[ppl1_index].append(approach_index)
    row_ptr = np.zeros(len(targets) + 1, dtype=np.int64)
    for index, outgoing in enumerate(targets):
        row_ptr[index + 1] = row_ptr[index] + len(outgoing)
    col_idx = np.asarray(
        [target for outgoing in targets for target in outgoing], dtype=np.int64
    )
    modulators = ModulatorState(len(cell_types))
    return cmap, modulators, cell_types, row_ptr, col_idx, pam_index, avoid_index


def present_odor(encoder: SmellEncoder, metabolism: EnergyState, n_kc: int) -> np.ndarray:
    """Encode the fruit odor and project its three ORN channels onto KCs."""
    odor_rates = encoder.encode({"fruit": 1.0}, {"fruit": 1.0}, batch=1)[0]
    odor_to_kc = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.8, 0.0, 0.2],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    kc_drive = odor_rates @ odor_to_kc.T
    kc_drive *= 0.5 + 0.5 * metabolism.hunger
    kc_drive /= max(float(kc_drive.max()), 1e-9)
    if len(kc_drive) != n_kc:
        raise RuntimeError("odor projection and KC count disagree")
    return kc_drive


def format_command(command: dict[str, float]) -> str:
    return f"turn={command['turn']:+.4f}, speed={command['speed']:.4f}"


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    n_kc = 6
    n_dan = 2
    n_mbon = 2

    # A hungry fly is more responsive to the odor and supplies the motor gain.
    metabolism = EnergyState(energy=0.25)
    if metabolism.state != "hungry":
        raise AssertionError(f"expected hungry state, got {metabolism.state}")

    smell = SmellEncoder(
        orn_glomeruli=("fruit", "yeast", "background"),
        orn_sides=(0, 0, 0),
        base_hz=0.0,
        max_hz=150.0,
        half_conc=0.5,
    )
    taste = TasteEncoder(2)
    sugar_rates = taste.encode(1.0, batch=1)[0]
    sugar_found = bool(np.any(sugar_rates > 0.0))
    if not sugar_found:
        raise AssertionError("sugar stimulus was not detected")

    cmap, modulators, cell_types, row_ptr, col_idx, pam_index, avoid_index = (
        make_dopamine_wiring(n_kc, n_dan, n_mbon)
    )

    # The two DAN compartments deliberately oppose one another: PAM is
    # attached to the avoidance MBON and PPL1 to the approach MBON, as in the
    # compartment rule used by DopaminergicPlasticity.
    initial_weights = np.asarray(
        [
            [0.45, 0.35],
            [0.55, 0.25],
            [0.50, 0.30],
            [0.40, 0.60],
            [0.60, 0.40],
            [0.50, 0.50],
        ],
        dtype=float,
    )
    mb = MushroomBody(
        initial_weights.copy(),
        cmap,
        cfg=MBConfig(apl_enabled=False),
    )
    plasticity = DopaminergicPlasticity(
        mb,
        PlasticityConfig(lr=0.08, tau_eligibility=30.0, recovery=0.0),
    )

    odor_rates = smell.encode({"fruit": 1.0}, {"fruit": 1.0}, batch=1)[0]
    kc_drive = present_odor(smell, metabolism, n_kc)
    kc_before = mb.encode(kc_drive)
    mbon_before = mb.readout(kc_before)
    weights_before = mb.W0.copy()
    hunger_gain = metabolism.gain()
    command_before = read_motor_commands(mbon_before, mb.mbon_valence, hunger_gain)

    print("Synthetic fly: 6 KCs, 2 DANs, 2 MBONs")
    print(f"Odor ORN rates (Hz): {odor_rates}")
    print(f"Sugar GRN rates (Hz): {sugar_rates}")
    print(
        f"State: {metabolism.state}, hunger={metabolism.hunger:.2f}, "
        f"sensory/motor gain={hunger_gain:.2f}"
    )
    print(f"MBON types: {cmap.mbon_types}, valence: {cmap.valence}")
    motor_backend = (
        "body.motor"
        if _body_motor is not None
        and hasattr(_body_motor, "mbon_valence_to_command")
        else "NumPy compatibility fallback"
    )
    print("Motor backend:", motor_backend)
    print("\nBEFORE learning")
    print("KC->MBON weights:\n", weights_before)
    print("MBON activity:", mbon_before)
    print("Motor command:", format_command(command_before))

    # Pair odor activity with sugar-reward dopamine for four trials.
    for trial in range(1, 5):
        trial_odor_rates = smell.encode({"fruit": 1.0}, {"fruit": 1.0}, batch=1)[0]
        trial_kc_drive = present_odor(smell, metabolism, n_kc)
        trial_kc = mb.encode(trial_kc_drive)
        plasticity.reset()
        plasticity.observe(trial_kc)

        spiked = np.zeros(len(cell_types), dtype=bool)
        if sugar_found:
            # A PAM DAN fires when the sweet-GRN signal reports sugar.
            spiked[pam_index] = True
        modulators.step(10.0, spiked, cell_types, row_ptr, col_idx)
        dopamine = float(modulators.dopamine[avoid_index])
        reward = float(np.clip(dopamine / 0.2, 0.0, 1.0))
        change = plasticity.teach(reward=reward, punishment=0.0)

        trial_mbon = mb.readout(trial_kc)
        trial_command = read_motor_commands(
            trial_mbon, mb.mbon_valence, metabolism.gain()
        )
        feeding = metabolism.step(
            0.1, speed=trial_command["speed"], tasting=sugar_found
        )
        print(
            f"trial {trial}: ORN={trial_odor_rates[0]:.3f} Hz, "
            f"dopamine={dopamine:.3f}, reward={reward:.2f}, "
            f"weight_move={change:.4f}, feeding={feeding}"
        )

    kc_after = mb.encode(present_odor(smell, metabolism, n_kc))
    mbon_after = mb.readout(kc_after)
    weights_after = mb.W_kc_mbon.copy()
    command_after = read_motor_commands(
        mbon_after, mb.mbon_valence, metabolism.gain()
    )

    print("\nAFTER learning")
    print("KC->MBON weights:\n", weights_after)
    print("MBON activity:", mbon_after)
    print("Motor command:", format_command(command_after))
    print(
        f"Approach weight change (odor KCs): "
        f"{(weights_after[:3, 1] - weights_before[:3, 1]).mean():+.4f}"
    )
    print(
        f"Avoidance weight change (odor KCs): "
        f"{(weights_after[:3, 0] - weights_before[:3, 0]).mean():+.4f}"
    )

    # These checks make the requested behavioral change explicit.
    if not np.all(np.isfinite(weights_after)):
        raise AssertionError("plasticity produced non-finite weights")
    if not weights_after[:3, 1].mean() > weights_before[:3, 1].mean():
        raise AssertionError("odor-to-approach weights did not increase")
    if not weights_after[:3, 0].mean() < weights_before[:3, 0].mean():
        raise AssertionError("odor-to-avoidance weights did not decrease")
    if not command_after["turn"] > command_before["turn"]:
        raise AssertionError("motor turn did not shift toward approach")
    if not command_after["speed"] > command_before["speed"]:
        raise AssertionError("motor speed did not increase with approach")

    print("DONE")
    print("Odor encoding and a sugar cue drove a hungry synthetic fly.")
    print("PAM dopamine release reached the MBON compartments during four trials.")
    print("KC weights and motor commands shifted from avoidance toward approach.")


if __name__ == "__main__":
    main()
