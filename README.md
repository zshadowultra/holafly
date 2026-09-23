# fly-master

Unified Drosophila brain simulation in Python (numpy only). Ports the mechanisms from seven open source fly projects into one stack: sensory encoding, neuromodulation, mushroom body plasticity, metabolism, motor readout.

## Quickstart

```
python3 demo.py
```

Runs a synthetic fly through odor conditioning. A hungry fly smells an odor, finds sugar, dopamine-gated plasticity shifts KC to MBON weights, and the motor output flips from avoidance to approach.

## Modules

| Module | Source | Function |
|---|---|---|
| `neuromod/modulators.py` | fwmc / mechabrain | Per-neuron DA, 5HT, OA release, decay, autoreceptor feedback |
| `neuromod/effects.py` | fwmc / mechabrain | Modulator effects on excitability and synaptic gain |
| `neuromod/gated_stdp.py` | fwmc / mechabrain | Dopamine-gated STDP, direct and eligibility-trace modes |
| `neuromod/stp.py` | fwmc / mechabrain | Tsodyks-Markram short-term plasticity, fly presets |
| `neuromod/gap.py` | fwmc / mechabrain | Gap junction currents, region-based builder |
| `neuromod/scaling.py` | fwmc / mechabrain | Homeostatic synaptic scaling |
| `neuromod/structural.py` | fwmc / mechabrain | Prune and sprout structural plasticity |
| `neuromod/calibration.py` | fwmc / mechabrain | Perturbation-based weight calibration |
| `plasticity/mushroom_body.py` | flytris, FlyBrain | KC to MBON dopamine learning rules, APL loop |
| `plasticity/compartments.py` | flytris | DAN to MBON compartment map, approach/avoid valence |
| `senses/encoders.py` | flyverse-core | Vision, smell, taste, wind to neural drive |
| `metabolism/hunger.py` | flyverse-core | Energy state, hunger-scaled foraging drive |
| `body/motor.py` | flyverse-core, FlyBrain | MBON activity to yaw, speed, lunge |
| `fixes/histamine.py` | fruit-fly | Photoreceptor sign correction |

Every module has a runnable smoke test under `if __name__ == "__main__"`. Parameter values and literature references are in the module docstrings.

## Caveats

- immortal-fruit-fly "starvation" is a game timer with no neural coupling. It was not used. `hunger.py` ports flyverse-core Metabolism instead.
- Hunger to behavior coupling is a multiplicative gain (engineering), not neural kinetics.
- Neuromodulator release keys off cell-type labels (DAN_PPL1, DAN_PAM, serotonergic, octopaminergic). For real connectome data these must come from FlyWire annotations.
- Gap junctions are not visible in EM data. Placement is region-based by density.
- flyverse-core time constants are demo scale (minutes). Real starvation takes days.
