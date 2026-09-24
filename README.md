# fly-master

A fruit fly brain simulation in Python (only needs numpy). It takes the working ideas from seven open source fly projects and puts them in one place: senses, brain chemicals, learning, hunger, movement. It can run on the real FlyWire brain wiring (138,639 neurons).

## Quickstart

```
python3 demo.py           # tiny hand-built circuit, runs in seconds
python3 demo_real.py      # real FlyWire wiring, 172 neurons, about 20 seconds
python3 demo_mb_full.py   # full mushroom body on real wiring: 2000 KCs, ~11 seconds
python3 senses/vision.py     # visual pipeline: 16x16 receptors, motion + looming readout
python3 body/arena.py        # 2D arena: fly seeks odor when approach drive is high
```

Both show a simulated fly learning. A hungry fly smells an odor, finds sugar, its brain links the smell to the reward, and it starts moving toward the smell instead of away from it.

## Modules

| Module | Taken from | What it does |
|---|---|---|
| `connectome/loader.py` | FlyWire (local v783 release) | Loads the real brain wiring: 138,639 neurons, 15M connections |
| `neuromod/modulators.py` | fwmc / mechabrain | Brain chemicals (dopamine, serotonin, octopamine): how they get released and fade out |
| `neuromod/effects.py` | fwmc / mechabrain | How those chemicals make neurons more or less excitable |
| `neuromod/gated_stdp.py` | fwmc / mechabrain | Learning rule: connections strengthen or weaken based on spike timing, steered by dopamine |
| `neuromod/stp.py` | fwmc / mechabrain | Short-term change in connection strength during bursts of firing |
| `neuromod/gap.py` | fwmc / mechabrain | Direct electrical links between neurons |
| `neuromod/scaling.py` | fwmc / mechabrain | Keeps neuron activity from running too hot or too quiet |
| `neuromod/structural.py` | fwmc / mechabrain | Weak connections get removed, new ones grow between neurons that fire together |
| `neuromod/calibration.py` | fwmc / mechabrain | Tunes connection strengths toward a target pattern |
| `plasticity/mushroom_body.py` | flytris, FlyBrain | The fly's learning center: smell in, dopamine teaches, approach or avoid comes out |
| `plasticity/compartments.py` | flytris | Map of which dopamine neurons talk to which output neurons |
| `senses/encoders.py` | flyverse-core | Turns light, smell, taste, and wind into brain signals |
| `metabolism/hunger.py` | flyverse-core | Energy level; hungrier flies search harder |
| `body/motor.py` | flyverse-core, FlyBrain | Turns brain output into turning, speed, and lunging |
| `fixes/histamine.py` | fruit-fly | Fix: light-sensing neurons had the wrong signal sign |

Every module has a small self-test at the bottom (run it directly with python3). Exact numbers and science references are in the comments inside each file.

## Things to know

- The full wiring is loaded sparse. A dense 138,639 x 138,639 matrix would need hundreds of GB, so the loader keeps it sparse and hands out dense blocks for small subcircuits.
- One repo's "starvation" was just a game countdown with no link to the brain. Skipped. Hunger here uses flyverse-core's energy model instead.
- Hunger changes behavior through a simple multiplier, not through real brain chemistry. The code says so where it matters.
- The chemical release needs to know which neurons are dopamine, serotonin, or octopamine neurons. For real brain data those labels must come from FlyWire's annotations.
- Electrical links between neurons cannot be seen in wiring data, so they are placed by brain region.
- The hunger timing is demo scale (minutes). Real flies take days to starve.
