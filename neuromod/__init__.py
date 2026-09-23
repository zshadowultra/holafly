"""fwmc neuromodulation engine, ported to Python (numpy only).

Modules
-------
modulators  per-neuron dopamine/serotonin/octopamine release + decay
effects     tonic excitability effects of the modulators
gated_stdp  dopamine-gated STDP, direct and eligibility-trace modes
stp         Tsodyks-Markram short-term plasticity + Drosophila presets
gap         gap junctions (electrical synapses) + region density builder
scaling     Turrigiano homeostatic synaptic scaling

Source: stanbot8/fwmc, engine in the bundled mechabrain headers.
Parameter values are taken verbatim from the fwmc survey spec.
"""

from .modulators import (
    ModulatorState,
    NeuromodulatorParams,
    DAN_PPL1,
    DAN_PAM,
    SEROTONERGIC,
    OCTOPAMINERGIC,
)
from .effects import apply_effects
from .gated_stdp import STDPParams, GatedSTDPDirect, GatedSTDEligibility
from .stp import (
    ShortTermPlasticity,
    nt_sign,
    PRESET_FACILITATING,
    PRESET_DEPRESSING,
    PRESET_COMBINED,
    NT_ACETYLCHOLINE,
    NT_GABA,
    NT_GLUTAMATE,
    NT_DOPAMINE,
    NT_SEROTONIN,
    NT_OCTOPAMINE,
    NT_HISTAMINE,
    NT_UNKNOWN,
)
from .gap import GapJunctions, build_from_region
from .scaling import SynapticScaling

__all__ = [
    "ModulatorState",
    "NeuromodulatorParams",
    "DAN_PPL1",
    "DAN_PAM",
    "SEROTONERGIC",
    "OCTOPAMINERGIC",
    "apply_effects",
    "STDPParams",
    "GatedSTDPDirect",
    "GatedSTDEligibility",
    "ShortTermPlasticity",
    "nt_sign",
    "PRESET_FACILITATING",
    "PRESET_DEPRESSING",
    "PRESET_COMBINED",
    "NT_ACETYLCHOLINE",
    "NT_GABA",
    "NT_GLUTAMATE",
    "NT_DOPAMINE",
    "NT_SEROTONIN",
    "NT_OCTOPAMINE",
    "NT_HISTAMINE",
    "NT_UNKNOWN",
    "GapJunctions",
    "build_from_region",
    "SynapticScaling",
]
