"""Mushroom-body dopamine learning: compartment map + plasticity rules."""

from .compartments import (
    CompartmentMap,
    build_compartment_map,
    DEFAULT_DAN_CLUSTERS,
)
from .mushroom_body import (
    MBConfig,
    MushroomBody,
    PlasticityConfig,
    DopaminergicPlasticity,
    FlyBrainConfig,
    SpikingPlasticity,
    ValenceBalance,
)

__all__ = [
    "CompartmentMap",
    "build_compartment_map",
    "DEFAULT_DAN_CLUSTERS",
    "MBConfig",
    "MushroomBody",
    "PlasticityConfig",
    "DopaminergicPlasticity",
    "FlyBrainConfig",
    "SpikingPlasticity",
    "ValenceBalance",
]
