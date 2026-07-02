"""
src/models/neuro — Neuroscience-inspired computational primitives.

Organised from single-neuron level up to network dynamics:

SINGLE NEURON (spiking_neurons.py):
    LIFNeuron, AdaptiveLIFNeuron, DendriticNeuron
SYNAPSE (synaptic_plasticity.py):
    HebbianLayer, STDPLayer, ShortTermPlasticity
MICROCIRCUIT (microcircuits.py):
    DivisiveNormalization, LateralInhibition, EIBalanceLayer, CorticalColumn
SPATIAL CELLS (spatial_cells.py):
    HeadDirectionCells, BoundaryVectorCells, SpeedCells, ConjunctiveSpatialCells,
    EgocentricObjectVectorCells
OSCILLATIONS (oscillations.py):
    ThetaOscillator, PhasePrecession, ThetaGammaCoupling, SharpWaveRipple
ATTRACTOR DYNAMICS (attractor.py):
    ContinuousAttractorNetwork, GridAttractorNetwork
GRID CELLS (grid_cells.py):
    (existing entorhinal grid encoder)
"""

from .spiking_neurons import LIFNeuron, AdaptiveLIFNeuron, DendriticNeuron
from .synaptic_plasticity import HebbianLayer, STDPLayer, ShortTermPlasticity, BTSPPlasticity
from .microcircuits import (
    DivisiveNormalization, LateralInhibition, EIBalanceLayer, CorticalColumn,
)
from .spatial_cells import (
    HeadDirectionCells, BoundaryVectorCells, SpeedCells, ConjunctiveSpatialCells,
    EgocentricObjectVectorCells, EgocentricCenterCells, LocalOrder3DGrid, ConjunctiveGridDirectionCells,
)
from .oscillations import (
    ThetaOscillator, PhasePrecession, ThetaGammaCoupling, SharpWaveRipple,
)
from .attractor import ContinuousAttractorNetwork, GridAttractorNetwork
from .theta_sweep import ThetaSweepSampler

__all__ = [
    "LIFNeuron", "AdaptiveLIFNeuron", "DendriticNeuron",
    "HebbianLayer", "STDPLayer", "ShortTermPlasticity", "BTSPPlasticity",
    "DivisiveNormalization", "LateralInhibition", "EIBalanceLayer", "CorticalColumn",
    "HeadDirectionCells", "BoundaryVectorCells", "SpeedCells", "ConjunctiveSpatialCells", "EgocentricObjectVectorCells", "EgocentricCenterCells", "LocalOrder3DGrid", "ConjunctiveGridDirectionCells",
    "ThetaOscillator", "PhasePrecession", "ThetaGammaCoupling", "SharpWaveRipple",
    "ContinuousAttractorNetwork", "GridAttractorNetwork", "ThetaSweepSampler",
]
