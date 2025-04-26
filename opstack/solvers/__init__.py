"""
Pre-implemented solvers for the OpStack framework.

This package provides ready-to-use solvers for common optimization problems.
"""

# Import from gradient-based solvers
from opstack.solvers.gradient_based import (
    GradientDescentSolver,
    MomentumGradientDescentSolver,
    AdamSolver
)

# Import from evolutionary solvers
from opstack.solvers.evolutionary import (
    GeneticAlgorithmSolver,
    EvolutionStrategySolver,
    DifferentialEvolutionSolver
)

# Import from stochastic solvers
from opstack.solvers.stochastic import (
    SimulatedAnnealingSolver, 
    RandomSearchSolver,
    ParticleSwarmSolver
)

# Import from direct search solvers
from opstack.solvers.direct import (
    GridSearchSolver,
    NelderMeadSolver
)

# Import from composite solvers
from opstack.solvers.composite import (
    HybridSolver
)

__all__ = [
    # Gradient-based solvers
    'GradientDescentSolver',
    'MomentumGradientDescentSolver',
    'AdamSolver',
    
    # Evolutionary solvers
    'GeneticAlgorithmSolver',
    'EvolutionStrategySolver',
    'DifferentialEvolutionSolver',
    
    # Stochastic solvers
    'SimulatedAnnealingSolver',
    'RandomSearchSolver',
    'ParticleSwarmSolver',
    
    # Direct search solvers
    'GridSearchSolver',
    'NelderMeadSolver',
    
    # Composite solvers
    'HybridSolver'
]