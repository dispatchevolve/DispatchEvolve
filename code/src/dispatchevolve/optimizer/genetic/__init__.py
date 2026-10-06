# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
DispatchEvolve Genetic Optimizer: An evolutionary program optimization engine
"""

from dispatchevolve.optimizer.genetic._version import __version__
from dispatchevolve.optimizer.genetic.config import Config
from dispatchevolve.optimizer.genetic.controller import GeneticOptimizer
from dispatchevolve.optimizer.genetic.api import (
    run_evolution,
    evolve_function,
    evolve_algorithm,
    evolve_code,
    EvolutionResult,
)

__all__ = [
    "Config",
    "GeneticOptimizer",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
]
