"""
OpStack: A simple modular framework to solve optimization problems.

This package provides a flexible interface for defining and solving
optimization problems using various solvers and objective functions.
"""

__version__ = "0.1.0"

# Import and expose core components
from opstack.core import Problem, Solver, Registry
from opstack.core.registry import registry

# Import and expose facade functions
from opstack.facade import (
    problem, solver,
    optimize, with_callback,
    create_progress_callback, quick_solve
)

__all__ = [
    # Core components
    'Problem', 'Solver', 'Registry', 'registry',
    
    # Facade decorators and functions
    'problem', 'solver',
    'optimize', 'with_callback',
    'create_progress_callback', 'quick_solve'
]