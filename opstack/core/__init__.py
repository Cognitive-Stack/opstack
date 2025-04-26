"""
Core components of the OpStack optimization framework.

This module contains the base classes for problems, objectives, and solvers.
"""

from .problem import Problem
from .solver import Solver
from .registry import Registry

__all__ = ['Problem', 'Solver', 'Registry']