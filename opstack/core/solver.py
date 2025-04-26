"""
Base class for optimization solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

from .problem import Problem


class Solver(ABC):
    """
    Base class for optimization solvers.
    
    A solver implements the algorithm to find optimal solutions
    for a given problem.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize a new solver.
        
        Args:
            name: A unique identifier for the solver
            **kwargs: Solver-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._callbacks = []
        self._problem = None
        
    def set_problem(self, problem: Problem) -> "Solver":
        """
        Set the problem to solve.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Solver: Self for method chaining
        """
        self._problem = problem
        return self
    
    def add_callback(self, callback: Callable) -> "Solver":
        """
        Add a callback function to be called during the optimization process.
        
        Args:
            callback: A function to call with the current state
            
        Returns:
            Solver: Self for method chaining
        """
        self._callbacks.append(callback)
        return self
        
    def _notify_callbacks(self, state: Dict[str, Any]):
        """
        Notify all callbacks with the current state.
        
        Args:
            state: The current solver state
        """
        for callback in self._callbacks:
            callback(state)
    
    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the optimization problem.
        
        Args:
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Dict[str, Any]: The solution and other solver outputs
        """
        if self._problem is None:
            raise ValueError("Problem not set. Call set_problem() first.")