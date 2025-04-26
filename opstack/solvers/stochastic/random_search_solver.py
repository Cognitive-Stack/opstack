"""
Random search optimization solver.

This module provides the RandomSearchSolver class, which implements
a simple stochastic method that randomly samples the solution space.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class RandomSearchSolver(Solver):
    """
    Random search solver.
    
    Simple stochastic method that generates random solutions within bounds
    and keeps track of the best solution found.
    """
    
    def __init__(self, name: str, num_iterations: int = 1000, **kwargs):
        """
        Initialize random search solver.
        
        Args:
            name: Name of the solver
            num_iterations: Number of random solutions to evaluate
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.num_iterations = num_iterations
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using random search.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Dict: Results including best solution and value
        """
        super().solve(**kwargs)
        
        # Store whether we're minimizing or maximizing
        minimizing = self._problem.direction == "minimize"
        self.best_value = float('inf') if minimizing else float('-inf')
        
        # Get problem bounds
        lower_bounds, upper_bounds = self._problem.bounds
        if not lower_bounds or not upper_bounds:
            raise ValueError("Random search requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        self.history = []
        
        for i in range(self.num_iterations):
            # Generate random solution within bounds
            solution = [random.uniform(lower_bounds[j], upper_bounds[j]) 
                       for j in range(dimensions)]
            
            # Evaluate solution
            result = self._problem.evaluate(solution)
            value = self._problem.calculate_objective(result)
            
            # Update best solution if better
            if self._problem.is_better(value, self.best_value):
                self.best_solution = solution.copy()
                self.best_value = value
                
            # Record every N solutions or improvements
            if i % 10 == 0 or self.best_solution == solution:
                self.history.append((solution.copy(), value))
                
            # Notify callbacks
            if i % 10 == 0:  # Don't notify for every iteration to improve performance
                self._notify_callbacks({
                    "iteration": i,
                    "solution": solution,
                    "value": value,
                    "best_solution": self.best_solution,
                    "best_value": self.best_value
                })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "iterations": self.num_iterations
        }