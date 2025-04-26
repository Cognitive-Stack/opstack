"""
Grid search optimization solver.

This module provides the GridSearchSolver class, which implements a systematic
exhaustive search through a specified subset of parameter space.
"""

import numpy as np
import itertools
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class GridSearchSolver(Solver):
    """
    Grid search solver.
    
    Systematic exhaustive search through a specified subset of parameter space.
    """
    
    def __init__(self, name: str, grid_points: int = 10, **kwargs):
        """
        Initialize grid search solver.
        
        Args:
            name: Name of the solver
            grid_points: Number of points to sample in each dimension
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.grid_points = grid_points
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using grid search.
        
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
            raise ValueError("Grid search requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Create grid points for each dimension
        grid_values = []
        for d in range(dimensions):
            values = np.linspace(lower_bounds[d], upper_bounds[d], self.grid_points).tolist()
            grid_values.append(values)
            
        # Generate all combinations using itertools.product
        total_points = self.grid_points ** dimensions
        
        # Warn if grid might be too large
        if total_points > 1000000:
            print(f"Warning: Grid search will evaluate {total_points} points. "
                  f"This may take a very long time.")
        
        # Initialize counters
        evaluated = 0
        self.history = []
        
        # Iterate through grid points
        for grid_point in itertools.product(*grid_values):
            solution = list(grid_point)
            
            # Evaluate solution
            result = self._problem.evaluate(solution)
            value = self._problem.calculate_objective(result)
            
            # Update best solution if better
            if self._problem.is_better(value, self.best_value):
                self.best_solution = solution.copy()
                self.best_value = value
                
                # Record improvements
                self.history.append((solution.copy(), value))
            
            evaluated += 1
            
            # Notify callbacks periodically
            if evaluated % 100 == 0 or evaluated == total_points:
                self._notify_callbacks({
                    "iteration": evaluated,
                    "total_points": total_points,
                    "progress": evaluated / total_points,
                    "solution": solution,
                    "value": value,
                    "best_solution": self.best_solution,
                    "best_value": self.best_value
                })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "evaluated_points": evaluated
        }