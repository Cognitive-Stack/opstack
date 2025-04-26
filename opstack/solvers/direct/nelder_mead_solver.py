"""
Nelder-Mead simplex optimization solver.

This module provides the NelderMeadSolver class, which implements
a popular direct search method that doesn't require gradient information.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class NelderMeadSolver(Solver):
    """
    Nelder-Mead simplex solver.
    
    A popular direct search method that doesn't require gradient information.
    """
    
    def __init__(self, name: str, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, alpha: float = 1.0,
                 gamma: float = 2.0, rho: float = 0.5, 
                 sigma: float = 0.5, **kwargs):
        """
        Initialize Nelder-Mead solver.
        
        Args:
            name: Name of the solver
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            alpha: Reflection parameter (default 1.0)
            gamma: Expansion parameter (default 2.0)
            rho: Contraction parameter (default 0.5)
            sigma: Shrinkage parameter (default 0.5)
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def _evaluate_simplex(self, simplex: List[List[float]]) -> List[float]:
        """Evaluate all points in the simplex."""
        values = []
        for point in simplex:
            result = self._problem.evaluate(point)
            values.append(self._problem.calculate_objective(result))
        return values
    
    def _centroid(self, simplex: List[List[float]], exclude_index: int) -> List[float]:
        """Calculate centroid of all points except the excluded one."""
        n = len(simplex) - 1  # Number of points to include
        dimensions = len(simplex[0])
        centroid = [0.0] * dimensions
        
        for i, point in enumerate(simplex):
            if i != exclude_index:
                for d in range(dimensions):
                    centroid[d] += point[d] / n
        
        return centroid
    
    def solve(self, initial_solution: Optional[List[float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve using Nelder-Mead simplex method.
        
        Args:
            initial_solution: Starting point (random if None)
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
            raise ValueError("Nelder-Mead requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize solution
        if initial_solution is None:
            initial_solution = [
                (lower_bounds[i] + upper_bounds[i]) / 2.0
                for i in range(dimensions)
            ]
        
        # Initialize simplex (n+1 points for n dimensions)
        simplex = [initial_solution.copy()]
        
        # Add additional points to form the simplex
        for i in range(dimensions):
            point = initial_solution.copy()
            if point[i] != 0:
                point[i] = point[i] * 1.05  # Non-zero perturbation
            else:
                point[i] = 0.00025  # Small non-zero value
            simplex.append(point)
        
        # Evaluate initial simplex
        values = self._evaluate_simplex(simplex)
        
        # Initialize history
        self.history = []
        
        # Track the best solution overall
        best_index = 0
        for i in range(len(values)):
            if self._problem.is_better(values[i], values[best_index]):
                best_index = i
        
        self.best_solution = simplex[best_index].copy()
        self.best_value = values[best_index]
        self.history.append((self.best_solution.copy(), self.best_value))
        
        # Main Nelder-Mead loop
        for iteration in range(self.max_iterations):
            # Sort points by objective value
            sorted_indices = sorted(range(len(values)), 
                                  key=lambda i: values[i],
                                  reverse=not minimizing)
            
            # Rearrange simplex and values
            simplex = [simplex[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            # Best, worst, and second-worst
            x_best = simplex[0]
            f_best = values[0]
            x_worst = simplex[-1]
            f_worst = values[-1]
            x_second_worst = simplex[-2]
            f_second_worst = values[-2]
            
            # Update best overall solution
            if self._problem.is_better(f_best, self.best_value):
                self.best_solution = x_best.copy()
                self.best_value = f_best
                self.history.append((self.best_solution.copy(), self.best_value))
            
            # Check for convergence (when simplex points are close enough)
            if np.std(values) < self.tolerance:
                break
                
            # Calculate centroid of all points except the worst
            x_centroid = self._centroid(simplex, len(simplex) - 1)
            
            # Reflection: reflect worst point through centroid
            x_reflected = [
                x_centroid[i] + self.alpha * (x_centroid[i] - x_worst[i])
                for i in range(dimensions)
            ]
            
            # Bound reflection point
            for i in range(dimensions):
                x_reflected[i] = max(min(x_reflected[i], upper_bounds[i]), lower_bounds[i])
                
            # Evaluate reflected point
            result = self._problem.evaluate(x_reflected)
            f_reflected = self._problem.calculate_objective(result)
            
            # Case 1: Reflection is better than second-worst but not better than best
            if ((minimizing and f_best <= f_reflected < f_second_worst) or 
                (not minimizing and f_best >= f_reflected > f_second_worst)):
                simplex[-1] = x_reflected
                values[-1] = f_reflected
                
            # Case 2: Reflection is better than the best
            elif self._problem.is_better(f_reflected, f_best):
                # Try expansion
                x_expanded = [
                    x_centroid[i] + self.gamma * (x_reflected[i] - x_centroid[i])
                    for i in range(dimensions)
                ]
                
                # Bound expansion point
                for i in range(dimensions):
                    x_expanded[i] = max(min(x_expanded[i], upper_bounds[i]), lower_bounds[i])
                
                # Evaluate expanded point
                result = self._problem.evaluate(x_expanded)
                f_expanded = self._problem.calculate_objective(result)
                
                # Choose better of reflection and expansion
                if self._problem.is_better(f_expanded, f_reflected):
                    simplex[-1] = x_expanded
                    values[-1] = f_expanded
                else:
                    simplex[-1] = x_reflected
                    values[-1] = f_reflected
                    
            # Case 3: Reflection is not better than second-worst
            else:
                # Try contraction
                x_contracted = [
                    x_centroid[i] + self.rho * (x_worst[i] - x_centroid[i])
                    for i in range(dimensions)
                ]
                
                # Bound contraction point
                for i in range(dimensions):
                    x_contracted[i] = max(min(x_contracted[i], upper_bounds[i]), lower_bounds[i])
                
                # Evaluate contracted point
                result = self._problem.evaluate(x_contracted)
                f_contracted = self._problem.calculate_objective(result)
                
                # If contraction is better than worst, use it
                if self._problem.is_better(f_contracted, f_worst):
                    simplex[-1] = x_contracted
                    values[-1] = f_contracted
                else:
                    # Shrink simplex towards best point
                    for i in range(1, len(simplex)):
                        for d in range(dimensions):
                            simplex[i][d] = x_best[d] + self.sigma * (simplex[i][d] - x_best[d])
                    
                    # Re-evaluate all points except the best
                    for i in range(1, len(simplex)):
                        result = self._problem.evaluate(simplex[i])
                        values[i] = self._problem.calculate_objective(result)
            
            # Notify callbacks
            self._notify_callbacks({
                "iteration": iteration,
                "simplex": simplex,
                "values": values,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "simplex_std": np.std(values)
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "iterations": iteration + 1
        }