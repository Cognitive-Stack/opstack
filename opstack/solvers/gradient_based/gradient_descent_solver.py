"""
Standard gradient descent optimization solver.

This module provides the GradientDescentSolver class, which implements
gradient descent for problems that provide gradient information.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class GradientDescentSolver(Solver):
    """
    Standard gradient descent solver.
    
    Iteratively updates solution in the direction of the negative gradient.
    Requires problems that provide gradient information in their evaluation results.
    """
    
    def __init__(self, name: str, learning_rate: float = 0.01, 
                 max_iterations: int = 1000, tol: float = 1e-6, **kwargs):
        """
        Initialize gradient descent solver.
        
        Args:
            name: Name of the solver
            learning_rate: Step size for gradient updates
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.best_solution = None
        self.best_value = float('inf') if self._problem else None  # Will be set in solve()
        self.history = []
        
    def solve(self, initial_solution: Optional[List[float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve using gradient descent.
        
        Args:
            initial_solution: Starting point (random if None)
            **kwargs: Additional parameters
            
        Returns:
            Dict: Results including best solution and value
            
        Raises:
            ValueError: If problem doesn't provide gradient information
        """
        super().solve(**kwargs)
        
        # Store whether we're minimizing or maximizing
        minimizing = self._problem.direction == "minimize"
        self.best_value = float('inf') if minimizing else float('-inf')
        
        # Get problem bounds
        lower_bounds, upper_bounds = self._problem.bounds or (None, None)
        if initial_solution is None:
            # Generate random initial solution within bounds if provided
            if lower_bounds and upper_bounds:
                dimensions = len(lower_bounds)
                initial_solution = np.random.uniform(lower_bounds, upper_bounds, dimensions).tolist()
            else:
                # Default to random near origin if no bounds
                initial_solution = np.random.uniform(-1.0, 1.0, 10).tolist()
        
        solution = initial_solution.copy()
        self.history = []
        
        for i in range(self.max_iterations):
            # Evaluate current solution
            result = self._problem.evaluate(solution)
            value = self._problem.calculate_objective(result)
            
            # Update best solution if better
            if self._problem.is_better(value, self.best_value):
                self.best_solution = solution.copy()
                self.best_value = value
                
            # Record history
            self.history.append((solution.copy(), value))
            
            # Check for gradient in result
            if 'gradient' not in result:
                raise ValueError("Problem must provide gradient information in evaluation results")
            
            gradient = result['gradient']
            gradient_norm = np.linalg.norm(gradient)
            
            # Check for convergence
            if gradient_norm < self.tol:
                break
                
            # Update solution (subtract gradient for minimization, add for maximization)
            sign = -1 if minimizing else 1
            solution = [x + sign * self.learning_rate * g for x, g in zip(solution, gradient)]
            
            # Apply bounds if provided
            if lower_bounds and upper_bounds:
                solution = [max(min(x, u), l) for x, l, u in zip(solution, lower_bounds, upper_bounds)]
            
            # Notify callbacks
            self._notify_callbacks({
                "iteration": i,
                "solution": solution,
                "value": value,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "gradient_norm": gradient_norm
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "iterations": len(self.history)
        }