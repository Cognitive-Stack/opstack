"""
Adam (Adaptive Moment Estimation) optimization solver.

This module provides the AdamSolver class, which implements the Adam optimizer
that combines ideas from momentum and RMSProp for adaptive learning rates.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class AdamSolver(Solver):
    """
    Adam (Adaptive Moment Estimation) solver.
    
    A popular optimizer that combines ideas of momentum and RMSprop,
    using both first and second moments of gradients for adaptive learning rates.
    """
    
    def __init__(self, name: str, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 max_iterations: int = 1000, tol: float = 1e-6, **kwargs):
        """
        Initialize Adam solver.
        
        Args:
            name: Name of the solver
            learning_rate: Step size for gradient updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def solve(self, initial_solution: Optional[List[float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve using Adam optimization.
        
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
        lower_bounds, upper_bounds = self._problem.bounds or (None, None)
        if initial_solution is None:
            if lower_bounds and upper_bounds:
                dimensions = len(lower_bounds)
                initial_solution = np.random.uniform(lower_bounds, upper_bounds, dimensions).tolist()
            else:
                # Default to random near origin if no bounds
                initial_solution = np.random.uniform(-1.0, 1.0, 10).tolist()
        
        solution = initial_solution.copy()
        dimensions = len(solution)
        
        # Initialize moment estimates
        m = [0.0] * dimensions  # First moment (mean)
        v = [0.0] * dimensions  # Second moment (variance)
        
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
                
            # Update biased moment estimates
            sign = -1 if minimizing else 1
            for j in range(dimensions):
                g = sign * gradient[j]
                m[j] = self.beta1 * m[j] + (1 - self.beta1) * g
                v[j] = self.beta2 * v[j] + (1 - self.beta2) * (g ** 2)
            
            # Correct bias in moment estimates
            t = i + 1
            m_hat = [m_j / (1 - self.beta1 ** t) for m_j in m]
            v_hat = [v_j / (1 - self.beta2 ** t) for v_j in v]
            
            # Update solution
            for j in range(dimensions):
                solution[j] += self.learning_rate * m_hat[j] / (np.sqrt(v_hat[j]) + self.epsilon)
            
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