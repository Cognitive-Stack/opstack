"""
Simulated Annealing optimization solver.

This module provides the SimulatedAnnealingSolver class, which implements
a probabilistic technique inspired by the annealing process in metallurgy.
"""

import numpy as np
import random
import math
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class SimulatedAnnealingSolver(Solver):
    """
    Simulated annealing solver.
    
    Probabilistic technique that uses a temperature parameter to control
    the acceptance of worse solutions as the algorithm progresses.
    """
    
    def __init__(self, name: str, initial_temp: float = 100.0, final_temp: float = 0.1, 
                 cooling_rate: float = 0.95, num_iterations: int = 1000, 
                 max_iterations_per_temp: int = 50, **kwargs):
        """
        Initialize simulated annealing solver.
        
        Args:
            name: Name of the solver
            initial_temp: Starting temperature
            final_temp: Final temperature (stopping criterion)
            cooling_rate: Rate at which temperature decreases
            num_iterations: Maximum total iterations
            max_iterations_per_temp: Maximum iterations at each temperature
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.max_iterations_per_temp = max_iterations_per_temp
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def _neighbor(self, solution: List[float], temp: float, 
                 lower_bounds: List[float], upper_bounds: List[float]) -> List[float]:
        """Generate a neighbor solution by perturbing current solution."""
        dimensions = len(solution)
        neighbor = solution.copy()
        
        # Choose a random dimension to perturb
        idx = random.randint(0, dimensions - 1)
        
        # Scale of perturbation based on temperature and bounds
        scale = temp / self.initial_temp  # High when hot, low when cold
        range_width = upper_bounds[idx] - lower_bounds[idx]
        perturbation = (random.random() * 2 - 1) * scale * range_width * 0.1
        
        neighbor[idx] += perturbation
        
        # Ensure within bounds
        neighbor[idx] = max(lower_bounds[idx], min(neighbor[idx], upper_bounds[idx]))
        
        return neighbor
    
    def solve(self, initial_solution: Optional[List[float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Solve using simulated annealing.
        
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
            raise ValueError("Simulated annealing requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize solution
        if initial_solution is None:
            current_solution = [random.uniform(lower_bounds[i], upper_bounds[i]) 
                               for i in range(dimensions)]
        else:
            current_solution = initial_solution.copy()
            
        # Evaluate initial solution
        result = self._problem.evaluate(current_solution)
        current_value = self._problem.calculate_objective(result)
        
        # Initialize best solution
        self.best_solution = current_solution.copy()
        self.best_value = current_value
        
        temp = self.initial_temp
        self.history = [(current_solution.copy(), current_value)]
        iteration = 0
        
        # Main simulated annealing loop
        while temp > self.final_temp and iteration < self.num_iterations:
            for _ in range(self.max_iterations_per_temp):
                # Generate neighbor solution
                neighbor_solution = self._neighbor(current_solution, temp, lower_bounds, upper_bounds)
                
                # Evaluate neighbor
                result = self._problem.evaluate(neighbor_solution)
                neighbor_value = self._problem.calculate_objective(result)
                
                # Calculate acceptance probability
                if minimizing:
                    delta = neighbor_value - current_value
                    accept_prob = math.exp(-delta / temp) if delta > 0 else 1.0
                else:
                    delta = current_value - neighbor_value
                    accept_prob = math.exp(-delta / temp) if delta > 0 else 1.0
                
                # Accept neighbor or not
                if random.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_value = neighbor_value
                    
                    # Update best solution if better
                    if self._problem.is_better(current_value, self.best_value):
                        self.best_solution = current_solution.copy()
                        self.best_value = current_value
                        # Record improvements
                        self.history.append((self.best_solution.copy(), self.best_value))
                
                iteration += 1
                if iteration >= self.num_iterations:
                    break
                    
            # Cool temperature
            temp *= self.cooling_rate
            
            # Notify callbacks
            self._notify_callbacks({
                "iteration": iteration,
                "temperature": temp,
                "solution": current_solution,
                "value": current_value,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "accept_prob": accept_prob
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "iterations": iteration,
            "final_temp": temp
        }