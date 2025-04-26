"""
Evolution Strategy (ES) optimization solver.

This module provides the EvolutionStrategySolver class, which implements
a (μ, λ) or (μ + λ) evolution strategy with self-adapting step sizes.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class EvolutionStrategySolver(Solver):
    """
    Evolution Strategy (ES) solver.
    
    Implements a (μ, λ) or (μ + λ) evolution strategy with self-adapting step sizes.
    """
    
    def __init__(self, name: str, mu: int = 20, lambda_: int = 100, 
                 plus_selection: bool = False, num_generations: int = 100,
                 initial_sigma: float = 1.0, sigma_decay: float = 0.99, **kwargs):
        """
        Initialize evolution strategy solver.
        
        Args:
            name: Name of the solver
            mu: Number of parents
            lambda_: Number of offspring
            plus_selection: If True, use (μ+λ) selection, else (μ,λ)
            num_generations: Number of generations
            initial_sigma: Initial step size for mutations
            sigma_decay: Decay factor for step size
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.mu = mu
        self.lambda_ = lambda_
        self.plus_selection = plus_selection
        self.num_generations = num_generations
        self.initial_sigma = initial_sigma
        self.sigma_decay = sigma_decay
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def _initialize_population(self, dimensions: int, 
                              lower_bounds: List[float], 
                              upper_bounds: List[float]) -> List[Tuple[List[float], float]]:
        """Initialize population randomly within bounds with strategy parameters."""
        population = []
        for _ in range(self.mu):
            # Generate random individual within bounds
            individual = [random.uniform(lower_bounds[i], upper_bounds[i]) 
                         for i in range(dimensions)]
            
            # Create strategy parameter (step size)
            sigma = self.initial_sigma
            
            population.append((individual, sigma))
        return population
    
    def _mutate(self, individual: List[float], sigma: float,
               lower_bounds: List[float], upper_bounds: List[float]) -> List[float]:
        """Mutate an individual using Gaussian perturbation."""
        dimensions = len(individual)
        mutated = []
        
        for i in range(dimensions):
            # Apply Gaussian mutation
            value = individual[i] + random.gauss(0, sigma)
            
            # Apply bounds
            value = max(lower_bounds[i], min(value, upper_bounds[i]))
            mutated.append(value)
            
        return mutated
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using evolution strategy.
        
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
            raise ValueError("Evolution strategy requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize population with strategy parameters
        population = self._initialize_population(dimensions, lower_bounds, upper_bounds)
        self.history = []
        
        # Main evolution loop
        for generation in range(self.num_generations):
            # Create offspring
            offspring = []
            for _ in range(self.lambda_):
                # Select random parent
                parent_idx = random.randint(0, self.mu - 1)
                parent, sigma = population[parent_idx]
                
                # Mutate strategy parameter (self-adaptation)
                new_sigma = sigma * self.sigma_decay
                
                # Mutate individual
                offspring_individual = self._mutate(parent, new_sigma, lower_bounds, upper_bounds)
                offspring.append((offspring_individual, new_sigma))
            
            # Evaluate combined population (for plus selection)
            combined_population = offspring
            if self.plus_selection:
                combined_population = population + offspring
            
            # Calculate fitness for combined population
            fitness = []
            for individual, _ in combined_population:
                result = self._problem.evaluate(individual)
                value = self._problem.calculate_objective(result)
                fitness.append(value)
                
                # Update best solution if better
                if self._problem.is_better(value, self.best_value):
                    self.best_solution = individual.copy()
                    self.best_value = value
            
            # Record history
            self.history.append((self.best_solution.copy(), self.best_value))
            
            # Sort by fitness
            sorted_indices = sorted(range(len(combined_population)), 
                                  key=lambda i: fitness[i],
                                  reverse=not minimizing)
            
            # Select best mu individuals
            new_population = [combined_population[i] for i in sorted_indices[:self.mu]]
            population = new_population
            
            # Notify callbacks
            self._notify_callbacks({
                "generation": generation,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "avg_sigma": sum(s for _, s in population) / len(population)
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "generations": len(self.history)
        }