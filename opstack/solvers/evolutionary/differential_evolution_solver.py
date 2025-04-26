"""
Differential Evolution optimization solver.

This module provides the DifferentialEvolutionSolver class, which implements
a powerful stochastic optimization technique using vector differences for mutation.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class DifferentialEvolutionSolver(Solver):
    """
    Differential Evolution (DE) solver.
    
    A powerful stochastic optimization technique that uses vector differences
    for mutation operations.
    """
    
    def __init__(self, name: str, population_size: int = 50, num_generations: int = 100,
                 crossover_prob: float = 0.7, differential_weight: float = 0.8, **kwargs):
        """
        Initialize differential evolution solver.
        
        Args:
            name: Name of the solver
            population_size: Size of the population
            num_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            differential_weight: Scale factor for mutation (F)
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.differential_weight = differential_weight
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
    
    def _initialize_population(self, dimensions: int, 
                              lower_bounds: List[float], 
                              upper_bounds: List[float]) -> List[List[float]]:
        """Initialize population randomly within bounds."""
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(lower_bounds[i], upper_bounds[i]) 
                         for i in range(dimensions)]
            population.append(individual)
        return population
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using differential evolution.
        
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
            raise ValueError("Differential evolution requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize population
        population = self._initialize_population(dimensions, lower_bounds, upper_bounds)
        fitness = [float('inf' if minimizing else '-inf')] * self.population_size
        
        # Evaluate initial population
        for i, individual in enumerate(population):
            result = self._problem.evaluate(individual)
            fitness[i] = self._problem.calculate_objective(result)
            
            # Update best solution if better
            if self._problem.is_better(fitness[i], self.best_value):
                self.best_solution = individual.copy()
                self.best_value = fitness[i]
        
        self.history = [(self.best_solution.copy(), self.best_value)]
        
        # Main DE loop
        for generation in range(self.num_generations):
            # Create new trial individuals
            for i in range(self.population_size):
                # Select three different individuals, all different from i
                candidates = list(range(self.population_size))
                candidates.remove(i)
                r1, r2, r3 = random.sample(candidates, 3)
                
                # Create mutant vector using DE/rand/1 strategy
                mutant = []
                for j in range(dimensions):
                    value = (population[r1][j] + 
                            self.differential_weight * 
                            (population[r2][j] - population[r3][j]))
                    
                    # Apply bounds
                    value = max(lower_bounds[j], min(value, upper_bounds[j]))
                    mutant.append(value)
                
                # Perform crossover (binomial crossover)
                trial = []
                j_rand = random.randint(0, dimensions - 1)  # Ensure at least one parameter is changed
                
                for j in range(dimensions):
                    if j == j_rand or random.random() < self.crossover_prob:
                        trial.append(mutant[j])
                    else:
                        trial.append(population[i][j])
                
                # Evaluate trial solution
                result = self._problem.evaluate(trial)
                trial_fitness = self._problem.calculate_objective(result)
                
                # Replace if better
                if self._problem.is_better(trial_fitness, fitness[i]):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update global best
                    if self._problem.is_better(trial_fitness, self.best_value):
                        self.best_solution = trial.copy()
                        self.best_value = trial_fitness
            
            # Record history
            self.history.append((self.best_solution.copy(), self.best_value))
            
            # Notify callbacks
            self._notify_callbacks({
                "generation": generation,
                "population": population,
                "fitness_values": fitness,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "avg_fitness": sum(fitness) / len(fitness)
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "generations": len(self.history)
        }