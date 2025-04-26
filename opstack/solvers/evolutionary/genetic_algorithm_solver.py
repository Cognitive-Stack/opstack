"""
Genetic Algorithm optimization solver.

This module provides the GeneticAlgorithmSolver class, which implements
evolutionary optimization using selection, crossover, and mutation operators.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class GeneticAlgorithmSolver(Solver):
    """
    Genetic algorithm solver.
    
    Uses evolutionary operators (selection, crossover, mutation) to
    evolve a population of solutions over generations.
    """
    
    def __init__(self, name: str, population_size: int = 100, num_generations: int = 100,
                crossover_prob: float = 0.8, mutation_prob: float = 0.1, 
                elite_size: int = 5, **kwargs):
        """
        Initialize genetic algorithm solver.
        
        Args:
            name: Name of the solver
            population_size: Size of the population
            num_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elite_size: Number of best individuals to preserve unchanged
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
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
    
    def _tournament_selection(self, fitness_values: List[float], tournament_size: int = 3) -> int:
        """Select an individual using tournament selection."""
        indices = random.sample(range(len(fitness_values)), tournament_size)
        best_idx = indices[0]
        best_fitness = fitness_values[best_idx]
        
        for idx in indices[1:]:
            if self._problem.is_better(fitness_values[idx], best_fitness):
                best_idx = idx
                best_fitness = fitness_values[idx]
                
        return best_idx
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
            
        dimensions = len(parent1)
        # Single-point crossover
        point = random.randint(1, dimensions - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutation(self, individual: List[float], 
                 lower_bounds: List[float], 
                 upper_bounds: List[float]) -> List[float]:
        """Perform mutation on an individual."""
        dimensions = len(individual)
        mutated = individual.copy()
        
        for i in range(dimensions):
            if random.random() < self.mutation_prob:
                # Random value within bounds
                mutated[i] = random.uniform(lower_bounds[i], upper_bounds[i])
                
        return mutated
    
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using genetic algorithm.
        
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
            raise ValueError("Genetic algorithm requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize population
        population = self._initialize_population(dimensions, lower_bounds, upper_bounds)
        self.history = []
        
        # Main evolution loop
        for generation in range(self.num_generations):
            # Evaluate population
            fitness_values = []
            for individual in population:
                result = self._problem.evaluate(individual)
                fitness = self._problem.calculate_objective(result)
                fitness_values.append(fitness)
                
                # Update best solution if better
                if self._problem.is_better(fitness, self.best_value):
                    self.best_solution = individual.copy()
                    self.best_value = fitness
            
            # Record history (best fitness in this generation)
            self.history.append((self.best_solution.copy(), self.best_value))
            
            # Notify callbacks
            self._notify_callbacks({
                "generation": generation,
                "population": population,
                "fitness_values": fitness_values,
                "best_solution": self.best_solution,
                "best_value": self.best_value,
                "avg_fitness": sum(fitness_values) / len(fitness_values)
            })
            
            # Exit if we've reached the last generation
            if generation == self.num_generations - 1:
                break
                
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            idx_sorted = sorted(range(self.population_size), 
                               key=lambda i: fitness_values[i],
                               reverse=not minimizing)
            for i in range(self.elite_size):
                new_population.append(population[idx_sorted[i]].copy())
            
            # Create rest of population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1_idx = self._tournament_selection(fitness_values)
                parent2_idx = self._tournament_selection(fitness_values)
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutation(child1, lower_bounds, upper_bounds)
                child2 = self._mutation(child2, lower_bounds, upper_bounds)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population
            population = new_population
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "generations": len(self.history)
        }