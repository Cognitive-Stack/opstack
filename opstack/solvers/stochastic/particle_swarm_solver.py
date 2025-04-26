"""
Particle Swarm Optimization (PSO) solver.

This module provides the ParticleSwarmSolver class, which implements
a population-based stochastic method inspired by social behavior of bird flocking.
"""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional, Callable

from opstack import solver
from opstack.core import Solver


@solver
class ParticleSwarmSolver(Solver):
    """
    Particle Swarm Optimization (PSO) solver.
    
    Population-based stochastic method inspired by social behavior of 
    bird flocking or fish schooling.
    """
    
    def __init__(self, name: str, num_particles: int = 30, num_iterations: int = 100,
                 inertia_weight: float = 0.7, cognitive_coef: float = 1.5,
                 social_coef: float = 1.5, **kwargs):
        """
        Initialize PSO solver.
        
        Args:
            name: Name of the solver
            num_particles: Number of particles in the swarm
            num_iterations: Maximum iterations
            inertia_weight: Inertia weight (w)
            cognitive_coef: Cognitive coefficient (c1)
            social_coef: Social coefficient (c2)
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.best_solution = None
        self.best_value = None  # Will be set in solve()
        self.history = []
        
    def solve(self, **kwargs) -> Dict[str, Any]:
        """
        Solve using PSO.
        
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
            raise ValueError("PSO requires problem bounds to be defined")
        
        dimensions = len(lower_bounds)
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_values = []
        
        for _ in range(self.num_particles):
            # Random position and velocity
            position = [random.uniform(lower_bounds[i], upper_bounds[i]) 
                       for i in range(dimensions)]
            velocity = [random.uniform(-1, 1) * (upper_bounds[i] - lower_bounds[i]) * 0.1
                       for i in range(dimensions)]
            
            particles.append(position)
            velocities.append(velocity)
            
            # Evaluate particle
            result = self._problem.evaluate(position)
            value = self._problem.calculate_objective(result)
            
            # Initialize personal best
            personal_best_positions.append(position.copy())
            personal_best_values.append(value)
            
            # Update global best
            if self._problem.is_better(value, self.best_value):
                self.best_solution = position.copy()
                self.best_value = value
                
        self.history = [(self.best_solution.copy(), self.best_value)]
        
        # Main PSO loop
        for iteration in range(self.num_iterations):
            # Update all particles
            for i in range(self.num_particles):
                # Evaluate current position
                result = self._problem.evaluate(particles[i])
                value = self._problem.calculate_objective(result)
                
                # Update personal best
                if self._problem.is_better(value, personal_best_values[i]):
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_values[i] = value
                    
                    # Update global best
                    if self._problem.is_better(value, self.best_value):
                        self.best_solution = particles[i].copy()
                        self.best_value = value
                        # Record improvements
                        self.history.append((self.best_solution.copy(), self.best_value))
                
                # Update velocity and position
                for d in range(dimensions):
                    # Random coefficients
                    r1 = random.random()
                    r2 = random.random()
                    
                    # Update velocity
                    velocities[i][d] = (
                        self.inertia_weight * velocities[i][d] +
                        self.cognitive_coef * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                        self.social_coef * r2 * (self.best_solution[d] - particles[i][d])
                    )
                    
                    # Apply velocity limits (10% of domain range)
                    v_max = 0.1 * (upper_bounds[d] - lower_bounds[d])
                    velocities[i][d] = max(-v_max, min(velocities[i][d], v_max))
                    
                    # Update position
                    particles[i][d] += velocities[i][d]
                    
                    # Apply bounds
                    particles[i][d] = max(lower_bounds[d], min(particles[i][d], upper_bounds[d]))
            
            # Notify callbacks
            self._notify_callbacks({
                "iteration": iteration,
                "particles": particles,
                "velocities": velocities,
                "personal_best_values": personal_best_values,
                "best_solution": self.best_solution,
                "best_value": self.best_value
            })
        
        return {
            "best_solution": self.best_solution,
            "best_value": self.best_value,
            "history": self.history,
            "iterations": self.num_iterations
        }