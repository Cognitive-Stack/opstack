"""
Constrained optimization example using OpStack.

This example demonstrates how to solve constrained optimization problems
by implementing the constraint-handling mechanism within the Problem class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable

from opstack import problem, solver
from opstack.core import Problem, Solver
from opstack.solvers.evolutionary import GeneticAlgorithmSolver, DifferentialEvolutionSolver
from opstack.solvers.stochastic import SimulatedAnnealingSolver


@problem(direction="minimize")
class EngineeringDesignProblem(Problem):
    """
    A constrained engineering design optimization problem.
    
    This problem represents a simplified engineering design task
    with multiple constraints. The objective is to minimize cost
    while satisfying constraints on performance metrics.
    """
    
    def __init__(self, name: str = "engineering_design", **kwargs):
        """
        Initialize the engineering design problem.
        
        Args:
            name: Problem name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
        # Define bounds for variables (e.g., thickness, length, width)
        self.bounds = (
            [0.1, 0.1, 0.1],  # Lower bounds
            [10.0, 10.0, 10.0]  # Upper bounds
        )
        
        # Add constraints
        self.add_constraint(self._constraint_strength)
        self.add_constraint(self._constraint_weight)
    
    def _constraint_strength(self, solution: List[float]) -> bool:
        """
        Check if the solution meets the strength constraint.
        
        Args:
            solution: Design parameters [thickness, length, width]
            
        Returns:
            bool: True if constraint is satisfied
        """
        thickness, length, width = solution
        strength = 0.2 * thickness * width**2 / length  # Simplified strength calculation
        return strength >= 1.0  # Minimum strength requirement
    
    def _constraint_weight(self, solution: List[float]) -> bool:
        """
        Check if the solution meets the weight constraint.
        
        Args:
            solution: Design parameters [thickness, length, width]
            
        Returns:
            bool: True if constraint is satisfied
        """
        thickness, length, width = solution
        weight = 7.8 * thickness * length * width  # Simplified weight calculation
        return weight <= 50.0  # Maximum weight limit
    
    def evaluate(self, solution: List[float]) -> Dict[str, Any]:
        """
        Evaluate a solution for the design problem.
        
        Args:
            solution: Design parameters [thickness, length, width]
            
        Returns:
            Dict with metrics and constraint information
        """
        thickness, length, width = solution
        
        # Calculate metrics
        strength = 0.2 * thickness * width**2 / length
        weight = 7.8 * thickness * length * width
        cost = 10 * thickness + 1 * length + 3 * width  # Simplified cost model
        
        # Penalty for constraint violations
        constraint_violation = 0.0
        if not self._constraint_strength(solution):
            constraint_violation += 100 * (1.0 - strength)
        if not self._constraint_weight(solution):
            constraint_violation += 100 * (weight - 50.0)
        
        total_cost = cost + constraint_violation
        
        return {
            "function_value": total_cost,
            "raw_cost": cost,
            "strength": strength,
            "weight": weight,
            "constraint_violation": constraint_violation,
            "constraints_satisfied": self.check_constraints(solution)
        }


def progress_callback(solver_state: Dict) -> None:
    """
    Callback function to monitor solver progress.
    
    Args:
        solver_state: Current state of the solver
    """
    generation = solver_state.get("generation", 0)
    best_value = solver_state.get("best_value", float("inf"))
    
    # Print progress only every 10 generations
    if generation % 10 == 0:
        print(f"Generation {generation}: Best value = {best_value:.6f}")


def compare_solvers():
    """Compare different solvers on the constrained optimization problem."""
    print("Running constrained optimization example...")
    
    # Create problem instance
    problem = EngineeringDesignProblem.create()
    
    # Create and configure solvers
    solvers = {
        "Genetic Algorithm": GeneticAlgorithmSolver.create(
            population_size=100,
            num_generations=100,
            crossover_prob=0.8,
            mutation_prob=0.1
        ),
        "Differential Evolution": DifferentialEvolutionSolver.create(
            population_size=50,
            num_generations=100,
            crossover_prob=0.7,
            differential_weight=0.8
        ),
        "Simulated Annealing": SimulatedAnnealingSolver.create(
            max_iterations=5000,
            initial_temp=100.0,
            cooling_rate=0.95
        )
    }
    
    results = {}
    
    # Run each solver and collect results
    for name, solver in solvers.items():
        print(f"\n=== Running {name} ===")
        
        # Set the problem
        solver.set_problem(problem)
        
        # Add progress callback
        solver.add_callback(progress_callback)
        
        # Solve the problem
        result = solver.solve()
        
        # Store results
        results[name] = {
            "best_solution": result["best_solution"],
            "best_value": result["best_value"],
            "history": result["history"] if "history" in result else None
        }
        
        # Extract solution parameters
        thickness, length, width = result["best_solution"]
        
        # Re-evaluate the solution to get all metrics
        metrics = problem.evaluate(result["best_solution"])
        
        print(f"Best solution: thickness={thickness:.3f}, length={length:.3f}, width={width:.3f}")
        print(f"Cost: {metrics['raw_cost']:.3f}")
        print(f"Strength: {metrics['strength']:.3f}")
        print(f"Weight: {metrics['weight']:.3f}")
        print(f"Constraints satisfied: {metrics['constraints_satisfied']}")
    
    # Plot convergence history for all solvers
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        if result["history"]:
            values = [h[1] for h in result["history"]]
            generations = list(range(len(values)))
            plt.plot(generations, values, label=name)
    
    plt.title('Optimization Convergence Comparison')
    plt.xlabel('Generation / Iteration')
    plt.ylabel('Best Objective Value')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Log scale often better for visualizing convergence
    plt.savefig('constrained_optimization_comparison.png')
    plt.close()
    print("\nConvergence plot saved as 'constrained_optimization_comparison.png'")
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Solver':<25} {'Best Value':<15} {'Constraints Satisfied':<20}")
    print("-" * 60)
    
    for name, result in results.items():
        metrics = problem.evaluate(result["best_solution"])
        print(f"{name:<25} {result['best_value']:<15.6f} {metrics['constraints_satisfied']}")


if __name__ == "__main__":
    compare_solvers()