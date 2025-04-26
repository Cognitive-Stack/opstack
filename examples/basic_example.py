"""
Basic example demonstrating how to use OpStack for function optimization.

This example optimizes the sphere function, which is a simple benchmark function
with a global minimum at [0, 0, ..., 0].
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from opstack import problem, quick_solve
from opstack.core import Problem
from opstack.solvers.evolutionary import GeneticAlgorithmSolver


@problem(direction="minimize")
class SphereProblem(Problem):
    """
    Sphere function optimization problem.
    
    The sphere function is a simple benchmark function defined as:
    f(x) = sum(x_i^2) for i=1 to n
    
    It has a global minimum at [0, 0, ..., 0] with a value of 0.
    """
    
    def __init__(self, name: str = "sphere", dimensions: int = 2, **kwargs):
        """
        Initialize the sphere problem.
        
        Args:
            name: Problem name
            dimensions: Number of dimensions
            **kwargs: Additional parameters
        """
        # Don't explicitly set direction here since it's already set in the decorator
        super().__init__(name, **kwargs)
        self.dimensions = dimensions
        
        # Define bounds for each dimension: [-5.12, 5.12]
        lower_bounds = [-5.12] * self.dimensions
        upper_bounds = [5.12] * self.dimensions
        self.bounds = (lower_bounds, upper_bounds)
    
    def evaluate(self, solution: List[float]) -> Dict[str, Any]:
        """
        Evaluate the sphere function for a given solution.
        
        Args:
            solution: A list of float values representing a point in the search space
            
        Returns:
            Dict with function_value and any other metrics
        """
        # Calculate sphere function: sum of squared components
        value = sum(x**2 for x in solution)
        
        return {
            "function_value": value,
            "solution_norm": np.linalg.norm(solution)
        }


def run_example():
    """Run the basic optimization example."""
    print("Running basic sphere function optimization example...")
    
    # Create problem instance
    dimensions = 3
    sphere = SphereProblem.create(dimensions=dimensions)
    
    # Method 1: Using quick_solve convenience function
    print("\nMethod 1: Using quick_solve...")
    result = quick_solve(
        SphereProblem, 
        GeneticAlgorithmSolver,
        problem_kwargs={"dimensions": dimensions},
        solver_kwargs={"population_size": 50, "num_generations": 100}
    )
    
    print(f"Best solution: {result['best_solution']}")
    print(f"Best value: {result['best_value']}")
    
    # Method 2: Creating solver instance and problem instance explicitly
    print("\nMethod 2: Using solver and problem instances...")
    sphere = SphereProblem.create(dimensions=dimensions)
    solver = GeneticAlgorithmSolver.create(
        population_size=50,
        num_generations=100
    )
    
    # Set the problem on the solver
    solver.set_problem(sphere)
    
    # Solve
    result = solver.solve()
    
    print(f"Best solution: {result['best_solution']}")
    print(f"Best value: {result['best_value']}")
    
    # Plot convergence history if this is a 2D problem
    if dimensions == 2:
        history = result["history"]
        generations = len(history)
        best_values = [h[1] for h in history]  # Extract best values from history tuples
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(generations), best_values)
        plt.title('Convergence History')
        plt.xlabel('Generation')
        plt.ylabel('Best Objective Value')
        plt.grid(True)
        plt.yscale('log')  # Log scale often better for visualizing convergence
        plt.savefig('sphere_convergence.png')
        plt.close()
        print("\nConvergence plot saved as 'sphere_convergence.png'")


if __name__ == "__main__":
    run_example()