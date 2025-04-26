"""
Hybrid optimization solver.

This module provides hybrid solver implementations that combine
multiple optimization strategies to solve complex problems.
"""

import time
from typing import Dict, Any, List, Tuple, Optional

from opstack import solver
from opstack.core import Solver
from opstack.solvers.evolutionary import GeneticAlgorithmSolver
from opstack.solvers.gradient_based import GradientDescentSolver


@solver
class HybridSolver(Solver):
    """
    A hybrid solver that combines multiple optimization strategies.
    
    This solver first uses an evolutionary approach to find a good
    starting point, then refines it using gradient-based optimization.
    """
    
    def __init__(self, name: str = "hybrid_solver", **kwargs):
        """
        Initialize hybrid solver.
        
        Args:
            name: Solver name
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
        # Create component solvers
        self.global_solver = GeneticAlgorithmSolver.create(
            instance_name="global_ga",
            population_size=50,
            num_generations=50
        )
        
        self.local_solver = GradientDescentSolver.create(
            instance_name="local_gd",
            max_iterations=200,
            learning_rate=0.1,
            momentum=0.9
        )
        
    def solve(self) -> Dict[str, Any]:
        """
        Solve the optimization problem using the hybrid approach.
        
        Returns:
            Dict with solution results
        """
        start_time = time.time()
        
        # Ensure problem is set
        if self._problem is None:
            raise ValueError("Problem must be set before solving")
            
        print("Starting global search with Genetic Algorithm...")
        
        # First phase: Global search
        self.global_solver.set_problem(self._problem)
        global_result = self.global_solver.solve()
        
        # Extract best solution from global search
        initial_solution = global_result["best_solution"]
        
        print(f"Global search completed. Best value: {global_result['best_value']:.6f}")
        print("Starting local refinement with Gradient Descent...")
        
        # Second phase: Local refinement
        self.local_solver.set_problem(self._problem)
        # Pass the initial_solution directly to the solve method
        local_result = self.local_solver.solve(initial_solution=initial_solution)
        
        # Final result
        final_value = local_result["best_value"]
        final_solution = local_result["best_solution"]
        
        print(f"Local refinement completed. Best value: {final_value:.6f}")
        
        # Combine history from both solvers
        history = []
        if "history" in global_result:
            history.extend(global_result["history"])
        
        # Offset the iterations in local_result history
        if "history" in local_result:
            start_iter = len(history)
            history.extend([(start_iter + i, v) for i, (_, v) in enumerate(local_result["history"])])
        
        end_time = time.time()
        
        return {
            "best_value": final_value,
            "best_solution": final_solution,
            "history": history,
            "solve_time": end_time - start_time
        }