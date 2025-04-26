"""
Advanced optimization examples for OpStack.

This example demonstrates:
1. Using gradient-based optimization
2. Customizing solver behavior for specific problems
3. Multi-objective optimization by scalarization
4. Combining multiple solvers in sequence
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import time

from opstack import problem, solver
from opstack.core import Problem, Solver
from opstack.solvers.gradient_based import GradientDescentSolver
from opstack.solvers.evolutionary import GeneticAlgorithmSolver
from opstack.solvers.direct import NelderMeadSolver
from opstack.solvers.composite import HybridSolver


@problem(direction="minimize")
class PortfolioOptimizationProblem(Problem):
    """
    Portfolio optimization problem using mean-variance approach.
    
    This implements Markowitz portfolio optimization, where we
    aim to balance expected returns and risk.
    """
    
    def __init__(self, name: str = "portfolio_optimization",
                 num_assets: int = 4, risk_aversion: float = 1.0,
                 expected_returns: Optional[List[float]] = None, 
                 covariance_matrix: Optional[List[List[float]]] = None,
                 **kwargs):
        """
        Initialize portfolio optimization problem.
        
        Args:
            name: Problem name
            num_assets: Number of assets in the portfolio
            risk_aversion: Risk aversion parameter (lambda)
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix for asset returns
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.num_assets = num_assets
        self.risk_aversion = risk_aversion
        
        # Create sample data if not provided
        if expected_returns is None:
            # Sample expected returns (higher mean for high expected return)
            self.expected_returns = np.array([0.05, 0.10, 0.15, 0.20][:num_assets])
        else:
            self.expected_returns = np.array(expected_returns)
            
        if covariance_matrix is None:
            # Sample covariance matrix
            # Diagonal elements are variances (risk), off-diagonals are correlations
            base_cov = np.array([
                [0.10, 0.03, 0.01, 0.02],
                [0.03, 0.15, 0.05, 0.06],
                [0.01, 0.05, 0.20, 0.04],
                [0.02, 0.06, 0.04, 0.25]
            ])[:num_assets, :num_assets]
            self.covariance_matrix = base_cov
        else:
            self.covariance_matrix = np.array(covariance_matrix)
            
        # Define bounds for each asset weight: [0, 1]
        self.bounds = (
            [0.0] * self.num_assets,  # Lower bounds
            [1.0] * self.num_assets   # Upper bounds
        )
        
        # Add constraint: weights sum to 1
        self.add_constraint(self._constraint_sum_to_one)
        
    def _constraint_sum_to_one(self, solution: List[float]) -> bool:
        """
        Check if the weights sum to 1.
        
        Args:
            solution: List of asset weights
            
        Returns:
            bool: True if the weights sum to approximately 1
        """
        return abs(sum(solution) - 1.0) < 1e-5
    
    def gradient(self, solution: List[float]) -> List[float]:
        """
        Calculate the gradient of the objective function.
        
        This is used by gradient-based solvers.
        
        Args:
            solution: List of asset weights
            
        Returns:
            Gradient vector
        """
        # Convert to numpy array for easier calculation
        weights = np.array(solution)
        
        # Gradient of -returns + lambda * risk
        grad_returns = -self.expected_returns
        grad_risk = 2 * self.risk_aversion * np.dot(self.covariance_matrix, weights)
        
        return (grad_returns + grad_risk).tolist()
    
    def evaluate(self, solution: List[float]) -> Dict[str, Any]:
        """
        Evaluate a portfolio allocation.
        
        Args:
            solution: List of asset weights
            
        Returns:
            Dict with evaluation metrics
        """
        # Convert to numpy array for easier calculation
        weights = np.array(solution)
        
        # Calculate expected return and risk
        expected_return = np.dot(weights, self.expected_returns)
        risk = np.dot(weights, np.dot(self.covariance_matrix, weights))
        
        # Penalty for constraint violations
        penalty = 0.0
        if not self._constraint_sum_to_one(solution):
            penalty = 100.0 * abs(sum(solution) - 1.0)
        
        # Objective function: -return + lambda * risk + penalty
        # (Negative return because we're minimizing)
        objective = -expected_return + self.risk_aversion * risk + penalty
        
        # Calculate gradient
        gradient = self.gradient(solution)
        
        return {
            "function_value": objective,
            "expected_return": expected_return,
            "risk": risk,
            "sharpe_ratio": expected_return / (np.sqrt(risk) + 1e-10),  # Avoid division by zero
            "weights": weights.tolist(),
            "constraints_satisfied": self._constraint_sum_to_one(solution),
            "gradient": gradient  # Include gradient information for gradient-based solvers
        }


def run_advanced_example():
    """
    Run the advanced optimization examples.
    """
    print("Running advanced portfolio optimization example...")
    
    # Create problem instance
    problem = PortfolioOptimizationProblem.create(
        num_assets=4,
        risk_aversion=1.0  # Balance between return and risk
    )
    
    # Try different solver strategies
    solvers = {
        "Gradient Descent": GradientDescentSolver.create(
            max_iterations=1000,
            learning_rate=0.01,
            momentum=0.9
        ),
        "Nelder-Mead": NelderMeadSolver.create(
            max_iterations=1000,
            initial_simplex_size=0.1
        ),
        "Hybrid Solver": HybridSolver.create()
    }
    
    results = {}
    
    # Run each solver and collect results
    for name, solver in solvers.items():
        print(f"\n=== Running {name} ===")
        
        # Set the problem
        solver.set_problem(problem)
        
        # Solve the problem
        start_time = time.time()
        result = solver.solve()
        end_time = time.time()
        
        # Store results
        results[name] = {
            "best_solution": result["best_solution"],
            "best_value": result["best_value"],
            "history": result["history"] if "history" in result else None,
            "solve_time": end_time - start_time
        }
        
        # Print detailed results
        weights = result["best_solution"]
        eval_result = problem.evaluate(weights)
        
        print(f"Asset Weights: {[f'{w:.4f}' for w in weights]}")
        print(f"Expected Return: {eval_result['expected_return']:.4f}")
        print(f"Risk (Variance): {eval_result['risk']:.4f}")
        print(f"Sharpe Ratio: {eval_result['sharpe_ratio']:.4f}")
        print(f"Solution time: {results[name]['solve_time']:.3f} seconds")
    
    # Visualize the results
    # Plot convergence history
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for name, result in results.items():
        if result["history"]:
            values = [h[1] for h in result["history"]]
            iterations = list(range(len(values)))
            plt.plot(iterations, values, label=name)
    
    plt.title('Optimization Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # Plot final portfolio allocations
    plt.subplot(2, 1, 2)
    asset_labels = [f'Asset {i+1}' for i in range(problem.num_assets)]
    x = np.arange(len(asset_labels))
    width = 0.8 / len(solvers)
    
    for i, (name, result) in enumerate(results.items()):
        weights = result["best_solution"]
        plt.bar(x + i * width - 0.4 + width/2, weights, width, label=name)
    
    plt.title('Portfolio Allocations')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(x, asset_labels)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('portfolio_optimization_results.png')
    plt.close()
    print("\nResults plot saved as 'portfolio_optimization_results.png'")
    
    # Print summary table
    print("\n=== Performance Summary ===")
    print(f"{'Solver':<15} {'Objective':<12} {'Return':<12} {'Risk':<12} {'Sharpe':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        eval_result = problem.evaluate(result["best_solution"])
        print(f"{name:<15} {result['best_value']:<12.6f} "
              f"{eval_result['expected_return']:<12.6f} {eval_result['risk']:<12.6f} "
              f"{eval_result['sharpe_ratio']:<12.6f} {result['solve_time']:<10.3f}")


if __name__ == "__main__":
    run_advanced_example()