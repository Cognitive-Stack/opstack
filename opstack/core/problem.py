"""
Base class for optimization problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Callable


class Problem(ABC):
    """
    Base class for all optimization problems.
    
    A problem defines the search space, constraints, objective function,
    and any problem-specific logic needed for optimization.
    """
    
    def __init__(self, name: str, direction: str = "minimize", **kwargs):
        """
        Initialize a new optimization problem.
        
        Args:
            name: A unique identifier for the problem
            direction: Either "minimize" or "maximize" for the objective
            **kwargs: Problem-specific configuration
        """
        self.name = name
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be either 'minimize' or 'maximize'")
        self.direction = direction
        self.config = kwargs
        self._constraints = []
        self._bounds = None
    
    @property
    def bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Get the bounds of the problem variables (lower, upper)."""
        return self._bounds
    
    @bounds.setter
    def bounds(self, bounds: Tuple[List[float], List[float]]):
        """Set the bounds of the problem variables."""
        self._bounds = bounds
    
    def add_constraint(self, constraint_func):
        """
        Add a constraint function to the problem.
        
        Args:
            constraint_func: A function that returns True if the constraint is satisfied
        """
        self._constraints.append(constraint_func)
    
    def check_constraints(self, solution) -> bool:
        """
        Check if a solution satisfies all constraints.
        
        Args:
            solution: The solution to check
            
        Returns:
            bool: True if all constraints are satisfied
        """
        return all(constraint(solution) for constraint in self._constraints)
    
    @abstractmethod
    def evaluate(self, solution) -> Dict[str, Any]:
        """
        Evaluate a solution and return metrics.
        
        Args:
            solution: The solution to evaluate
            
        Returns:
            Dict[str, Any]: A dictionary of evaluation metrics
        """
        pass
    
    def calculate_objective(self, evaluation_result: Dict[str, Any]) -> float:
        """
        Calculate the objective value from an evaluation result.
        
        Args:
            evaluation_result: The result from Problem.evaluate()
            
        Returns:
            float: The calculated objective value
        """
        # Default implementation, should be overridden by subclasses if needed
        return evaluation_result.get("function_value", 0.0)
    
    def is_better(self, value1: float, value2: float) -> bool:
        """
        Compare two objective values and determine if the first is better.
        
        Args:
            value1: The first objective value
            value2: The second objective value
            
        Returns:
            bool: True if value1 is better than value2 according to the objective direction
        """
        if self.direction == "minimize":
            return value1 < value2
        return value1 > value2