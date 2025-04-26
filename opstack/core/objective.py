"""
Base class for optimization objectives.
"""

from abc import ABC, abstractmethod
from typing import Any


class Objective(ABC):
    """
    Base class for optimization objectives.
    
    An objective defines the goal of the optimization process,
    such as minimization or maximization of a specific metric.
    """
    
    def __init__(self, name: str, direction: str = "minimize", **kwargs):
        """
        Initialize a new optimization objective.
        
        Args:
            name: A unique identifier for the objective
            direction: Either "minimize" or "maximize"
            **kwargs: Objective-specific configuration
        """
        self.name = name
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be either 'minimize' or 'maximize'")
        self.direction = direction
        self.config = kwargs
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        """
        Calculate the objective value.
        
        Returns:
            float: The calculated objective value
        """
        pass
        
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