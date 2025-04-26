"""
Registry for optimization components.

This module provides a central registry for problems and solvers.
"""

from typing import Dict, Any, Type, List, Optional, Union, Callable
import importlib
import logging

from .problem import Problem
from .solver import Solver


class Registry:
    """
    Central registry for optimization components.
    
    This class provides methods to register and retrieve problems
    and solvers.
    """
    
    def __init__(self):
        """Initialize a new registry."""
        self._problems = {}
        self._solvers = {}
        self._problem_classes = {}
        self._solver_classes = {}
        self.logger = logging.getLogger("opstack.registry")
    
    def register_problem(self, problem_class: Type[Problem]) -> None:
        """
        Register a problem class.
        
        Args:
            problem_class: The problem class to register
        """
        self._problem_classes[problem_class.__name__] = problem_class
        self.logger.debug(f"Registered problem class: {problem_class.__name__}")
    
    def register_solver(self, solver_class: Type[Solver]) -> None:
        """
        Register a solver class.
        
        Args:
            solver_class: The solver class to register
        """
        self._solver_classes[solver_class.__name__] = solver_class
        self.logger.debug(f"Registered solver class: {solver_class.__name__}")
    
    def register_problem_instance(self, problem: Problem) -> None:
        """
        Register a problem instance.
        
        Args:
            problem: The problem instance to register
        """
        self._problems[problem.name] = problem
        self.logger.debug(f"Registered problem instance: {problem.name}")
    
    def register_solver_instance(self, solver: Solver) -> None:
        """
        Register a solver instance.
        
        Args:
            solver: The solver instance to register
        """
        self._solvers[solver.name] = solver
        self.logger.debug(f"Registered solver instance: {solver.name}")
    
    def get_problem(self, name: str) -> Optional[Problem]:
        """
        Get a registered problem by name.
        
        Args:
            name: The name of the problem
            
        Returns:
            Optional[Problem]: The problem if found, None otherwise
        """
        return self._problems.get(name)
    
    def get_solver(self, name: str) -> Optional[Solver]:
        """
        Get a registered solver by name.
        
        Args:
            name: The name of the solver
            
        Returns:
            Optional[Solver]: The solver if found, None otherwise
        """
        return self._solvers.get(name)
    
    def create_problem(self, class_name: str, *args, **kwargs) -> Problem:
        """
        Create a new problem instance from a registered class.
        
        Args:
            class_name: The name of the problem class
            *args: Positional arguments for the constructor
            **kwargs: Keyword arguments for the constructor
            
        Returns:
            Problem: A new problem instance
            
        Raises:
            KeyError: If the class name is not registered
        """
        if class_name not in self._problem_classes:
            raise KeyError(f"Problem class not found: {class_name}")
        problem = self._problem_classes[class_name](*args, **kwargs)
        return problem
    
    def create_solver(self, class_name: str, *args, **kwargs) -> Solver:
        """
        Create a new solver instance from a registered class.
        
        Args:
            class_name: The name of the solver class
            *args: Positional arguments for the constructor
            **kwargs: Keyword arguments for the constructor
            
        Returns:
            Solver: A new solver instance
            
        Raises:
            KeyError: If the class name is not registered
        """
        if class_name not in self._solver_classes:
            raise KeyError(f"Solver class not found: {class_name}")
        solver = self._solver_classes[class_name](*args, **kwargs)
        return solver
    
    def list_problems(self) -> List[str]:
        """
        Get a list of all registered problem instances.
        
        Returns:
            List[str]: A list of problem names
        """
        return list(self._problems.keys())
    
    def list_solvers(self) -> List[str]:
        """
        Get a list of all registered solver instances.
        
        Returns:
            List[str]: A list of solver names
        """
        return list(self._solvers.keys())
    
    def list_problem_classes(self) -> List[str]:
        """
        Get a list of all registered problem classes.
        
        Returns:
            List[str]: A list of problem class names
        """
        return list(self._problem_classes.keys())
    
    def list_solver_classes(self) -> List[str]:
        """
        Get a list of all registered solver classes.
        
        Returns:
            List[str]: A list of solver class names
        """
        return list(self._solver_classes.keys())
    
    def register_from_module(self, module_name: str) -> None:
        """
        Register all problems and solvers from a module.
        
        Args:
            module_name: The name of the module to import
        """
        try:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    if issubclass(attr, Problem) and attr is not Problem:
                        self.register_problem(attr)
                    elif issubclass(attr, Solver) and attr is not Solver:
                        self.register_solver(attr)
        except ImportError as e:
            self.logger.error(f"Error importing module {module_name}: {e}")
            
# Create a global registry instance
registry = Registry()