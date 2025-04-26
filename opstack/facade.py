"""
Facade module for OpStack to simplify usage.

This module provides high-level functions and decorators to make
using the optimization framework more intuitive and concise.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar

from opstack.core import Problem, Solver
from opstack.core.registry import registry

T = TypeVar('T')


def problem(cls=None, *, name=None, direction="minimize"):
    """
    Decorator to register a problem class.
    
    Args:
        cls: The class to decorate
        name: Optional name for the problem instance
        direction: "minimize" or "maximize" for the objective
        
    Returns:
        The decorated class
        
    Examples:
        @problem
        class MyProblem(Problem):
            ...
            
        # Or with custom name and direction
        @problem(name="custom_name", direction="maximize")
        class MyProblem(Problem):
            ...
    """
    def decorator(cls):
        # Register the class
        registry.register_problem(cls)
        
        # Add a factory method to create instances easily
        @classmethod
        def create(cls_, instance_name=None, direction_=None, **kwargs):
            instance_name = instance_name or (name or cls_.__name__)
            instance = cls_(name=instance_name, direction=direction_ or direction, **kwargs)
            registry.register_problem_instance(instance)
            return instance
            
        cls.create = create
        return cls
        
    if cls is None:
        return decorator
    return decorator(cls)


def solver(cls=None, *, name=None):
    """
    Decorator to register a solver class.
    
    Args:
        cls: The class to decorate
        name: Optional name for the solver instance
        
    Returns:
        The decorated class
        
    Examples:
        @solver
        class MySolver(Solver):
            ...
            
        # Or with custom name
        @solver(name="custom_name")
        class MySolver(Solver):
            ...
    """
    def decorator(cls):
        # Register the class
        registry.register_solver(cls)
        
        # Add a factory method to create instances easily
        @classmethod
        def create(cls_, instance_name=None, **kwargs):
            instance_name = instance_name or (name or cls_.__name__)
            instance = cls_(name=instance_name, **kwargs)
            registry.register_solver_instance(instance)
            return instance
            
        cls.create = create
        return cls
        
    if cls is None:
        return decorator
    return decorator(cls)


def optimize(problem_instance: Problem, solver_instance: Solver, **solver_kwargs) -> Dict[str, Any]:
    """
    Convenience function to run an optimization in one line.
    
    Args:
        problem_instance: The problem to solve (includes objective functionality)
        solver_instance: The solver to use
        **solver_kwargs: Additional arguments for the solver
        
    Returns:
        Dict[str, Any]: The optimization result
        
    Example:
        result = optimize(my_problem, my_solver)
    """
    solver_instance.set_problem(problem_instance)
    return solver_instance.solve(**solver_kwargs)


def with_callback(callback_fn: Callable[[Dict[str, Any]], None]) -> Callable[[Solver], Solver]:
    """
    Decorator to add a callback function to a solver.
    
    Args:
        callback_fn: The callback function
        
    Returns:
        Callable: A decorator function
        
    Example:
        @with_callback(lambda state: print(f"Iteration {state['iteration']}: {state['best_value']}"))
        my_solver = MySolver.create()
    """
    def decorator(solver_instance: Solver) -> Solver:
        solver_instance.add_callback(callback_fn)
        return solver_instance
        
    return decorator


def create_progress_callback(frequency: int = 10, 
                            key_metrics: Optional[List[str]] = None) -> Callable:
    """
    Create a simple progress callback function.
    
    Args:
        frequency: How often to print progress (iterations)
        key_metrics: Which metrics to print
        
    Returns:
        Callable: A callback function
        
    Example:
        solver.add_callback(create_progress_callback(frequency=10, 
                                                   key_metrics=["iteration", "best_value"]))
    """
    def callback(state: Dict[str, Any]) -> None:
        # Get the iteration key (different solvers might use different names)
        iter_key = next((k for k in ["iteration", "generation", "epoch"] if k in state), None)
        
        if iter_key is None or state[iter_key] % frequency != 0:
            return
            
        # Format the output
        metrics = key_metrics or ["best_value"]
        metrics_str = ", ".join(f"{m}={state.get(m)}" for m in metrics if m in state)
        
        print(f"{iter_key.capitalize()} {state[iter_key]}: {metrics_str}")
        
    return callback


def quick_solve(problem_cls: Type[Problem], solver_cls: Type[Solver],
               problem_kwargs: Dict = None, solver_kwargs: Dict = None) -> Dict[str, Any]:
    """
    Create instances and solve in one function call.
    
    Args:
        problem_cls: Problem class (with integrated objective functionality)
        solver_cls: Solver class
        problem_kwargs: Arguments for problem creation
        solver_kwargs: Arguments for solver creation and solving
        
    Returns:
        Dict[str, Any]: Optimization result
        
    Example:
        result = quick_solve(MyProblem, MySolver, 
                          problem_kwargs={"capacity": 100},
                          solver_kwargs={"num_iterations": 1000})
    """
    # Create instances
    problem_instance = problem_cls.create(**(problem_kwargs or {}))
    solver_instance = solver_cls.create(**(solver_kwargs or {}))
    
    # Configure and run
    return optimize(problem_instance, solver_instance)