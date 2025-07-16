"""
Countdown number puzzle solver.

This module implements an efficient algorithm to solve Countdown number puzzles
where ALL numbers must be used exactly once.
"""

import itertools
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class CountdownSolver:
    """Solver for Countdown number puzzles where all numbers must be used exactly once."""
    
    def __init__(self):
        self.best_solution = None
        self.best_distance = float('inf')
        
    def solve(self, numbers: List[int], target: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Solve a Countdown puzzle using ALL numbers exactly once.
        
        Args:
            numbers: List of available numbers (all must be used)
            target: Target number to reach
            
        Returns:
            Tuple of (expression_string, human_explanation) or (None, None) if no solution
        """
        self.best_solution = None
        self.best_distance = float('inf')
        
        # Try to find a solution using all numbers
        result = self._solve_recursive(numbers, target)
        
        if result:
            expression, steps = result
            explanation = self._format_explanation(steps, target)
            return expression, explanation
        
        # If no exact solution, return the best we found (if any)
        if self.best_solution and self.best_distance < float('inf'):
            expression, steps, value = self.best_solution
            explanation = self._format_explanation(steps, value)
            explanation += f" This gives us {value}, which is {abs(value - target)} away from our target of {target}."
            return expression, explanation
        
        return None, None
    
    def _solve_recursive(self, numbers: List[int], target: int) -> Optional[Tuple[str, List[str]]]:
        """Recursively find a solution using all numbers."""
        # Base case: if we have one number left, check if it equals target
        if len(numbers) == 1:
            value = numbers[0]
            distance = abs(value - target)
            
            # Track best solution even if not exact
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_solution = (str(value), [], value)
            
            if value == target:
                return str(target), []
            else:
                return None
        
        # Try all pairs of numbers
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                a, b = numbers[i], numbers[j]
                remaining = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                
                # Try all operations
                operations = []
                
                # Addition
                operations.append((a + b, f"({a} + {b})", f"add {a} + {b} = {a + b}"))
                
                # Subtraction (both ways, but only if result is positive)
                if a > b:
                    operations.append((a - b, f"({a} - {b})", f"subtract {a} - {b} = {a - b}"))
                if b > a:
                    operations.append((b - a, f"({b} - {a})", f"subtract {b} - {a} = {b - a}"))
                
                # Multiplication
                operations.append((a * b, f"({a} * {b})", f"multiply {a} * {b} = {a * b}"))
                
                # Division (only if it results in a whole number)
                if b != 0 and a % b == 0:
                    operations.append((a // b, f"({a} / {b})", f"divide {a} / {b} = {a // b}"))
                if a != 0 and b % a == 0:
                    operations.append((b // a, f"({b} / {a})", f"divide {b} / {a} = {b // a}"))
                
                for result_value, expr_str, step in operations:
                    if result_value > 0:  # Only positive results allowed in Countdown
                        new_numbers = remaining + [result_value]
                        sub_solution = self._solve_recursive(new_numbers, target)
                        
                        if sub_solution:
                            sub_expr, sub_steps = sub_solution
                            
                            # Replace the result value in the sub-expression with our expression
                            if sub_expr == str(result_value):
                                # The result was the final answer
                                final_expr = expr_str
                            else:
                                # The result was used in further calculations
                                # Need to be careful about replacement to avoid replacing wrong numbers
                                final_expr = self._safe_replace(sub_expr, result_value, expr_str)
                            
                            final_steps = [step] + sub_steps
                            return final_expr, final_steps
                        
                        # Even if no exact solution, track if this path gets us closer
                        if len(remaining) == 0 and abs(result_value - target) < self.best_distance:
                            self.best_distance = abs(result_value - target)
                            self.best_solution = (expr_str, [step], result_value)
        
        return None
    
    def _safe_replace(self, expr: str, value: int, replacement: str) -> str:
        """Safely replace a value in an expression with another expression."""
        # This is a simplified version - in production you'd want a proper parser
        # For now, we'll do a simple string replacement
        value_str = str(value)
        
        # Look for the value as a standalone number (not part of a larger number)
        import re
        pattern = r'\b' + re.escape(value_str) + r'\b'
        return re.sub(pattern, replacement, expr, count=1)
    
    def _format_explanation(self, steps: List[str], final_value: int) -> str:
        """Format the solution steps into a human-readable explanation."""
        if not steps:
            return f"The answer is simply {final_value}."
        
        explanation_parts = []
        for i, step in enumerate(steps):
            if i == 0:
                explanation_parts.append(f"First, {step}")
            elif i == len(steps) - 1:
                explanation_parts.append(f"Finally, {step}")
            else:
                explanation_parts.append(f"Then {step}")
        
        return " ".join(explanation_parts) + "."

def solve_countdown_puzzle(numbers: List[int], target: int) -> Tuple[str, str]:
    """
    Convenience function to solve a countdown puzzle.
    All numbers must be used exactly once.
    
    Args:
        numbers: List of available numbers (all must be used)
        target: Target number to reach
        
    Returns:
        Tuple of (expression, explanation)
    """
    if len(numbers) == 0:
        return "No numbers provided", "Cannot solve puzzle without numbers."
    
    if len(numbers) == 1:
        if numbers[0] == target:
            return str(numbers[0]), f"The answer is simply {numbers[0]}."
        else:
            return str(numbers[0]), f"With only {numbers[0]} available, we cannot reach {target}. The result is {numbers[0]}."
    
    solver = CountdownSolver()
    expression, explanation = solver.solve(numbers, target)
    
    if expression is None:
        return "No solution found", f"No valid combination of {numbers} could reach {target} using all numbers exactly once."
    
    return expression, explanation