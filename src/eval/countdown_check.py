"""
Evaluation module for Countdown number puzzles.

This module provides functions to evaluate whether a proposed solution
to a Countdown puzzle is correct.
"""

import re
import ast
import operator
import logging
from typing import Optional, List, Union, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Define safe operators for expression evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(expression: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression containing only basic arithmetic.
    
    Args:
        expression: String containing mathematical expression
        
    Returns:
        The result of the expression, or None if evaluation fails
    """
    try:
        # Parse the expression into an AST
        node = ast.parse(expression, mode='eval')
        
        def _eval_node(node):
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            elif isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # For older Python versions
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                op = SAFE_OPERATORS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
                # Handle division by zero
                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError("Division by zero")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _eval_node(node.operand)
                op = SAFE_OPERATORS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported unary operator: {type(node.op)}")
                return op(operand)
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")
        
        result = _eval_node(node.body)
        return float(result)
    except Exception as e:
        logger.warning(f"Failed to evaluate expression '{expression}': {e}")
        return None

def extract_numbers_from_expression(expression: str) -> List[int]:
    """
    Extract all numbers used in an arithmetic expression.
    
    Args:
        expression: String containing mathematical expression
        
    Returns:
        List of numbers found in the expression
    """
    # Remove spaces and find all numbers
    numbers = re.findall(r'\d+', expression)
    return [int(num) for num in numbers]

def check_countdown_solution(
    expression: str,
    available_numbers: List[int],
    target: int,
    tolerance: float = 1e-9
) -> Dict[str, Any]:
    """
    Check if a proposed Countdown solution is valid.
    
    Args:
        expression: The arithmetic expression proposed as solution
        available_numbers: List of numbers available to use
        target: The target number to reach
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        Dictionary containing evaluation results
    """
    result = {
        'is_correct': False,
        'expression': expression,
        'target': target,
        'available_numbers': available_numbers,
        'computed_result': None,
        'numbers_used': [],
        'valid_number_usage': False,
        'valid_computation': False,
        'error_message': None
    }
    
    try:
        # Clean the expression - remove spaces and normalize
        clean_expr = expression.replace(' ', '')
        
        # Extract numbers used in the expression
        numbers_used = extract_numbers_from_expression(clean_expr)
        result['numbers_used'] = numbers_used
        
        # Check if all used numbers are available
        available_copy = available_numbers.copy()
        for num in numbers_used:
            if num in available_copy:
                available_copy.remove(num)
            else:
                result['error_message'] = f"Number {num} not available or used multiple times"
                return result
        
        result['valid_number_usage'] = True
        
        # Evaluate the expression
        computed_result = safe_eval(clean_expr)
        if computed_result is None:
            result['error_message'] = "Failed to evaluate expression"
            return result
        
        result['computed_result'] = computed_result
        result['valid_computation'] = True
        
        # Check if the result matches the target (within tolerance)
        if abs(computed_result - target) <= tolerance:
            result['is_correct'] = True
        else:
            result['error_message'] = f"Expression evaluates to {computed_result}, not {target}"
        
    except Exception as e:
        result['error_message'] = f"Error evaluating solution: {str(e)}"
        logger.error(f"Error in check_countdown_solution: {e}")
    
    return result

def parse_countdown_answer(answer: str) -> Optional[str]:
    """
    Parse a model's answer to extract the arithmetic expression.
    
    Args:
        answer: The raw answer from the model
        
    Returns:
        The cleaned arithmetic expression, or None if parsing fails
    """
    if not answer:
        return None
    
    # If the answer contains an equals sign, extract the left side
    if '=' in answer:
        expression = answer.split('=')[0].strip()
    else:
        expression = answer.strip()
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = ['expression:', 'answer:', 'solution:', 'final answer:']
    for prefix in prefixes_to_remove:
        if expression.lower().startswith(prefix):
            expression = expression[len(prefix):].strip()
    
    # Check if the expression contains arithmetic operators
    if not re.search(r'[\+\-\*\/]', expression):
        logger.warning(f"No arithmetic operators found in expression: {expression}")
        return None
    
    return expression

def evaluate_countdown_problem(
    model_answer: str,
    available_numbers: List[int],
    target: int
) -> Dict[str, Any]:
    """
    Evaluate a model's answer to a Countdown problem.
    
    Args:
        model_answer: The model's proposed solution
        available_numbers: Numbers available for the puzzle
        target: Target number to reach
        
    Returns:
        Dictionary containing evaluation results
    """
    if not model_answer:
        return {
            'is_correct': False,
            'error_message': 'No answer provided',
            'expression': None,
            'target': target,
            'available_numbers': available_numbers
        }
    
    # Parse the answer to extract the expression
    expression = parse_countdown_answer(model_answer)
    if not expression:
        return {
            'is_correct': False,
            'error_message': 'Could not parse arithmetic expression from answer',
            'expression': model_answer,
            'target': target,
            'available_numbers': available_numbers
        }
    
    # Check the solution
    return check_countdown_solution(expression, available_numbers, target)

def check_one_countdown_answer(
    model_ans: Optional[str],
    available_numbers: List[int],
    target: int,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Check a single Countdown answer (similar interface to check_one_latex_answer).
    
    Args:
        model_ans: The model's answer string
        available_numbers: List of available numbers
        target: Target number to reach
        debug: Whether to include debug information
        
    Returns:
        Dictionary with evaluation results
    """
    if model_ans is None:
        result = {
            'is_correct': False,
            'error': 'No answer provided',
            'target': target,
            'available_numbers': available_numbers
        }
        if debug:
            result['debug_info'] = 'Model answer is None'
        return result
    
    evaluation = evaluate_countdown_problem(model_ans, available_numbers, target)
    
    # Convert to format consistent with other answer checkers
    result = {
        'is_correct': evaluation['is_correct'],
        'target': target,
        'available_numbers': available_numbers,
        'model_answer': model_ans,
        'parsed_expression': evaluation.get('expression'),
        'computed_result': evaluation.get('computed_result'),
        'numbers_used': evaluation.get('numbers_used', []),
        'valid_number_usage': evaluation.get('valid_number_usage', False),
        'valid_computation': evaluation.get('valid_computation', False)
    }
    
    if not evaluation['is_correct']:
        result['error'] = evaluation.get('error_message', 'Solution is incorrect')
    
    if debug:
        result['debug_info'] = {
            'original_answer': model_ans,
            'full_evaluation': evaluation
        }
    
    return result