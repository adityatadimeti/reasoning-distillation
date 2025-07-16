"""
Evaluation module for Countdown number puzzles.

This module provides functions to evaluate whether a proposed solution
to a Countdown puzzle is correct using the provided scoring functions.
"""

import re
import logging
from typing import Optional, List, Union, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def extract_solution(solution_str):
    """Extract the last <answer>...</answer> equation from the solution string."""
    # Remove conversation prefixes
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Try standard <answer>...</answer> pattern first
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        # Handle edge cases with malformed tags
        
        # Case 1: Double closing tags </answer></answer>
        double_close_pattern = r"<answer>(.*?)</answer></answer>"
        matches = list(re.finditer(double_close_pattern, solution_str, re.DOTALL))
        if matches:
            final_answer = matches[-1].group(1).strip()
        
        # Case 2: Only closing tag </answer>
        elif "</answer>" in solution_str:
            # Find content before the last </answer>
            parts = solution_str.split("</answer>")
            if len(parts) > 1:
                # Get the last non-empty part before </answer>
                for i in range(len(parts)-2, -1, -1):
                    content = parts[i].strip()
                    if content:
                        # Extract the equation part (after any text)
                        equation_match = re.search(r"[(\d+\-*/÷x×\s]+[\d)]$", content)
                        if equation_match:
                            final_answer = equation_match.group(0).strip()
                            break
                else:
                    final_answer = "N/A"
            else:
                final_answer = "N/A"
        
        # Case 3: Only opening tag <answer>
        elif "<answer>" in solution_str:
            # Find content after the last <answer>
            parts = solution_str.split("<answer>")
            if len(parts) > 1:
                content = parts[-1].strip()
                # Extract equation until we hit non-equation text
                equation_match = re.match(r"^[(\d+\-*/÷x×\s]+[\d)]", content)
                if equation_match:
                    final_answer = equation_match.group(0).strip()
                else:
                    final_answer = "N/A"
            else:
                final_answer = "N/A"
        else:
            final_answer = "N/A"

    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Normalize operators first to ensure consistent number extraction
        normalized_equation = equation_str
        normalized_equation = normalized_equation.replace('÷', '/')
        normalized_equation = normalized_equation.replace('x', '*')
        normalized_equation = normalized_equation.replace('×', '*')
        
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", normalized_equation)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Normalize operators: convert ÷ to / and x/× to *
        normalized_equation = equation_str
        normalized_equation = normalized_equation.replace('÷', '/')
        normalized_equation = normalized_equation.replace('x', '*')
        normalized_equation = normalized_equation.replace('×', '*')
        
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, normalized_equation):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(normalized_equation, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: unused for now
        format_score: always zero in this implementation
        score: full score for a valid, correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["nums"]

    equation = extract_solution(solution_str=solution_str)
    do_print = False

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation == "N/A":
        if do_print:
            print("No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score  # Always 0.0

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score  # Always 0.0

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score  # Always 0.0
    except:
        if do_print:
            print("Error evaluating equation")
        return format_score  # Always 0.0


def check_one_countdown_answer(
    model_ans: Optional[str],
    available_numbers: List[int],
    target: int,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Check a single Countdown answer (compatible interface with other answer checkers).
    
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
    
    # Create ground truth dictionary for compute_score
    ground_truth = {
        'target': target,
        'nums': available_numbers
    }
    
    # Use the provided scoring function
    score = compute_score(model_ans, ground_truth)
    is_correct = score > 0
    
    # Extract additional information for debugging
    extracted_equation = extract_solution(model_ans)
    
    result = {
        'is_correct': is_correct,
        'score': score,
        'target': target,
        'available_numbers': available_numbers,
        'model_answer': model_ans,
        'extracted_equation': extracted_equation
    }
    
    if not is_correct:
        if extracted_equation == "N/A":
            result['error'] = 'No equation found in answer tags'
        elif not validate_equation(extracted_equation, available_numbers):
            result['error'] = 'Invalid equation - numbers not used correctly'
        else:
            # Try to evaluate to see what the result was
            eval_result = evaluate_equation(extracted_equation)
            if eval_result is None:
                result['error'] = 'Could not evaluate equation'
            else:
                result['error'] = f'Equation evaluates to {eval_result}, not {target}'
                result['evaluated_result'] = eval_result
    
    if debug:
        result['debug_info'] = {
            'original_answer': model_ans,
            'extracted_equation': extracted_equation,
            'ground_truth': ground_truth
        }
    
    return result