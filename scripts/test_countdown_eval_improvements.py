#!/usr/bin/env python3
"""
Test the improved countdown evaluation with better operator handling and robust answer extraction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.eval.countdown_check import extract_solution, evaluate_equation, validate_equation, compute_score

def test_operator_normalization():
    """Test that alternative operators are handled correctly."""
    print("=== Testing Operator Normalization ===")
    
    test_cases = [
        ("(10 ÷ 2) + 3", 8),              # Division with ÷
        ("5 x 4", 20),                     # Multiplication with x
        ("3 × 7", 21),                     # Multiplication with ×
        ("(15 ÷ 3) x 2", 10),            # Mixed operators
        ("100 ÷ 5 × 2", 40),              # Multiple alternative operators
        ("(8 + 2) × (15 ÷ 3)", 50),      # Complex expression
    ]
    
    for equation, expected in test_cases:
        result = evaluate_equation(equation)
        status = "✓" if result == expected else "✗"
        print(f"{status} {equation} = {result} (expected {expected})")
    
    print()

def test_answer_extraction():
    """Test robust answer extraction with malformed tags."""
    print("=== Testing Answer Extraction ===")
    
    test_cases = [
        # Standard case
        ("The answer is <answer>(5 + 3)</answer>", "(5 + 3)"),
        
        # Double closing tags
        ("The answer is <answer>(10 - 2)</answer></answer>", "(10 - 2)"),
        
        # Only closing tag
        ("The equation is (7 × 4) </answer>", "(7 × 4)"),
        
        # Only opening tag
        ("<answer> (12 ÷ 3) is the answer", "(12 ÷ 3)"),
        
        # Multiple answers (should take last)
        ("<answer>wrong</answer> Actually <answer>(5 + 5)</answer>", "(5 + 5)"),
        
        # With conversation prefix
        ("Assistant: <answer>(8 - 3)</answer>", "(8 - 3)"),
        
        # Complex equation with alternative operators
        ("<answer>(15 ÷ 3) × 2</answer>", "(15 ÷ 3) × 2"),
        
        # No tags at all
        ("Just some text without tags", "N/A"),
    ]
    
    for test_input, expected in test_cases:
        result = extract_solution(test_input)
        status = "✓" if result == expected else "✗"
        print(f"{status} Input: '{test_input[:50]}...' → '{result}' (expected '{expected}')")
    
    print()

def test_number_validation():
    """Test number validation with alternative operators."""
    print("=== Testing Number Validation ===")
    
    test_cases = [
        ("(5 × 3) + 2", [5, 3, 2], True),
        ("10 ÷ 2 + 3", [10, 2, 3], True),
        ("(7 × 4) - 1", [7, 4, 1], True),
        ("5 + 3", [5, 3, 2], False),  # Missing number 2
        ("5 + 3 + 2 + 2", [5, 3, 2], False),  # Using 2 twice
    ]
    
    for equation, numbers, expected in test_cases:
        result = validate_equation(equation, numbers)
        status = "✓" if result == expected else "✗"
        print(f"{status} Equation: '{equation}' with numbers {numbers} → {result} (expected {expected})")
    
    print()

def test_full_scoring():
    """Test complete scoring with various edge cases."""
    print("=== Testing Full Scoring ===")
    
    test_cases = [
        # Standard correct answer
        ("<answer>(5 + 3)</answer>", {"target": 8, "nums": [5, 3]}, 1.0),
        
        # Alternative operators
        ("<answer>10 ÷ 2</answer>", {"target": 5, "nums": [10, 2]}, 1.0),
        ("<answer>4 × 3</answer>", {"target": 12, "nums": [4, 3]}, 1.0),
        
        # Malformed tags but correct
        ("<answer>(7 + 2)</answer></answer>", {"target": 9, "nums": [7, 2]}, 1.0),
        
        # Wrong answer
        ("<answer>(5 + 3)</answer>", {"target": 10, "nums": [5, 3]}, 0.0),
        
        # Missing numbers
        ("<answer>5</answer>", {"target": 5, "nums": [5, 3]}, 0.0),
        
        # No valid answer
        ("No answer provided", {"target": 8, "nums": [5, 3]}, 0.0),
    ]
    
    for solution, ground_truth, expected_score in test_cases:
        score = compute_score(solution, ground_truth)
        status = "✓" if score == expected_score else "✗"
        target = ground_truth["target"]
        nums = ground_truth["nums"]
        print(f"{status} Target: {target}, Numbers: {nums}, Score: {score} (expected {expected_score})")
        print(f"   Solution: '{solution}'")
    
    print()

def main():
    """Run all tests."""
    print("Testing Countdown Evaluation Improvements")
    print("=" * 50)
    
    test_operator_normalization()
    test_answer_extraction()
    test_number_validation()
    test_full_scoring()
    
    print("=" * 50)
    print("All tests complete!")

if __name__ == "__main__":
    main()