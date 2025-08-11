#!/usr/bin/env python3
"""
Test script for the Countdown solver.

This script tests the countdown solver with various sample problems to ensure
it generates correct solutions and explanations.
"""

import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.countdown_solver import solve_countdown_puzzle, CountdownSolver
from src.eval.countdown_check import compute_score, extract_solution, evaluate_equation

def test_simple_cases():
    """Test simple countdown problems - all numbers must be used."""
    print("=== Testing Simple Cases (All Numbers Must Be Used) ===")
    
    test_cases = [
        ([1, 2, 3], 6),      # (1 + 2) + 3 = 6
        ([2, 4], 8),         # 2 * 4 = 8  
        ([10, 5], 2),        # 10 / 5 = 2
        ([7, 3], 4),         # 7 - 3 = 4
        ([1, 2, 3, 4], 10),  # ((1 + 2) + 3) + 4 = 10
        ([2, 3, 5], 11),     # (2 * 3) + 5 = 11
    ]
    
    for nums, target in test_cases:
        print(f"\nTesting: {nums} -> {target}")
        expression, explanation = solve_countdown_puzzle(nums, target)
        print(f"Expression: {expression}")
        print(f"Explanation: {explanation}")
        
        # Validate the solution
        if expression != "No solution found" and not "away from our target" in explanation:
            extracted = extract_solution(f"<answer>{expression}</answer>")
            result = evaluate_equation(extracted)
            print(f"Validation: {extracted} = {result} (target: {target})")
            
            # Check if all numbers are used exactly once
            import re
            numbers_in_expr = [int(x) for x in re.findall(r'\b\d+\b', extracted)]
            nums_sorted = sorted(nums)
            expr_nums_sorted = sorted(numbers_in_expr)
            
            if abs(result - target) < 1e-9 and nums_sorted == expr_nums_sorted:
                print("✓ PASS (correct result + all numbers used)")
            elif abs(result - target) < 1e-9:
                print("✗ FAIL (correct result but not all numbers used)")
                print(f"  Expected numbers: {nums_sorted}")
                print(f"  Numbers in expression: {expr_nums_sorted}")
            else:
                print("✗ FAIL (incorrect result)")
        else:
            print("No exact solution found")

def test_complex_cases():
    """Test more complex countdown problems."""
    print("\n=== Testing Complex Cases ===")
    
    test_cases = [
        ([25, 50, 3, 7, 2, 1], 765),  # Classic countdown problem
        ([1, 2, 3, 4, 5, 6], 100),    # Multiple ways to solve
        ([10, 9, 8], 72),             # 9 * 8 = 72
        ([5, 6, 2], 17),              # 5 * 6 - 2 * ? = need to check
    ]
    
    for nums, target in test_cases:
        print(f"\nTesting: {nums} -> {target}")
        expression, explanation = solve_countdown_puzzle(nums, target)
        print(f"Expression: {expression}")
        print(f"Explanation: {explanation}")
        
        # Validate the solution
        if expression != "No solution found":
            # Use the countdown evaluation function
            ground_truth = {'target': target, 'nums': nums}
            score = compute_score(f"<answer>{expression}</answer>", ground_truth)
            print(f"Score: {score}")
            if score > 0:
                print("✓ PASS")
            else:
                print("✗ FAIL")
        else:
            print("No solution generated")

def test_edge_cases():
    """Test edge cases."""
    print("\n=== Testing Edge Cases ===")
    
    test_cases = [
        ([1], 1),        # Single number that matches target
        ([5], 3),        # Single number that doesn't match target
        ([2, 3], 1),     # Only subtraction works: 3 - 2 = 1
        ([4, 2], 2),     # Division: 4 / 2 = 2
        ([1, 1], 2),     # Duplicate numbers: 1 + 1 = 2
    ]
    
    for nums, target in test_cases:
        print(f"\nTesting: {nums} -> {target}")
        expression, explanation = solve_countdown_puzzle(nums, target)
        print(f"Expression: {expression}")
        print(f"Explanation: {explanation}")

def test_impossible_cases():
    """Test cases that should have no solution."""
    print("\n=== Testing Impossible Cases ===")
    
    test_cases = [
        ([2, 4, 6], 5),   # All even numbers, odd target
        ([1], 10),        # Single number too small
        ([3, 5], 1),      # Can't make 1 with 3 and 5
    ]
    
    for nums, target in test_cases:
        print(f"\nTesting: {nums} -> {target}")
        expression, explanation = solve_countdown_puzzle(nums, target)
        print(f"Expression: {expression}")
        print(f"Explanation: {explanation}")
        
        if "closest" in explanation.lower():
            print("✓ PASS - Found closest solution")
        elif expression == "No solution found":
            print("✓ PASS - Correctly identified as unsolvable")
        else:
            print("? Check if this is actually solvable")

def main():
    """Run all tests."""
    print("Testing Countdown Solver")
    print("=" * 50)
    
    test_simple_cases()
    test_complex_cases()
    test_edge_cases()
    test_impossible_cases()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()