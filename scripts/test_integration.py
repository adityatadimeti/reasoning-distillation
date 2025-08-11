#!/usr/bin/env python3
"""
Integration test to verify the improved evaluation works with generated solutions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.countdown_solver import solve_countdown_puzzle
from src.eval.countdown_check import compute_score

def test_integration():
    """Test that generated solutions work with improved evaluation."""
    print("=== Integration Test: Generated Solutions + Improved Evaluation ===")
    
    test_cases = [
        ([5, 10, 3], 18),    # Should get (5 + 10) + 3 = 18
        ([12, 4, 2], 8),     # Should get 12 / 4 + 2 = 8  
        ([7, 3, 2], 17),     # Should get 7 * 3 - 2 = 17
    ]
    
    for numbers, target in test_cases:
        print(f"\nTesting: {numbers} -> {target}")
        
        # Generate solution
        expression, explanation = solve_countdown_puzzle(numbers, target)
        print(f"Generated expression: {expression}")
        
        # Create solution string with answer tags
        solution_str = f"<answer>{expression}</answer>"
        
        # Test with ground truth
        ground_truth = {"target": target, "nums": numbers}
        score = compute_score(solution_str, ground_truth)
        
        print(f"Score: {score}")
        status = "✓" if score == 1.0 else "✗"
        print(f"{status} Integration test passed" if score == 1.0 else f"{status} Integration test failed")
        
        # Test with alternative operators manually
        alt_expression = expression.replace('*', '×').replace('/', '÷')
        if alt_expression != expression:
            alt_solution_str = f"<answer>{alt_expression}</answer>"
            alt_score = compute_score(alt_solution_str, ground_truth)
            print(f"Alternative operators '{alt_expression}': Score {alt_score}")
    
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    test_integration()