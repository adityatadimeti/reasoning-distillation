#!/usr/bin/env python3
"""Test the countdown evaluation fixes"""

import pytest
from src.eval.countdown_check import extract_solution, validate_equation, evaluate_equation, compute_score

class TestCountdownEvalFixes:
    """Test suite for countdown evaluation fixes"""
    
    def test_equals_sign_handling(self):
        """Test that equations with = signs are cleaned properly"""
        # Test case 1: Standard equation with = target
        solution = "<answer>(130 - 43) = 87</answer>"
        extracted = extract_solution(solution)
        assert extracted == "(130 - 43)"
        
        # Test case 2: Complex equation with = target
        solution = "<answer>(130 - (63 - 20)) - (188 - 102 - 108 + 83 + 7 - 89) = 87</answer>"
        extracted = extract_solution(solution)
        assert extracted == "(130 - (63 - 20)) - (188 - 102 - 108 + 83 + 7 - 89)"
        
        # Test case 3: Just = target
        solution = "<answer>= 87</answer>"
        extracted = extract_solution(solution)
        assert extracted == ""
    
    def test_number_usage_validation(self):
        """Test validation ensures all numbers used exactly once"""
        # Test case 1: Correct usage
        equation = "(2-1)*3"
        nums = [1, 2, 3]
        assert validate_equation(equation, nums) == True
        
        # Test case 2: Missing number
        equation = "1+2"
        nums = [1, 2, 3]
        assert validate_equation(equation, nums) == False
        
        # Test case 3: Using number twice
        equation = "(3-2)+1+1"
        nums = [1, 2, 3]
        assert validate_equation(equation, nums) == False
        
        # Test case 4: Using extra number not in list
        equation = "1+2+3+4"
        nums = [1, 2, 3]
        assert validate_equation(equation, nums) == False
        
        # Test case 5: Just target number
        equation = "87"
        nums = [188, 130, 102, 83, 140, 108, 89, 63, 20, 7]
        assert validate_equation(equation, nums) == False
    
    def test_missing_answer_handling(self):
        """Test that missing answers are marked wrong"""
        # Test case 1: No answer tags
        solution = "I tried to solve but ran out of time"
        ground_truth = {"target": 87, "nums": [130, 43, 5, 10]}
        score = compute_score(solution, ground_truth)
        assert score == 0
        
        # Test case 2: Empty answer tags
        solution = "<answer></answer>"
        extracted = extract_solution(solution)
        assert extracted == ""
        
        # Test case 3: Cut off reasoning
        solution = "Let me work through this step by step..."
        ground_truth = {"target": 87, "nums": [130, 43, 5, 10]}
        score = compute_score(solution, ground_truth)
        assert score == 0
    
    def test_edge_cases_removed(self):
        """Test that problematic edge cases no longer extract partial equations"""
        # Test case 1: Only closing tag (should return N/A, not extract "87")
        solution = "The equation is (130 - 43) = 87</answer>"
        extracted = extract_solution(solution)
        assert extracted == "N/A"
        
        # Test case 2: Only opening tag (should return N/A)
        solution = "<answer>87 is the answer"
        extracted = extract_solution(solution)
        assert extracted == "N/A"
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation with various cases"""
        # Test case 1: Correct equation
        solution = "<answer>(2-1)*3</answer>"
        ground_truth = {"target": 3, "nums": [1, 2, 3]}
        score = compute_score(solution, ground_truth)
        assert score == 1.0
        
        # Test case 2: Equation with = (should strip and evaluate correctly)
        solution = "<answer>(130 - 43) = 87</answer>"
        ground_truth = {"target": 87, "nums": [130, 43]}
        score = compute_score(solution, ground_truth)
        assert score == 1.0
        
        # Test case 3: Not using all numbers
        solution = "<answer>130 - 43</answer>"
        ground_truth = {"target": 87, "nums": [130, 43, 5, 10]}
        score = compute_score(solution, ground_truth)
        assert score == 0
        
        # Test case 4: Using numbers multiple times
        solution = "<answer>(130 - 43) + 43 - 43</answer>"
        ground_truth = {"target": 87, "nums": [130, 43]}
        score = compute_score(solution, ground_truth)
        assert score == 0
        
        # Test case 5: Wrong answer
        solution = "<answer>(130 - 40)</answer>"
        ground_truth = {"target": 87, "nums": [130, 40]}
        score = compute_score(solution, ground_truth)
        assert score == 0
        
        # Test case 6: Just target number (even if in available numbers)
        solution = "<answer>87</answer>"
        ground_truth = {"target": 87, "nums": [87, 130, 43]}
        score = compute_score(solution, ground_truth)
        assert score == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])