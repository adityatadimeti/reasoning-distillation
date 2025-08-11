#!/usr/bin/env python3
"""
Test that existing countdown data still works with improved evaluation.
"""

import sys
import csv
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.eval.countdown_check import compute_score

def test_existing_data():
    """Test improved evaluation with existing countdown data."""
    print("=== Testing Existing Countdown Data ===")
    
    # Read first 10 problems from existing data
    data_file = Path("data/countdown.csv")
    if not data_file.exists():
        print(f"Data file {data_file} not found. Please generate it first.")
        return
    
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        problems = list(reader)[:10]  # First 10 problems
    
    passed = 0
    total = len(problems)
    
    for problem in problems:
        problem_id = problem['id']
        solution = problem['solution']
        target = int(problem['answer'])
        
        # Extract numbers from question
        import re
        match = re.search(r"Using the numbers ([\d, ]+), create", problem['question'])
        if match:
            nums_str = match.group(1)
            numbers = [int(n.strip()) for n in nums_str.split(',')]
        else:
            print(f"✗ Could not extract numbers from {problem_id}")
            continue
        
        # Test with answer tags
        solution_str = f"<answer>{solution}</answer>"
        ground_truth = {"target": target, "nums": numbers}
        score = compute_score(solution_str, ground_truth)
        
        status = "✓" if score == 1.0 else "✗"
        print(f"{status} {problem_id}: {solution} = {target} (score: {score})")
        
        if score == 1.0:
            passed += 1
    
    print(f"\nResults: {passed}/{total} problems passed")
    if passed == total:
        print("✓ All existing problems work with improved evaluation!")
    else:
        print("✗ Some existing problems failed - check for issues")

if __name__ == "__main__":
    test_existing_data()