"""
A simple script to verify that the filtering functionality in run_experiment.py works correctly.
This will load the problems and apply the filters, but not run the actual experiment.
"""

import pandas as pd
from run_experiment import load_problems, filter_problems

def test_filtering():
    # Load all problems
    data_path = "data/gpqa_diamond.csv"
    all_problems = load_problems(data_path)
    
    print(f"Total problems: {len(all_problems)}")
    
    # Test 1: Filter by index range
    index_range = "0-4"
    filtered = filter_problems(all_problems, index_range=index_range)
    print(f"\nIndex range {index_range}: {len(filtered)} problems")
    print("IDs:", [p["id"] for p in filtered])
    
    # Test 2: Filter by question IDs
    # Get first 2 IDs and one from the middle
    ids = [all_problems[0]["id"], all_problems[1]["id"], all_problems[10]["id"]]
    filtered = filter_problems(all_problems, question_ids=ids)
    print(f"\nSpecific IDs {ids}: {len(filtered)} problems")
    print("Filtered IDs:", [p["id"] for p in filtered])
    
    # Test 3: Invalid index range
    try:
        filter_problems(all_problems, index_range="invalid")
        print("\nTest 3 FAILED: Expected ValueError for invalid index range")
    except ValueError as e:
        print(f"\nTest 3 PASSED: Got expected error: {e}")
    
    # Test 4: Out of bounds index range
    try:
        filter_problems(all_problems, index_range=f"0-{len(all_problems)+10}")
        print("\nTest 4 FAILED: Expected ValueError for out of bounds index range")
    except ValueError as e:
        print(f"\nTest 4 PASSED: Got expected error: {e}")
    
    # Test 5: Both filters specified
    try:
        filter_problems(all_problems, question_ids=ids, index_range="0-4")
        print("\nTest 5 FAILED: Expected ValueError for specifying both filters")
    except ValueError as e:
        print(f"\nTest 5 PASSED: Got expected error: {e}")

if __name__ == "__main__":
    test_filtering() 