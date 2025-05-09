#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List, Optional

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy(results_data):
    """Calculate accuracy at different iterations."""
    # Handle both list and dict formats
    if isinstance(results_data, dict) and "results" in results_data:
        results_list = results_data["results"]
    else:
        results_list = results_data
    
    # Statistics
    total_problems = len(results_list)
    correct_by_iteration = {}
    max_iteration = 0
    
    # Process each problem
    for problem in results_list:
        correct_answer = problem.get("correct_answer", "")
        iterations = problem.get("iterations", [])
        
        # Track the maximum iteration
        max_iteration = max(max_iteration, len(iterations) - 1)
        
        # Check correctness at each iteration
        for i, iteration in enumerate(iterations):
            answer = iteration.get("answer")
            
            # Initialize counter for this iteration if needed
            if i not in correct_by_iteration:
                correct_by_iteration[i] = 0
            
            # Check if answer is correct
            if answer is not None and correct_answer is not None:
                if str(answer).strip() == str(correct_answer).strip():
                    correct_by_iteration[i] += 1
    
    return {
        "total_problems": total_problems,
        "correct_by_iteration": correct_by_iteration,
        "max_iteration": max_iteration
    }

def compare_results(original_file: str, fixed_file: str):
    """Compare accuracy between original and fixed results."""
    # Load both files
    original_data = load_results(original_file)
    fixed_data = load_results(fixed_file)
    
    # Calculate accuracy for both
    original_stats = calculate_accuracy(original_data)
    fixed_stats = calculate_accuracy(fixed_data)
    
    total_problems = original_stats["total_problems"]
    max_iteration = max(original_stats["max_iteration"], fixed_stats["max_iteration"])
    
    # Print results
    print(f"Analysis of {total_problems} problems")
    print("-" * 70)
    print(f"{'Iteration':<10} {'Original Correct':<20} {'Fixed Correct':<20} {'Improvement':<10}")
    print("-" * 70)
    
    # Show results for all iterations
    for i in range(max_iteration + 1):
        orig_correct = original_stats["correct_by_iteration"].get(i, 0)
        fixed_correct = fixed_stats["correct_by_iteration"].get(i, 0)
        improvement = fixed_correct - orig_correct
        
        orig_pct = orig_correct / total_problems * 100
        fixed_pct = fixed_correct / total_problems * 100
        
        print(f"{i:<10} {orig_correct} ({orig_pct:.1f}%){'':8} {fixed_correct} ({fixed_pct:.1f}%){'':8} {improvement:+d}")
    
    # Highlight first and last iteration
    print("\nKey Iterations:")
    first_iter = 0
    last_iter = max_iteration
    
    orig_first = original_stats["correct_by_iteration"].get(first_iter, 0)
    fixed_first = fixed_stats["correct_by_iteration"].get(first_iter, 0)
    orig_last = original_stats["correct_by_iteration"].get(last_iter, 0)
    fixed_last = fixed_stats["correct_by_iteration"].get(last_iter, 0)
    
    first_improvement = fixed_first - orig_first
    last_improvement = fixed_last - orig_last
    
    print(f"First iteration (0): Original {orig_first}/{total_problems} ({orig_first/total_problems*100:.1f}%), Fixed {fixed_first}/{total_problems} ({fixed_first/total_problems*100:.1f}%), Improvement: {first_improvement:+d}")
    print(f"Last iteration ({last_iter}): Original {orig_last}/{total_problems} ({orig_last/total_problems*100:.1f}%), Fixed {fixed_last}/{total_problems} ({fixed_last/total_problems*100:.1f}%), Improvement: {last_improvement:+d}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_accuracy.py <original_results.json> <fixed_results.json>")
        sys.exit(1)
    
    original_file = sys.argv[1]
    fixed_file = sys.argv[2]
    
    if not os.path.exists(original_file) or not os.path.exists(fixed_file):
        print(f"Error: Input file(s) not found")
        sys.exit(1)
    
    compare_results(original_file, fixed_file) 