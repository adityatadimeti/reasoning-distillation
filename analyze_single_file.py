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

def analyze_file(results_file: str):
    """Analyze accuracy for a single results file."""
    # Load file
    results_data = load_results(results_file)
    
    # Calculate accuracy
    stats = calculate_accuracy(results_data)
    
    total_problems = stats["total_problems"]
    max_iteration = stats["max_iteration"]
    
    # Print results
    print(f"Analysis of {total_problems} problems")
    print("-" * 70)
    print(f"{'Iteration':<10} {'Correct':<20} {'Accuracy':<10}")
    print("-" * 70)
    
    # Show results for all iterations
    for i in range(max_iteration + 1):
        correct = stats["correct_by_iteration"].get(i, 0)
        pct = correct / total_problems * 100
        print(f"{i:<10} {correct}/{total_problems}{'':<10} {pct:.1f}%")
    
    # Highlight first and last iteration
    print("\nKey Iterations:")
    first_iter = 0
    last_iter = max_iteration
    
    first_correct = stats["correct_by_iteration"].get(first_iter, 0)
    last_correct = stats["correct_by_iteration"].get(last_iter, 0)
    
    change = last_correct - first_correct
    change_pct = (last_correct - first_correct) / total_problems * 100
    
    print(f"First iteration (0): {first_correct}/{total_problems} ({first_correct/total_problems*100:.1f}%)")
    print(f"Last iteration ({last_iter}): {last_correct}/{total_problems} ({last_correct/total_problems*100:.1f}%)")
    print(f"Change: {change:+d} ({change_pct:+.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_single_file.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    analyze_file(results_file) 