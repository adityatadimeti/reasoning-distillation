#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List, Optional

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_transitions(results_file: str):
    """Analyze how problems transition between correct and incorrect across iterations."""
    # Load file
    data = load_results(results_file)
    
    # Handle both list and dict formats
    if isinstance(data, dict) and "results" in data:
        results_list = data["results"]
    else:
        results_list = data
    
    total_problems = len(results_list)
    
    # Track transitions
    improved = []      # Wrong at iter 0, right at iter 4
    regressed = []     # Right at iter 0, wrong at iter 4
    always_correct = []    # Right at both iter 0 and 4
    always_wrong = []      # Wrong at both iter 0 and 4
    
    for problem in results_list:
        problem_id = problem.get("problem_id", "unknown")
        correct_answer = problem.get("correct_answer", "unknown")
        iterations = problem.get("iterations", [])
        
        # Skip if not enough iterations
        if len(iterations) < 2:
            continue
            
        # Get first and last iteration
        first_iteration = iterations[0]
        last_iteration = iterations[-1]
        
        first_answer = first_iteration.get("answer")
        last_answer = last_iteration.get("answer")
        
        # Check if first answer is correct
        first_correct = False
        if first_answer is not None and correct_answer is not None:
            first_correct = str(first_answer).strip() == str(correct_answer).strip()
            
        # Check if last answer is correct
        last_correct = False
        if last_answer is not None and correct_answer is not None:
            last_correct = str(last_answer).strip() == str(correct_answer).strip()
        
        # Categorize the problem
        if first_correct and last_correct:
            always_correct.append(problem_id)
        elif not first_correct and not last_correct:
            always_wrong.append(problem_id)
        elif not first_correct and last_correct:
            improved.append(problem_id)
        elif first_correct and not last_correct:
            regressed.append(problem_id)
    
    # Print results
    print(f"Analysis of transitions between first and last iterations")
    print(f"Total problems: {total_problems}")
    print("-" * 70)
    
    print(f"Improved (wrong → right): {len(improved)} ({len(improved)/total_problems*100:.1f}%)")
    if improved:
        print(f"  Problem IDs: {', '.join(map(str, improved))}")
    
    print(f"Regressed (right → wrong): {len(regressed)} ({len(regressed)/total_problems*100:.1f}%)")
    if regressed:
        print(f"  Problem IDs: {', '.join(map(str, regressed))}")
    
    print(f"Always correct: {len(always_correct)} ({len(always_correct)/total_problems*100:.1f}%)")
    print(f"Always wrong: {len(always_wrong)} ({len(always_wrong)/total_problems*100:.1f}%)")
    
    # Calculate net gain
    net_gain = len(improved) - len(regressed)
    print(f"\nNet gain: {net_gain:+d} problems ({net_gain/total_problems*100:+.1f}%)")
    
    # Return the results for potential further analysis
    return {
        "improved": improved,
        "regressed": regressed,
        "always_correct": always_correct,
        "always_wrong": always_wrong,
        "total_problems": total_problems
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_transitions.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    analyze_transitions(results_file) 