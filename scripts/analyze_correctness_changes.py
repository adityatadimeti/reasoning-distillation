#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(json_data, dict) and "results" in json_data:
        results = json_data["results"]
    else:
        results = json_data
    
    return results

def analyze_correctness_changes(results_file: str) -> None:
    """Analyze problems where correctness changes across iterations."""
    results = load_results(results_file)
    
    print(f"Analyzing correctness changes in {len(results)} problems")
    print("-" * 80)
    
    # Statistics
    total_problems = len(results)
    problems_with_correctness_changes = 0
    problems_improved = 0
    problems_degraded = 0
    problems_with_both = 0
    
    # For each problem, find those with changing correctness
    for problem in results:
        problem_id = problem.get("problem_id", "unknown")
        correct_answer = problem.get("correct_answer", "")
        iterations = problem.get("iterations", [])
        
        if not iterations or len(iterations) < 2:  # Need at least 2 iterations to compare
            continue
            
        # Track answers and correctness for each iteration
        answers = []
        correctness = []
        
        for i, iteration in enumerate(iterations):
            answer = iteration.get("answer")
            answers.append(answer)
            
            # Check correctness
            is_correct = False
            if answer is not None and correct_answer is not None:
                is_correct = answer.strip() == correct_answer.strip()
            correctness.append(is_correct)
        
        # Check if correctness changes
        if len(set(correctness)) <= 1:  # All the same (all True or all False)
            continue
            
        # Found a problem with changing correctness
        problems_with_correctness_changes += 1
        
        print(f"Problem: {problem_id}")
        print(f"Correct answer: {correct_answer}")
        print(f"Answers across iterations: {answers}")
        print(f"Correctness across iterations: {correctness}")
        
        # Check improvement or degradation patterns
        improved = False
        degraded = False
        
        for i in range(1, len(correctness)):
            if not correctness[i-1] and correctness[i]:
                improved = True
            elif correctness[i-1] and not correctness[i]:
                degraded = True
        
        if improved and not degraded:
            problems_improved += 1
            print(f"  ✅ PATTERN: Overall improvement")
        elif degraded and not improved:
            problems_degraded += 1
            print(f"  ⚠️ PATTERN: Overall degradation")
        elif improved and degraded:
            problems_with_both += 1
            print(f"  🔄 PATTERN: Mixed (both improvement and degradation)")
            
        print("-" * 80)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total problems: {total_problems}")
    print(f"Problems with changing correctness: {problems_with_correctness_changes} ({problems_with_correctness_changes/total_problems*100:.1f}%)")
    print(f"Problems showing improvement: {problems_improved} ({problems_improved/total_problems*100:.1f}%)")
    print(f"Problems showing degradation: {problems_degraded} ({problems_degraded/total_problems*100:.1f}%)")
    print(f"Problems showing mixed patterns: {problems_with_both} ({problems_with_both/total_problems*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_correctness_changes.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    analyze_correctness_changes(results_file) 