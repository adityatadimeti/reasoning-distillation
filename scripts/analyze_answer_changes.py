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

def analyze_answer_changes(results_file: str) -> None:
    """Analyze problems where answers change across iterations."""
    results = load_results(results_file)
    
    print(f"Analyzing answer changes in {len(results)} problems")
    print("-" * 80)
    
    # Statistics
    total_problems = len(results)
    problems_with_changing_answers = 0
    problems_improved = 0  # Wrong → Right
    problems_degraded = 0  # Right → Wrong
    
    # For each problem, find those with changing answers
    for problem in results:
        problem_id = problem.get("problem_id", "unknown")
        correct_answer = problem.get("correct_answer", "")
        iterations = problem.get("iterations", [])
        
        if not iterations or len(iterations) == 0:
            continue
            
        # Track answers for each iteration
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
        
        # Check if answers change (ignoring None values)
        unique_answers = set([a for a in answers if a is not None])
        answers_change = len(unique_answers) > 1
        
        if not answers_change:
            continue
            
        # Only print problems where answers change
        problems_with_changing_answers += 1
        print(f"Problem: {problem_id}")
        print(f"Correct answer: {correct_answer}")
        print(f"Answers across iterations: {answers}")
        print(f"Correctness across iterations: {correctness}")
        print("-" * 80)
        
        # Check improvement/degradation
        if correctness and len(correctness) > 1:
            if not correctness[0] and any(correctness[1:]):
                problems_improved += 1
                print(f"  ✅ IMPROVED: Wrong → Right")
            elif correctness[0] and not all(correctness[1:]):
                problems_degraded += 1
                print(f"  ⚠️ DEGRADED: Right → Wrong")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total problems: {total_problems}")
    print(f"Problems with changing answers: {problems_with_changing_answers} ({problems_with_changing_answers/total_problems*100:.1f}%)")
    print(f"Problems that improved (wrong → right): {problems_improved} ({problems_improved/total_problems*100:.1f}%)")
    print(f"Problems that degraded (right → wrong): {problems_degraded} ({problems_degraded/total_problems*100:.1f}%)")
    
    if problems_with_changing_answers == 0:
        print("\nConclusion: No problems showed changing answers across iterations.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_answer_changes.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    analyze_answer_changes(results_file) 