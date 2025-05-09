#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List, Optional

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer_sequences(results_data):
    """Extract answer sequences for each problem."""
    # Handle both list and dict formats
    if isinstance(results_data, dict) and "results" in results_data:
        results_list = results_data["results"]
    else:
        results_list = results_data
    
    answer_sequences = {}
    
    for problem in results_list:
        problem_id = problem.get("problem_id", "unknown")
        correct_answer = problem.get("correct_answer", "unknown")
        
        # Extract answers from iterations
        iterations = problem.get("iterations", [])
        answers = [iteration.get("answer") for iteration in iterations]
        
        answer_sequences[problem_id] = {
            "correct": correct_answer,
            "answers": answers
        }
    
    return answer_sequences

def summarize_fixes(original_file: str, fixed_file: str):
    """Generate a summary of fixes."""
    # Load both files
    original_data = load_results(original_file)
    fixed_data = load_results(fixed_file)
    
    # Extract answer sequences
    original_sequences = extract_answer_sequences(original_data)
    fixed_sequences = extract_answer_sequences(fixed_data)
    
    # Track fix statistics
    total_problems = len(original_sequences)
    problems_fixed = 0
    total_iterations = 0
    iterations_fixed = 0
    
    # Track how answers change
    consistency_improvement = 0  # Problems where answer became more consistent
    final_answer_changes = 0     # Problems where final answer changed
    
    # Track effect on the last iteration specifically
    final_iteration_none_to_value = 0
    
    for problem_id in original_sequences:
        original_seq = original_sequences[problem_id]
        fixed_seq = fixed_sequences[problem_id]
        
        # Check if anything changed
        if original_seq["answers"] != fixed_seq["answers"]:
            problems_fixed += 1
            
            # Count iterations that were fixed (None -> value)
            for i, (orig, fixed) in enumerate(zip(original_seq["answers"], fixed_seq["answers"])):
                total_iterations += 1
                if orig is None and fixed is not None:
                    iterations_fixed += 1
                    # Check if it's the final iteration
                    if i == len(original_seq["answers"]) - 1:
                        final_iteration_none_to_value += 1
            
            # Check if consistency improved
            orig_non_null = [a for a in original_seq["answers"] if a is not None]
            fixed_non_null = [a for a in fixed_seq["answers"] if a is not None]
            
            if len(fixed_non_null) > len(orig_non_null):
                consistency_improvement += 1
            
            # Check if final answer changed
            if original_seq["answers"][-1] != fixed_seq["answers"][-1]:
                final_answer_changes += 1
    
    # Print summary
    print("Summary of Fixes")
    print("===============")
    print(f"Total problems analyzed: {total_problems}")
    print(f"Problems with fixes: {problems_fixed} ({problems_fixed/total_problems*100:.1f}%)")
    print(f"Iterations fixed (None → value): {iterations_fixed} out of {total_iterations} iterations")
    print()
    print("Impact Analysis")
    print("==============")
    print(f"Problems with improved answer consistency: {consistency_improvement}")
    print(f"Problems with final answer changes: {final_answer_changes}")
    print(f"Problems where final iteration changed from None to a value: {final_iteration_none_to_value}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python summarize_fixes.py <original_results.json> <fixed_results.json>")
        sys.exit(1)
    
    original_file = sys.argv[1]
    fixed_file = sys.argv[2]
    
    if not os.path.exists(original_file) or not os.path.exists(fixed_file):
        print(f"Error: Input file(s) not found")
        sys.exit(1)
    
    summarize_fixes(original_file, fixed_file) 