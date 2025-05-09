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

def analyze_progressions(original_file: str, fixed_file: str):
    """Analyze answer progressions between original and fixed results."""
    # Load both files
    original_data = load_results(original_file)
    fixed_data = load_results(fixed_file)
    
    # Extract answer sequences
    original_sequences = extract_answer_sequences(original_data)
    fixed_sequences = extract_answer_sequences(fixed_data)
    
    # Print header
    print(f"{'Problem ID':<15} {'Correct':<10} {'Original Sequence':<40} {'Fixed Sequence':<40}")
    print("-" * 110)
    
    # Sort by problem ID to ensure consistent output
    for problem_id in sorted(original_sequences.keys()):
        original_seq = original_sequences[problem_id]
        fixed_seq = fixed_sequences[problem_id]
        
        # Format the answer sequences
        orig_answers_str = str([a if a is not None else "None" for a in original_seq["answers"]])
        if len(orig_answers_str) > 40:
            orig_answers_str = orig_answers_str[:37] + "..."
            
        fixed_answers_str = str([a if a is not None else "None" for a in fixed_seq["answers"]])
        if len(fixed_answers_str) > 40:
            fixed_answers_str = fixed_answers_str[:37] + "..."
        
        # Print the comparison
        print(f"{problem_id:<15} {original_seq['correct']:<10} {orig_answers_str:<40} {fixed_answers_str:<40}")
        
        # Check if this problem had any change
        has_diff = original_seq["answers"] != fixed_seq["answers"]
        if has_diff:
            print("  CHANGED - Detailed progression:")
            print(f"  Original: {original_seq['answers']}")
            print(f"  Fixed:    {fixed_seq['answers']}")
            print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_answer_progression.py <original_results.json> <fixed_results.json>")
        sys.exit(1)
    
    original_file = sys.argv[1]
    fixed_file = sys.argv[2]
    
    if not os.path.exists(original_file) or not os.path.exists(fixed_file):
        print(f"Error: Input file(s) not found")
        sys.exit(1)
    
    analyze_progressions(original_file, fixed_file) 