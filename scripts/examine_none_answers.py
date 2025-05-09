#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List
import re

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

def examine_none_answers(results_file: str) -> None:
    """Examine cases where the answer is None in the continuation results."""
    results = load_results(results_file)
    
    print(f"Examining None answers in {len(results)} problems")
    print("-" * 80)
    
    none_examples = 0
    
    for problem in results:
        problem_id = problem.get("problem_id", "unknown")
        iterations = problem.get("iterations", [])
        
        for i, iteration in enumerate(iterations):
            answer = iteration.get("answer")
            reasoning_output = iteration.get("reasoning_output", "")
            
            if answer is None and reasoning_output:
                none_examples += 1
                print(f"Problem: {problem_id}, Iteration: {i}")
                print(f"Answer: None")
                
                # Show the end of the reasoning (last 500 chars)
                truncated = False
                reasoning_snippet = reasoning_output
                if len(reasoning_output) > 500:
                    reasoning_snippet = "..." + reasoning_output[-500:]
                    truncated = True
                
                print(f"Reasoning ending{' (truncated)' if truncated else ''}:")
                print("-" * 40)
                print(reasoning_snippet)
                print("-" * 40)
                
                # Check if it appears to be cut off
                cut_off = not reasoning_output.endswith("Wait") and not "</think>" in reasoning_output[-20:]
                print(f"Appears to be cut off: {cut_off}")
                
                # Check if there are think tags
                has_think_start = "<think>" in reasoning_output
                has_think_end = "</think>" in reasoning_output
                print(f"Has <think> tag: {has_think_start}")
                print(f"Has </think> tag: {has_think_end}")
                
                # Look for unfinished sentences
                last_sentence_ends_with_punctuation = re.search(r'[.!?]\s*$', reasoning_output.strip())
                print(f"Last sentence ends with punctuation: {bool(last_sentence_ends_with_punctuation)}")
                
                print("-" * 80)
                
                # Limit to 5 examples to keep output manageable
                if none_examples >= 5:
                    print(f"Only showing first 5 examples. There may be more.")
                    break
        
        if none_examples >= 5:
            break
    
    if none_examples == 0:
        print("No examples found where answer is None.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_none_answers.py <results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    examine_none_answers(results_file) 