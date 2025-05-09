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

def inspect_reasoning(cont_file: str, summ_file: str) -> None:
    """Extract and compare reasoning content from both experiment results."""
    cont_results = load_results(cont_file)
    summ_results = load_results(summ_file)
    
    # Get the first problem from each
    cont_problem = cont_results[0] if cont_results else None
    summ_problem = summ_results[0] if summ_results else None
    
    print("CONTINUATION EXPERIMENT")
    print("=" * 50)
    
    if cont_problem and "iterations" in cont_problem and cont_problem["iterations"]:
        iter0 = cont_problem["iterations"][0]
        print(f"Problem ID: {cont_problem.get('problem_id')}")
        print(f"Question: {cont_problem.get('question')[:200]}...")
        print(f"Correct answer: {cont_problem.get('correct_answer')}")
        
        # Basic stats
        print(f"\nIteration 0:")
        print(f"Extracted answer: {iter0.get('answer')}")
        print(f"Original correct: {iter0.get('correct')}")
        
        if "regraded_answer" in iter0:
            print(f"Regraded answer: {iter0.get('regraded_answer')}")
            print(f"Regraded correct: {iter0.get('regraded_correct')}")
        
        # Print the first few lines of the prompt
        print("\nPrompt (first 5 lines):")
        prompt_lines = iter0.get("prompt", "").split("\n")[:5]
        for line in prompt_lines:
            print(f"  {line}")
        
        # Print the first few lines of the model output
        print("\nModel Output (first 10 lines):")
        output_lines = iter0.get("reasoning_output", "").split("\n")[:10]
        for line in output_lines:
            print(f"  {line}")
    else:
        print("No continuation data available")
    
    print("\n\nSUMMARIZATION EXPERIMENT")
    print("=" * 50)
    
    if summ_problem and "iterations" in summ_problem and summ_problem["iterations"]:
        iter0 = summ_problem["iterations"][0]
        print(f"Problem ID: {summ_problem.get('problem_id')}")
        print(f"Question: {summ_problem.get('question')[:200]}...")
        print(f"Correct answer: {summ_problem.get('correct_answer')}")
        
        # Basic stats
        print(f"\nIteration 0:")
        print(f"Extracted answer: {iter0.get('answer')}")
        print(f"Original correct: {iter0.get('correct')}")
        
        # Print the first few lines of the reasoning
        print("\nReasoning (first 10 lines):")
        reasoning_lines = iter0.get("reasoning", "").split("\n")[:10]
        for line in reasoning_lines:
            print(f"  {line}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_reasoning.py <continuation_results.json> <summarization_results.json>")
        sys.exit(1)
    
    cont_file = sys.argv[1]
    summ_file = sys.argv[2]
    
    if not os.path.exists(cont_file):
        print(f"Error: Continuation results file {cont_file} not found")
        sys.exit(1)
    
    if not os.path.exists(summ_file):
        print(f"Error: Summarization results file {summ_file} not found")
        sys.exit(1)
    
    inspect_reasoning(cont_file, summ_file) 