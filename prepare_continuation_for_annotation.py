#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_for_annotation(results_file: str, output_file: str, start_iter: int = 1, end_iter: int = 4):
    """
    Prepare continuation experiment results for annotation by:
    1. Converting to the format expected by annotate_cog_behaviors.py
    2. Including only iterations from start_iter to end_iter
    3. Moving reasoning_output to reasoning field for compatibility
    """
    # Load results
    data = load_results(results_file)
    
    # Handle both list and dict formats
    if isinstance(data, dict) and "results" in data:
        results_list = data["results"]
    else:
        results_list = data
    
    # Create the output structure
    output_data = {
        "experiment_name": "continuation_aime_deepseek_qwen_14b_4iter",
        "problems": []
    }
    
    # Process each problem
    for problem in results_list:
        problem_id = problem.get("problem_id", "unknown")
        correct_answer = problem.get("correct_answer", "unknown")
        
        # Create output problem structure
        output_problem = {
            "problem_id": problem_id,
            "correct_answer": correct_answer,
            "problem_text": problem.get("problem_text", ""),
            "iterations": []
        }
        
        # Get only the iterations we want
        iterations = problem.get("iterations", [])
        for i, iteration in enumerate(iterations):
            # Skip iterations outside our range
            if i < start_iter or i > end_iter:
                continue
            
            # Get the reasoning_output and put it in the reasoning field
            reasoning_output = iteration.get("reasoning_output", "")
            if not reasoning_output:
                continue
                
            # Create output iteration structure
            output_iteration = {
                "iteration": i,
                "reasoning": reasoning_output,  # Map reasoning_output to reasoning for the annotation script
                "answer": iteration.get("answer")
            }
            
            output_problem["iterations"].append(output_iteration)
        
        # Only add problems that have the iterations we want
        if output_problem["iterations"]:
            output_data["problems"].append(output_problem)
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Prepared {len(output_data['problems'])} problems with iterations {start_iter}-{end_iter}")
    print(f"Output written to {output_file}")
    
    return output_data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_continuation_for_annotation.py <results.json> <output.json> [start_iter] [end_iter]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2]
    
    start_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    prepare_for_annotation(results_file, output_file, start_iter, end_iter) 