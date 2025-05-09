#!/usr/bin/env python
import json
import sys
import os
import argparse
import re

def load_results(file_path):
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_continuation_text(text):
    """
    Clean continuation text to make it start with a proper sentence:
    1. Remove leading commas, spaces, and other punctuation
    2. Capitalize the first letter
    """
    # Remove leading commas, spaces and other common punctuation
    cleaned = re.sub(r'^[,;\s]+', '', text)
    
    # Capitalize first letter if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    
    return cleaned

def extract_completions(results_file, output_file):
    """
    Extract just the reasoning_output for each iteration and format into the
    expected structure for annotation.
    """
    # Load results
    data = load_results(results_file)
    
    # Handle both list and dict formats
    if isinstance(data, dict) and "results" in data:
        results_list = data["results"]
    else:
        results_list = data
    
    # Get experiment name from filepath
    exp_name = os.path.basename(os.path.dirname(results_file))
    
    # Create the output structure
    output_data = {
        "experiment_name": exp_name,
        "problems": []
    }
    
    # Process each problem
    for problem in results_list:
        problem_id = problem.get("problem_id", "unknown")
        
        # Create output problem structure
        output_problem = {
            "problem_id": problem_id,
            "iterations": []
        }
        
        # Get iterations
        iterations = problem.get("iterations", [])
        for i, iteration in enumerate(iterations):
            # Get the reasoning_output
            reasoning_output = iteration.get("reasoning_output", "")
            if not reasoning_output:
                continue
            
            # For iterations > 0, clean the continuation text
            if i > 0:
                reasoning_output = clean_continuation_text(reasoning_output)
                
            # Create output iteration structure
            output_iteration = {
                "iteration": i,
                "reasoning": reasoning_output
            }
            
            output_problem["iterations"].append(output_iteration)
        
        # Only add problems that have iterations
        if output_problem["iterations"]:
            output_data["problems"].append(output_problem)
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Extracted completions for {len(output_data['problems'])} problems")
    print(f"Output written to {output_file}")
    
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract continuation completions for annotation')
    parser.add_argument('--results', required=True, help='Path to results.json file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found")
        sys.exit(1)
    
    extract_completions(args.results, args.output) 