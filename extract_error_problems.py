#!/usr/bin/env python3
"""
Extract problem IDs that resulted in errors from a results.json file.
Outputs a comma-separated list that can be used with the --question_ids parameter
in run_experiment.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def extract_error_problems(results_file):
    """
    Extract problem IDs that have errors from the results.json file.
    
    Args:
        results_file (str): Path to the results.json file
        
    Returns:
        list: List of problem IDs with errors
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}", file=sys.stderr)
        return []
    
    error_problems = []
    
    # Check if the file has the expected structure
    if 'results' in data and isinstance(data['results'], list):
        for problem in data['results']:
            # Check for explicit error status
            if problem.get('status') == 'error' or problem.get('error') is not None:
                error_problems.append(problem.get('problem_id'))
                continue
            
            # Check for missing question text
            if not problem.get('question'):
                error_problems.append(problem.get('problem_id'))
                continue
    else:
        # Alternative structure - direct list of problems
        for problem in data:
            if problem.get('status') == 'error' or problem.get('error') is not None:
                error_problems.append(problem.get('problem_id'))
                continue
            
            # Check for missing question text
            if not problem.get('question'):
                error_problems.append(problem.get('problem_id'))
                continue
    
    return error_problems


def main():
    parser = argparse.ArgumentParser(description="Extract problem IDs with errors from results.json")
    parser.add_argument("results_file", help="Path to the results.json file")
    parser.add_argument("--output", "-o", help="Output file (if not specified, prints to stdout)")
    
    args = parser.parse_args()
    
    # Validate the results file exists
    if not os.path.isfile(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Extract error problems
    error_problems = extract_error_problems(args.results_file)
    
    if not error_problems:
        print("No error problems found in the results file.", file=sys.stderr)
        sys.exit(0)
    
    # Format as comma-separated list
    output = ",".join(error_problems)
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Wrote {len(error_problems)} error problem IDs to {args.output}", file=sys.stderr)
    else:
        print(output)
        print(f"\nFound {len(error_problems)} error problem IDs", file=sys.stderr)
    
    # Also print the command that can be used with run_experiment.py
    experiment_dir = Path(args.results_file).parent.parent.name
    config_path = f"config/experiments/{experiment_dir}.yaml"
    
    print("\nTo rerun these problems, use:", file=sys.stderr)
    print(f"python run_experiment.py {config_path} --question_ids={output}", file=sys.stderr)


if __name__ == "__main__":
    main()
