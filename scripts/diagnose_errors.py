#!/usr/bin/env python3
"""
Diagnose errors in experiment results by examining the results.json file.
This script provides more detailed information about errors that occurred during experiments.
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path


def check_data_file(data_path, problem_ids):
    """
    Check the data file for issues with the specified problem IDs.
    
    Args:
        data_path (str): Path to the data file
        problem_ids (list): List of problem IDs to check
    """
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    print(f"\nChecking data file: {data_path}")
    
    # Try to read the data file with different CSV parsers
    try:
        # First try standard CSV reader
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = {row[0]: row for row in reader if row}
        
        # Check if problem IDs exist in the data file
        missing_ids = [pid for pid in problem_ids if pid not in data]
        if missing_ids:
            print(f"Problem IDs not found in data file: {', '.join(missing_ids)}")
        
        # Check for problems with missing question text
        for pid in problem_ids:
            if pid in data:
                row = data[pid]
                if len(row) < 2 or not row[1]:
                    print(f"Problem {pid} has no question text in the data file")
                elif len(row) < 3 or not row[2]:
                    print(f"Problem {pid} has no solution text in the data file")
                elif len(row) < 4 or not row[3]:
                    print(f"Problem {pid} has no answer in the data file")
    
    except Exception as e:
        print(f"Error parsing data file with standard CSV reader: {e}")
        print("This suggests there may be CSV formatting issues in the data file.")
        print("Try opening the file in a text editor to check for malformed CSV entries.")


def diagnose_errors(results_file, check_data=True):
    """
    Analyze the results.json file to diagnose errors.
    
    Args:
        results_file (str): Path to the results.json file
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}", file=sys.stderr)
        return
    
    # Check if the file has the expected structure
    results = []
    if 'results' in data and isinstance(data['results'], list):
        results = data['results']
    else:
        # Alternative structure - direct list of problems
        results = data if isinstance(data, list) else []
    
    if not results:
        print("No results found in the file.", file=sys.stderr)
        return
    
    print(f"Found {len(results)} total problems in results file.")
    
    # Count problems by status
    status_counts = {}
    for problem in results:
        status = problem.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nProblem status counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Analyze error problems
    error_problems = [p for p in results if p.get('status') == 'error' or p.get('error') is not None]
    
    if not error_problems:
        print("\nNo error problems found in the results file.")
        return
    
    print(f"\nFound {len(error_problems)} problems with errors:")
    
    # Group errors by error message
    error_groups = {}
    for problem in error_problems:
        error_msg = problem.get('error', 'Unknown error')
        if error_msg not in error_groups:
            error_groups[error_msg] = []
        error_groups[error_msg].append(problem.get('problem_id'))
    
    # Print error groups
    for error_msg, problem_ids in error_groups.items():
        print(f"\nError: {error_msg}")
        print(f"Affected problems ({len(problem_ids)}): {', '.join(problem_ids)}")
    
    # Check for problems with missing question text
    missing_question = [p for p in results if not p.get('question')]
    if missing_question:
        missing_ids = [p.get('problem_id') for p in missing_question]
        print(f"\nProblems with missing question text ({len(missing_ids)}): {', '.join(missing_ids)}")
    
    # Check for problems that might have timed out
    timeout_keywords = ['timeout', 'timed out', 'time limit']
    timeout_problems = []
    for problem in error_problems:
        error_msg = problem.get('error', '').lower()
        if any(keyword in error_msg for keyword in timeout_keywords):
            timeout_problems.append(problem.get('problem_id'))
    
    if timeout_problems:
        print(f"\nPossible timeout problems ({len(timeout_problems)}): {', '.join(timeout_problems)}")
    
    # Provide command to rerun just the error problems
    error_ids = [p.get('problem_id') for p in error_problems]
    error_ids_str = ','.join(error_ids)
    experiment_dir = Path(results_file).parent.parent.name
    config_path = f"config/experiments/{experiment_dir}.yaml"
    
    print("\nTo rerun these error problems, use:")
    print(f"python run_experiment.py {config_path} --question_ids={error_ids_str}")
    
    # Check the data file if requested
    if check_data:
        # Try to find the data file from the config
        config_file = f"config/experiments/{experiment_dir}.yaml"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        if line.startswith('data_path:'):
                            data_path = line.split(':', 1)[1].strip().strip('"').strip("'").replace('./', '')
                            check_data_file(data_path, error_ids)
                            break
            except Exception as e:
                print(f"Error reading config file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose errors in experiment results")
    parser.add_argument("results_file", help="Path to the results.json file")
    parser.add_argument("--no-check-data", action="store_true", help="Skip checking the data file")
    parser.add_argument("--data-file", help="Specify a data file to check")
    
    args = parser.parse_args()
    
    # Validate the results file exists
    if not os.path.isfile(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Diagnose errors
    diagnose_errors(args.results_file, check_data=not args.no_check_data)
    
    # Check specific data file if provided
    if args.data_file and os.path.exists(args.data_file):
        with open(args.results_file, 'r') as f:
            data = json.load(f)
        
        results = []
        if 'results' in data and isinstance(data['results'], list):
            results = data['results']
        else:
            results = data if isinstance(data, list) else []
        
        error_problems = [p for p in results if p.get('status') == 'error' or p.get('error') is not None]
        error_ids = [p.get('problem_id') for p in error_problems]
        
        check_data_file(args.data_file, error_ids)


if __name__ == "__main__":
    main()
