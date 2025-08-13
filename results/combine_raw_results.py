#!/usr/bin/env python3
"""
Script to combine multiple results.json files, preserving the exact format.

This script:
1. Takes multiple results.json files as input
2. Combines the results, filtering to only include problems with exactly 5 solutions/iterations
3. Handles overlapping problem_ids by prioritizing earlier files
4. Preserves the exact format of the original results without adding consensus/pass@k
"""

import argparse
import json
import hashlib
from typing import Dict, List, Any, Set

def load_results_file(file_path: str) -> Dict[str, Any]:
    """Load a results.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {"results": []}

def get_solution_hash(solution: Dict[str, Any]) -> str:
    """Generate a hash based on the entire solution reasoning to identify duplicates."""
    # Use the entire reasoning text to identify duplicates
    reasoning = solution.get("reasoning", "")
    return hashlib.md5(reasoning.encode('utf-8')).hexdigest()

def get_attempts_key(problem: Dict[str, Any]) -> str:
    """Determine whether a problem uses 'solutions' or 'iterations' for its attempts."""
    if "solutions" in problem and isinstance(problem["solutions"], list):
        return "solutions"
    elif "iterations" in problem and isinstance(problem["iterations"], list):
        return "iterations"
    else:
        # Default to solutions if neither is found
        return "solutions"

def get_attempts(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get the list of attempts (solutions or iterations) from a problem."""
    key = get_attempts_key(problem)
    return problem.get(key, [])

def combine_results(file_paths: List[str]) -> Dict[str, Any]:
    """
    Combine results from multiple files, preserving the exact original format.
    
    Args:
        file_paths: List of paths to results.json files, in order of priority
    
    Returns:
        Combined results dict
    """
    # Create a mapping of problem_id to problem data for easier merging
    all_problems = {}
    seen_solutions = set()  # Track solutions by hash to avoid duplicates
    
    # Process files in order (earlier files have priority for duplicate problem_ids)
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        results = load_results_file(file_path)
        
        for problem in results.get("results", []):
            problem_id = problem.get("problem_id", "unknown")
            
            # Determine the format (solutions or iterations)
            attempts_key = get_attempts_key(problem)
            attempts = get_attempts(problem)
            
            # Only include problems with exactly 5 attempts
            if len(attempts) != 5:
                print(f"Skipping problem {problem_id} from {file_path} - has {len(attempts)} {attempts_key}, not 5")
                continue
            
            # Skip if this problem_id was already processed (prioritize earlier files)
            if problem_id in all_problems:
                print(f"Skipping duplicate problem {problem_id} from {file_path} - already included from earlier file")
                continue
            
            # Create a deep copy of the problem
            problem_copy = {
                key: value for key, value in problem.items()
                if key != attempts_key  # We'll handle attempts separately
            }
            problem_copy[attempts_key] = []
            
            # Add all attempts from this problem, tracking hashes to avoid duplicates
            for attempt in attempts:
                solution_hash = get_solution_hash(attempt)
                
                if solution_hash not in seen_solutions:
                    seen_solutions.add(solution_hash)
                    problem_copy[attempts_key].append(attempt)
            
            # Only add the problem if we actually added some attempts
            if problem_copy[attempts_key]:
                all_problems[problem_id] = problem_copy
    
    # Create the combined results object, preserving the format
    experiment_name = "combined_results"
    
    combined_results = {
        "experiment_name": experiment_name,
        "results": list(all_problems.values())
    }
    
    return combined_results

def save_combined_results(combined_results: Dict[str, Any], output_file: str):
    """Save the combined results to a new file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nCombined results saved to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Combine multiple results.json files, preserving the exact format.")
    parser.add_argument("--files", "-f", nargs="+", help="Paths to results.json files (in order of priority)")
    parser.add_argument("--output", "-o", default="combined_raw_results.json", 
                        help="Output file path (default: combined_raw_results.json)")
    parser.add_argument("--use-default-files", "-d", action="store_true",
                        help="Use the default set of files specified in the script")
    
    args = parser.parse_args()
    
    # Define default file paths
    default_files = [
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6_20250508_211719/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6_20250505_004152/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6_20250505_171231/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6/harp_deepseek_qwen_14b_summ_base_sum_4iter_l6_20250508_183631/results.json"
    ]
    
    # Use either provided files or default files
    file_paths = args.files if args.files else default_files
    
    if args.use_default_files:
        file_paths = default_files
        
    if not file_paths:
        parser.error("No input files specified. Use --files to provide file paths or --use-default-files to use defaults.")
    
    print(f"Combining results from {len(file_paths)} files...")
    combined_results = combine_results(file_paths)
    
    # Print summary
    total_problems = len(combined_results.get("results", []))
    print(f"\nTotal problems in combined results: {total_problems}")
    
    # Save combined results
    save_combined_results(combined_results, args.output)

if __name__ == "__main__":
    main() 