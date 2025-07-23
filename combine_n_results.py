#!/usr/bin/env python3
"""
Script to combine and analyze multiple results.json files from PassK experiments.

This script:
1. Takes multiple results.json files as input
2. Combines the results, filtering to only include problems with exactly 5 solutions/iterations
3. Handles overlapping problem_ids by prioritizing earlier files
4. Tabulates statistics for each problem
5. Produces an overall summary of the combined results
6. Supports both "solutions" and "iterations" result formats for backward compatibility
"""

import argparse
import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

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

def combine_results(file_paths: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """
    Combine results from multiple files and calculate statistics for each problem.
    Supports both "solutions" and "iterations" formats.
    
    Args:
        file_paths: List of paths to results.json files, in order of priority
    
    Returns:
        Tuple containing the combined results dict and statistics dict
    """
    # Create a mapping of problem_id to problem data for easier merging
    all_problems = {}
    seen_solutions = {}  # Track solutions by hash to avoid duplicates
    
    # Statistics tracking
    stats = {}
    
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
            
            # Add this problem to our collection, preserving the original format
            all_problems[problem_id] = {
                "problem_id": problem_id,
                "question": problem.get("question", ""),
                "correct_answer": problem.get("correct_answer", ""),
                attempts_key: []  # Use the original key (solutions or iterations)
            }
            
            # Initialize statistics for this problem
            stats[problem_id] = {
                "correct": 0,
                "incorrect": 0,
                "error": 0,
                "length": 0,
                "total": 0,
                "format": attempts_key  # Track the format used for this problem
            }
            
            # Add all attempts from this problem, tracking hashes to avoid duplicates
            for attempt in attempts:
                solution_hash = get_solution_hash(attempt)
                
                if solution_hash not in seen_solutions:
                    seen_solutions[solution_hash] = True
                    all_problems[problem_id][attempts_key].append(attempt)
                    
                    # Update statistics
                    if attempt.get("correct", False):
                        stats[problem_id]["correct"] += 1
                    else:
                        stats[problem_id]["incorrect"] += 1
                        
                    finish_reason = attempt.get("finish_reason", "unknown")
                    if finish_reason == "error":
                        stats[problem_id]["error"] += 1
                    elif finish_reason == "length":
                        stats[problem_id]["length"] += 1
                        
                    stats[problem_id]["total"] += 1
    
    # Update consensus information for each problem based on the combined attempts
    for problem_id, problem_data in all_problems.items():
        attempts_key = stats[problem_id]["format"]
        attempts = problem_data[attempts_key]
        correct_answer = problem_data["correct_answer"]
        
        # Find consensus answer
        answers = [a.get("answer") for a in attempts if a.get("answer") is not None]
        consensus_data = find_consensus(answers)
        
        if consensus_data:
            consensus_answer, consensus_count = consensus_data
            consensus_correct = consensus_answer.strip() == correct_answer.strip()
        else:
            consensus_answer = None
            consensus_count = 0
            consensus_correct = False
        
        # Calculate pass@k
        num_correct = sum(1 for a in attempts if a.get("correct", False))
        pass_at_k = num_correct > 0
        
        # Update problem data with consensus information
        problem_data["consensus_answer"] = consensus_answer
        problem_data["consensus_correct"] = consensus_correct
        problem_data["consensus_count"] = consensus_count
        problem_data["pass_at_k"] = pass_at_k
        problem_data["num_correct"] = num_correct
    
    # Create the combined results object
    combined_results = {
        "experiment_name": "combined_results",
        "results": list(all_problems.values())
    }
    
    return combined_results, stats

def find_consensus(answers: List[str]) -> Optional[Tuple[str, int]]:
    """Find the most common answer and its count."""
    if not answers:
        return None
    
    # Count occurrences of each answer
    answer_counts = {}
    for answer in answers:
        if answer:
            answer = answer.strip()
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    # Find the most common answer
    if not answer_counts:
        return None
    
    most_common = max(answer_counts.items(), key=lambda x: x[1])
    return most_common

def print_statistics(stats: Dict[str, Dict[str, int]]):
    """Print statistics for all problems."""
    print("\n===== STATISTICS FOR COMBINED RESULTS =====")
    
    # Overall totals
    total_problems = len(stats)
    total_attempts = sum(problem_stats["total"] for problem_stats in stats.values())
    total_correct = sum(problem_stats["correct"] for problem_stats in stats.values())
    total_incorrect = sum(problem_stats["incorrect"] for problem_stats in stats.values())
    total_error = sum(problem_stats["error"] for problem_stats in stats.values())
    total_length = sum(problem_stats["length"] for problem_stats in stats.values())
    
    # Format counts
    solution_format_count = sum(1 for problem_stats in stats.values() if problem_stats.get("format") == "solutions")
    iteration_format_count = sum(1 for problem_stats in stats.values() if problem_stats.get("format") == "iterations")
    
    print(f"Total Problems: {total_problems}")
    print(f"  - With 'solutions' format: {solution_format_count}")
    print(f"  - With 'iterations' format: {iteration_format_count}")
    print(f"Total Attempts: {total_attempts}")
    print(f"Correct Attempts: {total_correct} ({total_correct/total_attempts:.2%})")
    print(f"Incorrect Attempts: {total_incorrect} ({total_incorrect/total_attempts:.2%})")
    print(f"Error Finishes: {total_error} ({total_error/total_attempts:.2%})")
    print(f"Length Finishes: {total_length} ({total_length/total_attempts:.2%})")
    
    # Problem-level statistics
    print("\nProblem-level Statistics:")
    print(f"{'Problem ID':<15} {'Format':<10} {'Total':<8} {'Correct':<10} {'Incorrect':<10} {'Error':<8} {'Length':<8}")
    print("-" * 75)
    
    for problem_id, problem_stats in sorted(stats.items()):
        total = problem_stats["total"]
        correct = problem_stats["correct"]
        incorrect = problem_stats["incorrect"]
        error = problem_stats["error"]
        length = problem_stats["length"]
        format_type = problem_stats.get("format", "unknown")
        
        print(f"{problem_id:<15} {format_type:<10} {total:<8} {correct:<10} {incorrect:<10} {error:<8} {length:<8}")

def save_combined_results(combined_results: Dict[str, Any], output_file: str):
    """Save the combined results to a new file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nCombined results saved to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Combine and analyze multiple results.json files.")
    parser.add_argument("--files", "-f", nargs="+", help="Paths to results.json files (in order of priority)")
    parser.add_argument("--output", "-o", default="combined_results.json", 
                        help="Output file path (default: combined_results.json)")
    parser.add_argument("--use-default-files", "-d", action="store_true",
                        help="Use the default set of files specified in the script")
    
    args = parser.parse_args()
    
    # Define default file paths
    default_files = [
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_pass_l6/harp_deepseek_qwen_14b_pass_l6_20250507_065058/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_pass_l6/harp_deepseek_qwen_14b_pass_l6_20250507_003243/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_pass_l6/harp_deepseek_qwen_14b_pass_l6_20250506_175633/results.json",
        "/sailhome/jshen3/research_projects/reasoning-distillation/results/harp_deepseek_qwen_14b_pass_l6/harp_deepseek_qwen_14b_pass_l6_20250506_004040/results.json"
    ]
    
    # Use either provided files or default files
    file_paths = args.files if args.files else default_files
    
    if args.use_default_files:
        file_paths = default_files
        
    if not file_paths:
        parser.error("No input files specified. Use --files to provide file paths or --use-default-files to use defaults.")
    
    print(f"Combining results from {len(file_paths)} files...")
    combined_results, stats = combine_results(file_paths)
    
    # Print statistics
    print_statistics(stats)
    
    # Save combined results
    save_combined_results(combined_results, args.output)

if __name__ == "__main__":
    main() 