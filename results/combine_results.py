#!/usr/bin/env python3
"""
Script to combine and analyze multiple results.json files from PassK experiments.

This script:
1. Takes two results.json files as input
2. Combines the results, avoiding duplicated reasoning traces
3. Tabulates statistics for each problem including:
   - Number of correct solutions
   - Number of incorrect solutions
   - Number of solutions with finish_reason = "error"
   - Number of solutions with finish_reason = "length"
4. Produces an overall summary of the combined results
"""

import argparse
import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

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

def combine_results(file1: str, file2: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """
    Combine results from two files and calculate statistics for each problem.
    
    Returns:
        Tuple containing the combined results dict and statistics dict
    """
    # Load both result files
    results1 = load_results_file(file1)
    results2 = load_results_file(file2)

    
    # Create a mapping of problem_id to problem data for easier merging
    all_problems = {}
    seen_solutions = {}  # Track solutions by hash to avoid duplicates
    
    # Statistics tracking
    stats = {}
    
    # Process first file
    for problem in results1.get("results", []):
        problem_id = problem.get("problem_id", "unknown")
        
        if problem_id not in all_problems:
            all_problems[problem_id] = {
                "problem_id": problem_id,
                "question": problem.get("question", ""),
                "correct_answer": problem.get("correct_answer", ""),
                "solutions": []
            }
            
            # Initialize statistics for this problem
            stats[problem_id] = {
                "correct": 0,
                "incorrect": 0,
                "error": 0,
                "length": 0,
                "total": 0
            }
        
        # Add solutions from this problem, tracking hashes to avoid duplicates
        for solution in problem.get("solutions", []):
            solution_hash = get_solution_hash(solution)
            
            if solution_hash not in seen_solutions:
                seen_solutions[solution_hash] = True
                all_problems[problem_id]["solutions"].append(solution)
                
                # Update statistics
                if solution.get("correct", False):
                    stats[problem_id]["correct"] += 1
                else:
                    stats[problem_id]["incorrect"] += 1
                    
                finish_reason = solution.get("finish_reason", "unknown")
                if finish_reason == "error":
                    stats[problem_id]["error"] += 1
                elif finish_reason == "length":
                    stats[problem_id]["length"] += 1
                    
                stats[problem_id]["total"] += 1
    
    # Process second file
    for problem in results2.get("results", []):
        problem_id = problem.get("problem_id", "unknown")
        
        if problem_id not in all_problems:
            all_problems[problem_id] = {
                "problem_id": problem_id,
                "question": problem.get("question", ""),
                "correct_answer": problem.get("correct_answer", ""),
                "solutions": []
            }
            
            # Initialize statistics for this problem
            stats[problem_id] = {
                "correct": 0,
                "incorrect": 0,
                "error": 0,
                "length": 0,
                "total": 0
            }
        
        # Add solutions from this problem, tracking hashes to avoid duplicates
        for solution in problem.get("solutions", []):
            solution_hash = get_solution_hash(solution)
            
            if solution_hash not in seen_solutions:
                seen_solutions[solution_hash] = True
                all_problems[problem_id]["solutions"].append(solution)
                
                # Update statistics
                if solution.get("correct", False):
                    stats[problem_id]["correct"] += 1
                else:
                    stats[problem_id]["incorrect"] += 1
                    
                finish_reason = solution.get("finish_reason", "unknown")
                if finish_reason == "error":
                    stats[problem_id]["error"] += 1
                elif finish_reason == "length":
                    stats[problem_id]["length"] += 1
                    
                stats[problem_id]["total"] += 1
    
    # Update consensus information for each problem based on the combined solutions
    for problem_id, problem_data in all_problems.items():
        solutions = problem_data["solutions"]
        correct_answer = problem_data["correct_answer"]
        
        # Find consensus answer
        answers = [s.get("answer") for s in solutions if s.get("answer") is not None]
        consensus_data = find_consensus(answers)
        
        if consensus_data:
            consensus_answer, consensus_count = consensus_data
            consensus_correct = consensus_answer.strip() == correct_answer.strip()
        else:
            consensus_answer = None
            consensus_count = 0
            consensus_correct = False
        
        # Calculate pass@k
        num_correct = sum(1 for s in solutions if s.get("correct", False))
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

def find_consensus(answers: List[str]) -> Tuple[str, int]:
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
    total_solutions = sum(problem_stats["total"] for problem_stats in stats.values())
    total_correct = sum(problem_stats["correct"] for problem_stats in stats.values())
    total_incorrect = sum(problem_stats["incorrect"] for problem_stats in stats.values())
    total_error = sum(problem_stats["error"] for problem_stats in stats.values())
    total_length = sum(problem_stats["length"] for problem_stats in stats.values())
    
    print(f"Total Problems: {total_problems}")
    print(f"Total Solutions: {total_solutions}")
    print(f"Correct Solutions: {total_correct} ({total_correct/total_solutions:.2%})")
    print(f"Incorrect Solutions: {total_incorrect} ({total_incorrect/total_solutions:.2%})")
    print(f"Error Finishes: {total_error} ({total_error/total_solutions:.2%})")
    print(f"Length Finishes: {total_length} ({total_length/total_solutions:.2%})")
    
    # Problem-level statistics
    print("\nProblem-level Statistics:")
    print(f"{'Problem ID':<15} {'Total':<8} {'Correct':<10} {'Incorrect':<10} {'Error':<8} {'Length':<8}")
    print("-" * 65)
    
    for problem_id, problem_stats in sorted(stats.items()):
        total = problem_stats["total"]
        correct = problem_stats["correct"]
        incorrect = problem_stats["incorrect"]
        error = problem_stats["error"]
        length = problem_stats["length"]
        
        print(f"{problem_id:<15} {total:<8} {correct:<10} {incorrect:<10} {error:<8} {length:<8}")

def save_combined_results(combined_results: Dict[str, Any], output_file: str):
    """Save the combined results to a new file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nCombined results saved to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Combine and analyze multiple results.json files.")
    parser.add_argument("file1", help="Path to first results.json file")
    parser.add_argument("file2", help="Path to second results.json file")
    parser.add_argument("--output", "-o", default="combined_results.json", 
                        help="Output file path (default: combined_results.json)")
    
    args = parser.parse_args()
    
    print(f"Combining results from {args.file1} and {args.file2}...")
    combined_results, stats = combine_results(args.file1, args.file2)
    
    # Print statistics
    print_statistics(stats)
    
    # Save combined results
    save_combined_results(combined_results, args.output)

if __name__ == "__main__":
    main()