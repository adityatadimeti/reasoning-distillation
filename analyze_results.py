#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

def load_results(results_path: str) -> Dict:
    """Load results from a JSON file."""
    with open(results_path) as f:
        data = json.load(f)
    return data.get('results', data)  # handle both raw results and wrapped results

def split_by_year(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split results into 2024 and 2025 problems."""
    results_2024 = [res for res in results if res['problem_id'].startswith('2024')]
    results_2025 = [res for res in results if not res['problem_id'].startswith('2024')]
    return results_2024, results_2025

def analyze_results(results: List[Dict]) -> Tuple[int, int, float, float]:
    """Analyze results and return initial/final correct counts and accuracies."""
    total_problems = len(results)
    if total_problems == 0:
        return 0, 0, 0.0, 0.0
        
    initial_correct = sum(1 for res in results if res['iterations'][0]['correct'])
    final_correct = sum(1 for res in results if res['iterations'][-1]['correct'])
    
    return (
        initial_correct,
        final_correct,
        initial_correct / total_problems * 100,
        final_correct / total_problems * 100
    )

def find_missing_problems(results: List[Dict], year: str) -> Set[str]:
    """Find missing problem IDs for a given year."""
    actual = {res['problem_id'] for res in results}
    
    if year == "2024":
        # Just report the count for 2024 since we don't know if it's AIME I or II
        if len(actual) != 30:
            return {f"Expected 30 problems, found {len(actual)}"}
        return set()
    else:  # 2025
        expected = {str(i) for i in range(1, 31)}
        return expected - actual

def analyze_experiment(results_dir: str, experiment_name: str) -> None:
    """Analyze a single experiment's results."""
    # Find the most recent results file
    exp_dir = Path(results_dir) / experiment_name
    if not exp_dir.exists():
        print(f"No results directory found for {experiment_name}")
        return

    # Find the most recent run directory
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print(f"No run directories found for {experiment_name}")
        return

    latest_run = max(run_dirs, key=lambda d: d.name)
    
    # Try both regular results.json and merged_results.json
    results_file = latest_run / "results.json"
    merged_results_file = latest_run / "merged_results.json"
    
    if merged_results_file.exists():
        results_path = merged_results_file
    elif results_file.exists():
        results_path = results_file
    else:
        print(f"No results file found for {experiment_name}")
        return

    results = load_results(str(results_path))
    results_2024, results_2025 = split_by_year(results)

    # Check for missing problems
    missing_2024 = find_missing_problems(results_2024, "2024")
    missing_2025 = find_missing_problems(results_2025, "2025")

    # Print analysis
    print(f"\nAnalyzing {experiment_name}")
    print(f"Using results from: {results_path}")
    print("\n2024 Results:")
    initial_correct, final_correct, initial_acc, final_acc = analyze_results(results_2024)
    print(f"Initial correct: {initial_correct}/30 ({initial_acc:.1f}%)")
    print(f"Final correct: {final_correct}/30 ({final_acc:.1f}%)")
    
    print("\n2025 Results:")
    initial_correct, final_correct, initial_acc, final_acc = analyze_results(results_2025)
    print(f"Initial correct: {initial_correct}/30 ({initial_acc:.1f}%)")
    print(f"Final correct: {final_correct}/30 ({final_acc:.1f}%)")

    # Report missing problems
    if missing_2024:
        print("\nIssue with 2024 problems:")
        print(", ".join(sorted(missing_2024)))
    if missing_2025:
        print("\nMissing 2025 problems:")
        print(", ".join(sorted(missing_2025)))

def main():
    results_dir = "./results"
    
    # List of experiments to analyze
    experiments = [
        # Seed 1
        "aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_1",
        "aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_1",
        "aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_1",
        "aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_1",
        
        # Seed 2
        "aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_2",
        "aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_2",
        "aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_2",
        
        # Seed 3
        "aime_deepseek_qwen_14b_baseline_post_think_sum_4iter_seed_3",
        "aime_deepseek_qwen_14b_summ_base_sum_4iter_seed_3",
        "aime_deepseek_qwen_14b_baseline_lastk_sum_4iter_seed_3",
        "aime_deepseek_qwen_14b_baseline_firstk_sum_4iter_seed_3"
    ]

    for experiment in experiments:
        analyze_experiment(results_dir, experiment)

if __name__ == "__main__":
    main() 