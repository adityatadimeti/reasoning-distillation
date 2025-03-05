"""
Baseline experiment for generating and evaluating reasoning traces.
"""
import os
import sys
import argparse
from pathlib import Path

from src.utils.config import Config, PROJECT_ROOT
from experiments.run_experiment import run_experiment

def run_baseline_experiment(
    config_path=None, 
    split="test", 
    max_problems=None, 
    problem_ids=None
):
    """
    Run the baseline reasoning experiment.
    
    Args:
        config_path: Path to configuration file (defaults to baseline config)
        split: Data split to use (train, test, all)
        max_problems: Maximum number of problems to process
        problem_ids: List of specific problem IDs to process
    
    Returns:
        Dictionary with experiment results
    """
    # Use default config if not specified
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "experiments" / "baseline.yaml"
    
    # Convert problem_ids list to comma-separated string if provided
    problem_ids_arg = None
    if problem_ids:
        if isinstance(problem_ids, list):
            problem_ids_arg = ",".join(problem_ids)
        else:
            problem_ids_arg = problem_ids
    
    # Run the experiment
    return run_experiment(
        config_path=config_path,
        split=split,
        max_problems=max_problems,
        problem_ids=problem_ids_arg
    )

def main():
    """Main entry point for baseline experiment."""
    parser = argparse.ArgumentParser(description="Run baseline reasoning experiment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--split", default="test", help="Data split to use (train, test, all)")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems to process")
    parser.add_argument("--problem-ids", help="Comma-separated list of problem IDs to process")
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        config_path=args.config,
        split=args.split,
        max_problems=args.max_problems,
        problem_ids=args.problem_ids
    )

if __name__ == "__main__":
    main()