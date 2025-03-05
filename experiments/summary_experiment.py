"""
Summary experiment for enhancing reasoning through summarization.
"""
import os
import sys
import argparse
from pathlib import Path

from src.utils.config import Config, PROJECT_ROOT
from experiments.run_experiment import run_experiment

def run_summary_experiment(
    config_path=None, 
    method="self",  # 'self' or 'external'
    split="test", 
    max_problems=None, 
    problem_ids=None,
    max_iterations=None
):
    """
    Run the summary reasoning experiment.
    
    Args:
        config_path: Path to configuration file (defaults to summary config)
        method: Summarization method ('self' or 'external')
        split: Data split to use (train, test, all)
        max_problems: Maximum number of problems to process
        problem_ids: List of specific problem IDs to process
        max_iterations: Maximum number of iterations
    
    Returns:
        Dictionary with experiment results
    """
    # Determine the appropriate config based on method
    if config_path is None:
        if method == "self":
            config_path = PROJECT_ROOT / "configs" / "experiments" / "self_summarization.yaml"
        else:
            config_path = PROJECT_ROOT / "configs" / "experiments" / "external_summarization.yaml"
    
    # Convert problem_ids list to comma-separated string if provided
    problem_ids_arg = None
    if problem_ids:
        if isinstance(problem_ids, list):
            problem_ids_arg = ",".join(problem_ids)
        else:
            problem_ids_arg = problem_ids
    
    # Prepare max_iterations argument
    max_iterations_arg = None
    if max_iterations:
        max_iterations_arg = str(max_iterations)
    
    # Run the experiment
    return run_experiment(
        config_path=config_path,
        split=split,
        max_problems=max_problems,
        problem_ids=problem_ids_arg,
        max_iterations=max_iterations_arg
    )

def main():
    """Main entry point for summary experiment."""
    parser = argparse.ArgumentParser(description="Run summary reasoning experiment")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--method", choices=["self", "external"], default="self", help="Summarization method")
    parser.add_argument("--split", default="test", help="Data split to use (train, test, all)")
    parser.add_argument("--max-problems", type=int, help="Maximum number of problems to process")
    parser.add_argument("--problem-ids", help="Comma-separated list of problem IDs to process")
    parser.add_argument("--max-iterations", type=int, help="Maximum number of iterations")
    
    args = parser.parse_args()
    
    run_summary_experiment(
        config_path=args.config,
        method=args.method,
        split=args.split,
        max_problems=args.max_problems,
        problem_ids=args.problem_ids,
        max_iterations=args.max_iterations
    )

if __name__ == "__main__":
    main()