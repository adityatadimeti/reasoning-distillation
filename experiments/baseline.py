"""
Baseline experiment runner for reasoning enhancement project.
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
    # Determine the appropriate config
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
    parser.add_argument("--model", help="Model to use (deepseek-r1, deepseek-r1-distill-qwen-7b, etc.)")
    parser.add_argument("--dataset", help="Dataset to use (aime2024, gsm8k, etc.)")
    
    args = parser.parse_args()
    
    # Create a config object if we need to override model or dataset
    config_path = args.config
    if args.model or args.dataset:
        # Start with default config if none specified
        if not config_path:
            config_path = PROJECT_ROOT / "configs" / "experiments" / "baseline.yaml"
                
        # Load the config
        config = Config(config_path)
        
        # Override model if specified
        if args.model:
            model_config_path = PROJECT_ROOT / "configs" / "models" / f"{args.model}.yaml"
            if model_config_path.exists():
                model_config = Config(model_config_path)
                config._merge_config(model_config.to_dict())
            else:
                print(f"Warning: Model config file {model_config_path} not found. Using default model.")
        
        # Override dataset if specified
        if args.dataset:
            dataset_config_path = PROJECT_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml"
            if dataset_config_path.exists():
                dataset_config = Config(dataset_config_path)
                config._merge_config(dataset_config.to_dict())
            else:
                print(f"Warning: Dataset config file {dataset_config_path} not found. Using default dataset.")
        
        # Save the modified config to a temporary file
        temp_config_path = PROJECT_ROOT / "configs" / "temp" / "baseline_experiment.yaml"
        os.makedirs(temp_config_path.parent, exist_ok=True)
        
        # Write config to file
        with open(temp_config_path, 'w') as f:
            import yaml
            yaml.dump(config.to_dict(), f)
        
        # Use the temporary config path
        config_path = temp_config_path
    
    run_baseline_experiment(
        config_path=config_path,
        split=args.split,
        max_problems=args.max_problems,
        problem_ids=args.problem_ids
    )

if __name__ == "__main__":
    main() 