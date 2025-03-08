import argparse
import json
import os
from typing import List, Dict, Any

from src.utils.config import load_config
from src.experiments.summarization import SummarizationExperiment

def load_problems(data_path: str) -> List[Dict[str, Any]]:
    """Load problems from a JSON file."""
    with open(data_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run reasoning experiments")
    parser.add_argument("--experiment", type=str, default="summarization",
                        help="Name of the experiment to run")
    parser.add_argument("--data", type=str, default="data/aime_2024/problems.json",
                        help="Path to the problems data file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a custom config file")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = load_config(args.experiment)
    
    # Load problems
    problems = load_problems(args.data)
    
    # Run experiment
    if args.experiment == "summarization":
        experiment = SummarizationExperiment(experiment_name=args.experiment, config=config)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    experiment.run(problems)
    experiment.save_results()

if __name__ == "__main__":
    main() 