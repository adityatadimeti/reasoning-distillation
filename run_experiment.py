import argparse
import os
import json
import csv
import logging
import time
import yaml
import asyncio
from typing import Dict, Any

from src.utils.config import load_config
from src.experiments.summarization import SummarizationExperiment
from src.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)

# NOTE: When running experiments, you only need to specify the config name, not the full path.
# Example: python run_experiment.py test --verbose
# This will automatically look for config/experiments/test.yaml
# DO NOT specify the full path like: python run_experiment.py config/experiments/test.yaml

def setup_logging(level: str = "INFO"):
    """
    Set up logging.
    
    Args:
        level: Logging level
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def load_problems(data_path: str) -> list:
    """
    Load problems from a CSV file.
    
    Args:
        data_path: Path to problems CSV file
        
    Returns:
        List of problem dictionaries
    """
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            # Create a normalized copy of the row with consistent keys
            normalized_row = {}
            for key, value in row.items():
                # Ensure ID field is available with both casings
                if key.lower() == 'id':
                    normalized_row['id'] = value
                    normalized_row['ID'] = value
                else:
                    normalized_row[key] = value
            problems.append(normalized_row)
    return problems

def load_prompt(prompt_type: str, version: str) -> str:
    """
    Load a prompt template from the prompts configuration.
    
    Args:
        prompt_type: Type of prompt (e.g., 'reasoning', 'summarization')
        version: Version of the prompt to use
        
    Returns:
        The prompt template string
    """
    prompt_path = os.path.join("config", "prompts", f"{prompt_type}.yaml")
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    if version not in prompts:
        raise ValueError(f"Prompt version '{version}' not found in {prompt_path}")
    
    return prompts[version]["template"]

def run_experiment(
    config_path: str, 
    use_dashboard: bool = False,
    verbose: bool = False,
    parallel: bool = False,
    max_concurrency: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an experiment with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        use_dashboard: Whether to use the dashboard
        verbose: Whether to log all LLM calls
        parallel: Whether to process problems in parallel
        max_concurrency: Maximum number of problems to process concurrently when parallel=True
        **kwargs: Additional configuration overrides
        
    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override configuration with kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    
    # Load prompt templates
    if "prompts" in config:
        for prompt_type, version in config["prompts"].items():
            template_key = f"{prompt_type}_prompt_template"
            config[template_key] = load_prompt(prompt_type, version)
    
    # Start dashboard if requested
    dashboard = None
    if use_dashboard:
        dashboard = DashboardServer(port=config.get("dashboard_port", 5000))
        dashboard.start(open_browser=True)
        
        # Initial status update
        logger.info("Sending initial experiment status with config")
        dashboard.update_experiment_status({
            "experiment_name": config.get("experiment_name", "Summarization"),
            "status": "Starting",
            "config": config  # Send full config only once
        })
        
        # Give the dashboard client time to connect
        time.sleep(2)
    
    # Load problems
    problems = load_problems(config["data_path"])
    
    # Show problem count
    logger.info(f"Loaded {len(problems)} problems")
    
    # When updating with problem count
    if dashboard:
        logger.info("Sending running status update with config")
        dashboard.update_experiment_status({
            "total": len(problems),
            "completed": 0,
            "status": "Running"
        })
    
    # Create experiment
    experiment = SummarizationExperiment(
        experiment_name=config.get("experiment_name", "summarization"),
        config=config,
        dashboard=dashboard,  # Pass dashboard to experiment
        verbose=verbose
    )
    
    # Run experiment
    if parallel and not use_dashboard:
        # For parallel processing, we need to use asyncio
        logger.info(f"Running experiment in parallel mode with max concurrency of {max_concurrency}")
        
        # Need to use asyncio.run to execute the coroutine
        results = asyncio.run(experiment.run_parallel(problems, max_concurrency=max_concurrency))
    else:
        if parallel and use_dashboard:
            logger.warning("Parallel processing is not compatible with dashboard. Using sequential processing.")
            
        # Run in sequential mode
        results = experiment.run(problems)
    
    # Final completion status
    if dashboard:
        logger.info("Sending completion status update with config")
        dashboard.update_experiment_status({
            "status": "Completed",
            "completed": len(problems)
        })
    
    # Save results
    experiment.save_results()
    
    return {
        "results": results,
        "config": config
    }

async def run_experiment_async(
    config_path: str, 
    verbose: bool = False,
    max_concurrency: int = 4,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an experiment with the specified configuration asynchronously.
    
    Args:
        config_path: Path to configuration file
        verbose: Whether to log all LLM calls
        max_concurrency: Maximum number of problems to process concurrently
        **kwargs: Additional configuration overrides
        
    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override configuration with kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    
    # Load prompt templates
    if "prompts" in config:
        for prompt_type, version in config["prompts"].items():
            template_key = f"{prompt_type}_prompt_template"
            config[template_key] = load_prompt(prompt_type, version)
    
    # Load problems
    problems = load_problems(config["data_path"])
    
    # Show problem count
    logger.info(f"Loaded {len(problems)} problems")
    
    # Create experiment
    experiment = SummarizationExperiment(
        experiment_name=config.get("experiment_name", "summarization"),
        config=config,
        dashboard=None,  # No dashboard in async mode
        verbose=verbose
    )
    
    # Run experiment in parallel
    results = await experiment.run_parallel(problems, max_concurrency=max_concurrency)
    
    # Save results
    experiment.save_results()
    
    return {
        "results": results,
        "config": config
    }

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run a reasoning enhancement experiment")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--verbose", action="store_true", help="log all LLM calls")
    parser.add_argument("--parallel", action="store_true", help="Process problems in parallel (incompatible with dashboard)")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum number of problems to process concurrently when parallel=True")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    try:
        run_experiment(
            args.config, 
            use_dashboard=args.dashboard,
            verbose=args.verbose,
            parallel=args.parallel,
            max_concurrency=args.concurrency
        )
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 