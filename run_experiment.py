import argparse
import os
import json
import csv
import logging
import time
import yaml
from typing import Dict, Any

from src.utils.config import load_config
from src.experiments.summarization import SummarizationExperiment
from src.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)

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
    model_params: Dict[str, str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an experiment with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        use_dashboard: Whether to use the dashboard
        verbose: Whether to log all LLM calls
        model_params: Dictionary of model parameters to override
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
    
    # Override with model parameters if provided
    if model_params:
        print(f"DEBUG: Applying model parameter overrides:")
        for key, value in model_params.items():
            if value is not None:
                print(f"  - Overriding {key}: {config.get(key, 'None')} -> {value}")
                config[key] = value
        
        # Ensure we print the final configuration
        print(f"DEBUG: Final configuration after overrides:")
        print(f"  - reasoning_model: {config.get('reasoning_model')}")
        print(f"  - summarizer_type: {config.get('summarizer_type')}")
        print(f"  - summarizer_model: {config.get('summarizer_model')}")
    
    # Load prompt templates
    if "prompts" in config:
        for prompt_type, version in config["prompts"].items():
            template_key = f"{prompt_type}_prompt_template"
            config[template_key] = load_prompt(prompt_type, version)
    
    # Start dashboard if requested
    dashboard = None
    if use_dashboard:
        dashboard = DashboardServer.get_instance(port=config.get("dashboard_port", 5000))
        
        # Initial status update
        logger.info("Sending initial experiment status with config")
        dashboard.update_experiment_status({
            "experiment_name": config.get("experiment_name", "Summarization"),
            "status": "Starting",
            "config": config  # Send full config only once
        })
    
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

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run a reasoning enhancement experiment")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--dashboard-only", action="store_true", help="Start dashboard without running experiment")
    parser.add_argument("--verbose", action="store_true", help="log all LLM calls")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    try:
        if args.dashboard_only:
            # Just start the dashboard server
            config_path = args.config
            config = load_config(config_path)
            dashboard = DashboardServer.get_instance(
                port=config.get("dashboard_port", 5000),
                config_path=config_path
            )
            dashboard.start(open_browser=True)
            
            # Keep the server running until interrupted
            logger.info(f"Dashboard started on port {config.get('dashboard_port', 5000)}. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down dashboard...")
                dashboard.stop()
        else:
            # Original behavior - start dashboard first, then run experiment
            config_path = args.config
            config = load_config(config_path)
            
            if args.dashboard:
                logger.info("Starting dashboard before running experiment...")
                dashboard = DashboardServer.get_instance(
                    port=config.get("dashboard_port", 5000),
                    config_path=config_path
                )
                dashboard.start(open_browser=True)
                
                # Give time for browser to open and connect
                logger.info("Waiting for dashboard to initialize...")
                time.sleep(3)  # Give browser a chance to open and connect
            
            # Run the experiment
            run_experiment(
                args.config, 
                use_dashboard=args.dashboard,
                verbose=args.verbose
            )
            logger.info("Experiment completed successfully")
            
            # If dashboard is running, keep it alive
            if args.dashboard:
                logger.info("Experiment completed. Dashboard remains active. Press Ctrl+C to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    dashboard.stop()
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 