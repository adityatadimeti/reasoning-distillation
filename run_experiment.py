import argparse
import os
import json
import csv
import logging
import time
import yaml
import asyncio
from typing import Dict, Any, List, Optional

from src.utils.config import load_config
from src.experiments.summarization import SummarizationExperiment
from src.experiments.passk import PassKExperiment
from src.dashboard.server import DashboardServer
from src.experiments.continuation import ContinuationExperiment

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

def filter_problems(problems: List[Dict], question_ids: Optional[List[str]] = None, exclude_question_ids: Optional[List[str]] = None, index_range: Optional[str] = None) -> List[Dict]:
    """
    Filter problems by question ID or index range.
    
    Args:
        problems: List of problem dictionaries
        question_ids: List of question IDs to include
        exclude_question_ids: List of question IDs to exclude
        index_range: Range of indices to include (e.g., "0-4" or "10-15")
        
    Returns:
        Filtered list of problem dictionaries
    """
    if question_ids and index_range:
        raise ValueError("Cannot specify both question_ids and index_range")
    
    filtered_problems = problems
    
    # First, apply inclusion filter if specified
    if question_ids:
        filtered_problems = [p for p in filtered_problems if p.get('id') in question_ids or p.get('ID') in question_ids]
    
    # Then, apply exclusion filter if specified
    if exclude_question_ids:
        filtered_problems = [p for p in filtered_problems if p.get('id') not in exclude_question_ids and p.get('ID') not in exclude_question_ids]
    
    # Finally, apply index range filter if specified
    if index_range:
        try:
            start, end = map(int, index_range.split('-'))
            # Ensure valid range
            if start < 0 or end >= len(problems) or start > end:
                raise ValueError(f"Invalid index range: {index_range}. Valid range is 0-{len(problems)-1}")
            
            # Apply index range filter to the already filtered problems
            # This is tricky because indices are based on the original list
            # So we need to map the indices to the filtered list
            original_indices = list(range(start, end+1))  # +1 to include the end index
            index_filtered = []
            
            for i, problem in enumerate(problems):
                if i in original_indices:
                    # Check if this problem is in the filtered list
                    problem_id = problem.get('id', problem.get('ID'))
                    for fp in filtered_problems:
                        if fp.get('id', fp.get('ID')) == problem_id:
                            index_filtered.append(fp)
                            break
            
            filtered_problems = index_filtered
        except ValueError as e:
            if "too many values to unpack" in str(e):
                raise ValueError(f"Invalid index range format: {index_range}. Expected format: 'start-end'")
            raise
    
    return filtered_problems

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
    question_ids: Optional[List[str]] = None,
    exclude_question_ids: Optional[List[str]] = None,
    index_range: Optional[str] = None,
    load_initial_reasoning: Optional[str] = None,
    load_checkpoint: Optional[str] = None,
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
        question_ids: List of question IDs to run (optional)
        exclude_question_ids: List of question IDs to exclude (optional)
        index_range: Range of question indices to run (e.g., "0-4") (optional)
        load_initial_reasoning: Path to a previous results file to load initial reasoning from (optional)
        load_checkpoint: Path to a previous results file to load full checkpoint from (optional)
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
    all_problems = load_problems(config["data_path"])
    
    # Filter problems if specified
    problems = filter_problems(all_problems, question_ids, exclude_question_ids, index_range)
    
    # Show problem count
    if len(problems) == len(all_problems):
        logger.info(f"Loaded {len(problems)} problems")
    else:
        logger.info(f"Loaded {len(problems)} of {len(all_problems)} problems")
    
    # When updating with problem count
    if dashboard:
        logger.info("Sending running status update with config")
        dashboard.update_experiment_status({
            "total": len(problems),
            "completed": 0,
            "status": "Running"
        })
    
    # Determine experiment type
    experiment_type = config.get("experiment_type", "summarize")
    
    # Create appropriate experiment based on type
    if experiment_type == "pass_k":
        experiment = PassKExperiment(
            experiment_name=config.get("experiment_name", "pass_k"),
            config=config,
            dashboard=dashboard,
            verbose=verbose
        )
    elif experiment_type == "continuation":
        experiment = ContinuationExperiment(
            experiment_name=config.get("experiment_name", "continuation"),
            config=config,
            dashboard=None,
            verbose=verbose
        )
    else:  # Default to summarization experiment
        experiment = SummarizationExperiment(
            experiment_name=config.get("experiment_name", "summarization"),
            config=config,
            dashboard=dashboard,
            verbose=verbose
        )
    
    # Initialize with previous results if specified (applies to Summarization and Continuation)
    if load_initial_reasoning:
        if not os.path.exists(load_initial_reasoning):
            raise FileNotFoundError(f"Initial reasoning file not found: {load_initial_reasoning}")
        
        logger.info(f"Loading initial reasoning from {load_initial_reasoning}")
        # Check if the experiment instance has the initialization method
        if hasattr(experiment, 'initialize_with_previous_results'):
            experiment.initialize_with_previous_results(load_initial_reasoning)
        else:
            logger.warning(f"Experiment type '{experiment_type}' does not support loading initial reasoning.")
    
    # Load full checkpoint if specified (currently only supported for SummarizationExperiment)
    if load_checkpoint:
        if not os.path.exists(load_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {load_checkpoint}")
        
        logger.info(f"Loading full checkpoint from {load_checkpoint}")
        # Check if the experiment instance has the initialization method
        if hasattr(experiment, 'initialize_with_checkpoint'):
            experiment.initialize_with_checkpoint(load_checkpoint)
        else:
            logger.warning(f"Experiment type '{experiment_type}' does not support loading full checkpoint.")
    
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
    
    # Calculate metrics including token usage and cost
    metrics = experiment.calculate_metrics()
    
    # Log token usage and cost information
    logger.info(f"Total token usage: {metrics['token_usage']}")
    logger.info(f"Total cost: ${metrics['cost_info']['total_cost']:.4f}")
    
    # Final completion status with token usage and cost
    if dashboard:
        logger.info("Sending completion status update with metrics")
        dashboard.update_experiment_status({
            "status": "Completed",
            "completed": len(problems),
            "token_usage": metrics['token_usage'],
            "cost_info": metrics['cost_info']
        })
    
    # Save results
    experiment.save_results()
    
    return {
        "results": results,
        "config": config,
        "metrics": metrics
    }

async def run_experiment_async(
    config_path: str, 
    verbose: bool = False,
    max_concurrency: int = 4,
    question_ids: Optional[List[str]] = None,
    exclude_question_ids: Optional[List[str]] = None,
    index_range: Optional[str] = None,
    load_initial_reasoning: Optional[str] = None,
    load_checkpoint: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an experiment with the specified configuration asynchronously.
    
    Args:
        config_path: Path to configuration file
        verbose: Whether to log all LLM calls
        max_concurrency: Maximum number of problems to process concurrently
        question_ids: List of question IDs to run (optional)
        exclude_question_ids: List of question IDs to exclude (optional)
        index_range: Range of question indices to run (e.g., "0-4") (optional)
        load_initial_reasoning: Path to a previous results file to load initial reasoning from (optional)
        load_checkpoint: Path to a previous results file to load full checkpoint from (optional)
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
    all_problems = load_problems(config["data_path"])
    
    # Filter problems if specified
    problems = filter_problems(all_problems, question_ids, exclude_question_ids, index_range)
    
    # Show problem count
    if len(problems) == len(all_problems):
        logger.info(f"Loaded {len(problems)} problems")
    else:
        logger.info(f"Loaded {len(problems)} of {len(all_problems)} problems")
    
    # Determine experiment type
    experiment_type = config.get("experiment_type", "summarize")
    
    # Create appropriate experiment based on type
    if experiment_type == "pass_k":
        experiment = PassKExperiment(
            experiment_name=config.get("experiment_name", "pass_k"),
            config=config,
            dashboard=None,  # No dashboard in async mode
            verbose=verbose
        )
    elif experiment_type == "continuation":
        experiment = ContinuationExperiment(
            experiment_name=config.get("experiment_name", "continuation"),
            config=config,
            dashboard=None,  # No dashboard in async mode
            verbose=verbose
        )
    else:  # Default to summarization experiment
        experiment = SummarizationExperiment(
            experiment_name=config.get("experiment_name", "summarization"),
            config=config,
            dashboard=None,  # No dashboard in async mode
            verbose=verbose
        )
    
    # Initialize with previous results if specified (applies to Summarization and Continuation)
    if load_initial_reasoning:
        if not os.path.exists(load_initial_reasoning):
            raise FileNotFoundError(f"Initial reasoning file not found: {load_initial_reasoning}")
        
        logger.info(f"Loading initial reasoning from {load_initial_reasoning}")
        # Check if the experiment instance has the initialization method
        if hasattr(experiment, 'initialize_with_previous_results'):
            experiment.initialize_with_previous_results(load_initial_reasoning)
        else:
            logger.warning(f"Experiment type '{experiment_type}' does not support loading initial reasoning in async mode.")
    
    # Load full checkpoint if specified (currently only supported for SummarizationExperiment)
    if load_checkpoint:
        if not os.path.exists(load_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {load_checkpoint}")
        
        logger.info(f"Loading full checkpoint from {load_checkpoint}")
        # Check if the experiment instance has the initialization method
        if hasattr(experiment, 'initialize_with_checkpoint'):
            experiment.initialize_with_checkpoint(load_checkpoint)
        else:
            logger.warning(f"Experiment type '{experiment_type}' does not support loading full checkpoint in async mode.")
    
    # Run experiment in parallel
    results = await experiment.run_parallel(problems, max_concurrency=max_concurrency)
    
    # Calculate metrics including token usage and cost
    metrics = experiment.calculate_metrics()
    
    # Log token usage and cost information
    logger.info(f"Total token usage: {metrics['token_usage']}")
    logger.info(f"Total cost: ${metrics['cost_info']['total_cost']:.4f}")
    
    # Save results
    experiment.save_results()
    
    return {
        "results": results,
        "config": config,
        "metrics": metrics
    }

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run a reasoning enhancement experiment")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--verbose", action="store_true", help="log all LLM calls")
    parser.add_argument("--parallel", action="store_true", help="Process problems in parallel (incompatible with dashboard)")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum number of problems to process concurrently when parallel=True")
    
    # Add arguments for filtering problems
    parser.add_argument("--question_ids", type=str, help="Comma-separated list of question IDs to run")
    parser.add_argument("--exclude_question_ids", type=str, help="Comma-separated list of question IDs to exclude")
    parser.add_argument("--index_range", type=str, help="Range of question indices to run (e.g., '0-4' or '10-15')")
    
    # Add argument for continuation mode
    parser.add_argument("--continuations_only", action="store_true", help="Only process continuations for truncated solutions")
    parser.add_argument("--results_file", type=str, help="Path to results file for continuation processing (optional)")
    
    # Add argument for loading initial reasoning from previous results
    parser.add_argument("--load_initial_reasoning", type=str, help="Path to a previous results file to load initial reasoning from")
    
    # Add argument for loading full checkpoint from previous results
    parser.add_argument("--load_checkpoint", type=str, help="Path to a previous results file to load all iterations and summaries from")
    
    args = parser.parse_args()

    # Process question IDs if provided
    question_ids = None
    if args.question_ids:
        question_ids = [id.strip() for id in args.question_ids.split(',')]

    # Process excluded question IDs if provided
    exclude_question_ids = None
    if args.exclude_question_ids:
        exclude_question_ids = [id.strip() for id in args.exclude_question_ids.split(',')]
    
    # Set up logging
    setup_logging()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        if args.continuations_only:
            # Only process continuations for truncated solutions
            logger.info("Running in continuations-only mode")
            
            # Create experiment instance
            reasoning_provider = config.get("reasoning_model_provider", None)
            
            # Override configuration with command-line args
            if args.verbose:
                config["verbose"] = True
            
            # Create the experiment instance
            experiment = PassExperiment(
                experiment_name=config.get("experiment_name", "pass_k"),
                config=config,
                dashboard=None,  # No dashboard in continuation mode
                verbose=args.verbose
            )
            
            # Run the continuations
            asyncio.run(experiment.process_continuations(args.results_file))
            
            logger.info("Continuation processing completed successfully")
        else:
            # Run full experiment
            run_experiment(
                args.config, 
                use_dashboard=args.dashboard,
                verbose=args.verbose,
                parallel=args.parallel,
                max_concurrency=args.concurrency,
                question_ids=question_ids,
                exclude_question_ids=exclude_question_ids,
                index_range=args.index_range,
                load_initial_reasoning=args.load_initial_reasoning,
                load_checkpoint=args.load_checkpoint
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