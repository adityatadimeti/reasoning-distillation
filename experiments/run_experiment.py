"""
Main experiment runner for reasoning enhancement project.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import importlib
import time
from src.utils.config import Config, PROJECT_ROOT
from src.data.dataset import Dataset
from src.pipeline.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

def setup_logging(config: Config) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration object
    """
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format
    )
    
    # Set up file logging if configured
    if log_config.get("log_file", False):
        log_path = Path(log_config.get("save_path", "logs"))
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        
        os.makedirs(log_path, exist_ok=True)
        
        # Create a timestamped log file
        log_file = log_path / f"experiment_{int(time.time())}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")

def get_pipeline_class(pipeline_type: str) -> type:
    """
    Get the pipeline class for a given pipeline type.
    
    Args:
        pipeline_type: Type of pipeline (e.g., 'baseline', 'summary', 'recursive')
        
    Returns:
        Pipeline class
        
    Raises:
        ValueError: If pipeline type is not supported
    """
    pipeline_mapping = {
        "baseline": "src.pipeline.baseline_pipeline.BaselineReasoningPipeline",
        "summary": "src.pipeline.summary_pipeline.SummaryReasoningPipeline",
        "recursive": "src.pipeline.recursive_pipeline.RecursiveReasoningPipeline"
    }
    
    if pipeline_type not in pipeline_mapping:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
    
    # Import the module and get the class
    module_path, class_name = pipeline_mapping[pipeline_type].rsplit('.', 1)
    module = importlib.import_module(module_path)
    pipeline_class = getattr(module, class_name)
    
    return pipeline_class

def run_experiment(config_path: str, **kwargs) -> Dict[str, Any]:
    """
    Run an experiment using the specified configuration.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to override configuration
        
    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    config = Config(config_path)
    
    # Override configuration with kwargs
    for key, value in kwargs.items():
        if key == "split" and value:
            config._merge_config({"pipeline": {"split": value}})
        elif key == "max_problems" and value:
            config._merge_config({"pipeline": {"max_problems": int(value)}})
        elif key == "problem_ids" and value:
            config._merge_config({"pipeline": {"problem_ids": value.split(",")}})
    
    # Set up logging
    setup_logging(config)
    
    # Log configuration
    logger.info(f"Running experiment with configuration from {config_path}")
    
    # Create dataset
    dataset_config = config.get("dataset", {})
    logger.info(f"Loading dataset: {dataset_config.get('name', 'unknown')}")
    dataset = Dataset(config)
    
    # Process raw data if needed
    logger.info("Processing dataset")
    dataset.load_processed()
    
    # Get the pipeline class and create instance
    pipeline_type = config.get("pipeline.type", "baseline")
    logger.info(f"Creating pipeline of type: {pipeline_type}")
    pipeline_class = get_pipeline_class(pipeline_type)
    pipeline = pipeline_class(config)
    
    # Run the pipeline
    logger.info("Running pipeline")
    results = pipeline.run(
        dataset=dataset,
        split=config.get("pipeline.split", "test"),
        max_problems=config.get("pipeline.max_problems", None),
        problem_ids=config.get("pipeline.problem_ids", None)
    )
    
    # Evaluate the results
    logger.info("Evaluating results")
    metrics = pipeline.evaluate(results)
    
    # Save the results
    logger.info("Saving results")
    pipeline.save_results(results, metrics)
    
    return {
        "results": results,
        "metrics": metrics
    }

def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run a reasoning enhancement experiment")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--split", help="Data split to use (train, test, all)")
    parser.add_argument("--max-problems", help="Maximum number of problems to process")
    parser.add_argument("--problem-ids", help="Comma-separated list of problem IDs to process")
    
    args = parser.parse_args()
    
    try:
        run_experiment(
            args.config,
            split=args.split,
            max_problems=args.max_problems,
            problem_ids=args.problem_ids
        )
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()