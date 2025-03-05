#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main experiment runner script.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.pipeline.baseline_pipeline import BaselinePipeline
from src.pipeline.summary_pipeline import SummaryPipeline
from src.pipeline.recursive_pipeline import RecursivePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run an experiment with the specified configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    return parser.parse_args()


def get_pipeline_class(pipeline_type):
    """Get the pipeline class based on the pipeline type."""
    pipeline_classes = {
        'baseline': BaselinePipeline,
        'summary': SummaryPipeline,
        'recursive': RecursivePipeline,
    }
    return pipeline_classes.get(pipeline_type)


def run_experiment(config_path, debug=False):
    """Run an experiment using the specified config."""
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.logs_dir, f"{config.experiment.name}_{timestamp}.log")
    log_level = logging.DEBUG if debug else getattr(logging, config.experiment.log_level)
    logger = setup_logger(config.experiment.name, log_file, log_level)
    
    logger.info(f"Starting experiment: {config.experiment.name}")
    logger.info(f"Configuration: {config_path}")
    
    # Create output directory
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline_class = get_pipeline_class(config.pipeline.type)
    if pipeline_class is None:
        logger.error(f"Unknown pipeline type: {config.pipeline.type}")
        return
    
    pipeline = pipeline_class(config)
    
    # Run experiment
    try:
        results = pipeline.run()
        
        # Evaluate and log results
        metrics = pipeline.evaluate(results)
        pipeline.log_results(results, metrics)
        
        logger.info(f"Experiment completed successfully. Results saved to {config.experiment.output_dir}")
        return results, metrics
    
    except Exception as e:
        logger.exception(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config, args.debug)
