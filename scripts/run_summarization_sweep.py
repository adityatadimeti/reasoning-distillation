"""
Script to run summarization sweep on reasoning traces.
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.utils.config import Config
from src.pipeline.baseline_pipeline import BaselinePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_summarization_sweep(
    problem_ids: List[str],
    config_path: str = "configs/fireworks.yaml",
    input_file: str = "results/baseline_reasoning_traces.json",
    output_dir: str = "results/summarization_sweep",
    summarization_mode: str = "append"
) -> None:
    """
    Run summarization sweep on reasoning traces for multiple problems.
    
    Args:
        problem_ids: List of problem IDs to process
        config_path: Path to config file
        input_file: Path to input JSON file with reasoning traces
        output_dir: Directory to save results
        summarization_mode: Mode of summarization ('append' or 'prepend')
    """
    # Load config
    config = Config(config_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reasoning traces
    with open(input_file, 'r') as f:
        reasoning_traces = json.load(f)
    
    # Initialize pipeline
    pipeline = BaselinePipeline(config)
    
    # Process each problem
    for problem_id in problem_ids:
        logger.info(f"Processing problem {problem_id}")
        
        # Find the reasoning trace for this problem
        problem_trace = None
        for trace in reasoning_traces.get("results", []):
            if trace.get("problem_id") == problem_id:
                problem_trace = trace
                break
        
        if not problem_trace:
            logger.warning(f"No reasoning trace found for problem {problem_id}")
            continue
        
        # Run summarization
        start_time = time.time()
        result = pipeline.run(
            problem_id=problem_id,
            reasoning_trace=problem_trace["reasoning"],
            summarization_mode=summarization_mode
        )
        processing_time = time.time() - start_time
        
        # Add processing time to result
        result["processing_time"] = processing_time
        
        # Save result
        output_file = os.path.join(
            output_dir, 
            f"{problem_id}_{summarization_mode}_summary.json"
        )
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved result to {output_file}")

if __name__ == "__main__":
    # List of problem IDs to process
    problem_ids = ["2024-I-8", "2024-I-12", "2024-I-11", "2024-I-13", 
                   "2024-II-15", "2024-II-9", "2024-I-14"]
    
    # Run summarization sweep
    run_summarization_sweep(problem_ids) 