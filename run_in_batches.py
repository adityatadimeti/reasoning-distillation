"""
Run an experiment in multiple batches, keeping all results in the same results directory.
This is useful for large datasets that you want to process in smaller chunks.
"""

import argparse
import sys
import subprocess
import time
import logging
import pandas as pd
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("batch_runner")

def run_batch(config: str, start_idx: int, end_idx: int, verbose: bool = False, parallel: bool = False):
    """Run a batch of questions from the dataset."""
    cmd = [
        "python", "run_experiment.py",
        config,
        "--index-range", f"{start_idx}-{end_idx}"
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    if parallel:
        cmd.append("--parallel")
    
    logger.info(f"Running batch for index range {start_idx}-{end_idx}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, text=True, capture_output=True, check=True)
        logger.info(f"Batch completed successfully: {start_idx}-{end_idx}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Batch failed with code {e.returncode}: {start_idx}-{end_idx}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run an experiment in batches")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=10, 
                        help="Number of questions to process in each batch")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Index to start from (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Index to end at (inclusive, defaults to the last question)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Delay in seconds between batches")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--parallel", action="store_true",
                        help="Run each batch with parallel processing")
    
    args = parser.parse_args()
    
    # Determine the data path from the config
    config_path = f"config/experiments/{args.config}.yaml"
    if not os.path.exists(config_path):
        config_path = args.config  # If full path was provided
        
    # Try to extract the data path from the config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config.get('data_path')
    if not data_path:
        logger.error(f"Could not find data_path in config: {config_path}")
        return 1
    
    # Check if the data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return 1
    
    # Count total questions
    df = pd.read_csv(data_path)
    total_questions = len(df)
    
    # Determine end index if not provided
    end_idx = args.end_idx if args.end_idx is not None else total_questions - 1
    
    # Validate indices
    if args.start_idx < 0 or args.start_idx >= total_questions:
        logger.error(f"Invalid start index: {args.start_idx}. Valid range: 0-{total_questions-1}")
        return 1
    
    if end_idx < args.start_idx or end_idx >= total_questions:
        logger.error(f"Invalid end index: {end_idx}. Valid range: {args.start_idx}-{total_questions-1}")
        return 1
    
    # Calculate batches
    batches = []
    current_start = args.start_idx
    
    while current_start <= end_idx:
        current_end = min(current_start + args.batch_size - 1, end_idx)
        batches.append((current_start, current_end))
        current_start = current_end + 1
    
    logger.info(f"Will process {end_idx - args.start_idx + 1} questions in {len(batches)} batches")
    
    # Run batches
    for i, (batch_start, batch_end) in enumerate(batches):
        logger.info(f"Starting batch {i+1}/{len(batches)}: {batch_start}-{batch_end}")
        
        success = run_batch(
            args.config, batch_start, batch_end, 
            verbose=args.verbose, parallel=args.parallel
        )
        
        if not success:
            logger.error(f"Batch {i+1} failed. Stopping.")
            return 1
        
        if i < len(batches) - 1:
            logger.info(f"Waiting {args.delay} seconds before starting next batch...")
            time.sleep(args.delay)
    
    logger.info(f"All {len(batches)} batches completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 