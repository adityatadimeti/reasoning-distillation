#!/usr/bin/env python3
"""
Script to run the DeepSeek model sweep on AIME 2024-5 dataset.
This script runs three experiments in parallel, each with a different DeepSeek model:
1. DeepSeek R1 Distill Qwen 1.5B
2. DeepSeek R1 Distill Qwen 14B
3. DeepSeek R1 Distill Llama 70B Free

Each experiment uses the same configuration parameters except for the reasoning model.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deepseek_sweep.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration for the experiments
EXPERIMENTS = [
    {
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "config": "deepseek_qwen_1_5b",
    },
    {
        "name": "DeepSeek R1 Distill Qwen 14B",
        "config": "deepseek_qwen_14b",
    },
    {
        "name": "DeepSeek R1 Distill Llama 70B Free",
        "config": "deepseek_llama_70b_free",
    }
]

def run_experiment(experiment):
    """Run a single experiment with the specified configuration."""
    name = experiment["name"]
    config = experiment["config"]
    
    logger.info(f"Starting experiment: {name}")
    start_time = time.time()
    
    # Build the command to run the experiment
    cmd = [
        "python", "run_experiment.py",
        config,
        "--parallel",
        "--concurrency", "8",  # 8 parallel workers per experiment
        "--verbose"
    ]
    
    # Run the experiment
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(f"[{name}] {line.strip()}")
        
        # Get the return code
        process.wait()
        
        if process.returncode != 0:
            # If there was an error, log the stderr
            stderr = process.stderr.read()
            logger.error(f"Experiment {name} failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Experiment {name} completed successfully in {duration:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment {name}: {str(e)}")
        return False

def run_all_experiments_parallel():
    """Run all experiments in parallel."""
    logger.info("Starting all experiments in parallel")
    
    with ProcessPoolExecutor(max_workers=len(EXPERIMENTS)) as executor:
        futures = {executor.submit(run_experiment, exp): exp["name"] for exp in EXPERIMENTS}
        
        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                success = future.result()
                if success:
                    logger.info(f"Experiment {exp_name} completed successfully")
                else:
                    logger.error(f"Experiment {exp_name} failed")
            except Exception as e:
                logger.error(f"Experiment {exp_name} raised an exception: {str(e)}")

def run_all_experiments_sequential():
    """Run all experiments sequentially."""
    logger.info("Starting all experiments sequentially")
    
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        if success:
            logger.info(f"Experiment {exp['name']} completed successfully")
        else:
            logger.error(f"Experiment {exp['name']} failed")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run DeepSeek model sweep on AIME 2024-5 dataset")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially instead of in parallel")
    args = parser.parse_args()
    
    logger.info("Starting DeepSeek model sweep")
    
    if args.sequential:
        run_all_experiments_sequential()
    else:
        run_all_experiments_parallel()
    
    logger.info("All experiments completed")

if __name__ == "__main__":
    main()
