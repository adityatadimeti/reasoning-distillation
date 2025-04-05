#!/usr/bin/env python3
"""
Script to run the DeepSeek model sweep on AIME 2024-5 dataset with resume functionality.
This script runs experiments with DeepSeek models and can resume from where it left off
if interrupted.

Key features:
- Tracks progress in a state file
- Skips problems that have already been completed
- Can resume from the last completed problem
- Uses Fireworks API for higher concurrency (up to 32 concurrent requests)
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
import subprocess
from datetime import datetime
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

# Path to the state file that tracks progress
STATE_FILE = "deepseek_sweep_state.json"

def load_state():
    """Load the current state from the state file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
            return {"experiments": {}}
    return {"experiments": {}}

def save_state(state):
    """Save the current state to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving state file: {e}")

def find_completed_problems(experiment_name):
    """Find all problems that have been completed for a given experiment."""
    completed_problems = set()
    
    # Look for results directories for this experiment
    results_dirs = glob.glob(f"results/{experiment_name}_*")
    
    for results_dir in results_dirs:
        results_file = os.path.join(results_dir, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    # Extract problem IDs from completed results
                    for result in data.get("results", []):
                        if result.get("status") != "error":
                            problem_id = result.get("problem_id")
                            if problem_id:
                                completed_problems.add(problem_id)
            except Exception as e:
                logger.warning(f"Error reading results file {results_file}: {e}")
    
    return completed_problems

def run_experiment(experiment, resume=True):
    """Run a single experiment with the specified configuration."""
    name = experiment["name"]
    config = experiment["config"]
    
    logger.info(f"Starting experiment: {name}")
    start_time = time.time()
    
    # Load state to check for progress
    state = load_state()
    experiment_state = state["experiments"].get(name, {})
    
    # Find completed problems if resuming
    completed_problems = set()
    if resume:
        completed_problems = find_completed_problems(name)
        if completed_problems:
            logger.info(f"Found {len(completed_problems)} already completed problems for {name}")
    
    # Convert completed problems to comma-separated string
    completed_str = ",".join(completed_problems)
    
    # Build the command to run the experiment
    # Determine appropriate concurrency based on provider
    if "together" in config.lower():
        concurrency = "4"  # Together API has lower rate limits
    else:
        concurrency = "32"  # Fireworks can handle up to 32 concurrent requests
    
    logger.info(f"Using concurrency {concurrency} for {name} based on API provider")
    
    cmd = [
        "python", "run_experiment.py",
        config,
        "--parallel",
        "--concurrency", concurrency,
        "--verbose"
    ]
    
    # Add exclude-question-ids parameter if we have completed problems
    if completed_problems and resume:
        cmd.extend(["--exclude-question-ids", completed_str])
    
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
            
            # Check for completion messages to update state
            if "Processing problem" in line:
                try:
                    # Extract problem ID from log line
                    parts = line.split("Processing problem ")
                    if len(parts) > 1:
                        problem_id = parts[1].split(" ")[0]
                        # Update the last processed problem in state
                        experiment_state["last_problem"] = problem_id
                        state["experiments"][name] = experiment_state
                        save_state(state)
                except Exception as e:
                    logger.warning(f"Error parsing problem ID from log: {e}")
        
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
        
        # Update state to mark experiment as completed
        experiment_state["completed"] = True
        experiment_state["completion_time"] = datetime.now().isoformat()
        state["experiments"][name] = experiment_state
        save_state(state)
        
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment {name}: {str(e)}")
        return False

def run_all_experiments_parallel(resume=True):
    """Run all experiments in parallel."""
    logger.info("Starting all experiments in parallel")
    
    with ProcessPoolExecutor(max_workers=len(EXPERIMENTS)) as executor:
        futures = {executor.submit(run_experiment, exp, resume): exp["name"] for exp in EXPERIMENTS}
        
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

def run_all_experiments_sequential(resume=True):
    """Run all experiments sequentially."""
    logger.info("Starting all experiments sequentially")
    
    for exp in EXPERIMENTS:
        success = run_experiment(exp, resume)
        if success:
            logger.info(f"Experiment {exp['name']} completed successfully")
        else:
            logger.error(f"Experiment {exp['name']} failed")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run DeepSeek model sweep on AIME 2024-5 dataset with resume capability")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially instead of in parallel")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous runs, start fresh")
    args = parser.parse_args()
    
    logger.info("Starting DeepSeek model sweep")
    
    # Check if we should resume
    resume = not args.no_resume
    if resume:
        logger.info("Resume mode enabled - will skip already completed problems")
    else:
        logger.info("Resume mode disabled - will run all problems")
    
    if args.sequential:
        run_all_experiments_sequential(resume)
    else:
        run_all_experiments_parallel(resume)
    
    logger.info("All experiments completed")

if __name__ == "__main__":
    main()
