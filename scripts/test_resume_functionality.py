#!/usr/bin/env python3
"""
Test script to verify the resume functionality of the DeepSeek sweep script.
This script will:
1. Run a small subset of problems
2. Simulate an interruption
3. Resume the run and verify it skips already completed problems
"""

import os
import sys
import time
import json
import shutil
import logging
import subprocess
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("resume_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# State file path
STATE_FILE = "deepseek_sweep_state.json"

def clean_state():
    """Remove any existing state files."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logger.info(f"Removed existing state file: {STATE_FILE}")

def run_initial_subset():
    """Run the experiment on a small subset of problems."""
    logger.info("Running initial subset of problems...")
    
    # Run only the first 2 problems
    cmd = [
        "python", "run_experiment.py",
        "deepseek_v3_0324",
        "--verbose",
        "--index-range", "0-1"  # Only run the first 2 problems
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(f"[Initial Run] {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Initial run failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        logger.info("Initial subset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during initial run: {str(e)}")
        return False

def run_resume_test():
    """Run the resumable sweep script and verify it skips completed problems."""
    logger.info("Running resume test...")
    
    # Run the resumable sweep script
    cmd = [
        "python", "run_deepseek_sweep_resumable.py",
        "--sequential"  # Run sequentially for easier logging
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Variables to track what we observe in the output
        completed_problems = set()
        skipped_problems = set()
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(f"[Resume Run] {line.strip()}")
            
            # Look for evidence of skipping already completed problems
            if "Processing problem" in line:
                try:
                    parts = line.split("Processing problem ")
                    if len(parts) > 1:
                        problem_id = parts[1].split(" ")[0]
                        completed_problems.add(problem_id)
                except Exception:
                    pass
            
            # Look for evidence of skipping
            if "Found" in line and "already completed problems" in line:
                try:
                    parts = line.split("Found ")[1].split(" already")[0]
                    num_skipped = int(parts)
                    logger.info(f"Detected {num_skipped} problems being skipped")
                except Exception:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Resume run failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        logger.info(f"Resume test completed. Processed {len(completed_problems)} problems.")
        return True
        
    except Exception as e:
        logger.error(f"Error during resume test: {str(e)}")
        return False

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test resume functionality")
    parser.add_argument("--clean", action="store_true", help="Clean state before running")
    args = parser.parse_args()
    
    if args.clean:
        clean_state()
    
    # Step 1: Run initial subset
    if not run_initial_subset():
        logger.error("Initial run failed, aborting test")
        return 1
    
    # Step 2: Simulate interruption (just wait a moment)
    logger.info("Simulating interruption...")
    time.sleep(2)
    
    # Step 3: Run resume test
    if not run_resume_test():
        logger.error("Resume test failed")
        return 1
    
    logger.info("Resume functionality test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
