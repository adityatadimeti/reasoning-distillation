#!/usr/bin/env python3
"""
Test script for DeepSeek V3-0324 with resume functionality.
This script will:
1. Run a small test with just 1-2 problems
2. Verify the results are saved correctly
3. Test that running again will skip already completed problems
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deepseek_v3_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = "deepseek_v3_0324"
STATE_FILE = "deepseek_sweep_state.json"

def clean_state():
    """Remove existing state file if it exists."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logger.info(f"Removed existing state file: {STATE_FILE}")

def run_initial_test(index_range="0-0"):
    """Run an initial test with a single problem."""
    logger.info(f"Running initial test with index range: {index_range}")
    
    cmd = [
        "python", "run_experiment.py",
        CONFIG,
        "--parallel",
        "--concurrency", "1",  # Just use 1 for the test
        "--verbose",
        "--index-range", index_range
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
            logger.info(f"[Initial Test] {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Initial test failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        logger.info("Initial test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during initial test: {str(e)}")
        return False

def find_results_dir():
    """Find the most recent results directory for the experiment."""
    results_dirs = list(Path("results").glob(f"{CONFIG}_*"))
    if not results_dirs:
        logger.error("No results directory found")
        return None
    
    # Sort by creation time (newest first)
    results_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return results_dirs[0]

def verify_results():
    """Verify that results were saved correctly."""
    results_dir = find_results_dir()
    if not results_dir:
        return False
    
    results_file = results_dir / "results.json"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return False
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Check that we have results
        if not data.get("results"):
            logger.error("No results found in results file")
            return False
        
        logger.info(f"Found {len(data['results'])} results in {results_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error reading results file: {e}")
        return False

def test_resume():
    """Test that running again will skip already completed problems."""
    logger.info("Testing resume functionality...")
    
    # Run with a slightly larger range that includes the already completed problem
    cmd = [
        "python", "run_deepseek_sweep_resumable.py",
        "--sequential"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Variables to track what we observe
        found_completed = False
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(f"[Resume Test] {line.strip()}")
            
            # Look for evidence of finding completed problems
            if "Found" in line and "already completed problems" in line:
                found_completed = True
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Resume test failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
        
        if found_completed:
            logger.info("Resume functionality working: detected already completed problems")
        else:
            logger.warning("Resume functionality might not be working: didn't detect already completed problems")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during resume test: {e}")
        return False

def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test DeepSeek V3-0324 with resume functionality")
    parser.add_argument("--clean", action="store_true", help="Clean state before running")
    parser.add_argument("--index-range", type=str, default="0-0", help="Index range for initial test")
    args = parser.parse_args()
    
    if args.clean:
        clean_state()
    
    # Step 1: Run initial test
    if not run_initial_test(args.index_range):
        logger.error("Initial test failed, aborting")
        return 1
    
    # Step 2: Verify results
    if not verify_results():
        logger.error("Results verification failed, aborting")
        return 1
    
    # Step 3: Test resume functionality
    if not test_resume():
        logger.error("Resume test failed")
        return 1
    
    logger.info("All tests completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
