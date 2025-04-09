#!/usr/bin/env python3
"""
Simplified test for the core resume functionality
"""
import os
import json
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# State file path
STATE_FILE = "deepseek_sweep_state.json"

def setup_mock_state():
    """Create a mock state file with some completed problems"""
    # Create a mock state with completed problems
    mock_state = {
        "experiments": {
            "DeepSeek V3-0324": {
                "last_problem": "problem_1",
                "completed_problems": ["problem_1", "problem_2"]
            }
        }
    }
    
    with open(STATE_FILE, "w") as f:
        json.dump(mock_state, f, indent=2)
    
    logger.info(f"Created mock state file with completed problems")
    return mock_state

def setup_mock_results_dir():
    """Create a mock results directory with completed problems"""
    # Create results directory
    results_dir = Path("results/deepseek_v3_0324_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mock results file
    results = {
        "results": [
            {"problem_id": "problem_1", "status": "success"},
            {"problem_id": "problem_2", "status": "success"}
        ]
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Created mock results directory with completed problems")
    return results

def test_find_completed_problems():
    """Test the find_completed_problems function directly"""
    # Run a Python command to execute the function
    cmd = [
        "python", "-c", 
        "from run_deepseek_sweep_resumable import find_completed_problems; "
        "import json; "
        "completed = find_completed_problems('deepseek_v3_0324'); "
        "print(json.dumps(list(completed)))"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        completed = json.loads(result.stdout.strip())
        logger.info(f"Found {len(completed)} completed problems: {completed}")
        return completed
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running find_completed_problems: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return []

def test_exclude_completed():
    """Test that the script correctly excludes completed problems"""
    # Run the script with --no-resume to check it doesn't exclude problems
    cmd = [
        "python", "run_deepseek_sweep_resumable.py",
        "--sequential",
        "--no-resume"
    ]
    
    try:
        # Just run with a timeout to see the initial output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        logger.info("Script started without resume, checking output...")
        
        # Check if it's trying to exclude problems
        if "Found" in result.stdout and "already completed problems" in result.stdout:
            logger.error("Script is excluding problems even with --no-resume flag")
            return False
    except subprocess.TimeoutExpired:
        # This is expected, we just wanted to see the initial output
        pass
    
    # Now run with resume to check it does exclude problems
    cmd = [
        "python", "run_deepseek_sweep_resumable.py",
        "--sequential"
    ]
    
    try:
        # Just run with a timeout to see the initial output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        logger.info("Script started with resume, checking output...")
        
        # Check if it's excluding problems
        if "Found" in result.stdout and "already completed problems" in result.stdout:
            logger.info("Script correctly excludes completed problems")
            return True
        else:
            logger.warning("Script doesn't seem to be excluding completed problems")
            return False
    except subprocess.TimeoutExpired:
        # This is expected, we just wanted to see the initial output
        pass
    
    return False

def cleanup():
    """Clean up test files"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    
    results_dir = Path("results/deepseek_v3_0324_test")
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    
    logger.info("Cleaned up test files")

def main():
    parser = argparse.ArgumentParser(description="Test core resume functionality")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files after running")
    args = parser.parse_args()
    
    try:
        # Setup test environment
        setup_mock_state()
        setup_mock_results_dir()
        
        # Test finding completed problems
        completed = test_find_completed_problems()
        
        # Test excluding completed problems
        exclude_works = test_exclude_completed()
        
        # Report results
        logger.info("\n--- Test Results ---")
        logger.info(f"Found completed problems: {len(completed) > 0}")
        logger.info(f"Exclude completed works: {exclude_works}")
        logger.info("Resume functionality appears to be working correctly" if exclude_works else "Resume functionality may not be working correctly")
        
    finally:
        if args.cleanup:
            cleanup()

if __name__ == "__main__":
    main()
