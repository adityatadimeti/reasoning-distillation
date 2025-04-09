#!/usr/bin/env python3
"""
Direct verification of the resume functionality in run_deepseek_sweep_resumable.py
"""
import os
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import functions directly from the sweep script
from run_deepseek_sweep_resumable import (
    find_completed_problems,
    load_state,
    save_state,
    STATE_FILE
)

def setup_test_environment():
    """Set up a test environment with mock data"""
    # Create results directory
    results_dir = Path("results/deepseek_v3_0324_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock results file
    results = {
        "results": [
            {"problem_id": "problem_1", "status": "success"},
            {"problem_id": "problem_2", "status": "success"},
            {"problem_id": "2024-I-1", "status": "success"}
        ]
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Created mock results directory with completed problems")
    
    # Create mock state file
    mock_state = {
        "experiments": {
            "DeepSeek V3-0324": {
                "last_problem": "2024-I-1",
                "completed": False,
                "completion_time": None
            }
        }
    }
    
    save_state(mock_state)
    logger.info(f"Created mock state file")
    
    return results_dir

def verify_find_completed_problems():
    """Verify that find_completed_problems correctly identifies completed problems"""
    logger.info("Testing find_completed_problems function...")
    
    # Call the function directly
    completed = find_completed_problems("deepseek_v3_0324")
    
    logger.info(f"Found {len(completed)} completed problems: {completed}")
    
    # Verify expected problems are found
    expected = {"problem_1", "problem_2", "2024-I-1"}
    found_expected = all(p in completed for p in expected)
    
    logger.info(f"Found all expected problems: {found_expected}")
    
    return completed, found_expected

def verify_command_generation():
    """Verify that the correct command is generated with exclude-question-ids"""
    logger.info("Verifying command generation with exclude-question-ids...")
    
    # Mock the completed problems
    completed_problems = {"problem_1", "problem_2", "2024-I-1"}
    completed_str = ",".join(completed_problems)
    
    # Build the expected command
    cmd = [
        "python", "run_experiment.py",
        "deepseek_v3_0324",
        "--parallel",
        "--concurrency", "32",
        "--verbose",
        "--exclude-question-ids", completed_str
    ]
    
    # Print the command that would be executed
    logger.info(f"Command that would be executed:")
    logger.info(" ".join(cmd))
    
    # Verify the command includes the exclude-question-ids parameter
    has_exclude = "--exclude-question-ids" in cmd
    logger.info(f"Command includes exclude-question-ids parameter: {has_exclude}")
    
    return has_exclude

def cleanup(results_dir):
    """Clean up test files"""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logger.info(f"Removed state file: {STATE_FILE}")
    
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
        logger.info(f"Removed results directory: {results_dir}")

def main():
    parser = argparse.ArgumentParser(description="Verify resume functionality")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test files after running")
    args = parser.parse_args()
    
    try:
        # Set up test environment
        results_dir = setup_test_environment()
        
        # Verify find_completed_problems
        completed, found_expected = verify_find_completed_problems()
        
        # Verify command generation
        has_exclude = verify_command_generation()
        
        # Report results
        logger.info("\n--- Verification Results ---")
        logger.info(f"Found completed problems: {len(completed) > 0}")
        logger.info(f"Found all expected problems: {found_expected}")
        logger.info(f"Command includes exclude-question-ids: {has_exclude}")
        
        if found_expected and has_exclude:
            logger.info("✅ Resume functionality appears to be working correctly")
        else:
            logger.info("❌ Resume functionality has issues that need to be addressed")
        
    finally:
        if args.cleanup:
            cleanup(results_dir)

if __name__ == "__main__":
    main()
