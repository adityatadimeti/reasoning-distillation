"""
Test script to verify that run_experiment.py can run with subsets of questions.
This script will test both index range and question ID filtering.
"""

import subprocess
import os
import sys

def run_command(cmd):
    """Run a command and print its output."""
    print(f"\nRunning command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("Command output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Test 1: Run with index range (first 2 questions)
    print("\n========== TEST 1: Run with index range 0-1 ==========")
    cmd1 = [
        "python", "run_experiment.py", 
        "gpqa_diamond", 
        "--index-range", "0-1",
        "--verbose"
    ]
    if not run_command(cmd1):
        print("Test 1 failed")
        return 1
    
    # Test 2: Run with specific question IDs
    print("\n========== TEST 2: Run with specific question IDs ==========")
    cmd2 = [
        "python", "run_experiment.py", 
        "gpqa_diamond", 
        "--question-ids", "c93b1e31,fd019ec3",
        "--verbose"
    ]
    if not run_command(cmd2):
        print("Test 2 failed")
        return 1
    
    print("\nBoth tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 