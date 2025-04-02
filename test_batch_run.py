"""
Simple test script to verify the batch runner works as expected.
This will run a very small test with just a few questions.
"""

import subprocess
import sys

def main():
    print("Testing batch run with 2 tiny batches of 2 questions each")
    
    # Run the first 4 questions in batches of 2
    cmd = [
        "python", "run_in_batches.py",
        "gpqa_diamond",
        "--batch-size", "2",
        "--end-idx", "3",  # Process indices 0-3 (first 4 questions)
        "--delay", "1",  # Just 1 second delay for testing
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("Test completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Test failed with code {e.returncode}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 