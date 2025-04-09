#!/usr/bin/env python3
"""
Quick test for resume functionality
"""
import os
import json
import argparse

# State file path
STATE_FILE = "deepseek_sweep_state.json"

def create_test_state():
    """Create a test state file with some completed problems"""
    state = {
        "completed_problems": {
            "problem_1": {"timestamp": "2025-04-05T15:00:00"},
            "problem_2": {"timestamp": "2025-04-05T15:01:00"}
        }
    }
    
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    
    print(f"Created test state file with {len(state['completed_problems'])} completed problems")

def read_state():
    """Read and display the current state"""
    if not os.path.exists(STATE_FILE):
        print("No state file exists")
        return
    
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    
    completed = state.get("completed_problems", {})
    print(f"Found {len(completed)} completed problems in state file:")
    for problem_id, data in completed.items():
        print(f"  - {problem_id}: {data.get('timestamp', 'unknown')}")

def main():
    parser = argparse.ArgumentParser(description="Quick resume functionality test")
    parser.add_argument("--create", action="store_true", help="Create a test state file")
    parser.add_argument("--read", action="store_true", help="Read the current state file")
    parser.add_argument("--clean", action="store_true", help="Remove the state file")
    
    args = parser.parse_args()
    
    if args.clean and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"Removed state file: {STATE_FILE}")
    
    if args.create:
        create_test_state()
    
    if args.read:
        read_state()

if __name__ == "__main__":
    main()
