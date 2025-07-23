#!/usr/bin/env python3
"""
Script to analyze combined results from multiple experiments.

This script analyzes combined results.json files produced by combine_n_results.py,
calculating metrics like pass@k and consensus correctness.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def get_attempts_key(problem: Dict[str, Any]) -> str:
    """Determine whether a problem uses 'solutions' or 'iterations' for its attempts."""
    if "solutions" in problem and isinstance(problem["solutions"], list):
        return "solutions"
    elif "iterations" in problem and isinstance(problem["iterations"], list):
        return "iterations"
    else:
        # Default to solutions if neither is found
        return "solutions"

def get_attempts(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get the list of attempts (solutions or iterations) from a problem."""
    key = get_attempts_key(problem)
    return problem.get(key, [])

def analyze_combined_results(combined_file_path: str):
    """
    Analyze combined results file, calculating metrics and producing visualizations.
    
    Args:
        combined_file_path: Path to the combined results JSON file
    """
    # Load the combined results file
    print(f"Loading combined results from {combined_file_path}...")
    with open(combined_file_path, 'r') as f:
        combined_data = json.load(f)

    # Extract all problems
    problems = combined_data.get("results", [])
    total_problems = len(problems)
    
    # Count problems by format
    solutions_format_count = sum(1 for p in problems if get_attempts_key(p) == "solutions")
    iterations_format_count = sum(1 for p in problems if get_attempts_key(p) == "iterations")
    
    print(f"Total problems: {total_problems}")
    print(f"  - With 'solutions' format: {solutions_format_count}")
    print(f"  - With 'iterations' format: {iterations_format_count}")

    # Calculate pass@k accuracy (at least one correct solution)
    pass_at_k_correct = sum(1 for problem in problems if problem.get("pass_at_k", False))
    pass_at_k_accuracy = pass_at_k_correct / total_problems if total_problems > 0 else 0

    # Calculate strong consensus (consensus_count >= 3)
    strong_consensus_correct = sum(1 for problem in problems if problem.get("consensus_count", 0) >= 3)
    strong_consensus_accuracy = strong_consensus_correct / total_problems if total_problems > 0 else 0

    # Calculate total attempts
    total_attempts = 0
    for problem in problems:
        attempts = get_attempts(problem)
        total_attempts += len(attempts)

    # Print the metrics
    print(f"Total attempts: {total_attempts}")
    print(f"Pass@k accuracy: {pass_at_k_correct}/{total_problems} = {pass_at_k_accuracy:.2%}")
    print(f"Strong consensus (count ≥ 3): {strong_consensus_correct}/{total_problems} = {strong_consensus_accuracy:.2%}")

    # Create a visualization of the metrics
    metrics = {
        'Metric': ['Pass@k', 'Strong Consensus (≥3)'],
        'Accuracy': [pass_at_k_accuracy, strong_consensus_accuracy],
        'Count': [f"{pass_at_k_correct}/{total_problems}", f"{strong_consensus_correct}/{total_problems}"]
    }

    df = pd.DataFrame(metrics)

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    bar = plt.bar(df['Metric'], df['Accuracy'], color=['#3498db', '#2ecc71'])
    plt.ylabel('Accuracy')
    plt.title('Performance Metrics for Combined Results')
    plt.ylim(0, 1.0)

    # Add value labels on top of bars
    for i, v in enumerate(df['Accuracy']):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
        plt.text(i, v/2, df['Count'][i], ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    print("Saved performance metrics visualization to 'performance_metrics.png'")
    plt.show()

    # Distribution of number of correct solutions per problem
    correct_counts = [problem.get("num_correct", 0) for problem in problems]
    max_correct = max(correct_counts) if correct_counts else 0

    plt.figure(figsize=(10, 6))
    plt.hist(correct_counts, bins=range(max_correct+2), align='left', rwidth=0.8)
    plt.xticks(range(max_correct+1))
    plt.xlabel('Number of Correct Attempts')
    plt.ylabel('Number of Problems')
    plt.title('Distribution of Correct Attempts per Problem')
    plt.tight_layout()
    plt.savefig('correct_distribution.png')
    print("Saved correct attempts distribution to 'correct_distribution.png'")
    plt.show()

    # Distribution of consensus counts
    consensus_counts = [problem.get("consensus_count", 0) for problem in problems]
    max_consensus = max(consensus_counts) if consensus_counts else 0
    
    plt.figure(figsize=(10, 6))
    plt.hist(consensus_counts, bins=range(max_consensus+2), align='left', rwidth=0.8)
    plt.xticks(range(max_consensus+1))
    plt.xlabel('Consensus Count (Number of Attempts Agreeing)')
    plt.ylabel('Number of Problems')
    plt.title('Distribution of Consensus Counts')
    plt.tight_layout()
    plt.savefig('consensus_distribution.png')
    print("Saved consensus distribution to 'consensus_distribution.png'")
    plt.show()

    return {
        'total_problems': total_problems,
        'total_attempts': total_attempts,
        'pass_at_k_accuracy': pass_at_k_accuracy,
        'strong_consensus_accuracy': strong_consensus_accuracy
    }

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze combined results from multiple experiments.")
    parser.add_argument("--input", "-i", default="combined_results.json", 
                        help="Input combined results file (default: combined_results.json)")
    
    args = parser.parse_args()
    
    analyze_combined_results(args.input)

if __name__ == "__main__":
    main()

# Jupyter notebook-friendly version
# To use in a notebook, copy and paste the following code:

"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def get_attempts_key(problem: Dict[str, Any]) -> str:
    \"\"\"Determine whether a problem uses 'solutions' or 'iterations' for its attempts.\"\"\"
    if "solutions" in problem and isinstance(problem["solutions"], list):
        return "solutions"
    elif "iterations" in problem and isinstance(problem["iterations"], list):
        return "iterations"
    else:
        # Default to solutions if neither is found
        return "solutions"

def get_attempts(problem: Dict[str, Any]) -> List[Dict[str, Any]]:
    \"\"\"Get the list of attempts (solutions or iterations) from a problem.\"\"\"
    key = get_attempts_key(problem)
    return problem.get(key, [])

# Load the combined results file
combined_file_path = "combined_harp_results.json"  # Update this path as needed

with open(combined_file_path, 'r') as f:
    combined_data = json.load(f)

# Extract all problems
problems = combined_data.get("results", [])
total_problems = len(problems)

# Count problems by format
solutions_format_count = sum(1 for p in problems if get_attempts_key(p) == "solutions")
iterations_format_count = sum(1 for p in problems if get_attempts_key(p) == "iterations")

print(f"Total problems: {total_problems}")
print(f"  - With 'solutions' format: {solutions_format_count}")
print(f"  - With 'iterations' format: {iterations_format_count}")

# Calculate pass@k accuracy (at least one correct solution)
pass_at_k_correct = sum(1 for problem in problems if problem.get("pass_at_k", False))
pass_at_k_accuracy = pass_at_k_correct / total_problems if total_problems > 0 else 0

# Calculate strong consensus (consensus_count >= 3)
strong_consensus_correct = sum(1 for problem in problems if problem.get("consensus_count", 0) >= 3)
strong_consensus_accuracy = strong_consensus_correct / total_problems if total_problems > 0 else 0

# Calculate total attempts
total_attempts = 0
for problem in problems:
    attempts = get_attempts(problem)
    total_attempts += len(attempts)

# Print the metrics
print(f"Total attempts: {total_attempts}")
print(f"Pass@k accuracy: {pass_at_k_correct}/{total_problems} = {pass_at_k_accuracy:.2%}")
print(f"Strong consensus (count ≥ 3): {strong_consensus_correct}/{total_problems} = {strong_consensus_accuracy:.2%}")

# Create a visualization of the metrics
metrics = {
    'Metric': ['Pass@k', 'Strong Consensus (≥3)'],
    'Accuracy': [pass_at_k_accuracy, strong_consensus_accuracy],
    'Count': [f"{pass_at_k_correct}/{total_problems}", f"{strong_consensus_correct}/{total_problems}"]
}

df = pd.DataFrame(metrics)

# Plot bar chart
plt.figure(figsize=(10, 6))
bar = plt.bar(df['Metric'], df['Accuracy'], color=['#3498db', '#2ecc71'])
plt.ylabel('Accuracy')
plt.title('Performance Metrics for Combined Results')
plt.ylim(0, 1.0)

# Add value labels on top of bars
for i, v in enumerate(df['Accuracy']):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    plt.text(i, v/2, df['Count'][i], ha='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# Distribution of number of correct solutions per problem
correct_counts = [problem.get("num_correct", 0) for problem in problems]
max_correct = max(correct_counts) if correct_counts else 0

plt.figure(figsize=(10, 6))
plt.hist(correct_counts, bins=range(max_correct+2), align='left', rwidth=0.8)
plt.xticks(range(max_correct+1))
plt.xlabel('Number of Correct Attempts')
plt.ylabel('Number of Problems')
plt.title('Distribution of Correct Attempts per Problem')
plt.tight_layout()
plt.show()

# Distribution of consensus counts
consensus_counts = [problem.get("consensus_count", 0) for problem in problems]
max_consensus = max(consensus_counts) if consensus_counts else 0

plt.figure(figsize=(10, 6))
plt.hist(consensus_counts, bins=range(max_consensus+2), align='left', rwidth=0.8)
plt.xticks(range(max_consensus+1))
plt.xlabel('Consensus Count (Number of Attempts Agreeing)')
plt.ylabel('Number of Problems')
plt.title('Distribution of Consensus Counts')
plt.tight_layout()
plt.show()
""" 