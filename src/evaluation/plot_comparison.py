import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def load_results(results_path: str):
    """Load results from a JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_model_name(path_str: str):
    """Extract model name from path."""
    # Look for model size pattern like 1p5b, 7b, 14b, 32b
    match = re.search(r'qwen_(\d+(?:p\d+)?b)', path_str.lower())
    if match:
        size = match.group(1)
        # Format size nicely
        if 'p' in size:
            size = size.replace('p', '.')
        return f"DeepSeek Qwen-{size.upper()}"
    return Path(path_str).parent.name

def calculate_per_iteration_accuracy(results):
    """Calculate the accuracy for each iteration (not cumulative)."""
    # Count the total number of problems
    total_problems = len(results['results'])
    
    # Find the maximum number of iterations across all problems
    max_iterations = max(len(problem['iterations']) for problem in results['results'])
    
    # Initialize counters for each iteration
    iteration_correct = {i: 0 for i in range(max_iterations)}
    iteration_total = {i: 0 for i in range(max_iterations)}
    
    # Process each problem
    for problem in results['results']:
        iterations = problem['iterations']
        
        # Process each iteration
        for i in range(max_iterations):
            # If we have data for this iteration
            if i < len(iterations):
                iteration = iterations[i]
                is_correct = iteration.get('correct', False)
                
                if is_correct:
                    iteration_correct[i] += 1
                
                iteration_total[i] += 1
    
    # Calculate accuracy for each iteration
    accuracies = {i: iteration_correct[i] / total_problems for i in range(max_iterations)}
    
    return accuracies, iteration_correct, iteration_total, total_problems, max_iterations

def plot_comparison(model_accuracies, output_path=None):
    """Generate a comparison plot of accuracy vs iteration for multiple models."""
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot data for each model
    for i, (model_name, data) in enumerate(model_accuracies.items()):
        iterations = sorted(data['accuracies'].keys())
        accuracy_values = [data['accuracies'][iter_idx] for iter_idx in iterations]
        
        plt.plot(iterations, accuracy_values, marker='o', linestyle='-', 
                 color=colors[i % len(colors)], linewidth=2, markersize=8,
                 label=f"{model_name}")
    
    # Set labels and title
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy on AIME 2024 & 2025 by Iteration', fontsize=16)
    
    # Set grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits from 0 to 1
    plt.ylim(0, 1.0)
    
    # Set x-axis ticks
    max_iters = max(max(data['accuracies'].keys()) for data in model_accuracies.values())
    plt.xticks(range(max_iters + 1))
    
    # Add legend with larger font
    plt.legend(fontsize=12, loc='lower right')
    
    # Tight layout to make sure everything fits
    plt.tight_layout()
    
    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot accuracy comparison for multiple models')
    parser.add_argument('results_paths', nargs='+', help='Paths to results.json files')
    parser.add_argument('--output', default='model_comparison.png', help='Path to save the plot')
    args = parser.parse_args()
    
    # Store data for all models
    model_accuracies = {}
    
    # Process each results file
    for path in args.results_paths:
        # Extract model name from path
        model_name = extract_model_name(path)
        print(f"Processing {model_name} from {path}")
        
        # Load results
        results = load_results(path)
        
        # Calculate per-iteration accuracy
        accuracies, iteration_correct, iteration_total, total_problems, max_iterations = calculate_per_iteration_accuracy(results)
        
        # Print the results
        print(f"  Total problems: {total_problems}")
        print(f"  Max iterations: {max_iterations}")
        print("  Accuracy by iteration:")
        for i, acc in sorted(accuracies.items()):
            print(f"    Iteration {i}: {acc:.4f} ({iteration_correct[i]}/{total_problems})")
        
        # Store data for this model
        model_accuracies[model_name] = {
            'accuracies': accuracies,
            'total_problems': total_problems,
            'max_iterations': max_iterations
        }
    
    # Generate comparison plot
    plot_comparison(model_accuracies, args.output)

if __name__ == '__main__':
    main() 