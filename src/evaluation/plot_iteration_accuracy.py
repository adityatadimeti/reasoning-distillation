import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_path: str):
    """Load results from a JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def calculate_per_iteration_accuracy(results):
    """Calculate the accuracy for each iteration (not cumulative)."""
    # Count the total number of problems
    total_problems = len(results['results'])
    print(f"Total problems: {total_problems}")
    
    # Find the maximum number of iterations across all problems
    max_iterations = max(len(problem['iterations']) for problem in results['results'])
    print(f"Max iterations: {max_iterations}")
    
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
                has_answer = iteration.get('answer') is not None
                
                if not has_answer:
                    print(f"Problem {problem['problem_id']} has no answer for iteration {i}")
                
                if is_correct:
                    iteration_correct[i] += 1
                
                iteration_total[i] += 1
    
    # Calculate accuracy for each iteration
    accuracies = {i: iteration_correct[i] / total_problems for i in range(max_iterations)}
    
    # Print out iteration counts for verification
    for i in range(max_iterations):
        print(f"Iteration {i}: {iteration_total[i]} problems processed, {iteration_correct[i]} correct")
    
    return accuracies, iteration_correct, iteration_total, total_problems

def plot_iteration_accuracy(accuracies, output_path=None):
    """Generate a plot of accuracy vs iteration."""
    iterations = sorted(accuracies.keys())
    accuracy_values = [accuracies[i] for i in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracy_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Iteration')
    plt.xticks(iterations)
    plt.ylim(0, max(accuracy_values) * 1.2)  # Add some space above the highest point
    plt.grid(True)
    
    # Add values above each point
    for i, acc in zip(iterations, accuracy_values):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot accuracy vs iteration')
    parser.add_argument('results_path', help='Path to results.json file')
    parser.add_argument('--output', default=None, help='Path to save the plot')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_path)
    
    # Calculate per-iteration accuracy
    accuracies, iteration_correct, iteration_total, total_problems = calculate_per_iteration_accuracy(results)
    
    # Print the results
    print("\nAccuracy by iteration:")
    for i, acc in sorted(accuracies.items()):
        print(f"Iteration {i}: {acc:.4f} ({iteration_correct[i]}/{total_problems})")
    
    # Generate the plot
    output_path = args.output
    if not output_path:
        # Create default output path based on input file
        input_path = Path(args.results_path)
        output_dir = input_path.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'{input_path.stem}_iteration_accuracy.png'
    
    plot_iteration_accuracy(accuracies, output_path)

if __name__ == '__main__':
    main() 