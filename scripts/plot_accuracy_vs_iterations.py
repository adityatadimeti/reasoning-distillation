import json
import argparse
import matplotlib.pyplot as plt
import os

def compute_accuracy(results_path):
    """
    Compute accuracy at each iteration index.
    Accuracy = (# correct at iteration) / (total problems)
    """
    with open(results_path, 'r') as f:
        data = json.load(f)
    results = data.get('results', [])
    if not results:
        print("No results found in file.")
        return []

    total = len(results)
    # Determine maximum number of iterations across all problems
    max_iters = max(len(p.get('iterations', [])) for p in results)
    accuracies = []
    for i in range(max_iters):
        correct_count = 0
        for p in results:
            iters = p.get('iterations', [])
            if i < len(iters) and iters[i].get('correct', False):
                correct_count += 1
        accuracies.append(correct_count / total)
    return accuracies


def plot_accuracy(accuracies, output_path=None):
    """
    Plot accuracy vs iteration and save or show the figure.
    """
    iterations = list(range(len(accuracies)))
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, accuracies, marker='o', linestyle='-')
    plt.title('Accuracy vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xticks(iterations)
    plt.ylim(0, 1)
    plt.grid(True)

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved accuracy plot to {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot performance (accuracy) vs iterations from a results JSON file'
    )
    parser.add_argument(
        'results_path', type=str,
        help='Path to the results JSON file'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='accuracy_vs_iterations.png',
        help='Output path for the plot image'
    )
    args = parser.parse_args()

    acc = compute_accuracy(args.results_path)
    if acc:
        plot_accuracy(acc, args.output)
    else:
        print('No accuracies computed.')
