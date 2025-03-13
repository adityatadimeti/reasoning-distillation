import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(results_path: str) -> Dict:
    """Load results from a JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_iterations(results: Dict) -> Tuple[Dict, Dict, List, Dict, Dict, Dict]:
    """Analyze the effectiveness of iterations in improving answer correctness."""
    
    # Track metrics
    iteration_accuracy = defaultdict(lambda: {'correct': set(), 'total': 0})
    problem_progression = {'improved': [], 'regressed': []}
    iterations_to_correct = []
    correct_distribution = defaultdict(int)
    problem_details = {}
    missing_answers = defaultdict(list)  # Track problems with missing answers per iteration
    
    total_problems = len(results['results'])
    max_iterations = max(len(problem['iterations']) for problem in results['results'])
    
    for problem in results['results']:
        problem_id = problem['problem_id']
        iterations = problem['iterations']
        
        # Track accuracy per iteration
        initial_correct = None
        found_correct = False
        iterations_needed = -1
        answer_status = []
        
        # For each possible iteration (even if this problem didn't reach it)
        for i in range(max_iterations):
            # Count total problems for each iteration
            iteration_accuracy[i]['total'] = total_problems
            
            # If we have data for this iteration
            if i < len(iterations):
                iteration = iterations[i]
                is_correct = iteration.get('correct', False)
                has_answer = iteration.get('answer') is not None
                
                if not has_answer:
                    missing_answers[i].append(problem_id)
                
                if is_correct and not found_correct:
                    found_correct = True
                    iterations_needed = i
                    correct_distribution[i] += 1
                
                if i == 0:
                    initial_correct = is_correct
                
                answer_status.append({
                    'has_answer': has_answer,
                    'answer': iteration.get('answer'),
                    'is_correct': is_correct
                })
            
            # If problem was correct in this or any previous iteration
            if found_correct:
                iteration_accuracy[i]['correct'].add(problem_id)
        
        # Track problem progression
        if iterations:
            final_correct = found_correct
            if not initial_correct and final_correct:
                problem_progression['improved'].append(problem_id)
            elif initial_correct and not final_correct:
                problem_progression['regressed'].append(problem_id)
        
        # Store iterations needed
        if found_correct:
            iterations_to_correct.append(iterations_needed)
        
        # Store problem details
        problem_details[problem_id] = {
            'initial_correct': initial_correct,
            'final_correct': final_correct if iterations else None,
            'iterations_to_correct': iterations_needed,
            'num_iterations': len(iterations),
            'cumulative_correct': [i for i in range(max_iterations) 
                                 if i >= iterations_needed and iterations_needed != -1],
            'answer_status': answer_status
        }
    
    # Convert sets to counts for the final report
    iteration_accuracy_counts = {
        i: {
            'correct': len(acc['correct']),
            'total': acc['total']
        } for i, acc in iteration_accuracy.items()
    }
    
    # Convert missing_answers to counts per iteration
    missing_answers_summary = {
        i: {
            'count': len(problems),
            'problems': problems
        } for i, problems in missing_answers.items()
    }
    
    return (iteration_accuracy_counts, problem_progression, 
            iterations_to_correct, correct_distribution, 
            problem_details, missing_answers_summary)

def plot_metrics(iteration_accuracy: Dict, correct_distribution: Dict, 
                missing_answers: Dict, output_dir: str):
    """Generate plots for the metrics."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot accuracy per iteration
    iterations = sorted(iteration_accuracy.keys())
    accuracies = [iteration_accuracy[i]['correct'] / iteration_accuracy[i]['total'] 
                  for i in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o')
    plt.title('Cumulative Accuracy by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Accuracy')
    plt.grid(True)
    plt.savefig(f'{output_dir}/accuracy_by_iteration.png')
    plt.close()
    
    # Plot distribution of first correct answers
    iterations = sorted(correct_distribution.keys())
    counts = [correct_distribution[i] for i in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.bar(iterations, counts)
    plt.title('Distribution of First Correct Answers')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{output_dir}/correct_distribution.png')
    plt.close()
    
    # Plot missing answers by iteration
    iterations = sorted(missing_answers.keys())
    counts = [missing_answers[i]['count'] for i in iterations]
    
    plt.figure(figsize=(10, 6))
    plt.bar(iterations, counts, color='red', alpha=0.6)
    plt.title('Missing Answers by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{output_dir}/missing_answers.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze iteration effectiveness')
    parser.add_argument('results_path', help='Path to results.json file')
    parser.add_argument('--output-dir', default='eval_output',
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_path)
    
    # Analyze iterations
    (iteration_accuracy, problem_progression, iterations_to_correct,
     correct_distribution, problem_details, missing_answers) = analyze_iterations(results)
    
    # Calculate summary statistics
    total_problems = len(results['results'])
    avg_iterations_to_correct = (sum(iterations_to_correct) / len(iterations_to_correct) 
                               if iterations_to_correct else 0)
    
    # Generate report
    report = {
        'total_problems': total_problems,
        'iteration_accuracy': {i: {
            'accuracy': acc['correct'] / acc['total'],
            'correct': acc['correct'],
            'total': acc['total']
        } for i, acc in iteration_accuracy.items()},
        'problems_improved': len(problem_progression['improved']),
        'problems_regressed': len(problem_progression['regressed']),
        'avg_iterations_to_correct': avg_iterations_to_correct,
        'correct_distribution': correct_distribution,
        'problem_details': problem_details,
        'missing_answers': missing_answers
    }
    
    # Save report
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{args.output_dir}/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate plots
    plot_metrics(iteration_accuracy, correct_distribution, missing_answers, args.output_dir)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total problems analyzed: {total_problems}")
    print("\nCumulative accuracy by iteration:")
    for i, acc in sorted(iteration_accuracy.items()):
        accuracy = acc['correct'] / acc['total']
        print(f"Iteration {i}: {accuracy:.2%} ({acc['correct']}/{acc['total']})")
    
    if len(iteration_accuracy) > 1:
        improvement = (iteration_accuracy[1]['correct'] - iteration_accuracy[0]['correct']) / total_problems
        print(f"\nAbsolute improvement from iteration 0 to 1: {improvement:.2%}")
    
    print(f"\nProblems that improved after iteration 0: {len(problem_progression['improved'])}")
    print(f"Problems that regressed after iteration 0: {len(problem_progression['regressed'])}")
    
    if iterations_to_correct:
        print(f"\nAverage iterations to first correct answer: {avg_iterations_to_correct:.2f}")
    
    print("\nDistribution of first correct answers:")
    for i, count in sorted(correct_distribution.items()):
        print(f"Iteration {i}: {count} problems")
    
    print("\nMissing answers by iteration:")
    for i, data in sorted(missing_answers.items()):
        print(f"Iteration {i}: {data['count']} problems")
        if data['count'] > 0:
            print(f"  Problem IDs: {', '.join(data['problems'])}")

if __name__ == '__main__':
    main() 