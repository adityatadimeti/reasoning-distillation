import pandas as pd
import json
import os
import glob
from collections import Counter

def categorize_problem(iterations, use_summary=False):
    """
    Categorize a problem based on its iterations
    
    Parameters:
    - iterations: List of iteration data
    - use_summary: If True, use summary answers for categorization instead of regular answers
    
    Returns:
    - category: Category name
    - first_correct: Whether the first iteration was correct
    - last_correct: Whether the last iteration was correct
    """
    # Track correctness patterns across iterations
    first_iteration_correct = False
    any_later_iteration_correct = False
    any_later_iteration_incorrect = False
    all_correct = True
    all_incorrect = True
    last_iteration_correct = False
    
    # Analyze each iteration
    for i, iteration in enumerate(iterations):
        # Determine which answer/correctness to use based on the use_summary flag
        if use_summary and 'summary_correct' in iteration:
            is_correct = iteration.get('summary_correct', False)
        else:
            is_correct = iteration.get('correct', False)
        
        # Update tracking variables
        if i == 0:
            first_iteration_correct = is_correct
        else:
            if is_correct:
                any_later_iteration_correct = True
            if not is_correct:
                any_later_iteration_incorrect = True
        
        # Track last iteration
        if i == len(iterations) - 1:
            last_iteration_correct = is_correct
        
        if is_correct:
            all_incorrect = False
        if not is_correct:
            all_correct = False
    
    # Determine category based on correctness pattern
    category = None
    if all_correct:
        category = 'all_correct'
    elif all_incorrect:
        category = 'all_incorrect'
    elif not first_iteration_correct and any_later_iteration_correct:
        if last_iteration_correct:
            category = 'improved_final_correct'
        else:
            category = 'improved_final_incorrect'
    elif first_iteration_correct and any_later_iteration_incorrect:
        if last_iteration_correct:
            category = 'regressed_final_correct'
        else:
            category = 'regressed_final_incorrect'
    
    return category, first_iteration_correct, last_iteration_correct

def find_category_differences(results_path):
    """
    Find problems that have different categorizations when using regular vs summary answers
    
    Parameters:
    - results_path: Path to the results JSON file
    
    Returns:
    - diff_problems: List of problems with different categorizations
    """
    # Load the results JSON
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract the results list
    results = data.get('results', [])
    
    categories = {
        'all_correct': '🟩 All Correct',
        'all_incorrect': '🟥 All Incorrect',
        'improved_final_incorrect': '🟦 Improved (Final Incorrect)',
        'improved_final_correct': '👑 Improved (Final Correct)',
        'regressed_final_incorrect': '🟪 Regressed (Final Incorrect)',
        'regressed_final_correct': '🧼 Regressed (Final Correct)'
    }
    
    diff_problems = []
    
    for problem in results:
        problem_id = problem.get('problem_id', 'Unknown')
        iterations = problem.get('iterations', [])
        
        if not iterations:
            continue
        
        # Categorize using regular answers
        regular_category, regular_first_correct, regular_last_correct = categorize_problem(iterations, use_summary=False)
        
        # Categorize using summary answers
        summary_category, summary_first_correct, summary_last_correct = categorize_problem(iterations, use_summary=True)
        
        # Check if categorizations differ
        if regular_category != summary_category:
            # Find the specific iterations that caused the difference
            iteration_differences = []
            for i, iteration in enumerate(iterations):
                regular_correct = iteration.get('correct', False)
                summary_correct = iteration.get('summary_correct', False) if 'summary_correct' in iteration else regular_correct
                
                if regular_correct != summary_correct:
                    iteration_differences.append({
                        'iteration': i,
                        'regular_answer': iteration.get('answer', 'No answer'),
                        'summary_answer': iteration.get('summary_answer', 'No answer') if 'summary_answer' in iteration else iteration.get('answer', 'No answer'),
                        'regular_correct': regular_correct,
                        'summary_correct': summary_correct
                    })
            
            diff_problems.append({
                'problem_id': problem_id,
                'regular_category': regular_category,
                'regular_category_display': categories.get(regular_category, 'Uncategorized'),
                'summary_category': summary_category,
                'summary_category_display': categories.get(summary_category, 'Uncategorized'),
                'iteration_differences': iteration_differences,
                'iterations': iterations
            })
    
    # Print the differences
    print(f"\nFound {len(diff_problems)} problems with different categorizations:")
    for problem in diff_problems:
        print(f"\nProblem {problem['problem_id']}:")
        print(f"  Regular category: {problem['regular_category_display']}")
        print(f"  Summary category: {problem['summary_category_display']}")
        
        print("  Iteration sequence (regular):")
        for i, iteration in enumerate(problem['iterations']):
            is_correct = iteration.get('correct', False)
            answer = iteration.get('answer', 'No answer')
            print(f"    Iteration {i}: {'✅' if is_correct else '❌'} {answer}")
        
        print("  Iteration sequence (summary):")
        for i, iteration in enumerate(problem['iterations']):
            if 'summary_correct' in iteration:
                is_correct = iteration.get('summary_correct', False)
                answer = iteration.get('summary_answer', 'No answer')
            else:
                is_correct = iteration.get('correct', False)
                answer = iteration.get('answer', 'No answer')
            print(f"    Iteration {i}: {'✅' if is_correct else '❌'} {answer}")
        
        if problem['iteration_differences']:
            print("  Differences in iterations:")
            for diff in problem['iteration_differences']:
                print(f"    Iteration {diff['iteration']}:")
                print(f"      Regular: {'✅' if diff['regular_correct'] else '❌'} {diff['regular_answer']}")
                print(f"      Summary: {'✅' if diff['summary_correct'] else '❌'} {diff['summary_answer']}")
    
    return diff_problems

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find problems with different categorizations when using regular vs summary answers')
    parser.add_argument('--results_path', type=str, help='Path to results JSON file')
    
    args = parser.parse_args()
    
    # If results_path is provided, use it; otherwise try to find the most recent results file
    if args.results_path:
        results_path = args.results_path
    else:
        # Look for the most recent results file in the results directory
        results_files = glob.glob('./results/**/results.json', recursive=True)
        if not results_files:
            print("No results files found. Please specify a path with --results_path.")
            exit(1)
        
        # Sort by modification time (most recent first)
        results_path = max(results_files, key=os.path.getmtime)
        print(f"Using most recent results file: {results_path}")
    
    # Find problems with different categorizations
    diff_problems = find_category_differences(results_path)
