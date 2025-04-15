import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

def analyze_iterations(results_path, use_summary=False):
    """
    Analyze iterations from results JSON file using similar logic to dashboard-iterations.js
    
    Parameters:
    - results_path: Path to the results JSON file
    - use_summary: If True, use summary answers for categorization instead of regular answers
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
    
    categorized_problems = []
    answer_differences = []
    
    for problem in results:
        problem_id = problem.get('problem_id', 'Unknown')
        iterations = problem.get('iterations', [])
        
        if not iterations:
            continue
        
        # Track correctness patterns across iterations
        first_iteration_correct = False
        any_later_iteration_correct = False
        any_later_iteration_incorrect = False
        all_correct = True
        all_incorrect = True
        last_iteration_correct = False
        
        # Track differences between regular and summary answers
        problem_differences = []
        
        # Analyze each iteration
        for i, iteration in enumerate(iterations):
            # Determine which answer/correctness to use based on the use_summary flag
            if use_summary and 'summary_correct' in iteration:
                is_correct = iteration.get('summary_correct', False)
                answer = iteration.get('summary_answer', 'No answer')
            else:
                is_correct = iteration.get('correct', False)
                answer = iteration.get('answer', 'No answer')
            
            # Check for differences between regular and summary answers
            if 'summary_answer' in iteration and 'answer' in iteration:
                regular_answer = iteration.get('answer')
                summary_answer = iteration.get('summary_answer')
                regular_correct = iteration.get('correct', False)
                summary_correct = iteration.get('summary_correct', False)
                
                if regular_answer != summary_answer or regular_correct != summary_correct:
                    problem_differences.append({
                        'iteration': i,
                        'regular_answer': regular_answer,
                        'summary_answer': summary_answer,
                        'regular_correct': regular_correct,
                        'summary_correct': summary_correct,
                        'difference_type': 'answer_mismatch' if regular_answer != summary_answer else 'correctness_mismatch'
                    })
            
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
        
        # Add to categorized problems
        categorized_problems.append({
            'problem_id': problem_id,
            'category': category,
            'category_display': categories.get(category, 'Uncategorized'),
            'iterations_count': len(iterations),
            'first_correct': first_iteration_correct,
            'last_correct': last_iteration_correct,
            'final_correct': problem.get('final_correct', last_iteration_correct),
            'iterations': iterations
        })
        
        # Add any differences to the list
        if problem_differences:
            answer_differences.append({
                'problem_id': problem_id,
                'differences': problem_differences
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(categorized_problems)
    
    # Print category counts
    category_counts = Counter(df['category_display'])
    print(f"\nProblem Categories ({'Using Summary' if use_summary else 'Using Regular'} Answers):")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")
    
    # Calculate some overall statistics
    print("\nOverall Statistics:")
    print(f"Total problems: {len(df)}")
    if len(df) > 0:
        print(f"First iteration correct: {df['first_correct'].sum()} ({df['first_correct'].mean() * 100:.1f}%)")
        print(f"Last iteration correct: {df['last_correct'].sum()} ({df['last_correct'].mean() * 100:.1f}%)")
        print(f"Final answer correct: {df['final_correct'].sum()} ({df['final_correct'].mean() * 100:.1f}%)")
    
    return df, category_counts, answer_differences

def plot_categories(regular_counts, summary_counts=None):
    """
    Create a bar chart of problem categories
    
    Parameters:
    - regular_counts: Counter object with regular answer category counts
    - summary_counts: Counter object with summary answer category counts (optional)
    """
    if summary_counts:
        # Plot side by side comparison
        categories = list(set(list(regular_counts.keys()) + list(summary_counts.keys())))
        categories.sort()
        
        regular_values = [regular_counts.get(cat, 0) for cat in categories]
        summary_values = [summary_counts.get(cat, 0) for cat in categories]
        
        x = range(len(categories))
        width = 0.35
        
        plt.figure(figsize=(12, 7))
        bars1 = plt.bar([i - width/2 for i in x], regular_values, width, label='Regular Answers', color='#2196F3')
        bars2 = plt.bar([i + width/2 for i in x], summary_values, width, label='Summary Answers', color='#FF9800')
        
        plt.title('Problem Categories Distribution: Regular vs Summary')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        
        # Add count labels on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('category_comparison.png')
        print("Comparison plot saved as 'category_comparison.png'")
    else:
        # Plot single category distribution
        categories = list(regular_counts.keys())
        counts = list(regular_counts.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, counts, color=['#4CAF50', '#F44336', '#2196F3', '#FFD700', '#9C27B0', '#00BCD4'])
        
        plt.title('Problem Categories Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('category_distribution.png')
        print("Plot saved as 'category_distribution.png'")
    
    # If running in a notebook environment or with display capability
    try:
        plt.show()
    except:
        pass

def list_examples_by_category(df, category, max_examples=3, use_summary=False):
    """
    List example problems for a specific category
    
    Parameters:
    - df: DataFrame with categorized problems
    - category: Category to list examples for
    - max_examples: Maximum number of examples to show
    - use_summary: If True, show summary answers instead of regular answers
    """
    category_df = df[df['category'] == category]
    if len(category_df) == 0:
        print(f"No examples found for category: {category}")
        return
    
    print(f"\nExamples for {categories[category]} (up to {max_examples}):")
    for i, (idx, row) in enumerate(category_df.iloc[:max_examples].iterrows()):
        print(f"  Problem {row['problem_id']}:")
        for j, iteration in enumerate(row['iterations']):
            if use_summary and 'summary_correct' in iteration:
                is_correct = iteration.get('summary_correct', False)
                answer = iteration.get('summary_answer', 'No answer')
                print(f"    Iteration {j}: {'✅' if is_correct else '❌'} {answer} (summary)")
            else:
                is_correct = iteration.get('correct', False)
                answer = iteration.get('answer', 'No answer')
                print(f"    Iteration {j}: {'✅' if is_correct else '❌'} {answer}")
        if i < min(max_examples, len(category_df)) - 1:
            print()

def analyze_answer_differences(differences):
    """
    Analyze and print differences between regular and summary answers
    
    Parameters:
    - differences: List of problem differences
    """
    if not differences:
        print("\nNo differences found between regular and summary answers.")
        return
    
    print("\nDifferences between Regular and Summary Answers:")
    print(f"Found differences in {len(differences)} problems")
    
    # Count types of differences
    answer_mismatches = 0
    correctness_mismatches = 0
    both_mismatches = 0
    
    for problem in differences:
        problem_id = problem['problem_id']
        has_answer_mismatch = any(d['difference_type'] == 'answer_mismatch' for d in problem['differences'])
        has_correctness_mismatch = any(d['difference_type'] == 'correctness_mismatch' for d in problem['differences'])
        
        if has_answer_mismatch and has_correctness_mismatch:
            both_mismatches += 1
        elif has_answer_mismatch:
            answer_mismatches += 1
        elif has_correctness_mismatch:
            correctness_mismatches += 1
    
    print(f"Problems with answer mismatches only: {answer_mismatches}")
    print(f"Problems with correctness mismatches only: {correctness_mismatches}")
    print(f"Problems with both types of mismatches: {both_mismatches}")
    
    # Print examples of differences
    print("\nExamples of differences (up to 5 problems):")
    for i, problem in enumerate(differences[:5]):
        problem_id = problem['problem_id']
        print(f"\n  Problem {problem_id}:")
        
        for diff in problem['differences']:
            iteration = diff['iteration']
            regular_answer = diff['regular_answer']
            summary_answer = diff['summary_answer']
            regular_correct = diff['regular_correct']
            summary_correct = diff['summary_correct']
            
            print(f"    Iteration {iteration}:")
            print(f"      Regular: {'✅' if regular_correct else '❌'} {regular_answer}")
            print(f"      Summary: {'✅' if summary_correct else '❌'} {summary_answer}")
            
            if regular_answer != summary_answer and regular_correct != summary_correct:
                print(f"      Difference: Both answer and correctness mismatch")
            elif regular_answer != summary_answer:
                print(f"      Difference: Answer mismatch")
            elif regular_correct != summary_correct:
                print(f"      Difference: Correctness mismatch")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze iteration results for problem-solving experiments')
    parser.add_argument('--results_path', type=str, help='Path to results JSON file')
    parser.add_argument('--examples', type=int, default=3, help='Number of examples to show for each category')
    parser.add_argument('--compare_summary', action='store_true', help='Compare regular and summary answers')
    
    args = parser.parse_args()
    
    # Define categories
    categories = {
        'all_correct': '🟩 All Correct',
        'all_incorrect': '🟥 All Incorrect',
        'improved_final_incorrect': '🟦 Improved (Final Incorrect)',
        'improved_final_correct': '👑 Improved (Final Correct)',
        'regressed_final_incorrect': '🟪 Regressed (Final Incorrect)',
        'regressed_final_correct': '🧼 Regressed (Final Correct)'
    }
    
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
    
    # Analyze the results using regular answers
    print("\n=== Analysis using Regular Answers ===")
    df_regular, category_counts_regular, answer_differences = analyze_iterations(results_path, use_summary=False)
    
    if args.compare_summary:
        # Analyze the results using summary answers
        print("\n=== Analysis using Summary Answers ===")
        df_summary, category_counts_summary, _ = analyze_iterations(results_path, use_summary=True)
        
        # Plot category comparison
        if len(df_regular) > 0 and len(df_summary) > 0:
            plot_categories(category_counts_regular, category_counts_summary)
            
            # Analyze differences between regular and summary answers
            analyze_answer_differences(answer_differences)
    else:
        # Plot regular category distribution only
        if len(df_regular) > 0:
            plot_categories(category_counts_regular)
    
    # Show examples for each category
    for category in categories:
        list_examples_by_category(df_regular, category, max_examples=args.examples)
