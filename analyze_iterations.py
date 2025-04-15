import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

def analyze_iterations(results_path):
    """
    Analyze iterations from results JSON file using similar logic to dashboard-iterations.js
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
        
        # Analyze each iteration
        for i, iteration in enumerate(iterations):
            is_correct = iteration.get('correct', False)
            answer = iteration.get('answer', 'No answer')
            
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
    
    # Convert to DataFrame
    df = pd.DataFrame(categorized_problems)
    
    # Print category counts
    category_counts = Counter(df['category_display'])
    print("Problem Categories:")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")
    
    # Calculate some overall statistics
    print("\nOverall Statistics:")
    print(f"Total problems: {len(df)}")
    if len(df) > 0:
        print(f"First iteration correct: {df['first_correct'].sum()} ({df['first_correct'].mean() * 100:.1f}%)")
        print(f"Last iteration correct: {df['last_correct'].sum()} ({df['last_correct'].mean() * 100:.1f}%)")
        print(f"Final answer correct: {df['final_correct'].sum()} ({df['final_correct'].mean() * 100:.1f}%)")
    
    return df, category_counts

def plot_categories(category_counts):
    """
    Create a bar chart of problem categories
    """
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
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
    
    # Save figure
    plt.savefig('category_distribution.png')
    print("Plot saved as 'category_distribution.png'")
    
    # If running in a notebook environment or with display capability
    try:
        plt.show()
    except:
        pass

def list_examples_by_category(df, category, max_examples=3):
    """
    List example problems for a specific category
    """
    category_df = df[df['category'] == category]
    if len(category_df) == 0:
        print(f"No examples found for category: {category}")
        return
    
    print(f"\nExamples for {categories[category]} (up to {max_examples}):")
    for i, (idx, row) in enumerate(category_df.iloc[:max_examples].iterrows()):
        print(f"  Problem {row['problem_id']}:")
        for j, iteration in enumerate(row['iterations']):
            print(f"    Iteration {j}: {'✅' if iteration.get('correct', False) else '❌'} {iteration.get('answer', 'No answer')}")
        if i < min(max_examples, len(category_df)) - 1:
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze iteration results for problem-solving experiments')
    parser.add_argument('--results_path', type=str, help='Path to results JSON file')
    parser.add_argument('--examples', type=int, default=3, help='Number of examples to show for each category')
    
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
    
    # Analyze the results
    df, category_counts = analyze_iterations(results_path)
    
    # Plot category distribution
    if len(df) > 0:
        plot_categories(category_counts)
        
        # Show examples for each category
        for category in categories:
            list_examples_by_category(df, category, max_examples=args.examples)
