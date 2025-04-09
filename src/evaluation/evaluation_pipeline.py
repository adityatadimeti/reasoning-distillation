#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np

# Add the project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the evaluation functions from math.py with an alias to avoid name conflicts
import math_utils

def load_results(results_path: str) -> Dict:
    """Load results from a JSON file.
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        Dictionary containing the loaded results
    """
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_llm_answer(reasoning: str) -> Optional[str]:
    """Extract the LLM's answer from the reasoning text.
    
    Args:
        reasoning: The full reasoning text from the LLM
        
    Returns:
        Extracted answer as a string, or None if no answer is found
    """
    # Try to extract the answer from a boxed environment
    boxed_answer = math_utils.last_boxed_only_string(reasoning)
    return boxed_answer

def analyze_error_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns in incorrect answers.
    
    Args:
        df: DataFrame containing the evaluation results
        
    Returns:
        Dictionary with error pattern analysis
    """
    if df.empty:
        return {}
    
    # Filter to only incorrect answers where we have both LLM answer and correct answer
    incorrect_df = df[(df['is_correct'] == False) & 
                      (df['llm_answer'].notna()) & 
                      (df['correct_answer'].notna())]
    
    if incorrect_df.empty:
        return {'message': 'No analyzable incorrect answers found'}
    
    # Initialize error pattern counters
    error_patterns = {
        'off_by_one': 0,
        'wrong_operation': 0,
        'close_answer': 0,
        'completely_different': 0,
        'total_errors': len(incorrect_df)
    }
    
    # For each incorrect answer, try to categorize the error
    for _, row in incorrect_df.iterrows():
        llm_answer = row['llm_answer']
        correct_answer = row['correct_answer']
        
        # Try to convert to numbers for comparison when possible
        try:
            llm_num = float(llm_answer)
            correct_num = float(correct_answer)
            
            # Off by one errors
            if abs(llm_num - correct_num) == 1:
                error_patterns['off_by_one'] += 1
            # Close answers (within 10% of correct value)
            elif abs(llm_num - correct_num) <= 0.1 * abs(correct_num):
                error_patterns['close_answer'] += 1
            # Wrong operation might have been used
            elif abs(llm_num - correct_num) < max(abs(correct_num), 100):
                error_patterns['wrong_operation'] += 1
            else:
                error_patterns['completely_different'] += 1
        except (ValueError, TypeError):
            # If we can't convert to numbers, assume completely different
            error_patterns['completely_different'] += 1
    
    # Calculate percentages
    total = error_patterns['total_errors']
    for key in ['off_by_one', 'wrong_operation', 'close_answer', 'completely_different']:
        error_patterns[f'{key}_pct'] = error_patterns[key] / total if total > 0 else 0
    
    return error_patterns

def classify_problem_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify problems according to the following criteria:
    - All Correct [Green]: Correct from Iteration 0 onwards.
    - All Incorrect [Red]: Incorrect through all iterations.
    - Improved (Final Incorrect) [Blue]: Initially Incorrect -> Correct at some middle iteration -> Ended Incorrect.
    - Improved (Final Correct) [Gold]: Initially Incorrect -> Ended Correct (ideal case).
    - Regressed (Final Incorrect) [Purple]: Started Correct -> Ended Incorrect.
    - Regressed (Final Correct) [Teal]: Started Correct -> Incorrect at some middle iteration -> Ended Correct.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        DataFrame with problem classifications added
    """
    if df.empty:
        return df
    
    # Get unique problem IDs
    problem_ids = df['problem_id'].unique()
    
    # Dictionary to store classification for each problem
    problem_classifications = {}
    
    for problem_id in problem_ids:
        # Get the data for this problem, sorted by iteration
        problem_df = df[df['problem_id'] == problem_id].sort_values('iteration')
        
        # Extract correctness history
        correctness_history = problem_df['is_correct'].tolist()
        
        if len(correctness_history) == 0:
            continue
            
        # Get first and last iteration correctness
        first_correct = correctness_history[0]
        last_correct = correctness_history[-1]
        
        # Check middle iterations (if any)
        middle_iterations = correctness_history[1:-1] if len(correctness_history) > 2 else []
        any_middle_correct = any(middle_iterations) if middle_iterations else False
        any_middle_incorrect = not all(middle_iterations) if middle_iterations else False
        
        # Classify based on the criteria
        if all(correctness_history):
            classification = "All Correct"
            color = "green"
        elif not any(correctness_history):
            classification = "All Incorrect"
            color = "red"
        elif not first_correct and last_correct:
            classification = "Improved (Final Correct)"
            color = "gold"
        elif not first_correct and not last_correct and any_middle_correct:
            classification = "Improved (Final Incorrect)"
            color = "blue"
        elif first_correct and not last_correct:
            classification = "Regressed (Final Incorrect)"
            color = "purple"
        elif first_correct and last_correct and any_middle_incorrect:
            classification = "Regressed (Final Correct)"
            color = "teal"
        else:
            classification = "Unclassified"
            color = "gray"
            
        # Store the classification for this problem
        problem_classifications[problem_id] = {
            'outcome': classification,
            'color': color
        }
    
    # Create a new DataFrame with problem IDs and classifications
    classification_df = pd.DataFrame.from_dict(
        problem_classifications, 
        orient='index'
    ).reset_index().rename(columns={'index': 'problem_id'})
    
    # Merge this with the original DataFrame
    df = df.merge(classification_df, on='problem_id', how='left')
    
    return df

def evaluate_results(results: Dict) -> pd.DataFrame:
    """Evaluate each result in the results dictionary.
    
    Args:
        results: Dictionary containing the results
        
    Returns:
        DataFrame containing the evaluation results
    """
    evaluation_records = []
    
    # Iterate through all results
    for problem in tqdm(results.get('results', []), desc="Evaluating problems"):
        problem_id = problem.get('problem_id', 'unknown')
        question = problem.get('question', '')
        correct_answer = problem.get('correct_answer', '')
        
        # Process each iteration for this problem
        for i, iteration in enumerate(problem.get('iterations', [])):
            reasoning = iteration.get('reasoning', '')
            llm_answer = iteration.get('answer')  # Use the already extracted answer if available
            
            # If no answer is extracted, try to extract it
            if llm_answer is None:
                llm_answer_raw = extract_llm_answer(reasoning)
            else:
                llm_answer_raw = llm_answer
            
            # Compute score using the evaluation function
            score = 0.0
            if llm_answer_raw is not None:
                score = math_utils.compute_score(reasoning, correct_answer)
            
            # Is the answer equivalent to the correct answer?
            is_correct = False
            if llm_answer is not None:
                is_correct = math_utils.is_equiv(llm_answer, correct_answer)
            
            # Create a record for this evaluation
            record = {
                'problem_id': problem_id,
                'iteration': i,
                'question': question,
                'llm_answer_raw': llm_answer_raw,
                'llm_answer': llm_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'score': score
            }
            
            evaluation_records.append(record)
    
    # Create a DataFrame from the evaluation records
    df = pd.DataFrame(evaluation_records)
    
    # Apply problem classification
    if not df.empty:
        df = classify_problem_outcomes(df)
    
    # Calculate aggregate statistics
    if not df.empty:
        # Calculate accuracy by iteration
        accuracy_by_iteration = df.groupby('iteration')['is_correct'].mean()
        print("\nAccuracy by iteration:")
        for iteration, accuracy in accuracy_by_iteration.items():
            print(f"Iteration {iteration}: {accuracy:.2%}")
        
        # Overall accuracy
        overall_accuracy = df['is_correct'].mean()
        print(f"\nOverall accuracy: {overall_accuracy:.2%}")
        
        # Analyze error patterns
        error_patterns = analyze_error_patterns(df)
        print("\nError pattern analysis:")
        for key, value in error_patterns.items():
            if key.endswith('_pct'):
                print(f"- {key.replace('_pct', '')}: {value:.2%}")
            elif key != 'total_errors' and key != 'message':
                print(f"- {key}: {value}")
        
        # Print problem outcome classifications
        if 'outcome' in df.columns:
            outcome_counts = df.drop_duplicates('problem_id')['outcome'].value_counts()
            print("\nProblem outcome classifications:")
            for outcome, count in outcome_counts.items():
                print(f"- {outcome}: {count} problems ({count/len(df.drop_duplicates('problem_id')):.2%})")
    
    return df

def save_results(df: pd.DataFrame, output_path: str):
    """Save the evaluation results.
    
    Args:
        df: DataFrame containing the evaluation results
        output_path: Path to save the output
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Also save as Excel if pandas has openpyxl, but sanitize math expressions first
    # try:
    #     # excel_path = output_path.replace('.csv', '.xlsx')
        
    #     # Create a copy of the DataFrame to sanitize for Excel
    #     excel_df = df.copy()
        
    #     # Sanitize mathematical expressions in string columns
    #     for col in excel_df.select_dtypes(include=['object']).columns:
    #         if col in ['question', 'llm_answer', 'llm_answer_raw', 'correct_answer']:
    #             # Replace LaTeX math expressions with plain text
    #             excel_df[col] = excel_df[col].astype(str).apply(
    #                 lambda x: x.replace('$', '').replace('\\frac', 'frac').replace('\\boxed', 'boxed')
    #             )
        
    #     # Save to Excel
    #     excel_df.to_excel(excel_path, index=False)
    #     print(f"Results also saved to {excel_path}")
    # except Exception as e:
    #     print(f"Could not save as Excel (math expressions sanitized): {e}")
    #     # Try saving without the problematic columns
    #     try:
    #         excel_path = output_path.replace('.csv', '_basic.xlsx')
    #         # Take just the basic columns
    #         basic_cols = ['problem_id', 'iteration', 'is_correct', 'score', 'outcome']
    #         basic_df = df[[col for col in basic_cols if col in df.columns]]
    #         basic_df.to_excel(excel_path, index=False)
    #         print(f"Basic results saved to {excel_path}")
    #     except Exception as e2:
    #         print(f"Could not save basic Excel file either: {e2}")
    
    # Create visualizations
    create_visualizations(df, output_dir)

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualizations of the evaluation results.
    
    Args:
        df: DataFrame containing the evaluation results
        output_dir: Directory to save the visualizations
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Accuracy by iteration
    plt.figure(figsize=(10, 6))
    accuracy_by_iteration = df.groupby('iteration')['is_correct'].mean()
    ax = accuracy_by_iteration.plot(kind='bar', color='skyblue')
    plt.title('Accuracy by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy percentages on top of bars
    for i, v in enumerate(accuracy_by_iteration):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_iteration.png'))
    plt.close()
    
    # 2. Problem improvement heatmap
    problem_improvement = pd.pivot_table(
        df, 
        values='is_correct',
        index='problem_id',
        columns='iteration', 
        aggfunc=lambda x: 1 if any(x) else 0,
        fill_value=0
    )
    
    if not problem_improvement.empty and problem_improvement.shape[1] > 1:
        plt.figure(figsize=(12, max(8, len(problem_improvement) * 0.4)))
        sns.heatmap(problem_improvement, cmap='RdYlGn', cbar_kws={'label': 'Correct'})
        plt.title('Problem Solution Status by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Problem ID')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'problem_improvement.png'))
        plt.close()
    
    # 3. Confusion matrix of improvement/regression
    if df['iteration'].nunique() > 1:
        # Get first and last iteration results for each problem
        first_iterations = df[df['iteration'] == df['iteration'].min()].set_index('problem_id')['is_correct']
        last_iterations = df[df['iteration'] == df['iteration'].max()].set_index('problem_id')['is_correct']
        
        # Combine into a DataFrame
        improvement_df = pd.DataFrame({
            'first_correct': first_iterations,
            'last_correct': last_iterations
        })
        
        # Count different categories
        improved = (improvement_df['first_correct'] == False) & (improvement_df['last_correct'] == True)
        regressed = (improvement_df['first_correct'] == True) & (improvement_df['last_correct'] == False)
        stayed_correct = (improvement_df['first_correct'] == True) & (improvement_df['last_correct'] == True)
        stayed_incorrect = (improvement_df['first_correct'] == False) & (improvement_df['last_correct'] == False)
        
        # Create labels and sizes for pie chart
        labels = ['Improved', 'Regressed', 'Stayed Correct', 'Stayed Incorrect']
        sizes = [
            improved.sum(),
            regressed.sum(),
            stayed_correct.sum(),
            stayed_incorrect.sum()
        ]
        
        # Create pie chart with the specified color scheme
        plt.figure(figsize=(10, 8))
        # Use the user-specified color scheme:
        # Green for "Stayed Correct" (All Correct)
        # Red for "Stayed Incorrect" (All Incorrect)
        # Gold for "Improved" (matching Improved (Final Correct))
        # Purple for "Regressed" (matching Regressed (Final Incorrect))
        colors = ['gold', 'purple', 'green', 'red']
        explode = (0.1, 0.1, 0, 0)  # explode the first two slices (improved and regressed)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Problem Outcome Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outcome_distribution.png'))
        plt.close()
    
    # 4. Problem outcome classifications - Bar Chart
    if 'outcome' in df.columns:
        # Get unique problems with their classifications
        problem_outcomes = df.drop_duplicates('problem_id')[['problem_id', 'outcome', 'color']]
        
        # Count by outcome type
        outcome_counts = problem_outcomes['outcome'].value_counts()
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(
            outcome_counts.index, 
            outcome_counts.values,
            color=[problem_outcomes[problem_outcomes['outcome'] == outcome]['color'].iloc[0] 
                   for outcome in outcome_counts.index]
        )
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{height}',
                ha='center', va='bottom'
            )
        
        plt.title('Problem Outcome Classifications')
        plt.xlabel('Outcome Category')
        plt.ylabel('Number of Problems')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'problem_outcome_classifications.png'))
        plt.close()
        
        # 5. NEW: Problem outcome classifications - Pie Chart
        plt.figure(figsize=(12, 10))
        colors = [problem_outcomes[problem_outcomes['outcome'] == outcome]['color'].iloc[0] 
                  for outcome in outcome_counts.index]
        
        # Create pie chart with percentages
        plt.pie(
            outcome_counts.values,
            labels=outcome_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(outcome_counts)  # Slight explosion for all slices
        )
        plt.axis('equal')
        plt.title('Problem Outcome Classifications Distribution', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'problem_outcome_pie.png'))
        plt.close()
        
        # 6. NEW: Correctness trajectory visualization
        # Create a visualization showing correctness trajectory for each problem
        if df['iteration'].nunique() > 1:
            # Get all problem IDs
            problem_ids = df['problem_id'].unique()
            
            # Limit to at most 50 problems to keep the plot readable
            if len(problem_ids) > 50:
                # If we have color classification, ensure we include some of each type
                if 'outcome' in df.columns:
                    # Get problems from each outcome category
                    sampled_problems = []
                    for outcome in problem_outcomes['outcome'].unique():
                        # Select up to 10 problems from each category
                        category_problems = problem_outcomes[problem_outcomes['outcome'] == outcome]['problem_id'].values
                        if len(category_problems) > 0:
                            # Take up to 10 problems or all if fewer
                            sample_size = min(10, len(category_problems))
                            sampled_problems.extend(np.random.choice(category_problems, sample_size, replace=False))
                    
                    # If we need more problems to reach 50, randomly sample from remaining
                    if len(sampled_problems) < 50:
                        remaining = [p for p in problem_ids if p not in sampled_problems]
                        if remaining:
                            additional = min(50 - len(sampled_problems), len(remaining))
                            sampled_problems.extend(np.random.choice(remaining, additional, replace=False))
                    
                    # If still too many, truncate to 50
                    if len(sampled_problems) > 50:
                        sampled_problems = sampled_problems[:50]
                    
                    problem_ids = np.array(sampled_problems)
                else:
                    # Random sampling if no classification available
                    problem_ids = np.random.choice(problem_ids, 50, replace=False)
            
            # Set up the plot
            plt.figure(figsize=(15, max(8, len(problem_ids) * 0.3)))
            
            # Track the problems to include in legend
            legend_elements = []
            legend_labels = []
            
            # Plot each problem's correctness trajectory
            for i, problem_id in enumerate(problem_ids):
                problem_data = df[df['problem_id'] == problem_id].sort_values('iteration')
                iterations = problem_data['iteration'].values
                correctness = problem_data['is_correct'].values
                
                # Get color for this problem if available
                if 'outcome' in problem_data.columns:
                    color = problem_data['color'].iloc[0]
                    outcome = problem_data['outcome'].iloc[0]
                else:
                    color = 'gray'
                    outcome = 'Unknown'
                
                # Plot correctness as a line
                line, = plt.plot(iterations, correctness, 'o-', color=color, linewidth=2, markersize=8)
                
                # Only add to legend if this outcome hasn't been added yet
                if outcome not in legend_labels:
                    legend_elements.append(line)
                    legend_labels.append(outcome)
                
                # Add problem ID as text
                plt.text(
                    iterations[-1] + 0.1,
                    correctness[-1],
                    str(problem_id),
                    fontsize=8,
                    verticalalignment='center'
                )
            
            # Create custom legend with one entry per outcome category
            plt.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
            
            plt.title('Correctness Trajectory by Problem', fontsize=16)
            plt.xlabel('Iteration')
            plt.ylabel('Correct (1) / Incorrect (0)')
            plt.yticks([0, 1], ['Incorrect', 'Correct'])
            plt.xlim(-0.5, df['iteration'].max() + 0.5)
            plt.ylim(-0.1, 1.1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correctness_trajectory.png'))
            plt.close()
            
        # 7. NEW: Transition diagram - Sankey plot showing transitions between iterations
        if df['iteration'].nunique() > 1:
            try:
                # Try to import the required packages for Sankey diagrams
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Get iteration transitions (0→1, 1→2, etc.)
                transitions = []
                values = []
                
                max_iteration = df['iteration'].max()
                for i in range(max_iteration):
                    # Get dataframe with both current and next iteration
                    df_iter = df[df['iteration'].isin([i, i+1])].sort_values(['problem_id', 'iteration'])
                    
                    # Group by problem_id and check if we have both iterations
                    grouped = df_iter.groupby('problem_id')
                    transitions_df = []
                    
                    for _, group in grouped:
                        if len(group) == 2:  # Has both iterations
                            prev_correct = group.iloc[0]['is_correct']
                            next_correct = group.iloc[1]['is_correct']
                            
                            if prev_correct and next_correct:
                                transition = f"Correct (Iter {i}) → Correct (Iter {i+1})"
                                color = "green"
                            elif prev_correct and not next_correct:
                                transition = f"Correct (Iter {i}) → Incorrect (Iter {i+1})"
                                color = "purple"
                            elif not prev_correct and next_correct:
                                transition = f"Incorrect (Iter {i}) → Correct (Iter {i+1})"
                                color = "gold"
                            else:
                                transition = f"Incorrect (Iter {i}) → Incorrect (Iter {i+1})"
                                color = "red"
                            
                            transitions_df.append({
                                "transition": transition,
                                "color": color
                            })
                    
                    # Count the transitions
                    if transitions_df:
                        transition_df = pd.DataFrame(transitions_df)
                        transition_counts = transition_df['transition'].value_counts()
                        
                        for transition, count in transition_counts.items():
                            source = transition.split(" → ")[0]
                            target = transition.split(" → ")[1]
                            color = transition_df[transition_df['transition'] == transition]['color'].iloc[0]
                            
                            transitions.append((source, target, color))
                            values.append(count)
                
                # Create Sankey diagram if we have transitions
                if transitions:
                    # Create nodes and links for Sankey diagram
                    nodes = []
                    node_colors = []
                    links = []
                    
                    # Create unique nodes with colors
                    for source, target, color in transitions:
                        if source not in nodes:
                            nodes.append(source)
                            node_colors.append('lightgray')
                        if target not in nodes:
                            nodes.append(target)
                            node_colors.append('lightgray')
                    
                    # Create links
                    for (source, target, color), value in zip(transitions, values):
                        source_idx = nodes.index(source)
                        target_idx = nodes.index(target)
                        links.append({
                            'source': source_idx,
                            'target': target_idx,
                            'value': value,
                            'color': color
                        })
                    
                    # Create Sankey figure
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=nodes,
                            color=node_colors
                        ),
                        link=dict(
                            source=[link['source'] for link in links],
                            target=[link['target'] for link in links],
                            value=[link['value'] for link in links],
                            color=[link['color'] for link in links]
                        )
                    )])
                    
                    fig.update_layout(
                        title="Problem Transitions Between Iterations",
                        font=dict(size=12),
                        width=1200,
                        height=800
                    )
                    
                    # Save figure
                    fig.write_image(os.path.join(output_dir, 'transition_sankey.png'))
                    fig.write_html(os.path.join(output_dir, 'transition_sankey.html'))
            except ImportError:
                print("Could not create Sankey diagram - plotly not installed")
            except Exception as e:
                print(f"Error creating Sankey diagram: {str(e)}")
                
            # 8. NEW: Heatmap showing correctness by iteration with color coding
            # Create a DataFrame with one row per problem and columns for each iteration
            heatmap_data = pd.pivot_table(
                df, 
                values='is_correct',
                index='problem_id',
                columns='iteration'
            ).fillna(-1)  # Fill NA with -1 to indicate missing data
            
            if not heatmap_data.empty:
                plt.figure(figsize=(14, max(8, len(heatmap_data) * 0.25)))
                
                # Create custom colormap: red for incorrect, green for correct, gray for missing
                cmap = plt.cm.colors.ListedColormap(['lightgray', 'red', 'green'])
                bounds = [-1.5, -0.5, 0.5, 1.5]
                norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
                
                # Create heatmap
                ax = plt.gca()
                cax = ax.matshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')
                
                # Add outcome classification as row colors if available
                if 'outcome' in df.columns:
                    # Get outcome for each problem
                    problem_outcomes = df.drop_duplicates('problem_id')[['problem_id', 'outcome', 'color']]
                    
                    # Add row colors matching outcomes
                    row_colors = []
                    for problem_id in heatmap_data.index:
                        if problem_id in problem_outcomes['problem_id'].values:
                            color = problem_outcomes[problem_outcomes['problem_id'] == problem_id]['color'].iloc[0]
                            row_colors.append(color)
                        else:
                            row_colors.append('lightgray')
                    
                    # Add colored rectangles for each row
                    for i, color in enumerate(row_colors):
                        rect = plt.Rectangle(
                            xy=(-1.5, i - 0.5),
                            width=1,
                            height=1,
                            facecolor=color,
                            edgecolor='none',
                            alpha=0.8
                        )
                        ax.add_patch(rect)
                
                # Configure axes and labels
                colorbar = plt.colorbar(cax, ticks=[-1, 0, 1])
                colorbar.set_ticklabels(['Missing', 'Incorrect', 'Correct'])
                plt.title('Correctness by Problem and Iteration', fontsize=16)
                plt.xlabel('Iteration')
                plt.ylabel('Problem ID')
                
                # Set x and y labels
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels(heatmap_data.columns)
                ax.set_yticks(range(len(heatmap_data)))
                ax.set_yticklabels(heatmap_data.index)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'correctness_heatmap.png'))
                plt.close()

def generate_report(df: pd.DataFrame, output_dir: str):
    """Generate a detailed analysis report in markdown format.
    
    Args:
        df: DataFrame containing the evaluation results
        output_dir: Directory to save the report
    """
    if df.empty:
        print("No data to generate report")
        return
    
    # Calculate key metrics
    total_problems = df['problem_id'].nunique()
    total_iterations = df['iteration'].nunique()
    
    # Accuracy by iteration
    accuracy_by_iteration = df.groupby('iteration')['is_correct'].mean()
    
    # Error pattern analysis
    error_patterns = analyze_error_patterns(df)
    
    # Problem outcome classifications
    if 'outcome' in df.columns:
        outcome_counts = df.drop_duplicates('problem_id')['outcome'].value_counts()
    
    # Improvement analysis
    if total_iterations > 1:
        first_iterations = df[df['iteration'] == df['iteration'].min()].set_index('problem_id')['is_correct']
        last_iterations = df[df['iteration'] == df['iteration'].max()].set_index('problem_id')['is_correct']
        
        improvement_df = pd.DataFrame({
            'first_correct': first_iterations,
            'last_correct': last_iterations
        })
        
        improved = (improvement_df['first_correct'] == False) & (improvement_df['last_correct'] == True)
        regressed = (improvement_df['first_correct'] == True) & (improvement_df['last_correct'] == False)
        stayed_correct = (improvement_df['first_correct'] == True) & (improvement_df['last_correct'] == True)
        stayed_incorrect = (improvement_df['first_correct'] == False) & (improvement_df['last_correct'] == False)
        
        improved_count = improved.sum()
        regressed_count = regressed.sum()
        stayed_correct_count = stayed_correct.sum()
        stayed_incorrect_count = stayed_incorrect.sum()
        
        # List of improved problems
        improved_problems = improvement_df[improved].index.tolist()
        regressed_problems = improvement_df[regressed].index.tolist()
    
    # Start building the report
    report = [
        "# AIME Math Problem Evaluation Report\n",
        f"## Summary\n",
        f"- Total problems evaluated: {total_problems}",
        f"- Total iterations per problem: {total_iterations}",
        f"- Overall accuracy: {df['is_correct'].mean():.2%}\n",
        
        f"## Accuracy by Iteration\n"
    ]
    
    # Add accuracy by iteration
    for iteration, accuracy in accuracy_by_iteration.items():
        report.append(f"- Iteration {iteration}: {accuracy:.2%}")
    
    # Add problem outcome classifications
    if 'outcome' in df.columns:
        report.extend([
            f"\n## Problem Outcome Classifications\n"
        ])
        
        for outcome, count in outcome_counts.items():
            report.append(f"- {outcome}: {count} problems ({count/total_problems:.2%})")
        
        # Add lists of problems by category
        for outcome in outcome_counts.index:
            problems_in_category = df[df['outcome'] == outcome].drop_duplicates('problem_id')['problem_id'].tolist()
            if problems_in_category:
                report.append(f"\n### {outcome} Problems\n")
                report.append(", ".join(str(pid) for pid in sorted(problems_in_category)))
    
    # Add error pattern analysis
    if 'message' not in error_patterns:
        report.extend([
            f"\n## Error Pattern Analysis\n",
            f"- Total incorrect answers analyzed: {error_patterns.get('total_errors', 0)}",
            f"- Off by one errors: {error_patterns.get('off_by_one', 0)} ({error_patterns.get('off_by_one_pct', 0):.2%})",
            f"- Wrong operation errors: {error_patterns.get('wrong_operation', 0)} ({error_patterns.get('wrong_operation_pct', 0):.2%})",
            f"- Close answers: {error_patterns.get('close_answer', 0)} ({error_patterns.get('close_answer_pct', 0):.2%})",
            f"- Completely different: {error_patterns.get('completely_different', 0)} ({error_patterns.get('completely_different_pct', 0):.2%})"
        ])
    
    # Add improvement analysis if multiple iterations
    if total_iterations > 1:
        report.extend([
            f"\n## Basic Improvement Analysis\n",
            f"- Problems that improved: {improved_count} ({improved_count/total_problems:.2%})",
            f"- Problems that regressed: {regressed_count} ({regressed_count/total_problems:.2%})",
            f"- Problems that stayed correct: {stayed_correct_count} ({stayed_correct_count/total_problems:.2%})",
            f"- Problems that stayed incorrect: {stayed_incorrect_count} ({stayed_incorrect_count/total_problems:.2%})"
        ])
    
    # Write the report to a file
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Report saved to {report_path}")

def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate LLM answers to AIME math problems')
    parser.add_argument('--results_path', type=str, required=True, 
                        help='Path to the results.json file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the evaluation results')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}")
    results = load_results(args.results_path)
    
    # Set default output path if not provided and create appropriate subfolder
    if args.output_path is None:
        # Extract the parent folder name of results.json
        path_parts = args.results_path.split('/')
        if 'results.json' in path_parts[-1]:
            # Get the parent folder name (the experiment run folder)
            experiment_folder = path_parts[-2]
            # Create a subfolder in eval_output with this name
            output_dir = f"./eval_output/{experiment_folder}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            args.output_path = f"{output_dir}/evaluation_results.csv"
        else:
            # Extract the part after "results/" if it exists
            if "results/" in args.results_path:
                results_file = args.results_path.split("results/")[1]
                # Replace .json with .csv if present
                if results_file.endswith(".json"):
                    results_file = results_file[:-5]
                # Create subfolder
                output_dir = f"./eval_output/{results_file}"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                args.output_path = f"{output_dir}/evaluation_results.csv"
            else:
                # Default to a generic name
                args.output_path = "./eval_output/evaluation_results.csv"
    
    print(f"Output will be saved to: {args.output_path}")
    
    # Evaluate results
    print("Evaluating results...")
    df = evaluate_results(results)
    
    # Save results
    output_dir = os.path.dirname(args.output_path)
    save_results(df, args.output_path)
    
    # Generate detailed report
    generate_report(df, output_dir)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
