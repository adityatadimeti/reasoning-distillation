#!/usr/bin/env python
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results_file(file_path):
    """Load a behavior analysis results file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_behaviors(data, iterations=None):
    """
    Extract behavior counts from the results data
    
    Args:
        data: The loaded JSON data
        iterations: List of iterations to include, or None for all
    
    Returns:
        DataFrame with behavior counts
    """
    records = []
    
    for problem in data.get("results", []):
        problem_id = problem.get("problem_id")
        
        for iteration_data in problem.get("iterations", []):
            iteration = iteration_data.get("iteration")
            
            # Skip iterations not in the specified list
            if iterations is not None and iteration not in iterations:
                continue
                
            field = iteration_data.get("field_analyzed")
            behavior_counts = iteration_data.get("behavior_counts", {})
            
            record = {
                "problem_id": problem_id,
                "iteration": iteration,
                "field": field
            }
            
            # Add behavior counts to the record
            for behavior, count in behavior_counts.items():
                record[behavior] = count
                
            records.append(record)
    
    if not records:
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df

def calculate_statistics(df):
    """Calculate statistics on the behavior counts"""
    if df is None or df.empty:
        return None
    
    # Identify behavior columns
    behavior_cols = [col for col in df.columns if col not in ["problem_id", "iteration", "field"]]
    
    # Calculate per-iteration statistics
    stats_by_iteration = df.groupby("iteration")[behavior_cols].agg(['mean', 'sum', 'count']).reset_index()
    
    # Calculate overall statistics
    overall_stats = pd.DataFrame({
        'behavior': behavior_cols,
        'mean': [df[col].mean() for col in behavior_cols],
        'sum': [df[col].sum() for col in behavior_cols],
        'count': [df[col].count() for col in behavior_cols]
    })
    
    return {
        'overall': overall_stats,
        'by_iteration': stats_by_iteration
    }

def plot_behaviors(stats, output_file=None):
    """Create visualizations of the behavior counts"""
    if stats is None:
        return
    
    # Plot overall statistics
    overall = stats['overall']
    
    plt.figure(figsize=(10, 6))
    plt.bar(overall['behavior'], overall['sum'])
    plt.title('Total Behavior Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(f"{output_file}_overall.png")
    else:
        plt.show()
    
    # Plot by iteration if there are multiple iterations
    by_iteration = stats['by_iteration']
    if len(by_iteration) > 1:
        behaviors = overall['behavior'].tolist()
        iterations = by_iteration['iteration'].unique()
        
        plt.figure(figsize=(12, 8))
        
        for behavior in behaviors:
            values = [by_iteration.loc[by_iteration['iteration'] == it, (behavior, 'sum')].values[0] 
                     for it in iterations]
            plt.plot(iterations, values, marker='o', label=behavior)
            
        plt.title('Behavior Counts by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(f"{output_file}_by_iteration.png")
        else:
            plt.show()

def print_statistics(stats):
    """Print statistics in a readable format"""
    if stats is None:
        print("No data to display")
        return
    
    print("\n=== OVERALL BEHAVIOR STATISTICS ===")
    print(stats['overall'][['behavior', 'sum', 'mean']].to_string(index=False))
    
    print("\n=== BEHAVIOR STATISTICS BY ITERATION ===")
    by_iteration = stats['by_iteration']
    
    for iteration in by_iteration['iteration'].unique():
        print(f"\nIteration {iteration}:")
        iteration_data = by_iteration[by_iteration['iteration'] == iteration]
        
        # Get behavior columns and reshape for prettier printing
        behaviors = stats['overall']['behavior'].tolist()
        data = []
        
        for behavior in behaviors:
            total = iteration_data[(behavior, 'sum')].values[0]
            mean = iteration_data[(behavior, 'mean')].values[0]
            count = iteration_data[(behavior, 'count')].values[0]
            data.append([behavior, total, mean, count])
        
        # Print as table
        print(pd.DataFrame(data, columns=['Behavior', 'Total', 'Mean', 'Count']).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Analyze behavior counts from results files")
    parser.add_argument("--input", required=True, help="Path to the results JSON file")
    parser.add_argument("--iterations", help="Comma-separated list of iterations to analyze, e.g., '1,2,3' (default: all)")
    parser.add_argument("--output", help="Base name for output files (without extension)")
    parser.add_argument("--plot", action="store_true", help="Generate plots of the behavior counts")
    
    args = parser.parse_args()
    
    # Load the results file
    data = load_results_file(args.input)
    
    # Parse iterations if provided
    iterations_to_analyze = None
    if args.iterations and args.iterations.lower() != "all":
        iterations_to_analyze = [int(it.strip()) for it in args.iterations.split(",")]
    
    # Extract and analyze behavior counts
    behavior_df = extract_behaviors(data, iterations_to_analyze)
    
    if behavior_df is None or behavior_df.empty:
        print(f"No matching data found in {args.input} for the specified iterations")
        return
    
    # Calculate statistics
    stats = calculate_statistics(behavior_df)
    
    # Print statistics
    print_statistics(stats)
    
    # Generate plots if requested
    if args.plot:
        plot_behaviors(stats, args.output)
    
    # Export to CSV if an output file is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Export the raw data
        behavior_df.to_csv(f"{args.output}_raw.csv", index=False)
        
        # Export the statistics
        stats['overall'].to_csv(f"{args.output}_overall_stats.csv", index=False)
        stats['by_iteration'].to_csv(f"{args.output}_iteration_stats.csv", index=False)
        
        print(f"Results exported to {args.output}_*.csv")

if __name__ == "__main__":
    main() 