#!/usr/bin/env python
import json
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_results_file(file_path):
    """Load a behavior analysis results file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_behaviors_from_file(file_path, file_label=None, iterations=None):
    """
    Extract behavior counts from a results file
    
    Args:
        file_path: Path to the results file
        file_label: Optional label for the file source
        iterations: List of iterations to include, or None for all
    
    Returns:
        DataFrame with behavior counts
    """
    data = load_results_file(file_path)
    
    records = []
    
    # Get the field type from the analysis parameters
    text_fields = data.get("analysis_parameters", {}).get("text_fields_searched", [])
    field_type = text_fields[0] if text_fields else "unknown"
    
    # Use the filename as label if not provided
    if file_label is None:
        file_label = Path(file_path).stem
    
    for problem in data.get("results", []):
        problem_id = problem.get("problem_id")
        
        for iteration_data in problem.get("iterations", []):
            iteration = iteration_data.get("iteration")
            
            # Skip iterations not in the specified list
            if iterations is not None and iteration not in iterations:
                continue
                
            field = iteration_data.get("field_analyzed", field_type)
            behavior_counts = iteration_data.get("behavior_counts", {})
            
            record = {
                "problem_id": problem_id,
                "iteration": iteration,
                "field": field,
                "file_source": file_label
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

def combine_behavior_files(file_paths, labels=None, iterations=None):
    """
    Combine behavior counts from multiple results files
    
    Args:
        file_paths: List of paths to results files
        labels: Optional list of labels for the files
        iterations: List of iterations to include, or None for all
    
    Returns:
        DataFrame with combined behavior counts
    """
    # Create labels if not provided
    if labels is None:
        labels = [Path(file_path).stem for file_path in file_paths]
    
    # Extract behaviors from each file
    dataframes = []
    for file_path, label in zip(file_paths, labels):
        df = extract_behaviors_from_file(file_path, label, iterations)
        if df is not None:
            dataframes.append(df)
    
    # Combine the dataframes
    if not dataframes:
        return None
        
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def calculate_statistics(df, group_by=None):
    """
    Calculate statistics on the behavior counts
    
    Args:
        df: DataFrame with behavior counts
        group_by: Column to group statistics by (e.g., 'iteration', 'field', 'file_source')
    
    Returns:
        Dictionary with statistics
    """
    if df is None or df.empty:
        return None
    
    # Identify behavior columns
    behavior_cols = [col for col in df.columns if col not in ["problem_id", "iteration", "field", "file_source"]]
    
    # If no group_by, just calculate overall statistics
    if group_by is None:
        overall_stats = pd.DataFrame({
            'behavior': behavior_cols,
            'mean': [df[col].mean() for col in behavior_cols],
            'sum': [df[col].sum() for col in behavior_cols],
            'count': [df[col].count() for col in behavior_cols]
        })
        
        return {
            'overall': overall_stats,
            'by_group': None
        }
    
    # Calculate statistics grouped by the specified column
    stats_by_group = df.groupby(group_by)[behavior_cols].agg(['mean', 'sum', 'count']).reset_index()
    
    # Calculate overall statistics
    overall_stats = pd.DataFrame({
        'behavior': behavior_cols,
        'mean': [df[col].mean() for col in behavior_cols],
        'sum': [df[col].sum() for col in behavior_cols],
        'count': [df[col].count() for col in behavior_cols]
    })
    
    return {
        'overall': overall_stats,
        'by_group': stats_by_group,
        'group_by': group_by
    }

def plot_statistics(stats, output_file=None):
    """Create visualizations of the behavior statistics"""
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
    
    # Plot by group if available
    if stats['by_group'] is not None and len(stats['by_group']) > 1:
        group_by = stats['group_by']
        by_group = stats['by_group']
        behaviors = overall['behavior'].tolist()
        groups = by_group[group_by].unique()
        
        # For each behavior, plot counts by group
        for behavior in behaviors:
            plt.figure(figsize=(10, 6))
            
            values = [by_group.loc[by_group[group_by] == g, (behavior, 'sum')].values[0] 
                     for g in groups]
            
            plt.bar(groups, values)
            plt.title(f'{behavior} by {group_by}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(f"{output_file}_{behavior}_by_{group_by}.png")
            else:
                plt.show()
                
        # If group_by is 'iteration', create line plots showing behavior trends
        if group_by == 'iteration':
            plt.figure(figsize=(12, 8))
            
            for behavior in behaviors:
                values = [by_group.loc[by_group[group_by] == g, (behavior, 'sum')].values[0] 
                         for g in groups]
                plt.plot(groups, values, marker='o', label=behavior)
                
            plt.title('Behavior Trends by Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(f"{output_file}_trends_by_iteration.png")
            else:
                plt.show()

def print_statistics(stats):
    """Print statistics in a readable format"""
    if stats is None:
        print("No data to display")
        return
    
    print("\n=== OVERALL BEHAVIOR STATISTICS ===")
    print(stats['overall'][['behavior', 'sum', 'mean']].to_string(index=False))
    
    if stats['by_group'] is not None:
        group_by = stats['group_by']
        print(f"\n=== BEHAVIOR STATISTICS BY {group_by.upper()} ===")
        by_group = stats['by_group']
        
        for group_val in by_group[group_by].unique():
            print(f"\n{group_by.capitalize()}: {group_val}")
            group_data = by_group[by_group[group_by] == group_val]
            
            # Get behavior columns and reshape for prettier printing
            behaviors = stats['overall']['behavior'].tolist()
            data = []
            
            for behavior in behaviors:
                total = group_data[(behavior, 'sum')].values[0]
                mean = group_data[(behavior, 'mean')].values[0]
                count = group_data[(behavior, 'count')].values[0]
                data.append([behavior, total, mean, count])
            
            # Print as table
            print(pd.DataFrame(data, columns=['Behavior', 'Total', 'Mean', 'Count']).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Combine and analyze behavior counts from multiple results files")
    parser.add_argument("--input", required=True, help="Comma-separated list of paths to results JSON files")
    parser.add_argument("--labels", help="Comma-separated list of labels for the input files (default: filenames)")
    parser.add_argument("--iterations", help="Comma-separated list of iterations to analyze, e.g., '1,2,3' (default: all)")
    parser.add_argument("--group-by", choices=["iteration", "field", "file_source"], default="file_source",
                      help="Column to group statistics by (default: file_source)")
    parser.add_argument("--output", help="Base name for output files (without extension)")
    parser.add_argument("--plot", action="store_true", help="Generate plots of the behavior counts")
    
    args = parser.parse_args()
    
    # Parse input files
    file_paths = [path.strip() for path in args.input.split(",")]
    
    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]
    
    # Parse iterations if provided
    iterations_to_analyze = None
    if args.iterations and args.iterations.lower() != "all":
        iterations_to_analyze = [int(it.strip()) for it in args.iterations.split(",")]
    
    # Combine behavior data from all files
    combined_df = combine_behavior_files(file_paths, labels, iterations_to_analyze)
    
    if combined_df is None or combined_df.empty:
        print(f"No matching data found in the specified input files")
        return
    
    # Calculate statistics
    stats = calculate_statistics(combined_df, args.group_by)
    
    # Print statistics
    print_statistics(stats)
    
    # Generate plots if requested
    if args.plot:
        plot_statistics(stats, args.output)
    
    # Export to CSV if an output file is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Export the combined data
        combined_df.to_csv(f"{args.output}_combined.csv", index=False)
        
        # Export the statistics
        stats['overall'].to_csv(f"{args.output}_overall_stats.csv", index=False)
        
        if stats['by_group'] is not None:
            stats['by_group'].to_csv(f"{args.output}_by_{args.group_by}_stats.csv", index=False)
        
        print(f"Results exported to {args.output}_*.csv")

if __name__ == "__main__":
    main() 