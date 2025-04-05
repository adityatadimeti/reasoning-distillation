#!/usr/bin/env python3
"""
Script to merge multiple experiment result files into a single combined file.
This allows analyzing results from separate experiment runs (e.g., different index ranges)
using the same analysis tools like show_answer_progression.py.
"""

import json
import argparse
import os
import glob
from datetime import datetime
import copy

def merge_results(input_files, output_file=None):
    """
    Merge multiple result JSON files into a single combined file.
    
    Args:
        input_files: List of paths to input JSON result files
        output_file: Path to save the merged result file
    
    Returns:
        Path to the merged result file
    """
    if not input_files:
        print("Error: No input files provided")
        return None
    
    # Load the first file to get the base structure
    print(f"Loading base structure from {input_files[0]}")
    try:
        with open(input_files[0], 'r') as f:
            base_results = json.load(f)
    except Exception as e:
        print(f"Error loading {input_files[0]}: {e}")
        return None
    
    # Make a deep copy to avoid modifying original data
    merged_results = copy.deepcopy(base_results)
    
    # Create a mapping of existing problem IDs for quick lookup
    existing_problems = {p['problem_id']: True for p in merged_results['results']}
    
    # Initialize counters for statistics
    total_problems_added = 0
    duplicates_skipped = 0
    
    # Process other input files
    for file_path in input_files[1:]:
        print(f"Merging results from {file_path}")
        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # Add problems from this file if not already in the base results
            problems_added = 0
            for problem in file_data['results']:
                problem_id = problem['problem_id']
                if problem_id not in existing_problems:
                    merged_results['results'].append(problem)
                    existing_problems[problem_id] = True
                    problems_added += 1
                else:
                    duplicates_skipped += 1
            
            total_problems_added += problems_added
            print(f"  Added {problems_added} new problems")
            
            # Merge metadata if needed
            if 'config' in file_data and file_data['config'] != merged_results.get('config'):
                # Handle config differences if needed
                pass
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort results by problem_id for consistency
    merged_results['results'].sort(key=lambda x: x['problem_id'])
    
    # Update timestamp 
    merged_results['timestamp'] = datetime.now().timestamp()
    
    # Generate output file path if not provided
    if not output_file:
        dirname = os.path.dirname(input_files[0])
        basename = f"merged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = os.path.join(dirname, basename)
    
    # Save merged results
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_results, f, indent=2)
        print(f"\nSuccessfully merged {len(input_files)} result files:")
        print(f"- Base file: {input_files[0]}")
        print(f"- Additional files: {len(input_files)-1}")
        print(f"- Total problems: {len(merged_results['results'])}")
        print(f"- New problems added: {total_problems_added}")
        print(f"- Duplicate problems skipped: {duplicates_skipped}")
        print(f"\nMerged results saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving merged results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Merge multiple experiment result files")
    parser.add_argument("--input", "-i", nargs="+", help="Paths to input result JSON files")
    parser.add_argument("--output", "-o", help="Path to save the merged result file (optional)")
    parser.add_argument("--latest", "-l", type=int, default=0, 
                       help="Use the latest N result files from the most recent experiment directory (optional)")
    parser.add_argument("--experiment", "-e", 
                       help="Experiment name to search for (e.g., gpqa_diamond_mc)")
    parser.add_argument("--results-dir", "-d", default="./results",
                       help="Base results directory (default: ./results)")
    
    args = parser.parse_args()
    
    input_files = []
    
    # Handle the case where we want to use the latest N result files
    if args.latest > 0:
        if not args.experiment:
            print("Error: --experiment is required when using --latest")
            return
        
        # Find directories related to the experiment
        search_path = os.path.join(args.results_dir, f"*{args.experiment}*")
        exp_dirs = sorted(glob.glob(search_path), key=os.path.getmtime, reverse=True)
        
        if not exp_dirs:
            print(f"Error: No experiment directories found matching {args.experiment}")
            return
        
        # Find the latest subdirectory in the experiment directory
        latest_exp_dir = exp_dirs[0]
        subdirs = sorted(glob.glob(os.path.join(latest_exp_dir, "*")), key=os.path.getmtime, reverse=True)
        
        if not subdirs:
            print(f"Error: No subdirectories found in {latest_exp_dir}")
            return
        
        # Find result.json files in these directories
        result_files = []
        for subdir in subdirs[:args.latest]:
            result_path = os.path.join(subdir, "results.json")
            if os.path.exists(result_path):
                result_files.append(result_path)
        
        if not result_files:
            print(f"Error: No result.json files found in the latest subdirectories")
            return
        
        input_files = result_files
        print(f"Using the latest {len(input_files)} result files from experiment {args.experiment}")
    
    # Use explicitly provided input files if specified
    elif args.input:
        input_files = args.input
    
    # Final check
    if not input_files:
        print("Error: No input files specified. Use --input or --latest")
        return
    
    # Merge the results
    merge_results(input_files, args.output)

if __name__ == "__main__":
    main() 