#!/usr/bin/env python3
"""
Script to strip verbose fields from results.json files to create a smaller version
without the lengthy text fields (reasoning, summary, post_think_summary, initial_reasoning).
"""

import json
import argparse
import os
from datetime import datetime
import copy

def strip_verbose_fields(input_file, output_file=None):
    """
    Create a copy of the results.json file without the verbose text fields.
    
    Args:
        input_file: Path to input JSON result file
        output_file: Path to save the stripped result file
    
    Returns:
        Path to the output file
    """
    print(f"Loading results file from {input_file}")
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return None
    
    # Make a deep copy to avoid modifying original data
    stripped_results = copy.deepcopy(results)
    
    # Fields to strip from each iteration
    iteration_verbose_fields = ['reasoning', 'summary', 'post_think_summary']
    
    # Fields to strip at the problem level
    problem_verbose_fields = ['initial_reasoning', 'summary', 'improved_reasoning']
    
    # Count of original file size before stripping
    original_size = os.path.getsize(input_file)
    
    # Strip verbose fields from each problem
    for problem in stripped_results['results']:
        # Strip problem-level verbose fields
        for field in problem_verbose_fields:
            if field in problem:
                problem[field] = "[content stripped]"
        
        # Strip iteration-level verbose fields
        if 'iterations' in problem:
            for iteration in problem['iterations']:
                for field in iteration_verbose_fields:
                    if field in iteration:
                        iteration[field] = "[content stripped]"
    
    # Generate output file path if not provided
    if not output_file:
        dirname = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        basename_no_ext = os.path.splitext(basename)[0]
        output_file = os.path.join(dirname, f"{basename_no_ext}_stripped.json")
    
    # Save stripped results
    try:
        with open(output_file, 'w') as f:
            json.dump(stripped_results, f, indent=2)
        
        # Calculate size reduction
        new_size = os.path.getsize(output_file)
        size_reduction = original_size - new_size
        size_reduction_percent = (size_reduction / original_size) * 100
        
        print(f"\nSuccessfully stripped verbose fields from results file:")
        print(f"- Original file: {input_file}")
        print(f"- Original size: {original_size / 1024:.2f} KB")
        print(f"- New size: {new_size / 1024:.2f} KB")
        print(f"- Size reduction: {size_reduction / 1024:.2f} KB ({size_reduction_percent:.2f}%)")
        print(f"\nStripped results saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving stripped results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Strip verbose fields from results.json files")
    parser.add_argument("input_file", help="Path to the results.json file")
    parser.add_argument("--output", "-o", help="Path to save the stripped result file (optional)")
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Strip verbose fields and save to output file
    output_file = strip_verbose_fields(args.input_file, args.output)
    if output_file:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main()) 