#!/usr/bin/env python3
"""
Script to extract only the reasoning and post_think_summary fields from a results JSON file.
This creates a simplified JSON containing just these fields for analysis.
"""

import json
import argparse
import os
from datetime import datetime

def extract_reasoning_summaries(input_file, output_file=None):
    """
    Extract only the reasoning, summary, and post_think_summary fields from a results JSON file.
    
    Args:
        input_file: Path to input JSON result file
        output_file: Path to save the extracted data
    
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
    
    # Create a new structure to hold only the extracted data
    extracted_data = {
        "experiment_name": results.get("experiment_name", ""),
        "problems": []
    }
    
    total_entries = 0
    
    # Process each problem
    for problem_idx, problem in enumerate(results['results']):
        problem_id = problem.get('problem_id', str(problem_idx))
        question = problem.get('question', '')
        
        problem_data = {
            "problem_id": problem_id,
            "question": question,
            "iterations": []
        }
        
        # Process iterations if they exist
        if 'iterations' in problem:
            for iteration in problem['iterations']:
                iteration_num = iteration.get('iteration', 0)
                
                # Extract only the reasoning and post_think_summary
                iteration_data = {
                    "iteration": iteration_num
                }
                
                # Add reasoning if it exists
                if 'reasoning' in iteration:
                    iteration_data["reasoning"] = iteration['reasoning']
                    total_entries += 1

                if 'summary' in iteration:
                    iteration_data["summary"] = iteration['summary']
                    total_entries += 1

                # Add post_think_summary if it exists
                if 'post_think_summary' in iteration:
                    iteration_data["post_think_summary"] = iteration['post_think_summary']
                    total_entries += 1
                
                problem_data["iterations"].append(iteration_data)
        
        extracted_data["problems"].append(problem_data)
    
    # Generate output file path if not provided
    if not output_file:
        dirname = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        basename_no_ext = os.path.splitext(basename)[0]
        output_file = os.path.join(dirname, f"{basename_no_ext}_reasoning_summaries.json")
    
    # Save extracted data
    try:
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        # Calculate size
        size = os.path.getsize(output_file)
        
        print(f"\nSuccessfully extracted reasoning, summary, and post_think_summary fields:")
        print(f"- Original file: {input_file}")
        print(f"- Total entries extracted: {total_entries}")
        print(f"- Output size: {size / 1024:.2f} KB")
        print(f"\nExtracted data saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving extracted data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract only the reasoning, summary, and post_think_summary fields from a results JSON file"
    )
    parser.add_argument("input_file", help="Path to the results.json file")
    parser.add_argument("--output", "-o", help="Path to save the extracted data (optional)")
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Extract fields and save to output file
    output_file = extract_reasoning_summaries(args.input_file, args.output)
    if output_file:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main()) 