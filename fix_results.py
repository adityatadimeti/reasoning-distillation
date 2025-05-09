#!/usr/bin/env python
import json
import sys
import os
import re
import copy
from typing import Dict, Any, List, Optional

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_boxed_content(text: str) -> Optional[str]:
    """Extract boxed content using regex."""
    if not text:
        return None
        
    # Look for \boxed{content} pattern
    pattern = r'\\boxed\{([^{}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # Return the last match
    return None

def fix_results(input_file: str, output_file: str) -> Dict[str, Any]:
    """Fix results by using regex extraction when math extractor returns None."""
    results_data = load_results(input_file)
    
    # Make a deep copy to avoid modifying the original
    fixed_results = copy.deepcopy(results_data)
    
    # Handle both list and dict formats
    if isinstance(fixed_results, dict) and "results" in fixed_results:
        results_list = fixed_results["results"]
    else:
        results_list = fixed_results
    
    none_answers_fixed = 0
    total_iterations = 0
    
    # Process each problem
    for problem in results_list:
        iterations = problem.get("iterations", [])
        total_iterations += len(iterations)
        
        # Process each iteration
        for iteration in iterations:
            # If the stored answer is None, try to extract with regex
            if iteration.get("answer") is None:
                # First try reasoning_output
                reasoning_output = iteration.get("reasoning_output", "")
                extracted_answer = check_boxed_content(reasoning_output)
                
                # If that fails, try the full extraction text if available
                if extracted_answer is None and "reasoning_full_for_extraction" in iteration:
                    full_text = iteration.get("reasoning_full_for_extraction", "")
                    extracted_answer = check_boxed_content(full_text)
                
                # If we found an answer with regex, update it
                if extracted_answer is not None:
                    iteration["answer"] = extracted_answer
                    none_answers_fixed += 1
    
    # Save the fixed results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_results, f, indent=2)
    
    print(f"Results fixed and saved to {output_file}")
    print(f"Fixed {none_answers_fixed} None answers out of {total_iterations} total iterations")
    
    return fixed_results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_results.py <input_results.json> <output_results.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    
    fix_results(input_file, output_file) 