#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(json_data, dict) and "results" in json_data:
        results = json_data["results"]
    else:
        results = json_data
    
    return results

def check_iteration0_match(continuation_file: str, converted_file: str) -> None:
    """Check if iteration 0 in continuation results matches the converted summarization data."""
    continuation_results = load_results(continuation_file)
    converted_results = load_results(converted_file)
    
    # Create mappings from problem_id to results
    cont_map = {p.get("problem_id", ""): p for p in continuation_results}
    conv_map = {p.get("problem_id", ""): p for p in converted_results}
    
    # Find common problem IDs
    common_ids = set(cont_map.keys()).intersection(set(conv_map.keys()))
    total_common = len(common_ids)
    
    print(f"Found {total_common} common problems between the two result sets")
    
    # Check matching metrics
    matching_reasoning = 0
    matching_answers = 0
    
    # Sample problems to display
    sample_count = 0
    max_samples = 3
    
    for problem_id in common_ids:
        cont_problem = cont_map[problem_id]
        conv_problem = conv_map[problem_id]
        
        # Get iteration 0 data
        cont_iter0 = cont_problem.get("iterations", [])[0] if cont_problem.get("iterations") else {}
        conv_iter0 = conv_problem.get("iterations", [])[0] if conv_problem.get("iterations") else {}
        
        # Skip if either is missing
        if not cont_iter0 or not conv_iter0:
            continue
        
        # Check if iteration 0 reasoning matches
        is_matching_reasoning = False
        if "reasoning_output" in cont_iter0 and "reasoning_output" in conv_iter0:
            is_matching_reasoning = cont_iter0["reasoning_output"] == conv_iter0["reasoning_output"]
            
        # Check if answers match
        is_matching_answer = False
        if "answer" in cont_iter0 and "answer" in conv_iter0:
            is_matching_answer = cont_iter0["answer"] == conv_iter0["answer"]
        
        # Count matches
        if is_matching_reasoning:
            matching_reasoning += 1
        
        if is_matching_answer:
            matching_answers += 1
        
        # Display a few sample problems for inspection
        if sample_count < max_samples:
            print(f"\nProblem ID: {problem_id}")
            print(f"  Reasoning match: {is_matching_reasoning}")
            print(f"  Answer match: {is_matching_answer}")
            
            # If not matching, show first few chars of each
            if not is_matching_reasoning:
                cont_reasoning = cont_iter0.get("reasoning_output", "")[:100] + "..."
                conv_reasoning = conv_iter0.get("reasoning_output", "")[:100] + "..."
                print(f"  Continuation reasoning: {cont_reasoning}")
                print(f"  Converted reasoning: {conv_reasoning}")
            
            sample_count += 1
    
    # Print overall statistics
    print(f"\nSummary statistics:")
    print(f"  Total common problems: {total_common}")
    print(f"  Matching reasoning: {matching_reasoning}/{total_common} ({matching_reasoning/total_common*100:.1f}%)")
    print(f"  Matching answers: {matching_answers}/{total_common} ({matching_answers/total_common*100:.1f}%)")
    
    if matching_reasoning == total_common:
        print("\n✅ All iteration 0 reasoning matches perfectly, confirming the continuation experiment used the converted reasoning data.")
    else:
        print("\n⚠️ Not all iteration 0 reasoning matches. The continuation experiment may not have used the converted data correctly.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_iter0_match.py <continuation_results.json> <converted_input.json>")
        sys.exit(1)
    
    continuation_file = sys.argv[1]
    converted_file = sys.argv[2]
    
    if not os.path.exists(continuation_file):
        print(f"Error: Continuation results file {continuation_file} not found")
        sys.exit(1)
    
    if not os.path.exists(converted_file):
        print(f"Error: Converted summarization file {converted_file} not found")
        sys.exit(1)
    
    check_iteration0_match(continuation_file, converted_file) 