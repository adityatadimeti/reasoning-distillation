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

def compare_iteration0(cont_file: str, summ_file: str) -> None:
    """Compare iteration 0 data between continuation and summarization results."""
    cont_results = load_results(cont_file)
    summ_results = load_results(summ_file)
    
    # Create mappings from problem_id to results
    cont_map = {p.get("problem_id", ""): p for p in cont_results}
    summ_map = {p.get("problem_id", ""): p for p in summ_results}
    
    # Find common problem IDs
    common_ids = set(cont_map.keys()).intersection(set(summ_map.keys()))
    
    print(f"Found {len(common_ids)} common problems between the two result sets")
    
    # Check if iteration 0 prompts are the same
    print("\nComparing prompt templates:")
    
    # Take a sample problem to compare
    if common_ids:
        sample_id = list(common_ids)[0]
        
        # Get iteration 0 data
        cont_iter0 = cont_map[sample_id].get("iterations", [])[0] if cont_map[sample_id].get("iterations") else {}
        summ_iter0 = summ_map[sample_id].get("iterations", [])[0] if summ_map[sample_id].get("iterations") else {}
        
        # Get prompt for the first 100 chars to see format
        cont_prompt = cont_iter0.get("prompt", "")[:100] + "..." if cont_iter0.get("prompt") else "N/A"
        summ_prompt = summ_iter0.get("reasoning", "")[:100] + "..." if summ_iter0.get("reasoning") else "N/A"
        
        print(f"Continuation prompt format: {cont_prompt}")
        print(f"Summarization prompt format: {summ_prompt}")
        
        # Check if they might be the same format but different implementations
        if cont_prompt != "N/A" and summ_prompt != "N/A":
            print(f"Prompt format similarity: {'SIMILAR' if len(set(cont_prompt[:50]).intersection(set(summ_prompt[:50]))) > 25 else 'DIFFERENT'}")
    
    # Compare iteration 0 correctness rates
    cont_correct = sum(1 for p in cont_results if p.get("iterations") and 
                      p.get("iterations")[0].get("correct", False))
    summ_correct = sum(1 for p in summ_results if p.get("iterations") and 
                      p.get("iterations")[0].get("correct", False))
    
    print(f"\nIteration 0 correct answers:")
    print(f"Continuation: {cont_correct}/{len(cont_results)} ({cont_correct/len(cont_results)*100:.1f}%)")
    print(f"Summarization: {summ_correct}/{len(summ_results)} ({summ_correct/len(summ_results)*100:.1f}%)")
    
    # Compare regraded continuation results
    cont_regraded_file = cont_file.replace(".json", "_regraded.json")
    if os.path.exists(cont_regraded_file):
        cont_regraded = load_results(cont_regraded_file)
        cont_regraded_correct = sum(1 for p in cont_regraded if p.get("iterations") and 
                                  p.get("iterations")[0].get("regraded_correct", False))
        print(f"Continuation (regraded): {cont_regraded_correct}/{len(cont_regraded)} ({cont_regraded_correct/len(cont_regraded)*100:.1f}%)")
    
    # Check answer extraction field names
    print("\nAnswer extraction field names:")
    if common_ids:
        sample_id = list(common_ids)[0]
        
        # Get iteration 0 data again
        cont_iter0 = cont_map[sample_id].get("iterations", [])[0] if cont_map[sample_id].get("iterations") else {}
        summ_iter0 = summ_map[sample_id].get("iterations", [])[0] if summ_map[sample_id].get("iterations") else {}
        
        print(f"Continuation has fields: {', '.join(cont_iter0.keys())}")
        print(f"Summarization has fields: {', '.join(summ_iter0.keys())}")
        
        # Compare answer fields
        cont_answer_field = "answer" if "answer" in cont_iter0 else "N/A"
        summ_answer_field = "answer" if "answer" in summ_iter0 else "N/A"
        
        print(f"\nContinuation stores answers in: {cont_answer_field}")
        print(f"Summarization stores answers in: {summ_answer_field}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_iteration0.py <continuation_results.json> <summarization_results.json>")
        sys.exit(1)
    
    cont_file = sys.argv[1]
    summ_file = sys.argv[2]
    
    if not os.path.exists(cont_file):
        print(f"Error: Continuation results file {cont_file} not found")
        sys.exit(1)
    
    if not os.path.exists(summ_file):
        print(f"Error: Summarization results file {summ_file} not found")
        sys.exit(1)
    
    compare_iteration0(cont_file, summ_file) 