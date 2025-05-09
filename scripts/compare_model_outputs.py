#!/usr/bin/env python
import json
import sys
import os
import re
from typing import Dict, Any, List
from difflib import SequenceMatcher

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

def clean_text(text: str) -> str:
    """Clean text by removing tags and normalizing whitespace."""
    # Remove <think> tags
    text = re.sub(r'</?think>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()

def compare_model_outputs(cont_file: str, summ_file: str, num_problems: int = 5) -> None:
    """Compare model outputs between continuation and summarization experiments."""
    cont_results = load_results(cont_file)
    summ_results = load_results(summ_file)
    
    # Create mappings from problem_id to results
    cont_map = {p.get("problem_id", ""): p for p in cont_results}
    summ_map = {p.get("problem_id", ""): p for p in summ_results}
    
    # Find common problem IDs
    common_ids = list(set(cont_map.keys()).intersection(set(summ_map.keys())))
    
    print(f"Found {len(common_ids)} common problems between the two result sets")
    
    # Limit to requested number of problems
    sample_ids = common_ids[:min(num_problems, len(common_ids))]
    
    overall_similarities = []
    
    # Compare outputs for each problem
    for problem_id in sample_ids:
        cont_problem = cont_map[problem_id]
        summ_problem = summ_map[problem_id]
        
        # Get iteration 0 data
        cont_iter0 = cont_problem.get("iterations", [])[0] if cont_problem.get("iterations") else {}
        summ_iter0 = summ_problem.get("iterations", [])[0] if summ_problem.get("iterations") else {}
        
        # Get model outputs
        cont_output = cont_iter0.get("reasoning_output", "")
        summ_output = summ_iter0.get("reasoning", "")
        
        # Clean texts
        cont_cleaned = clean_text(cont_output)
        summ_cleaned = clean_text(summ_output)
        
        # Calculate similarity
        similarity = text_similarity(cont_cleaned, summ_cleaned)
        overall_similarities.append(similarity)
        
        print(f"\nProblem ID: {problem_id}")
        print(f"Text similarity: {similarity:.2f}")
        
        # Extract first few lines for comparison
        cont_first_lines = '\n'.join(cont_output.split('\n')[:3])
        summ_first_lines = '\n'.join(summ_output.split('\n')[:3])
        
        print("\nContinuation (first 3 lines):")
        print(cont_first_lines)
        
        print("\nSummarization (first 3 lines):")
        print(summ_first_lines)
        
        # Check if both extracted the same answer
        cont_answer = cont_iter0.get("answer", "None")
        summ_answer = summ_iter0.get("answer", "None")
        print(f"\nAnswers: Continuation='{cont_answer}', Summarization='{summ_answer}'")
    
    # Calculate average similarity
    if overall_similarities:
        avg_similarity = sum(overall_similarities) / len(overall_similarities)
        print(f"\nAverage text similarity across {len(overall_similarities)} problems: {avg_similarity:.2f}")
        print(f"Interpretation: 0.0=completely different, 1.0=identical")
        
        if avg_similarity > 0.8:
            print("The model outputs are very similar despite different prompts.")
        elif avg_similarity > 0.5:
            print("The model outputs share significant content but differ in expression.")
        else:
            print("The model outputs are substantially different.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_model_outputs.py <continuation_results.json> <summarization_results.json> [num_problems]")
        sys.exit(1)
    
    cont_file = sys.argv[1]
    summ_file = sys.argv[2]
    num_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if not os.path.exists(cont_file):
        print(f"Error: Continuation results file {cont_file} not found")
        sys.exit(1)
    
    if not os.path.exists(summ_file):
        print(f"Error: Summarization results file {summ_file} not found")
        sys.exit(1)
    
    compare_model_outputs(cont_file, summ_file, num_problems) 