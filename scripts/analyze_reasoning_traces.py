#!/usr/bin/env python3
"""
This script extracts and analyzes reasoning traces for problems where the model changed its answers.
It helps understand what the model was thinking when it changed its approach.
"""

import json
import sys
import os
import re
from collections import defaultdict

# Path to the results file
RESULTS_PATH = "results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond/summarization_8_iter_rzn-R1_summ-V3-approach_focused_gpqa_diamond_20250402_102708/results.json"

def extract_key_reasoning(full_reasoning, max_chars=300):
    """Extract a snippet of the reasoning for display."""
    # Remove think tags and extra whitespace
    clean_text = re.sub(r'</?think>', '', full_reasoning)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # If reasoning is short enough, return it all
    if len(clean_text) <= max_chars:
        return clean_text
    
    # Otherwise return a snippet from the beginning and end
    half_len = max_chars // 2 - 3  # Leave room for ellipsis
    return clean_text[:half_len] + "..." + clean_text[-half_len:]

def analyze_changing_answers(results_path, problem_id=None):
    """Analyze reasoning traces for problems where answers changed across iterations."""
    print(f"Analyzing reasoning traces from: {results_path}")
    
    # Load the results file
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Results file contains invalid JSON")
        return
    
    results = data.get('results', [])
    if not results:
        print("No results found in the data")
        return
    
    # Filter to specific problem if requested
    if problem_id:
        results = [r for r in results if r.get('problem_id') == problem_id]
        if not results:
            print(f"No problem found with ID: {problem_id}")
            return
    
    print(f"\nTotal problems analyzed: {len(results)}")
    
    # Get problems with changing answers
    problems_with_changes = []
    for problem in results:
        problem_id = problem.get('problem_id', 'unknown')
        iterations = problem.get('iterations', [])
        
        if not iterations:
            continue
            
        # Check for answer changes
        answers = [iter.get('answer', '') for iter in iterations]
        unique_answers = set(answers)
        changed_answers = len(unique_answers) > 1
        
        if changed_answers:
            problems_with_changes.append(problem)
    
    print(f"Found {len(problems_with_changes)} problems with changing answers\n")
    
    for problem in problems_with_changes:
        problem_id = problem.get('problem_id', 'unknown')
        question = problem.get('question', '')
        correct_answer = problem.get('correct_answer', 'unknown')
        iterations = problem.get('iterations', [])
        
        # Print problem details
        print(f"===== PROBLEM {problem_id} =====")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        print(f"Expected answer: {correct_answer}")
        print("\nAnswers across iterations:")
        
        # Find iterations where answer changed
        answer_changes = []
        prev_answer = None
        for i, iter in enumerate(iterations):
            current_answer = iter.get('answer', '')
            if i == 0 or current_answer != prev_answer:
                answer_changes.append(i)
                prev_answer = current_answer
        
        # Print answers and reasoning where changes occurred
        for i in answer_changes:
            iter = iterations[i]
            answer = iter.get('answer', '')
            reasoning = iter.get('reasoning', '')
            is_correct = iter.get('correct', False)
            
            print(f"\n--- ITERATION {i} {'✓' if is_correct else '✗'} ---")
            print(f"Answer: {answer}")
            print("\nReasoning snippet:")
            
            # Extract a snippet of reasoning to show
            if reasoning:
                print(extract_key_reasoning(reasoning))
            else:
                print("No reasoning trace available")
            
            # If this is the first iteration, show the summary too
            if i > 0 and 'summary' in iterations[i-1]:
                print("\nPrevious summary that led to this change:")
                summary = iterations[i-1].get('summary', '')
                print(extract_key_reasoning(summary))
        
        print("\n" + "=" * 80 + "\n")

def analyze_specific_iteration_change(results_path, problem_id, from_iter, to_iter):
    """Analyze how reasoning changed between specific iterations for a problem."""
    print(f"Analyzing change for problem {problem_id} from iteration {from_iter} to {to_iter}")
    
    # Load the results file
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return
    
    # Find the specific problem
    problem = None
    for p in data.get('results', []):
        if p.get('problem_id') == problem_id:
            problem = p
            break
    
    if not problem:
        print(f"No problem found with ID: {problem_id}")
        return
    
    iterations = problem.get('iterations', [])
    if from_iter >= len(iterations) or to_iter >= len(iterations):
        print(f"Iteration index out of range. Problem has {len(iterations)} iterations.")
        return
    
    # Get iterations
    iter_from = iterations[from_iter]
    iter_to = iterations[to_iter]
    
    # Get summaries
    summary = None
    if from_iter < to_iter and from_iter < len(iterations) - 1:
        summary = iterations[from_iter].get('summary', '')
    
    # Print detailed comparison
    print("\n===== DETAILED COMPARISON =====")
    print(f"Problem: {problem.get('question', '')[:200]}...")
    print(f"Expected answer: {problem.get('correct_answer', '')}")
    
    print(f"\n--- ITERATION {from_iter} ---")
    print(f"Answer: {iter_from.get('answer', '')}")
    print("\nFull Reasoning:")
    print(iter_from.get('reasoning', 'No reasoning available'))
    
    if summary:
        print("\n--- SUMMARY ---")
        print(summary)
    
    print(f"\n--- ITERATION {to_iter} ---")
    print(f"Answer: {iter_to.get('answer', '')}")
    print("\nFull Reasoning:")
    print(iter_to.get('reasoning', 'No reasoning available'))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze reasoning traces from experiment results")
    parser.add_argument("--results", default=RESULTS_PATH, help="Path to results.json file")
    parser.add_argument("--problem", help="Focus on a specific problem ID")
    parser.add_argument("--from-iter", type=int, help="Compare from this iteration")
    parser.add_argument("--to-iter", type=int, help="Compare to this iteration")
    
    args = parser.parse_args()
    
    if args.problem and args.from_iter is not None and args.to_iter is not None:
        analyze_specific_iteration_change(args.results, args.problem, args.from_iter, args.to_iter)
    else:
        analyze_changing_answers(args.results, args.problem) 