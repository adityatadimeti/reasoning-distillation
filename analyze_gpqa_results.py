#!/usr/bin/env python3
"""
Analysis script for GPQA Diamond experiment results.
This script analyzes:
1. Which problems were answered correctly vs incorrectly
2. Whether the model changes its answers across iterations
"""

import json
import os
import sys
from collections import defaultdict

# Path to the results file
RESULTS_PATH = "results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond/summarization_8_iter_rzn-R1_summ-V3-approach_focused_gpqa_diamond_20250402_102708/results.json"

def analyze_results(results_path):
    """Analyze the experiment results."""
    print(f"Analyzing results from: {results_path}")
    
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
    
    print(f"\nTotal problems analyzed: {len(results)}")
    
    # Track correct vs incorrect problems
    correct_problems = []
    incorrect_problems = []
    
    # Track answer changes
    problems_with_answer_changes = []
    
    # Track correct answers by iteration
    correct_by_iteration = defaultdict(int)
    
    # Analyze each problem
    for problem in results:
        problem_id = problem.get('problem_id', 'unknown')
        correct_answer = problem.get('correct_answer', 'unknown')
        iterations = problem.get('iterations', [])
        
        if not iterations:
            print(f"Warning: Problem {problem_id} has no iterations")
            continue
        
        # Check for answer changes
        answers = [iter.get('answer', '') for iter in iterations]
        unique_answers = set(answers)
        changed_answers = len(unique_answers) > 1
        
        # Get final answer correctness
        final_iteration = iterations[-1]
        final_answer = final_iteration.get('answer', '')
        is_correct = final_iteration.get('correct', False)
        
        # Track problem status
        if is_correct:
            correct_problems.append({
                'id': problem_id,
                'expected': correct_answer,
                'answer': final_answer,
                'changed_answers': changed_answers,
                'unique_answers': len(unique_answers)
            })
        else:
            incorrect_problems.append({
                'id': problem_id,
                'expected': correct_answer,
                'answer': final_answer,
                'changed_answers': changed_answers,
                'unique_answers': len(unique_answers)
            })
        
        # Track if answers changed
        if changed_answers:
            answer_sequence = []
            for i, iter in enumerate(iterations):
                answer_sequence.append({
                    'iteration': i, 
                    'answer': iter.get('answer', ''),
                    'correct': iter.get('correct', False)
                })
                
            problems_with_answer_changes.append({
                'id': problem_id, 
                'expected': correct_answer,
                'answer_sequence': answer_sequence
            })
        
        # Track correct answers by iteration
        for iter in iterations:
            if iter.get('correct', False):
                correct_by_iteration[iter.get('iteration', 0)] += 1
    
    # Print correct problems
    print(f"\n=== CORRECTLY SOLVED: {len(correct_problems)}/{len(results)} ===")
    for i, prob in enumerate(correct_problems):
        print(f"{i+1}. Problem {prob['id']}")
        print(f"   Expected: {prob['expected']}")
        print(f"   Answer: {prob['answer']}")
        if prob['changed_answers']:
            print(f"   ⚠️ Changed answers {prob['unique_answers']} times across iterations")
        print()
    
    # Print incorrect problems
    print(f"\n=== INCORRECTLY SOLVED: {len(incorrect_problems)}/{len(results)} ===")
    for i, prob in enumerate(incorrect_problems):
        print(f"{i+1}. Problem {prob['id']}")
        print(f"   Expected: {prob['expected']}")
        print(f"   Answer: {prob['answer']}")
        if prob['changed_answers']:
            print(f"   ⚠️ Changed answers {prob['unique_answers']} times across iterations")
        print()
    
    # Print problems with changing answers
    print(f"\n=== PROBLEMS WITH CHANGING ANSWERS: {len(problems_with_answer_changes)}/{len(results)} ===")
    for i, prob in enumerate(problems_with_answer_changes):
        print(f"{i+1}. Problem {prob['id']}")
        print(f"   Expected: {prob['expected']}")
        print(f"   Answer sequence:")
        for ans in prob['answer_sequence']:
            correct_mark = "✓" if ans['correct'] else "✗"
            print(f"     Iteration {ans['iteration']}: {ans['answer']} {correct_mark}")
        print()
    
    # Print correct answers by iteration
    print("\n=== CORRECT ANSWERS BY ITERATION ===")
    iterations = sorted(correct_by_iteration.keys())
    for i in iterations:
        percent = (correct_by_iteration[i] / len(results)) * 100
        print(f"Iteration {i}: {correct_by_iteration[i]}/{len(results)} correct ({percent:.1f}%)")

if __name__ == "__main__":
    # Allow overriding results path from command line
    results_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_PATH
    analyze_results(results_path) 