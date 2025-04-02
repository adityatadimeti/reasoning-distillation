#!/usr/bin/env python3
"""
A simple script to show the progression of answers across iterations for each problem.
This helps visualize how the model's answers change (or don't change) over iterations.
"""

import json
import sys
from collections import defaultdict

# Path to the results file
RESULTS_PATH = "results/summarization_8_iter_rzn-R1_summ-V3_gpqa_diamond/summarization_8_iter_rzn-R1_summ-V3-approach_focused_gpqa_diamond_20250402_102708/results.json"

def show_answer_progression(results_path):
    """Show how answers change across iterations for each problem."""
    print(f"Loading results from: {results_path}")
    
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
    print("=" * 80)
    
    # Analyze each problem
    for problem_idx, problem in enumerate(results):
        problem_id = problem.get('problem_id', 'unknown')
        question = problem.get('question', '')
        correct_answer = problem.get('correct_answer', 'unknown')
        iterations = problem.get('iterations', [])
        
        # Display problem overview
        print(f"\nPROBLEM {problem_idx+1}: {problem_id}")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        
        # Prominently display the ground truth
        print("\n📌 GROUND TRUTH ANSWER:")
        print(f"   {correct_answer}")
        
        if not iterations:
            print("No iterations data available for this problem")
            continue
        
        # Track if the answer changes across iterations
        answers = [iter.get('answer', '') for iter in iterations]
        unique_answers = set(answers)
        answer_changed = len(unique_answers) > 1
        
        # Track answer correctness
        correctness = [iter.get('correct', False) for iter in iterations]
        final_correct = correctness[-1] if correctness else False
        
        # Print a clear table header
        print("\nAnswer progression:")
        print("Iteration | Correct? | Answer" + " " * 30 + "| Match with Ground Truth?")
        print("-" * 100)
        
        # Show each iteration's answer
        for i, iter in enumerate(iterations):
            answer = iter.get('answer', '')
            is_correct = iter.get('correct', False)
            
            # Highlight if the answer changed from the previous iteration
            changed_marker = ""
            if i > 0 and answer != iterations[i-1].get('answer', ''):
                changed_marker = " [CHANGED]"
            
            # Format the answer for display
            formatted_answer = answer
            if len(formatted_answer) > 40:
                formatted_answer = formatted_answer[:37] + "..."
            
            # Compare with ground truth (simplistic comparison)
            # This will only catch exact matches or obvious equivalents
            match_marker = "❌ No"
            if is_correct:
                match_marker = "✅ Yes"
            elif answer == correct_answer:  # Check if strings match exactly
                match_marker = "✅ Yes (but not marked as correct)"
            elif correct_answer.replace(" ", "") == answer.replace(" ", ""):  # Try without spaces
                match_marker = "✅ Yes (format differs)"
            # Special case for TeV to GeV conversion (problem 7)
            elif "TeV" in answer and "GeV" in correct_answer:
                try:
                    # Extract the number from the TeV answer
                    tev_value = ''.join(filter(lambda x: x.isdigit() or x == '.', answer.split("TeV")[0]))
                    tev_value = float(tev_value)
                    # Extract the number from the GeV answer
                    gev_str = correct_answer.replace("*1e", "e")
                    gev_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', 'e', '+', '-'], gev_str.split("GeV")[0]))
                    gev_value = float(gev_value)
                    # Compare (1 TeV = 1000 GeV)
                    if abs(tev_value * 1000 - gev_value) < 1:
                        match_marker = "✅ Yes (unit conversion: TeV to GeV)"
                except:
                    pass
            
            # Print with clear alignment
            print(f"{i:9} | {'✓' if is_correct else '✗':8} | {formatted_answer:<40}{changed_marker} | {match_marker}")
        
        # Summary for this problem
        print("\nSummary:")
        if answer_changed:
            print(f"- Answer changed {len(unique_answers)-1} times across iterations")
        else:
            print("- Answer remained the same across all iterations")
        
        print(f"- Final answer correct: {'Yes' if final_correct else 'No'}")
        print(f"- Iterations with correct answers: {sum(correctness)}/{len(iterations)}")
        print("=" * 80)

if __name__ == "__main__":
    # Allow overriding results path from command line
    results_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_PATH
    show_answer_progression(results_path) 