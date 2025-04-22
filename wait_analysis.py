import json
import argparse
import os
from collections import defaultdict
import re

def normalize_answer(answer: str) -> str:
    """Normalize the answer by removing certain characters and extra spaces."""
    if answer is None:
        return None
    # Remove $, commas, and percentage signs
    answer = str(answer).replace('$', '').replace(',', '').replace('%', '')
    # Remove leading/trailing whitespace
    answer = answer.strip()
    # Standardize spacing around common math symbols if needed (optional)
    # Example: answer = re.sub(r'\s*([+\-*/=()])\s*', r'\1', answer) 
    return answer

def compare_answers(extracted_answer: str, correct_answer: str) -> bool:
    """Compare extracted answer with correct answer after normalization."""
    norm_extracted = normalize_answer(extracted_answer)
    norm_correct = normalize_answer(correct_answer)
    
    if norm_extracted is None:
        return False
        
    # Simple equality check after normalization
    return norm_extracted == norm_correct

def analyze_iteration_accuracy(results_file_path: str):
    """
    Analyzes the results.json file to calculate accuracy per iteration.

    Args:
        results_file_path: Path to the results.json file.
    """
    if not os.path.exists(results_file_path):
        print(f"Error: Results file not found at {results_file_path}")
        return

    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {results_file_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading file {results_file_path}: {e}")
        return

    if 'results' not in data:
        print("Error: 'results' key not found in JSON data.")
        return

    correct_counts = defaultdict(int)
    attempt_counts = defaultdict(int)
    flagged_issues = [] # To store details of errors or missing answers

    for problem in data.get('results', []):
        problem_id = problem.get('problem_id', 'Unknown ID')
        correct_answer_raw = problem.get('correct_answer')
        
        if correct_answer_raw is None:
             print(f"Warning: Problem {problem_id} missing 'correct_answer'. Skipping.")
             continue

        iterations_data = problem.get('iterations', [])
        if not isinstance(iterations_data, list):
             print(f"Warning: Problem {problem_id} has invalid 'iterations' data type. Skipping.")
             continue

        for iteration_data in iterations_data:
            iter_num = iteration_data.get('iteration')
            if iter_num is None:
                flagged_issues.append({
                    "problem_id": problem_id,
                    "iteration": "Unknown",
                    "issue": "Iteration number missing",
                    "details": iteration_data
                })
                continue
                
            # Count every iteration entry as an attempt for that iteration number
            attempt_counts[iter_num] += 1

            # Check for errors during the iteration
            if 'error' in iteration_data:
                flagged_issues.append({
                    "problem_id": problem_id,
                    "iteration": iter_num,
                    "issue": "Error during processing",
                    "details": iteration_data.get('error'),
                    "traceback": iteration_data.get('traceback')
                })
                # Error means incorrect
                continue 

            extracted_answer = iteration_data.get('answer')

            # Check for missing answer
            if extracted_answer is None:
                 flagged_issues.append({
                    "problem_id": problem_id,
                    "iteration": iter_num,
                    "issue": "No answer extracted",
                    "reasoning": iteration_data.get('reasoning', 'N/A')[-200:] # Last 200 chars of reasoning
                 })
                 # Missing answer means incorrect
                 continue
            
            # Use pre-calculated 'correct' field if available and valid, otherwise compare
            # Note: The ContinuationExperiment stores a 'correct' field directly
            is_correct = False
            if 'correct' in iteration_data and isinstance(iteration_data['correct'], bool):
                 is_correct = iteration_data['correct']
            else:
                # Fallback to direct comparison if 'correct' field is missing/invalid
                is_correct = compare_answers(extracted_answer, correct_answer_raw)

            if is_correct:
                correct_counts[iter_num] += 1
            # else: # Optionally flag incorrect answers too
            #      flagged_issues.append({
            #         "problem_id": problem_id,
            #         "iteration": iter_num,
            #         "issue": "Incorrect answer",
            #         "extracted": extracted_answer,
            #         "expected": correct_answer_raw
            #      })


    if not attempt_counts:
        print("No iterations found to analyze.")
        return

    print("\n--- Iteration Accuracy ---")
    max_iteration = max(attempt_counts.keys())
    for i in range(max_iteration + 1):
        attempts = attempt_counts.get(i, 0)
        correct = correct_counts.get(i, 0)
        if attempts > 0:
            accuracy = (correct / attempts) * 100
            print(f"Iteration {i}: {accuracy:.2f}% (Correct: {correct}, Attempts: {attempts})")
        else:
            print(f"Iteration {i}: No attempts found.")

    if flagged_issues:
        print("\n--- Flagged Issues (Errors or Missing Answers) ---")
        for issue in flagged_issues:
            print(f"Problem ID: {issue['problem_id']}, Iteration: {issue['iteration']}, Issue: {issue['issue']}")
            if 'details' in issue:
                 print(f"  Details: {issue['details']}")
            if 'traceback' in issue:
                 print(f"  Traceback: {issue['traceback'][:500]}...") # Show start of traceback
            if issue['issue'] == "No answer extracted":
                 print(f"  Reasoning (end): ...{issue.get('reasoning')[-10:]}")
            # if issue['issue'] == "Incorrect answer":
            #      print(f"  Extracted: {issue['extracted']}")
            #      print(f"  Expected:  {issue['expected']}")

    else:
        print("\nNo errors or missing answers flagged during analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze iteration accuracy from experiment results.")
    parser.add_argument("results_file", help="Path to the results.json file")
    args = parser.parse_args()
    analyze_iteration_accuracy(args.results_file)