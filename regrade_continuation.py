#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, List, Any
import re

def extract_answer(text: str) -> str:
    """Simple answer extraction using regex pattern matching."""
    # Common patterns for answer extraction
    answer_patterns = [
        r"answer[^\w\d]*is[^\w\d]*([A-Z0-9]+)",  # Matches "answer is X"
        r"answer[^\w\d]*([A-Z0-9]+)",            # Matches "answer X"
        r"the answer is[^\w\d]*([A-Z0-9]+)",     # Matches "the answer is X"
        r"final answer[^\w\d]*is[^\w\d]*([A-Z0-9]+)",  # Matches "final answer is X"
        r"final answer[^\w\d]*([A-Z0-9]+)",      # Matches "final answer X"
        r"I'll choose[^\w\d]*([A-Z0-9]+)",       # Matches "I'll choose X"
        r"I choose[^\w\d]*([A-Z0-9]+)",          # Matches "I choose X"
        r"my answer is[^\w\d]*([A-Z0-9]+)",      # Matches "my answer is X"
        r"After all calculations,[^\w\d]*([A-Z0-9]+)",  # Matches "After all calculations, X"
        r"The correct answer is[^\w\d]*([A-Z0-9]+)",    # Matches "The correct answer is X"
        r"Thus, the answer is[^\w\d]*([A-Z0-9]+)",      # Matches "Thus, the answer is X"
        r"Therefore, the answer is[^\w\d]*([A-Z0-9]+)",  # Matches "Therefore, the answer is X"
        r"Therefore,[^\w\d]*([A-Z0-9]+)",        # Matches "Therefore, X"
        r"So, the answer is[^\w\d]*([A-Z0-9]+)", # Matches "So, the answer is X"
        r"The answer to the problem is[^\w\d]*([A-Z0-9]+)", # Matches "The answer to the problem is X" 
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return the last match
    
    return None

def regrade_results(results_file: str) -> Dict[str, Any]:
    """Regrade continuation experiment results using full reasoning context."""
    with open(results_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both list and dict formats (with 'results' key)
    if isinstance(json_data, dict) and "results" in json_data:
        results = json_data["results"]
    else:
        results = json_data

    # Ensure results is a list
    if not isinstance(results, list):
        raise ValueError(f"Unexpected results format in {results_file}")
    
    # Keep track of stats
    total_problems = len(results)
    completed_problems = 0
    problems_by_iteration_count = {}
    problems_with_answer_changes = 0
    
    # Track different change patterns
    started_correct_stayed_correct = 0
    started_incorrect_stayed_incorrect = 0
    improved = 0  # wrong → right
    regressed = 0  # right → wrong
    oscillating = 0  # multiple changes
    
    # Track accuracy by iteration
    accuracy_by_iteration = {}
    max_iterations = 0
    
    # Process each problem
    for problem in results:
        if problem.get("status") != "completed":
            continue
        
        completed_problems += 1
        correct_answer = problem.get("correct_answer")
        iterations = problem.get("iterations", [])
        
        # Count iterations for this problem
        num_iterations = len(iterations)
        problems_by_iteration_count[num_iterations] = problems_by_iteration_count.get(num_iterations, 0) + 1
        max_iterations = max(max_iterations, num_iterations - 1)  # 0-indexed
        
        # Track correctness for each iteration
        previous_correctness = None
        correctness_changes = 0
        first_correct = None
        last_correct = None
        
        for i, iteration in enumerate(iterations):
            # Get the full reasoning context
            full_reasoning = iteration.get("reasoning_full_for_extraction")
            if not full_reasoning:
                continue
                
            # Extract answer using the full reasoning context
            extracted_answer = extract_answer(full_reasoning)
            is_correct = False
            if extracted_answer is not None:
                is_correct = extracted_answer.strip() == correct_answer.strip()
            
            # Update iteration with regraded results
            iteration["regraded_answer"] = extracted_answer
            iteration["regraded_correct"] = is_correct
            
            # Update accuracy stats
            iteration_idx = iteration.get("iteration", i)
            if iteration_idx not in accuracy_by_iteration:
                accuracy_by_iteration[iteration_idx] = {"correct": 0, "total": 0}
            accuracy_by_iteration[iteration_idx]["total"] += 1
            if is_correct:
                accuracy_by_iteration[iteration_idx]["correct"] += 1
            
            # Track correctness changes
            if previous_correctness is None:
                previous_correctness = is_correct
                first_correct = is_correct
            elif previous_correctness != is_correct:
                correctness_changes += 1
                previous_correctness = is_correct
            
            # Track final correctness
            if i == len(iterations) - 1:
                last_correct = is_correct
        
        # Record answer change patterns
        if correctness_changes > 0:
            problems_with_answer_changes += 1
            
        if correctness_changes > 1:
            oscillating += 1
        elif first_correct and last_correct:
            started_correct_stayed_correct += 1
        elif not first_correct and not last_correct:
            started_incorrect_stayed_incorrect += 1
        elif not first_correct and last_correct:
            improved += 1
        elif first_correct and not last_correct:
            regressed += 1
    
    # Calculate accuracy percentages
    accuracy_percentages = {}
    for iteration, stats in accuracy_by_iteration.items():
        accuracy_percentages[iteration] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    # Calculate net change in accuracy
    initial_accuracy = accuracy_percentages.get(0, 0)
    final_accuracy = accuracy_percentages.get(max_iterations, 0)
    net_change = final_accuracy - initial_accuracy
    
    # Create summary report
    report = {
        "total_problems": total_problems,
        "completed_problems": completed_problems,
        "problems_by_iteration_count": problems_by_iteration_count,
        "problems_with_answer_changes": problems_with_answer_changes,
        "percentage_with_changes": (problems_with_answer_changes / completed_problems * 100) if completed_problems > 0 else 0,
        "change_patterns": {
            "started_correct_stayed_correct": started_correct_stayed_correct,
            "started_incorrect_stayed_incorrect": started_incorrect_stayed_incorrect,
            "improved": improved,
            "regressed": regressed,
            "oscillating": oscillating,
        },
        "accuracy_by_iteration": accuracy_percentages,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "net_change": net_change,
    }
    
    # Save regraded results
    output_file = results_file.replace(".json", "_regraded.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return report

def print_report(report: Dict[str, Any]):
    """Print a formatted report of the regrading results."""
    print("==================================================")
    print("CONTINUATION EXPERIMENT REGRADING ANALYSIS")
    print("==================================================")
    print()
    print(f"Processed {report['total_problems']} problems ({report['completed_problems']} completed)")
    print()
    
    print("Problems by iteration count:")
    for iters, count in sorted(report['problems_by_iteration_count'].items()):
        print(f"  {iters} iterations: {count} problems")
    print()
    
    print(f"Problems with answer changes: {report['problems_with_answer_changes']} ({report['percentage_with_changes']:.1f}% of completed)")
    print()
    
    patterns = report['change_patterns']
    completed = report['completed_problems']
    print("Change patterns:")
    print(f"  Started correct, stayed correct: {patterns['started_correct_stayed_correct']} ({patterns['started_correct_stayed_correct']/completed*100:.1f}%)")
    print(f"  Started incorrect, stayed incorrect: {patterns['started_incorrect_stayed_incorrect']} ({patterns['started_incorrect_stayed_incorrect']/completed*100:.1f}%)")
    print(f"  Improved (wrong → right): {patterns['improved']} ({patterns['improved']/completed*100:.1f}%)")
    print(f"  Regressed (right → wrong): {patterns['regressed']} ({patterns['regressed']/completed*100:.1f}%)")
    print(f"  Oscillating: {patterns['oscillating']} ({patterns['oscillating']/completed*100:.1f}%)")
    print()
    
    print("Accuracy by iteration:")
    for iteration, accuracy in sorted(report['accuracy_by_iteration'].items()):
        correct = report['accuracy_by_iteration'][iteration] * completed / 100
        print(f"  Iteration {iteration}: {accuracy:.1f}% ({int(correct)}/{completed})")
    print()
    
    print(f"Net change in accuracy: {report['net_change']:.1f}% (Initial: {report['initial_accuracy']:.1f}%, Final: {report['final_accuracy']:.1f}%)")
    print()
    
    if report['net_change'] > 0:
        print("✅ Continuation improved overall accuracy")
    elif report['net_change'] < 0:
        print("❌ Continuation decreased overall accuracy")
    else:
        print("⚠️ Continuation had no effect on overall accuracy")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regrade_continuation.py <path/to/results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    report = regrade_results(results_file)
    print_report(report)
    
    output_file = results_file.replace(".json", "_regraded.json")
    print(f"Regraded results saved to: {output_file}") 