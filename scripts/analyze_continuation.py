import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def load_results(results_path: str) -> Dict:
    """Load results from the specified path."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle structure where results are nested under a 'results' key or are top-level list
    if isinstance(data, dict) and 'results' in data:
        return data
    else:
        return {"results": data} if isinstance(data, list) else data

def analyze_continuation_results(results_path: str, verbose: bool = False) -> Dict:
    """Analyze continuation experiment results to track answer changes and accuracy."""
    data = load_results(results_path)
    results = data.get("results", [])
    
    # Tracking variables
    stats = {
        "total_problems": 0,
        "completed_problems": 0,
        "accuracy_by_iteration": defaultdict(lambda: {"correct": 0, "total": 0}),
        "changes": {
            "improved": 0,     # Wrong → Right
            "regression": 0,   # Right → Wrong
            "stable_correct": 0, # Right throughout
            "stable_incorrect": 0, # Wrong throughout
            "oscillating": 0,   # Changed multiple times
        },
        "answer_changes": 0,    # Number of problems where answer changed at least once
        "problems_by_iterations": defaultdict(int),  # Count of problems with N iterations
    }
    
    # Problem-by-problem analysis
    problem_details = []
    
    for result in results:
        problem_id = result.get("problem_id", "unknown")
        
        # Skip problems with errors
        if result.get("status") == "error":
            if verbose:
                print(f"Skipping problem {problem_id} due to error: {result.get('error', 'Unknown error')}")
            continue
        
        iterations = result.get("iterations", [])
        stats["total_problems"] += 1
        
        if not iterations:
            if verbose:
                print(f"Problem {problem_id} has no iterations")
            continue
            
        stats["completed_problems"] += 1
        stats["problems_by_iterations"][len(iterations)] += 1
        
        # Track answers and correctness across iterations
        answers = []
        correct_flags = []
        
        for iteration in iterations:
            iter_num = iteration.get("iteration", -1)
            answer = iteration.get("answer")
            is_correct = iteration.get("correct", False)
            
            # Store info
            answers.append(answer)
            correct_flags.append(is_correct)
            
            # Update iteration stats
            stats["accuracy_by_iteration"][iter_num]["total"] += 1
            if is_correct:
                stats["accuracy_by_iteration"][iter_num]["correct"] += 1
        
        # Check for answer changes
        unique_answers = set(str(a) for a in answers if a is not None)
        had_answer_change = len(unique_answers) > 1
        
        if had_answer_change:
            stats["answer_changes"] += 1
        
        # Categorize change pattern
        if len(correct_flags) > 0:
            initial_correct = correct_flags[0]
            final_correct = correct_flags[-1]
            
            if initial_correct and final_correct:
                # Started right, ended right
                stats["changes"]["stable_correct"] += 1
            elif not initial_correct and not final_correct:
                # Started wrong, ended wrong
                stats["changes"]["stable_incorrect"] += 1
            elif not initial_correct and final_correct:
                # Started wrong, ended right - improvement!
                stats["changes"]["improved"] += 1
            elif initial_correct and not final_correct:
                # Started right, ended wrong - regression!
                stats["changes"]["regression"] += 1
            
            # Check for oscillations (multiple changes)
            changes_count = sum(1 for i in range(1, len(correct_flags)) if correct_flags[i] != correct_flags[i-1])
            if changes_count > 1:
                stats["changes"]["oscillating"] += 1
        
        # Store detailed problem info for optional display
        problem_details.append({
            "problem_id": problem_id,
            "iterations": len(iterations),
            "answers": answers,
            "correct_flags": correct_flags,
            "initial_correct": correct_flags[0] if correct_flags else None,
            "final_correct": correct_flags[-1] if correct_flags else None,
            "had_answer_change": had_answer_change
        })
    
    # Calculate accuracies as percentages
    for iter_num, counts in stats["accuracy_by_iteration"].items():
        if counts["total"] > 0:
            counts["accuracy"] = (counts["correct"] / counts["total"]) * 100
    
    return {
        "stats": stats,
        "problem_details": problem_details,
        "config": data.get("config", {})
    }

def format_results(analysis: Dict, show_problems: bool = False) -> str:
    """Format analysis results into a readable report."""
    stats = analysis["stats"]
    
    report = []
    report.append("=" * 50)
    report.append("CONTINUATION EXPERIMENT ANALYSIS")
    report.append("=" * 50)
    
    # Basic stats
    report.append(f"\nProcessed {stats['total_problems']} problems ({stats['completed_problems']} completed)")
    
    # Iteration distribution
    report.append("\nProblems by iteration count:")
    for iter_count, count in sorted(stats["problems_by_iterations"].items()):
        report.append(f"  {iter_count} iterations: {count} problems")
    
    # Answer change stats
    report.append(f"\nProblems with answer changes: {stats['answer_changes']} ({stats['answer_changes']/stats['completed_problems']*100:.1f}% of completed)")
    
    # Change patterns
    report.append("\nChange patterns:")
    report.append(f"  Started correct, stayed correct: {stats['changes']['stable_correct']} ({stats['changes']['stable_correct']/stats['completed_problems']*100:.1f}%)")
    report.append(f"  Started incorrect, stayed incorrect: {stats['changes']['stable_incorrect']} ({stats['changes']['stable_incorrect']/stats['completed_problems']*100:.1f}%)")
    report.append(f"  Improved (wrong → right): {stats['changes']['improved']} ({stats['changes']['improved']/stats['completed_problems']*100:.1f}%)")
    report.append(f"  Regressed (right → wrong): {stats['changes']['regression']} ({stats['changes']['regression']/stats['completed_problems']*100:.1f}%)")
    report.append(f"  Oscillating: {stats['changes']['oscillating']} ({stats['changes']['oscillating']/stats['completed_problems']*100:.1f}%)")
    
    # Accuracy by iteration
    report.append("\nAccuracy by iteration:")
    for iter_num, counts in sorted(stats["accuracy_by_iteration"].items()):
        report.append(f"  Iteration {iter_num}: {counts.get('accuracy', 0):.1f}% ({counts['correct']}/{counts['total']})")
    
    # Net improvement
    initial_acc = stats["accuracy_by_iteration"][0].get("accuracy", 0)
    final_iter = max(stats["accuracy_by_iteration"].keys())
    final_acc = stats["accuracy_by_iteration"][final_iter].get("accuracy", 0)
    
    report.append(f"\nNet change in accuracy: {final_acc - initial_acc:.1f}% (Initial: {initial_acc:.1f}%, Final: {final_acc:.1f}%)")
    
    # Summary judgment
    if final_acc > initial_acc:
        report.append("\n✅ Continuation improved overall accuracy")
    elif final_acc < initial_acc:
        report.append("\n⚠️ Continuation reduced overall accuracy")
    else:
        report.append("\n⚖️ Continuation had no net effect on accuracy")
    
    # Problem details (optional)
    if show_problems:
        report.append("\n" + "=" * 50)
        report.append("PROBLEM DETAILS")
        report.append("=" * 50)
        
        for prob in analysis["problem_details"]:
            report.append(f"\nProblem {prob['problem_id']} ({prob['iterations']} iterations):")
            for i, (ans, correct) in enumerate(zip(prob["answers"], prob["correct_flags"])):
                status = "✓" if correct else "✗"
                report.append(f"  Iter {i}: {status} {ans}")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze continuation experiment results")
    parser.add_argument("results_path", help="Path to the results.json file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose logging")
    parser.add_argument("--details", "-d", action="store_true", help="Show problem-by-problem details")
    parser.add_argument("--output", "-o", help="Save report to output file")
    
    args = parser.parse_args()
    
    try:
        analysis = analyze_continuation_results(args.results_path, args.verbose)
        report = format_results(analysis, args.details)
        
        print(report)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()

if __name__ == "__main__":
    main()