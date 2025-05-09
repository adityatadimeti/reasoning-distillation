import json
import argparse
import os

def count_iteration0_correct(results_path: str) -> dict:
    """
    Count the number of correct answers in iteration 0 of a results json file.
    
    Args:
        results_path: Path to the results.json file
        
    Returns:
        Dictionary with counts and percentages
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Load the results file
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get the results list
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unexpected JSON structure. Expected list or dict with 'results' key.")
    
    # Initialize counters
    total_problems = 0
    has_iteration0 = 0
    correct_answers = 0
    
    # Process each problem
    for result in results:
        if not isinstance(result, dict):
            continue
            
        total_problems += 1
        
        # Check if the problem has iterations
        iterations = result.get("iterations", [])
        if not iterations:
            continue
            
        # Check if the first iteration exists
        if len(iterations) > 0 and isinstance(iterations[0], dict):
            has_iteration0 += 1
            # Check if the answer in iteration 0 is correct
            if iterations[0].get("correct", False):
                correct_answers += 1
    
    # Calculate percentages
    percent_correct = (correct_answers / has_iteration0 * 100) if has_iteration0 > 0 else 0
    
    return {
        "total_problems": total_problems,
        "problems_with_iteration0": has_iteration0,
        "correct_in_iteration0": correct_answers,
        "percent_correct": percent_correct
    }

def main():
    parser = argparse.ArgumentParser(description="Count correct answers in iteration 0")
    parser.add_argument("results_path", help="Path to the results.json file")
    
    args = parser.parse_args()
    
    try:
        counts = count_iteration0_correct(args.results_path)
        
        print(f"\nResults from {args.results_path}:")
        print(f"Total problems: {counts['total_problems']}")
        print(f"Problems with iteration 0: {counts['problems_with_iteration0']}")
        print(f"Correct answers in iteration 0: {counts['correct_in_iteration0']}")
        print(f"Accuracy in iteration 0: {counts['percent_correct']:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()