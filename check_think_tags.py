import json
import argparse
import os
from typing import List, Dict, Any, Tuple

DEFAULT_THINK_TAG = "</think>"

def check_think_tags(
    results_path: str, 
    think_tag: str = DEFAULT_THINK_TAG
) -> Tuple[int, int, int, List[str]]:
    """
    Checks iteration 0 reasoning output in a results JSON file for a closing think tag.

    Args:
        results_path: Path to the results.json file.
        think_tag: The closing tag to search for (defaults to </think>).

    Returns:
        A tuple containing: 
         (total_problems_checked, found_count, missing_count, missing_ids)
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {results_path}: {e}")

    # Handle structure where results are nested under a 'results' key or are top-level list
    if isinstance(data, dict) and 'results' in data:
        results_list = data['results']
    elif isinstance(data, list):
        results_list = data
    else:
        raise ValueError(f"Unexpected JSON structure in {results_path}. Expected list or dict with 'results' key.")

    total_checked = 0
    found_count = 0
    missing_count = 0
    missing_ids = []

    for result in results_list:
        if not isinstance(result, dict):
            print(f"Warning: Skipping non-dictionary item in results: {result}")
            continue

        problem_id = result.get("problem_id", "UNKNOWN")
        iterations = result.get("iterations")

        if not iterations or not isinstance(iterations, list) or len(iterations) == 0:
            print(f"Warning: Skipping problem {problem_id}: No iterations found or invalid format.")
            continue

        iter0_data = iterations[0]
        if not isinstance(iter0_data, dict):
             print(f"Warning: Skipping problem {problem_id}: Iteration 0 is not a dictionary.")
             continue

        # Check the correct field based on how ContinuationExperiment loads/stores it
        # It loads 'reasoning' from the source file.
        reasoning_text = iter0_data.get("reasoning") 
        
        # Fallback check (might not be needed if source always uses 'reasoning')
        # if reasoning_text is None:
        #    reasoning_text = iter0_data.get("reasoning_output") 
        
        total_checked += 1 # Count problems that have at least a structure to check

        if reasoning_text is None:
            print(f"Warning: Skipping problem {problem_id}: Iteration 0 missing 'reasoning' key.")
            missing_count += 1
            missing_ids.append(str(problem_id))
        elif isinstance(reasoning_text, str):
            if think_tag in reasoning_text:
                found_count += 1
            else:
                missing_count += 1
                missing_ids.append(str(problem_id))
        else:
            print(f"Warning: Skipping problem {problem_id}: 'reasoning' is not a string ({type(reasoning_text)}).")
            missing_count += 1
            missing_ids.append(str(problem_id))

    return total_checked, found_count, missing_count, missing_ids

def main():
    parser = argparse.ArgumentParser(description=f"Check for closing think tags ('{DEFAULT_THINK_TAG}') in iteration 0 of results.json.")
    parser.add_argument("results_path", help="Path to the results.json file to check.")
    parser.add_argument("--tag", default=DEFAULT_THINK_TAG, help=f"The closing tag to search for (default: {DEFAULT_THINK_TAG})")
    parser.add_argument("--list-missing", action="store_true", help="List the IDs of problems missing the tag.")

    args = parser.parse_args()

    try:
        total, found, missing, missing_ids = check_think_tags(args.results_path, args.tag)
        
        print(f"--- Think Tag Check Summary ({args.results_path}) ---")
        print(f"Tag Searched For: '{args.tag}'")
        print(f"Total Problems Checked (with Iteration 0): {total}")
        print(f"  Problems WITH Tag in Iteration 0: {found}")
        print(f"  Problems MISSING Tag in Iteration 0: {missing}")
        
        if args.list_missing and missing_ids:
            print("\nProblem IDs Missing Tag:")
            # Print 10 per line for readability
            for i in range(0, len(missing_ids), 10):
                print(", ".join(missing_ids[i:i+10]))
        elif not args.list_missing and missing_ids:
            print("\n(Use --list-missing to see the specific IDs)")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except (ValueError, TypeError) as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main() 