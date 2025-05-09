import json
import os

def merge_json_results(original_file_path, new_results_file_path, output_file_path, problem_ids_to_replace):
    """
    Merges results from new_results_file_path into original_file_path for specified problem_ids.

    Args:
        original_file_path (str): Path to the original full results JSON file.
        new_results_file_path (str): Path to the JSON file containing updated results for a subset of problems.
        output_file_path (str): Path to save the merged results JSON file.
        problem_ids_to_replace (list): A list of problem_id strings to be replaced.
    """
    try:
        with open(original_file_path, 'r') as f:
            original_data = json.load(f)
        print(f"Successfully loaded original data from: {original_file_path}")
    except FileNotFoundError:
        print(f"Error: Original results file not found at {original_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {original_file_path}")
        return

    try:
        with open(new_results_file_path, 'r') as f:
            new_results_data = json.load(f)
        print(f"Successfully loaded new results data from: {new_results_file_path}")
    except FileNotFoundError:
        print(f"Error: New results file not found at {new_results_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {new_results_file_path}")
        return

    # Create a dictionary of the new results for quick lookup
    # Ensure 'results' key exists and is a list
    if 'results' not in new_results_data or not isinstance(new_results_data['results'], list):
        print(f"Error: 'results' key is missing or not a list in {new_results_file_path}")
        return
        
    new_results_map = {item['problem_id']: item for item in new_results_data['results']}
    print(f"Found {len(new_results_map)} items in the new results file.")

    # Create a copy of the original data to modify
    merged_data = original_data.copy() # Shallow copy is fine for top-level keys like 'config'

    # Ensure 'results' key exists and is a list in original data
    if 'results' not in merged_data or not isinstance(merged_data['results'], list):
        print(f"Error: 'results' key is missing or not a list in {original_file_path}")
        return

    updated_problem_count = 0
    # Iterate through the original results and update if necessary
    merged_results_list = []
    for original_item in merged_data['results']:
        problem_id = original_item.get('problem_id')
        if problem_id in problem_ids_to_replace and problem_id in new_results_map:
            merged_results_list.append(new_results_map[problem_id])
            updated_problem_count += 1
            print(f"Replaced problem_id: {problem_id}")
        else:
            merged_results_list.append(original_item)
    
    merged_data['results'] = merged_results_list
    print(f"Total problems updated: {updated_problem_count}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(output_file_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"Successfully wrote merged results to: {output_file_path}")
    except IOError:
        print(f"Error: Could not write merged results to {output_file_path}")


if __name__ == "__main__":
    original_file = "results/aime_deepseek_qwen_14b_summ_base_sum_4iter/aime_deepseek_qwen_14b_summ_base_sum_4iter_20250504_173244/results.json"
    new_subset_file = "results/aime_deepseek_qwen_14b_summ_base_sum_4iter_new/aime_deepseek_qwen_14b_summ_base_sum_4iter_new_20250508_132930/results.json"
    
    # Define a path for the output file. Let's put it in a new directory to keep things clean.
    output_merged_file = "results/aime_deepseek_qwen_14b_summ_base_sum_4iter_rerun_merged/results.json"

    # The problem IDs you specified for replacement
    problem_ids_to_update = ["11", "29", "26", "21", "13", "2024-I-5", "4", "2024-II-8", "12", "5", "2024-I-8"]

    print("Starting merge process...")
    merge_json_results(original_file, new_subset_file, output_merged_file, problem_ids_to_update)
    print("Merge process finished.") 