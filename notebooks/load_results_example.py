import json
import pandas as pd
import os

# Path to the results file
results_path = "../results/aime_deepseek_qwen_14b_diff_self_sum/aime_deepseek_qwen_14b_diff_self_sum_20250414_002913/results.json"

# Load the JSON data using the json module first
with open(results_path, 'r') as f:
    data = json.load(f)

# Extract the results array which contains the problem data
problems = data['results']

# Option 1: Convert just the problems list to a DataFrame
# This will create a DataFrame where each row is a problem
df_problems = pd.json_normalize(problems)
print("DataFrame shape:", df_problems.shape)
print("DataFrame columns:", df_problems.columns.tolist())

# Option 2: If you want to analyze iteration data:
# Create a flattened list of iterations
all_iterations = []
for problem in problems:
    problem_id = problem['problem_id']
    question = problem['question']
    correct_answer = problem.get('correct_answer', '')
    
    # Extract iterations if they exist
    iterations = problem.get('iterations', [])
    for iter_data in iterations:
        # Add problem metadata to each iteration
        iter_data['problem_id'] = problem_id
        iter_data['question'] = question
        iter_data['correct_answer'] = correct_answer
        all_iterations.append(iter_data)

# Convert iterations list to DataFrame
df_iterations = pd.DataFrame(all_iterations)
if not df_iterations.empty:
    print("\nIterations DataFrame shape:", df_iterations.shape)
    print("Iterations DataFrame columns:", df_iterations.columns.tolist())

# Save processed DataFrames to CSV for easier analysis
df_problems.to_csv('../results/aime_deepseek_qwen_14b_diff_self_sum_problems.csv', index=False)
if not df_iterations.empty:
    df_iterations.to_csv('../results/aime_deepseek_qwen_14b_diff_self_sum_iterations.csv', index=False)

print("\nDataFrames saved as CSV files")
