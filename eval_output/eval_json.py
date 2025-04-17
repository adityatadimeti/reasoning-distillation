import json

#path = "results/aime_2024_consensus_update_test_7b/aime_2024_pass_consensus_update_test_7b_20250415_140805/test_results.json"
path = "results/aime_2024_consensus_update_test_14b/aime_2024_pass_consensus_update_test_14b_20250415_140046/test_results.json"

# Read the JSON file
with open(path, 'r') as file:
    data = json.load(file)

# Now 'data' contains the JSON content as a Python data structure

# (typically a dictionary or a list depending on the JSON structure)

results = data["results"]

pass_at_5 = 0
consensus_at_5 = 0
for problem in results: # 60 total problems
    total_correct = 0
    for solution in problem["solutions"]: # 5 individual solutions
        if solution["correct"]: # individual solution is true, update pass value
            total_correct += 1
    if total_correct > 0:
        pass_at_5 += 1
    if total_correct > 2:
        consensus_at_5 += 1

breakpoint()

