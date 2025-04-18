import json
import sys
import os

def check_pass_vs_consensus(results_path):
    with open(results_path, 'r') as f:
        data = json.load(f)
    results = data["results"] if isinstance(data, dict) and "results" in data else data

    disagreements = []
    for r in results:
        if r.get("pass_at_k", False) and not r.get("consensus_correct", False):
            disagreements.append({
                "problem_id": r.get("problem_id"),
                "answers": [s.get("answer") for s in r.get("solutions", [])],
                "correct_answer": r.get("correct_answer"),
                "consensus_answer": r.get("consensus_answer"),
            })

    print(f"Number of disagreements (pass@k True, consensus@k False): {len(disagreements)}")
    if disagreements:
        for d in disagreements:
            print(f"Problem ID: {d['problem_id']}")
            print(f"  Correct Answer: {d['correct_answer']}")
            print(f"  Consensus Answer: {d['consensus_answer']}")
            print(f"  All Answers: {d['answers']}")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <results_path>")
        sys.exit(1)
    check_pass_vs_consensus(sys.argv[1])
