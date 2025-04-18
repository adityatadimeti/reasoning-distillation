import json
import sys
import os


def aggregate_passk_results(results_path):
    """
    Aggregates pass@k and consensus@k results from a results.json file produced by PassKExperiment.
    Args:
        results_path (str): Path to the results.json file
    Prints:
        Total problems, pass@k count and percentage, consensus@k count and percentage
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Support both formats: {"results": [...]} or just a list
    results = data["results"] if isinstance(data, dict) and "results" in data else data

    total = len(results)
    pass_count = sum(1 for r in results if r.get("pass_at_k", False))
    consensus_count = sum(1 for r in results if r.get("consensus_correct", False))

    pass_pct = (pass_count / total) * 100 if total > 0 else 0.0
    consensus_pct = (consensus_count / total) * 100 if total > 0 else 0.0

    print(f"Total problems: {total}")
    print(f"Pass@k: {pass_count} ({pass_pct:.2f}%)")
    print(f"Consensus@k: {consensus_count} ({consensus_pct:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <results_path>")
        sys.exit(1)
    aggregate_passk_results(sys.argv[1])
