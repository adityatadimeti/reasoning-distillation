# analyze_annotations.py  (run after chunk_annotate.py has produced *_annotated.json)
import json, collections, itertools, argparse
from pathlib import Path
import statistics
import sys

LABELS = [
    "initializing", "deduction", "adding-knowledge",
    "example-testing", "uncertainty-estimation", "backtracking"
]

def extract_labels(annotated: str):
    """Return the list of labels in order of appearance."""
    import re
    pat = re.compile(r'\["([^"]+)"\]')
    allowed_labels = set(LABELS)
    
    labels = []
    for m in pat.finditer(annotated):
        if m.group(1) == "end-section":
            continue
            
        # Look for any of the valid labels within the captured text
        # This handles cases like "1. deduction" or "4. uncertainty-estimation"
        for label in allowed_labels:
            if label in m.group(1).lower():
                labels.append(label)
                break
                
    return labels

def collapse_runs(labels: list) -> list:
    """Collapse consecutive runs of the same label into a single instance."""
    if not labels:
        return []
    
    collapsed = [labels[0]]
    runs_collapsed = 0
    current_run = 1
    
    for lbl in labels[1:]:
        if lbl == collapsed[-1]:
            current_run += 1
        else:
            if current_run > 1:
                runs_collapsed += 1
                print(f"Collapsed run of {current_run} '{collapsed[-1]}' labels")
            collapsed.append(lbl)
            current_run = 1
            
    # Handle final run
    if current_run > 1:
        runs_collapsed += 1
        print(f"Collapsed run of {current_run} '{collapsed[-1]}' labels")
        
    if runs_collapsed > 0:
        print(f"Total runs collapsed: {runs_collapsed}")
        
    return collapsed

def label_counts(lbls):
    return collections.Counter(lbls)

def rel_freq(counter):
    total = sum(counter.values())
    return {k: counter[k]/total for k in LABELS}

def lcs(a, b):
    # classical DP LCS length
    from functools import lru_cache
    @lru_cache(None)
    def f(i,j):
        if i==len(a) or j==len(b): return 0
        if a[i]==b[j]: return 1+f(i+1,j+1)
        return max(f(i+1,j), f(i,j+1))
    return f(0,0)

def first_occurrence_pos(seq):
    pos = {}
    for i,lab in enumerate(seq):
        if lab not in pos: pos[lab] = i
    return pos

def print_stats(global_stats):
    """Print summary statistics from the analysis"""
    if not global_stats:
        print("No data to analyze")
        return
    
    print("\n=== BEHAVIOR ANNOTATION STATISTICS ===")
    print(f"Total Samples: {len(global_stats)}")
    
    # Calculate averages
    avg_trace_len = sum(stat["trace_len"] for stat in global_stats) / len(global_stats)
    avg_sum_len = sum(stat["sum_len"] for stat in global_stats) / len(global_stats)
    avg_compression = sum(stat["compression"] for stat in global_stats) / len(global_stats)
    avg_coverage = sum(stat["coverage"] for stat in global_stats) / len(global_stats)
    avg_order_sim = sum(stat["order_sim"] for stat in global_stats) / len(global_stats)
    
    print(f"\nAverages:")
    print(f"  Trace Length: {avg_trace_len:.2f}")
    print(f"  Summary Length: {avg_sum_len:.2f}")
    print(f"  Compression Ratio: {avg_compression:.2f}")
    print(f"  Coverage: {avg_coverage:.2f}")
    print(f"  Order Similarity: {avg_order_sim:.2f}")
    
    # Calculate behavior distributions
    trace_behaviors = {label: sum(stat[f"trace_{label}"] for stat in global_stats) for label in LABELS}
    sum_behaviors = {label: sum(stat[f"sum_{label}"] for stat in global_stats) for label in LABELS}
    
    total_trace = sum(trace_behaviors.values())
    total_sum = sum(sum_behaviors.values())
    
    print("\nBehavior Distribution in Reasoning Traces:")
    for label in LABELS:
        percentage = (trace_behaviors[label] / total_trace * 100) if total_trace else 0
        print(f"  {label}: {trace_behaviors[label]} ({percentage:.1f}%)")
    
    print("\nBehavior Distribution in Summaries:")
    for label in LABELS:
        percentage = (sum_behaviors[label] / total_sum * 100) if total_sum else 0
        print(f"  {label}: {sum_behaviors[label]} ({percentage:.1f}%)")
    
    print("\n=== BEHAVIOR TRANSITIONS ===")
    # Calculate common transitions using stored collapsed sequences
    transitions = collections.Counter()
    for stat in global_stats:
        if stat["trace_len"] < 2:
            continue
        
        # Use the stored collapsed label sequence
        r_lbl = stat["trace_labels"]
        
        for i in range(len(r_lbl) - 1):
            transitions[(r_lbl[i], r_lbl[i+1])] += 1
    
    # Print top transitions
    print("Top Behavior Transitions:")
    for (from_behavior, to_behavior), count in transitions.most_common(10):
        percentage = (count / sum(transitions.values()) * 100)
        print(f"  {from_behavior} → {to_behavior}: {count} ({percentage:.1f}%)")
    
    print("\n=== FULL STATISTICS SAVED TO CSV ===")

def main():
    parser = argparse.ArgumentParser(description="Analyze behavior annotations in reasoning traces")
    parser.add_argument("input", help="Path to the annotated JSON file")
    parser.add_argument("--output", "-o", help="Output CSV file (default: input file with .behaviour_stats.csv suffix)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Don't print statistics to the console")
    parser.add_argument("--collapse-runs", dest="collapse_runs", action="store_true", 
                        help="Collapse consecutive identical labels into a single instance (default: True)")
    parser.add_argument("--no-collapse-runs", dest="collapse_runs", action="store_false",
                        help="Don't collapse consecutive identical labels")
    parser.set_defaults(collapse_runs=True)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        return 1
    
    global data
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_path}: {e}", file=sys.stderr)
        return 1
    
    global_stats = []
    
    for prob in data["problems"]:
        for it in prob["iterations"]:
            # Check if both annotated fields exist
            if not ("reasoning_annotated" in it and "post_think_summary_annotated" in it):
                continue
                
            r_lbl = extract_labels(it["reasoning_annotated"])
            s_lbl = extract_labels(it["post_think_summary_annotated"])
            
            # Optionally collapse consecutive identical labels
            if args.collapse_runs:
                r_lbl = collapse_runs(r_lbl)
                s_lbl = collapse_runs(s_lbl)
                
            rc, sc = label_counts(r_lbl), label_counts(s_lbl)
    
            lcs_len = lcs(r_lbl, s_lbl)
            order_sim = lcs_len / len(r_lbl) if r_lbl else 0
    
            coverage = len(set(sc) & set(rc)) / len(set(rc)) if rc else 0
    
            global_stats.append({
                "problem": prob["problem_id"],
                "iter": it["iteration"],
                "trace_len": len(r_lbl),
                "sum_len": len(s_lbl),
                "compression": len(s_lbl)/len(r_lbl) if r_lbl else 0,
                "coverage": coverage,
                "order_sim": order_sim,
                # Store the actual label sequences for transition analysis
                "trace_labels": r_lbl,
                "summary_labels": s_lbl,
                **{f"trace_{k}": rc[k] for k in LABELS},
                **{f"sum_{k}": sc[k] for k in LABELS},
            })
    
    # Print statistics unless --quiet is specified
    if not args.quiet:
        print_stats(global_stats)
    
    # Determine output path
    out_path = args.output if args.output else input_path.with_suffix(".behaviour_stats.csv")
    
    # Write CSV for quick inspection
    import csv
    with open(out_path, "w", newline="") as fp:
        if global_stats:
            w = csv.DictWriter(fp, fieldnames=global_stats[0].keys())
            w.writeheader()
            w.writerows(global_stats)
    
    print(f"Wrote statistics to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())