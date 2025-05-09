#!/usr/bin/env python
import json
import os
import re
import argparse
import tempfile
import sys
import time
import random
from pathlib import Path
import datetime
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError
import dotenv

# Fireworks AI configuration
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v3-0324"

# Retry configuration
MAX_RETRIES = 10
RETRY_BASE_DELAY = 0.5  # seconds

# Reasoning behavior categories for continuation analysis
BEHAVIORS = [
    "initializing",          # Model is stating task and initial thoughts
    "deduction",             # Model is performing deduction steps
    "adding_knowledge",      # Model is enriching with recalled facts
    "example_testing",       # Model tests its approach with examples
    "uncertainty_estimation", # Model expresses uncertainty
    "backtracking",          # Model identifies errors and changes approach
]

# Prompt template for behavior counting
BEHAVIOR_COUNT_PROMPT = """Here is a chain-of-reasoning that a Language Model generated while solving a math problem.

Please count how many instances of each reasoning behavior appear in this continuation fragment:

1. Initializing: The model is rephrasing the given task and states initial thoughts.
2. Deduction: The model is performing a deduction step based on its current approach and assumptions.
3. Adding Knowledge: The model is enriching the current approach with recalled facts.
4. Example Testing: The model generates examples to test its current approach.
5. Uncertainty Estimation: The model is stating its own uncertainty.
6. Backtracking: The model decides to change its approach.

Continuation fragment to analyze:
"{text}"

For each behavior, respond with a count in this exact format:
<deduction>COUNT</deduction>
<adding_knowledge>COUNT</adding_knowledge>
<example_testing>COUNT</example_testing>
<uncertainty_estimation>COUNT</uncertainty_estimation>
<backtracking>COUNT</backtracking>
"""

def call_fireworks_with_retry(client, prompt, model=DEFAULT_MODEL):
    """Call Fireworks AI chat completion with retry logic for transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        
        except (APIError, RateLimitError, APIConnectionError) as e:
            # Don't retry on last attempt
            if attempt == MAX_RETRIES - 1:
                print(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                raise
            
            # Calculate exponential backoff with jitter
            delay = min(RETRY_BASE_DELAY * (2 ** attempt) + (0.1 * random.random()), 60)
            print(f"API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(delay)
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
            
    # Should never reach here due to the raise in the loop
    raise RuntimeError("Failed to get a response after multiple retries")

class ContinuationAnalyzer:
    def __init__(self, api_key=None, model=DEFAULT_MODEL):
        """Initialize the analyzer with API key and model."""
        self.client = OpenAI(
            api_key=api_key,
            base_url=FIREWORKS_API_BASE
        )
        self.model = model

    def analyze_fragment(self, text):
        """Analyze a single reasoning fragment and count behaviors."""
        prompt = BEHAVIOR_COUNT_PROMPT.format(text=text)
        
        print("\n====== ANALYZING FRAGMENT ======")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        result = call_fireworks_with_retry(self.client, prompt, self.model)
        
        print("\n====== ANALYSIS RESULT ======")
        print(result)
        
        # Extract counts from the response
        counts = {}
        for behavior in BEHAVIORS:
            pattern = f"<{behavior}>(\d+)</{behavior}>"
            match = re.search(pattern, result)
            counts[behavior] = int(match.group(1)) if match else 0
            
        return counts, result

    def analyze_continuations(self, input_file, output_dir, iterations=None):
        """
        Analyze all continuation fragments in the input file.
        
        Args:
            input_file: Path to the results_for_annotation.json file
            output_dir: Directory to write results
            iterations: List of iteration numbers to analyze, or None for all
        """
        print(f"Analyzing continuations from {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create output structure
        results = {
            "experiment_name": data.get("experiment_name", "continuation_analysis"),
            "analysis_parameters": {
                "input_file": input_file,
                "model": self.model,
                "iterations_analyzed": iterations if iterations else "all",
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "problems": []
        }
        
        # Analyze each problem
        for problem_idx, problem in enumerate(data.get("problems", [])):
            print(f"Processing problem {problem_idx+1}/{len(data.get('problems', []))}...")
            
            problem_id = problem.get("problem_id", f"problem_{problem_idx}")
            output_problem = {
                "problem_id": problem_id,
                "iterations": []
            }
            
            # Analyze each iteration for this problem
            for iteration in problem.get("iterations", []):
                it_no = iteration.get("iteration", 0)
                
                # Skip iterations that aren't requested
                if iterations is not None and it_no not in iterations:
                    continue
                
                # Only analyze iterations > 0 (continuations)
                if it_no > 0:
                    reasoning = iteration.get("reasoning", "")
                    if not reasoning:
                        continue
                    
                    print(f"  Analyzing iteration {it_no} for problem {problem_id}...")
                    counts, raw_response = self.analyze_fragment(reasoning)
                    
                    output_iteration = {
                        "iteration": it_no,
                        "behavior_counts": counts,
                        "raw_llm_analysis": raw_response
                    }
                    output_problem["iterations"].append(output_iteration)
            
            # Only add problems that have analyzed iterations
            if output_problem["iterations"]:
                results["problems"].append(output_problem)
                
                # Save incremental progress
                os.makedirs(output_dir, exist_ok=True)
                temp_output_path = os.path.join(output_dir, f"temp_{Path(input_file).stem}_analysis.json")
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
        
        # Calculate aggregate statistics
        aggregate_stats = self.calculate_aggregate_stats(results)
        results["aggregate_statistics"] = aggregate_stats
        
        # Write final output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(input_file).stem}_behavior_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete! Results written to {output_path}")
        return results
        
    def calculate_aggregate_stats(self, results):
        """Calculate aggregate statistics across all problems and iterations."""
        total_counts = {behavior: 0 for behavior in BEHAVIORS}
        iterations_by_number = {}
        
        # Count behaviors across all problems
        num_continuations = 0
        for problem in results.get("problems", []):
            for iteration in problem.get("iterations", []):
                it_no = iteration.get("iteration", 0)
                counts = iteration.get("behavior_counts", {})
                
                # Add to total counts
                for behavior in BEHAVIORS:
                    total_counts[behavior] += counts.get(behavior, 0)
                
                # Track by iteration number
                if it_no not in iterations_by_number:
                    iterations_by_number[it_no] = {behavior: 0 for behavior in BEHAVIORS}
                    iterations_by_number[it_no]["count"] = 0
                
                iterations_by_number[it_no]["count"] += 1
                for behavior in BEHAVIORS:
                    iterations_by_number[it_no][behavior] += counts.get(behavior, 0)
                
                num_continuations += 1
        
        # Calculate averages
        avg_counts = {}
        if num_continuations > 0:
            for behavior in BEHAVIORS:
                avg_counts[f"avg_{behavior}"] = total_counts[behavior] / num_continuations
        
        # Calculate per-iteration averages
        iteration_averages = {}
        for it_no, data in iterations_by_number.items():
            count = data["count"]
            if count > 0:
                iteration_averages[str(it_no)] = {
                    "count": count,
                }
                for behavior in BEHAVIORS:
                    iteration_averages[str(it_no)][f"total_{behavior}"] = data[behavior]
                    iteration_averages[str(it_no)][f"avg_{behavior}"] = data[behavior] / count
        
        return {
            "total_continuations_analyzed": num_continuations,
            "total_counts": total_counts,
            "average_counts": avg_counts,
            "by_iteration": iteration_averages
        }

def main():
    parser = argparse.ArgumentParser(description="Analyze reasoning behaviors in continuation fragments")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with continuation data")
    parser.add_argument("--output-dir", "-o", default="analysis_results", help="Output directory for results")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Fireworks AI model to use for analysis")
    parser.add_argument("--iters", default="all", help="Iterations to analyze: 'all' or e.g. 1,2,3,4")
    parser.add_argument("--max-problems", type=int, default=None, help="Maximum number of problems to analyze")
    args = parser.parse_args()
    
    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment variables", file=sys.stderr)
        return 1
    
    # Parse iterations
    iterations = None
    if args.iters != "all":
        iterations = [int(x) for x in args.iters.split(",")]
    
    # Create and run analyzer
    analyzer = ContinuationAnalyzer(api_key=api_key, model=args.model)
    analyzer.analyze_continuations(args.input, args.output_dir, iterations)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 