#!/usr/bin/env python
import json
import os
import re
import argparse
import tempfile
import sys
import time
import random
import asyncio
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

# Reasoning behavior categories
BEHAVIORS = [
    "deduction",             # Model is performing deduction steps
    "adding_knowledge",      # Model is enriching with recalled facts
    "example_testing",       # Model tests its approach with examples
    "uncertainty_estimation", # Model expresses uncertainty
    "backtracking",          # Model identifies errors and changes approach
]

BEHAVIOR_PROMPTS = {
    "deduction": """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that deduction appears in this chain-of-reasoning. Deduction is when the model is performing a deduction step based on its current approach and assumptions. The chain-of-reasoning to analyze: {text}
Count the number of distinct deduction instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any deduction behavior, please provide a count of 0 as <count>0</count>.""",

    "adding_knowledge": """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that adding knowledge appears in this chain-of-reasoning. Adding knowledge is when the model is enriching the current approach with recalled facts. The chain-of-reasoning to analyze: {text}
Count the number of distinct adding knowledge instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any adding knowledge behavior, please provide a count of 0 as <count>0</count>.""",

    "example_testing": """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that example testing appears in this chain-of-reasoning. Example testing is when the model generates examples to test its current approach. The chain-of-reasoning to analyze: {text}
Count the number of distinct example testing instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any example testing behavior, please provide a count of 0 as <count>0</count>.""",

    "uncertainty_estimation": """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that uncertainty estimation appears in this chain-of-reasoning. Uncertainty estimation is when the model is stating its own uncertainty. The chain-of-reasoning to analyze: {text}
Count the number of distinct uncertainty estimation instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any uncertainty estimation behavior, please provide a count of 0 as <count>0</count>.""",

    "backtracking": """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that backtracking appears in this chain-of-reasoning. Backtracking is when the model decides to change its approach. The chain-of-reasoning to analyze: {text}
Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>."""
}

# Separate prompt for initializing behavior (used when requested)
INIT_PROMPT = """Here is a chain-of-reasoning that a Language Model generated while solving a math problem. Please count the number of distinct instances that initializing appears in this chain-of-reasoning. Initializing is when the model is rephrasing the given task and states initial thoughts. The chain-of-reasoning to analyze: {text}
Count the number of distinct initializing instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any initializing behavior, please provide a count of 0 as <count>0</count>."""

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

async def analyze_behavior(client, text, behavior, model=DEFAULT_MODEL, verbose=False):
    """Analyze a single behavior in the text using a separate API call."""
    prompt = BEHAVIOR_PROMPTS[behavior].format(text=text)
    
    print(f"\n====== ANALYZING {behavior.upper()} ======")
    if verbose:
        print(text)
    else:
        print(text[:100] + "..." if len(text) > 100 else text)
    
    result = call_fireworks_with_retry(client, prompt, model)
    
    print(f"\n====== {behavior.upper()} RESULT ======")
    print(result)
    
    # Extract count from the response
    pattern = r"<count>(\d+)</count>"
    match = re.search(pattern, result)
    count = int(match.group(1)) if match else 0
    
    return behavior, count, result

async def analyze_initializing(client, text, model=DEFAULT_MODEL, verbose=False):
    """Analyze initializing behavior separately (optional)."""
    prompt = INIT_PROMPT.format(text=text)
    
    print("\n====== ANALYZING INITIALIZING ======")
    if verbose:
        print(text)
    else:
        print(text[:100] + "..." if len(text) > 100 else text)
    
    result = call_fireworks_with_retry(client, prompt, model)
    
    print("\n====== INITIALIZING RESULT ======")
    print(result)
    
    # Extract count from the response
    pattern = r"<count>(\d+)</count>"
    match = re.search(pattern, result)
    count = int(match.group(1)) if match else 0
    
    return "initializing", count, result

class SeparateBehaviorAnalyzer:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, include_initializing=False, verbose=False):
        """Initialize the analyzer with API key and model."""
        self.client = OpenAI(
            api_key=api_key,
            base_url=FIREWORKS_API_BASE
        )
        self.model = model
        self.include_initializing = include_initializing
        self.verbose = verbose

    async def analyze_fragment(self, text):
        """Analyze a reasoning fragment with separate API calls for each behavior."""
        print("\n====== STARTING FRAGMENT ANALYSIS ======")
        
        # Create tasks for each behavior
        tasks = []
        for behavior in BEHAVIORS:
            behavior, count, result = await analyze_behavior(self.client, text, behavior, self.model, self.verbose)
            print(f"  {behavior}: {count}")
            breakpoint()
            tasks.append((behavior, count, result))
            
        # Optionally analyze initializing behavior
        if self.include_initializing:
            initializing, count, result = await analyze_initializing(self.client, text, self.model, self.verbose)
            tasks.append((initializing, count, result))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Process results
        counts = {}
        raw_results = {}
        
        for behavior, count, raw_result in results:
            counts[behavior] = count
            raw_results[behavior] = raw_result
            
        return counts, raw_results

    async def analyze_continuations(self, input_file, output_dir, iterations=None, text_fields=None, analyze_all_iterations=False):
        """
        Analyze all text fragments in the input file with separate API calls per behavior.
        
        Args:
            input_file: Path to the results JSON file
            output_dir: Directory to write results
            iterations: List of iteration numbers to analyze, or None for all
            text_fields: List of field names to check for text to analyze (in order of priority)
            analyze_all_iterations: If True, analyze iteration 0 as well (not just continuations)
        """
        print(f"Analyzing text fragments from {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Default text field names to look for, in order of priority
        if text_fields is None:
            text_fields = ["reasoning", "summary", "reasoning_output"]
        
        # Create output structure
        results = {
            "experiment_name": data.get("experiment_name", "separate_behavior_analysis"),
            "analysis_parameters": {
                "input_file": input_file,
                "model": self.model,
                "iterations_analyzed": iterations if iterations else "all",
                "text_fields_searched": text_fields,
                "analyze_all_iterations": analyze_all_iterations,
                "separate_api_calls": True,
                "include_initializing": self.include_initializing,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "results": []
        }
        
        # Analyze each problem
        for problem_idx, problem in enumerate(data.get("results", [])):
            print(f"Processing problem {problem_idx+1}/{len(data.get('results', []))}...")
            
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
                
                # Skip iteration 0 unless analyze_all_iterations is True
                if not analyze_all_iterations and it_no == 0:
                    continue
                
                # Find text to analyze by checking each field in priority order
                text_to_analyze = None
                field_used = None
                
                for field in text_fields:
                    if field in iteration and iteration[field]:
                        text_to_analyze = iteration[field]
                        field_used = field
                        break
                
                if not text_to_analyze:
                    print(f"  No text found for iteration {it_no} in problem {problem_id}. Checked fields: {text_fields}")
                    continue
                
                print(f"  Analyzing iteration {it_no} for problem {problem_id} (field: {field_used})...")
                counts, raw_responses = await self.analyze_fragment(text_to_analyze)
                
                output_iteration = {
                    "iteration": it_no,
                    "field_analyzed": field_used,
                    "behavior_counts": counts,
                    "raw_llm_analyses": raw_responses
                }
                output_problem["iterations"].append(output_iteration)
            
            # Only add problems that have analyzed iterations
            if output_problem["iterations"]:
                results["results"].append(output_problem)
                
                # Save incremental progress
                os.makedirs(output_dir, exist_ok=True)
                temp_output_path = os.path.join(output_dir, f"temp_{Path(input_file).stem}_separate_analysis.json")
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
        
        # Calculate aggregate statistics
        aggregate_stats = self.calculate_aggregate_stats(results)
        results["aggregate_statistics"] = aggregate_stats
        
        # Write final output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(input_file).stem}_behavior_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete! Results written to {output_path}")
        return results
        
    def calculate_aggregate_stats(self, results):
        """Calculate aggregate statistics across all problems and iterations."""
        behaviors_to_analyze = BEHAVIORS.copy()
        if self.include_initializing:
            behaviors_to_analyze.append("initializing")
            
        total_counts = {behavior: 0 for behavior in behaviors_to_analyze}
        iterations_by_number = {}
        
        # Count behaviors across all problems
        num_fragments = 0
        for problem in results.get("results", []):
            for iteration in problem.get("iterations", []):
                it_no = iteration.get("iteration", 0)
                counts = iteration.get("behavior_counts", {})
                
                # Add to total counts
                for behavior in behaviors_to_analyze:
                    total_counts[behavior] += counts.get(behavior, 0)
                
                # Track by iteration number
                if it_no not in iterations_by_number:
                    iterations_by_number[it_no] = {behavior: 0 for behavior in behaviors_to_analyze}
                    iterations_by_number[it_no]["count"] = 0
                
                iterations_by_number[it_no]["count"] += 1
                for behavior in behaviors_to_analyze:
                    iterations_by_number[it_no][behavior] += counts.get(behavior, 0)
                
                num_fragments += 1
        
        # Calculate averages
        avg_counts = {}
        if num_fragments > 0:
            for behavior in behaviors_to_analyze:
                avg_counts[f"avg_{behavior}"] = total_counts[behavior] / num_fragments
        
        # Calculate per-iteration averages
        iteration_averages = {}
        for it_no, data in iterations_by_number.items():
            count = data["count"]
            if count > 0:
                iteration_averages[str(it_no)] = {
                    "count": count,
                }
                for behavior in behaviors_to_analyze:
                    iteration_averages[str(it_no)][f"total_{behavior}"] = data[behavior]
                    iteration_averages[str(it_no)][f"avg_{behavior}"] = data[behavior] / count
        
        return {
            "total_fragments_analyzed": num_fragments,
            "total_counts": total_counts,
            "average_counts": avg_counts,
            "by_iteration": iteration_averages
        }

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze reasoning behaviors with separate API calls per behavior")
    parser.add_argument("--input", required=True, help="Path to the input file (results JSON)")
    parser.add_argument("--output-dir", default="behavior_analysis", help="Directory to save analysis results")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for analysis")
    parser.add_argument("--iterations", help="Comma-separated list of iterations to analyze, e.g., '1,2,3'")
    parser.add_argument("--include-initializing", action="store_true", help="Include analysis of initializing behavior")
    parser.add_argument("--text-fields", default="reasoning,summary,reasoning_output", 
                        help="Comma-separated list of field names to check for text to analyze (in order of priority)")
    parser.add_argument("--analyze-all-iterations", action="store_true", 
                        help="Analyze iteration 0 as well (not just continuations)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full text being analyzed instead of truncated version")
    
    args = parser.parse_args()
    
    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("ERROR: FIREWORKS_API_KEY environment variable not set")
        sys.exit(1)
    
    # Parse iterations if provided
    iterations_to_analyze = None
    if args.iterations:
        iterations_to_analyze = [int(it.strip()) for it in args.iterations.split(",")]
    
    # Parse text fields
    text_fields = [field.strip() for field in args.text_fields.split(",")]
    
    # Create analyzer and run analysis
    analyzer = SeparateBehaviorAnalyzer(
        api_key=api_key, 
        model=args.model,
        include_initializing=args.include_initializing,
        verbose=args.verbose
    )
    
    await analyzer.analyze_continuations(
        input_file=args.input,
        output_dir=args.output_dir,
        iterations=iterations_to_analyze,
        text_fields=text_fields,
        analyze_all_iterations=args.analyze_all_iterations
    )

if __name__ == "__main__":
    asyncio.run(main()) 