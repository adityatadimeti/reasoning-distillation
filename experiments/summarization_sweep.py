"""
Script to run summarization sweep on existing reasoning traces.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import re
from typing import List, Optional, Dict, Any

from src.utils.config import Config, PROJECT_ROOT, load_config
from src.pipeline.summary_pipeline import SummaryReasoningPipeline      

def extract_reasoning_trace(completion: str) -> str:
    """
    Extract the reasoning trace from a model completion.
    The reasoning trace is everything before the </think> tag.
    
    Args:
        completion: The full model completion text
        
    Returns:
        The extracted reasoning trace
    """
    # Split on </think> and take everything before it
    parts = completion.split('</think>')
    if len(parts) > 1:
        return parts[0].strip()
    
    # If no </think> tag found, return the whole completion
    return completion.strip()

def load_reasoning_traces(problem_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load reasoning traces from the DataFrame.
    
    Args:
        problem_ids: Optional list of problem IDs to load
        
    Returns:
        Dictionary mapping problem IDs to their reasoning traces
    """
    # Load the DataFrame
    df_path = PROJECT_ROOT / "data" / "processed" / "reasoning_checkpoint_final.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Reasoning traces DataFrame not found at {df_path}")
    
    df = pd.read_csv(df_path)
    
    # Filter by problem IDs if specified
    if problem_ids:
        df = df[df['ID'].isin(problem_ids)]
        if len(df) != len(problem_ids):
            missing = set(problem_ids) - set(df['ID'].unique())
            raise ValueError(f"Missing reasoning traces for problems: {missing}")
    
    # Convert DataFrame to dictionary format
    traces = {}
    for _, row in df.iterrows():
        problem_id = row['ID']
        
        # Extract reasoning trace from model completion
        reasoning_trace = extract_reasoning_trace(row['model_completion'])
        
        traces[problem_id] = {
            'problem': row['Problem'],
            'solution': row['Solution'],
            'answer': row['Answer'],
            'model_completion': row['model_completion'],
            'reasoning_trace': reasoning_trace,  # Add extracted reasoning trace
            'extracted_answer': row['extracted_answer'],
            'summarized_completion': row['summarized_completion'],
            'summarized_extracted_answer': row['summarized_extracted_answer'],
            'iterations': row['iterations'],
            'summarized_reasoning_trace': row['summarized_reasoning_trace'],
            'new_completion': row['new_completion'],
            'new_extracted_answer': row['new_extracted_answer'],
            'new_prompt': row['new_prompt']
        }
    
    return traces

def run_summarization_sweep(
    config_path: str,
    method: str = "self",
    max_iterations: int = 2,
    problem_ids: Optional[str] = None,
    summarization_mode: str = "append"
) -> Dict[str, Any]:
    """
    Run summarization sweep on existing reasoning traces.
    
    Args:
        config_path: Path to configuration file
        method: Summarization method ('self' or 'external')
        max_iterations: Maximum number of iterations
        problem_ids: Comma-separated list of problem IDs to process
        summarization_mode: How to include summary in prompt ('append' or 'prompt')
        
    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Update config with summarization mode
    config["pipeline"]["summarization_mode"] = summarization_mode
    
    # Parse problem IDs
    problem_id_list = problem_ids.split(",") if problem_ids else None
    
    # Load reasoning traces
    traces = load_reasoning_traces(problem_id_list)
    
    # Create a dataset from the traces
    from src.data.dataset import Dataset
    dataset = Dataset(config)
    dataset.problems = [
        {
            'id': problem_id,
            'question': trace['problem'],
            'solution': trace['solution'],
            'answer': trace['answer'],
            'reasoning_trace': trace['reasoning_trace'],
            'ground_truth': trace['answer']  # Use the original answer as ground truth
        }
        for problem_id, trace in traces.items()
    ]
    
    # Initialize pipeline
    pipeline = SummaryReasoningPipeline(config, method=method)
    
    # Run summarization for each problem
    results = {}
    for problem_id in traces:
        print(f"Processing problem {problem_id}...")
        try:
            result = pipeline.run(
                dataset=dataset,
                problem_ids=[problem_id],
                max_iterations=max_iterations
            )
            results[problem_id] = result
            
            # Save intermediate results
            output_dir = PROJECT_ROOT / "results" / "summarization_sweep"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{problem_id}_{summarization_mode}_summary.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error processing problem {problem_id}: {str(e)}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run summarization sweep on reasoning traces")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--method", default="self", choices=["self", "external"], help="Summarization method")
    parser.add_argument("--max-iterations", type=int, default=2, help="Maximum number of iterations")
    parser.add_argument("--problem-ids", help="Comma-separated list of problem IDs to process")
    parser.add_argument("--summarization-mode", default="append", choices=["append", "prompt"], 
                       help="How to include summary in prompt ('append' or 'prompt')")
    
    args = parser.parse_args()
    
    try:
        results = run_summarization_sweep(
            config_path=args.config,
            method=args.method,
            max_iterations=args.max_iterations,
            problem_ids=args.problem_ids,
            summarization_mode=args.summarization_mode
        )
        print(f"Successfully processed {len(results)} problems")
    except Exception as e:
        print(f"Error running summarization sweep: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 