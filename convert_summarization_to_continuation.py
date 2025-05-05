#!/usr/bin/env python
import json
import sys
import os
import re
import argparse
from typing import Dict, Any, List

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(json_data, dict) and "results" in json_data:
        results = json_data["results"]
    else:
        results = json_data
    
    return results

def create_dummy_api_calls(tokens: int = 1000, cost: float = 0.001):
    """Create dummy API call data structure."""
    return [{
        "reused": True,
        "tokens": {
            "prompt_tokens": tokens // 3,
            "completion_tokens": tokens // 3 * 2,
            "total_tokens": tokens
        },
        "cost": {
            "prompt_cost": cost / 3,
            "completion_cost": cost / 3 * 2,
            "total_cost": cost
        }
    }]

def create_formatted_prompt(reasoning: str) -> str:
    """Create a mock formatted prompt that's compatible with continuation format."""
    # Chat template like format similar to what we saw in the continuation format
    prompt = "<|im_start|>user\n"
    
    # Find the first line after initial whitespace as a kind of "question"
    question_match = re.search(r'^\s*(.+?)$', reasoning, re.MULTILINE)
    if question_match:
        prompt += question_match.group(1) + "\n"
    else:
        prompt += "Solve this problem.\n"
    
    return prompt

def convert_summarization_to_continuation(summ_file: str, output_file: str = None) -> None:
    """Convert summarization experiment results to continuation format."""
    summ_results = load_results(summ_file)
    cont_results = []
    
    conversion_count = 0
    
    for problem in summ_results:
        problem_id = problem.get("problem_id", "")
        question = problem.get("question", "")
        correct_answer = problem.get("correct_answer", "")
        
        # Skip if no iterations
        if not problem.get("iterations"):
            print(f"Skipping problem {problem_id}: No iterations")
            continue
        
        # Get iteration 0
        iter0 = problem["iterations"][0]
        
        # Essential fields for continuation format
        reasoning = iter0.get("reasoning", "")
        answer = iter0.get("answer", "")
        correct = iter0.get("correct", False)
        finish_reason = iter0.get("finish_reason", "stop")
        
        # Skip if missing essential fields
        if not reasoning:
            print(f"Skipping problem {problem_id}: Missing reasoning")
            continue
        
        # Create formatted prompt
        formatted_prompt = create_formatted_prompt(reasoning)
        
        # Create new problem entry compatible with continuation format
        new_problem = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "iterations": [
                {
                    "iteration": 0,
                    "prompt": formatted_prompt,
                    "reasoning_output": reasoning,
                    "reasoning_full_for_extraction": formatted_prompt + reasoning,
                    "answer": answer,
                    "correct": correct,
                    "final_finish_reason": finish_reason,
                    "api_calls": create_dummy_api_calls()
                }
            ],
            "timestamp": problem.get("timestamp", 0),
            "status": "completed",
            "final_answer": answer,
            "final_correct": correct
        }
        
        cont_results.append(new_problem)
        conversion_count += 1
    
    # Write output
    if output_file is None:
        output_file = summ_file.replace(".json", "_converted_for_continuation.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cont_results, f, indent=2)
    
    print(f"Converted {conversion_count}/{len(summ_results)} problems from summarization to continuation format")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert summarization experiment results to continuation format")
    parser.add_argument("summ_file", help="Path to summarization results JSON file")
    parser.add_argument("--output", "-o", help="Output file path (default: input_converted_for_continuation.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.summ_file):
        print(f"Error: Summarization results file {args.summ_file} not found")
        sys.exit(1)
    
    convert_summarization_to_continuation(args.summ_file, args.output) 