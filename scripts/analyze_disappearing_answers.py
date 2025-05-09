#!/usr/bin/env python
import json
import sys
import os
from typing import Dict, Any, List
import re

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

def find_and_analyze_problem(results_file: str, specific_problem_id: str = None) -> None:
    """Find and analyze a problem with answers that disappear in later iterations."""
    results = load_results(results_file)
    
    print(f"Searching for problems with disappearing answers in {len(results)} problems")
    print("-" * 80)
    
    # Find a suitable problem if none specified
    target_problem = None
    
    if specific_problem_id:
        # Find the specified problem
        for problem in results:
            if problem.get("problem_id") == specific_problem_id:
                target_problem = problem
                break
        
        if not target_problem:
            print(f"Problem with ID {specific_problem_id} not found in results")
            return
    else:
        # Find the first problem where an answer is present then disappears
        for problem in results:
            iterations = problem.get("iterations", [])
            if len(iterations) >= 2:  # Need at least 2 iterations
                # Check for pattern where answer exists then disappears
                has_pattern = False
                for i in range(len(iterations) - 1):
                    if iterations[i].get("answer") is not None and iterations[i+1].get("answer") is None:
                        has_pattern = True
                        break
                
                if has_pattern:
                    target_problem = problem
                    break
    
    if not target_problem:
        print("No problems found with the pattern of answers disappearing in later iterations")
        return
    
    # Analyze the target problem
    problem_id = target_problem.get("problem_id", "unknown")
    iterations = target_problem.get("iterations", [])
    
    print(f"Analyzing problem: {problem_id}")
    print(f"Correct answer: {target_problem.get('correct_answer')}")
    print(f"Total iterations: {len(iterations)}")
    
    # Analyze each iteration
    for i, iteration in enumerate(iterations):
        answer = iteration.get("answer")
        reasoning_output = iteration.get("reasoning_output", "")
        prompt = iteration.get("prompt", "")
        
        # Check for think tags and positions
        has_think_end = "</think>" in reasoning_output
        think_end_pos = reasoning_output.find("</think>")
        
        # Check for boxed content
        boxed_match = re.search(r'\\boxed\{([^{}]+)\}', reasoning_output)
        has_boxed = bool(boxed_match)
        boxed_content = boxed_match.group(1) if boxed_match else None
        boxed_pos = boxed_match.start() if boxed_match else -1
        
        # Determine if boxed is after think tag
        boxed_after_think = (has_think_end and has_boxed and boxed_pos > think_end_pos)
        
        # Compute sizes
        prompt_length = len(prompt)
        output_length = len(reasoning_output)
        
        print(f"\nIteration {i}:")
        print(f"  Answer extracted: {answer}")
        print(f"  Has </think> tag: {has_think_end}")
        print(f"  Has \\boxed{{}}: {has_boxed}")
        if has_boxed:
            print(f"  Boxed content: {boxed_content}")
        if has_think_end and has_boxed:
            print(f"  Boxed is after </think>: {boxed_after_think}")
        print(f"  Prompt length: {prompt_length} chars")
        print(f"  Output length: {output_length} chars")
        
        # If this is the iteration where answer disappears, show relevant details
        if i > 0 and iterations[i-1].get("answer") is not None and answer is None:
            print("\n  ANSWER DISAPPEARED IN THIS ITERATION")
            print("  Analyzing truncation:")
            
            # Check if previous iteration's boxed content would have been truncated
            prev_reasoning = iterations[i-1].get("reasoning_output", "")
            prev_think_end_pos = prev_reasoning.find("</think>")
            
            curr_prompt_start = prompt[:100] + "..." if len(prompt) > 100 else prompt
            prev_prompt_start = iterations[i-1].get("prompt", "")[:100] + "..." if len(iterations[i-1].get("prompt", "")) > 100 else iterations[i-1].get("prompt", "")
            
            print(f"  Previous iteration output length: {len(prev_reasoning)} chars")
            if prev_think_end_pos >= 0:
                print(f"  Previous </think> position: {prev_think_end_pos}")
                print(f"  Content after </think> (truncated): {prev_reasoning[prev_think_end_pos:prev_think_end_pos+100]}...")
            
            print(f"  Previous prompt start: {prev_prompt_start}")
            print(f"  Current prompt start: {curr_prompt_start}")
    
    # Show the most important reasoning outputs
    for i, iteration in enumerate(iterations):
        if i > 0 and iterations[i-1].get("answer") is not None and iteration.get("answer") is None:
            # Show the end of previous iteration's output
            prev_output = iterations[i-1].get("reasoning_output", "")
            print("\nEnd of previous iteration's output (with answer):")
            print("-" * 60)
            print(prev_output[-500:])
            print("-" * 60)
            
            # Show the beginning of current iteration's prompt
            curr_prompt = iteration.get("prompt", "")
            print("\nBeginning of current iteration's prompt (truncated at </think>):")
            print("-" * 60)
            print(curr_prompt[:500])
            print("-" * 60)
            
            # Show the current iteration's output
            curr_output = iteration.get("reasoning_output", "")
            print("\nCurrent iteration's output (no answer extracted):")
            print("-" * 60)
            print(curr_output[:500])
            print("-" * 60)
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_disappearing_answers.py <results.json> [problem_id]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    specific_problem_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    find_and_analyze_problem(results_file, specific_problem_id) 