#!/usr/bin/env python
import json
import sys
import os
import re
from typing import Dict, Any, List, Optional

# Import the actual math extractor
sys.path.append('.')  # Add current directory to path
try:
    from src.reasoning.extractor import extract_math_answer
    from src.reasoning.math import last_boxed_only_string, remove_boxed
    IMPORTS_SUCCEEDED = True
except ImportError as e:
    print(f"Warning: Failed to import from src.reasoning: {e}")
    IMPORTS_SUCCEEDED = False

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

def check_boxed_content(text: str) -> Optional[str]:
    """Manually check for boxed content using regex."""
    if not text:
        return None
        
    # Look for \boxed{content} pattern
    pattern = r'\\boxed\{([^{}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # Return the last match
    return None

def inspect_answer_extraction(results_file: str, problem_id: str) -> None:
    """Inspect why answer extraction failed on a specific problem."""
    results = load_results(results_file)
    
    # Find the specified problem
    target_problem = None
    for problem in results:
        if str(problem.get("problem_id", "")) == str(problem_id):
            target_problem = problem
            break
    
    if not target_problem:
        print(f"Problem with ID {problem_id} not found in results")
        return
    
    iterations = target_problem.get("iterations", [])
    print(f"Analyzing answer extraction for problem {problem_id}")
    print(f"Correct answer: {target_problem.get('correct_answer')}")
    print(f"Total iterations: {len(iterations)}")
    
    # Check each iteration
    for i, iteration in enumerate(iterations):
        print(f"\n{'-'*20} ITERATION {i} {'-'*20}")
        
        # Get the stored answer from results
        stored_answer = iteration.get("answer")
        print(f"Stored answer in results.json: {stored_answer}")
        
        # Get the reasoning text that should have been extracted from
        reasoning_output = iteration.get("reasoning_output", "")
        full_extraction_text = iteration.get("reasoning_full_for_extraction", "")
        
        # Simple check for \boxed pattern with our own regex
        boxed_in_output = check_boxed_content(reasoning_output)
        boxed_in_full = check_boxed_content(full_extraction_text)
        
        print(f"Basic regex finds boxed in reasoning_output: {boxed_in_output}")
        print(f"Basic regex finds boxed in full extraction text: {boxed_in_full}")
        
        # Only proceed with imported functions if imports succeeded
        if IMPORTS_SUCCEEDED:
            try:
                # Try to extract boxed content using the actual implementation
                answer_from_output = extract_math_answer(reasoning_output)
                answer_from_full = extract_math_answer(full_extraction_text) if full_extraction_text else None
                
                print(f"Math extractor result from reasoning_output: {answer_from_output}")
                print(f"Math extractor result from full extraction text: {answer_from_full}")
                
                # Show if there's a discrepancy
                if stored_answer != answer_from_output and boxed_in_output:
                    print(f"⚠️ DISCREPANCY: Math extractor found '{answer_from_output}' but stored answer is '{stored_answer}'")
            except Exception as e:
                print(f"Error using math extractor: {e}")
        
        # Count instances of \boxed
        output_boxed_count = reasoning_output.count("\\boxed{")
        full_boxed_count = full_extraction_text.count("\\boxed{") if full_extraction_text else 0
        
        print(f"Number of \\boxed{{}} in reasoning_output: {output_boxed_count}")
        print(f"Number of \\boxed{{}} in full extraction text: {full_boxed_count}")
        
        # If the output has boxed content but answer is None, show context
        if output_boxed_count > 0 and stored_answer is None:
            print("\n⚠️ ISSUE DETECTED: \\boxed{} found but answer is None")
            
            # Find and show all boxed expressions
            print("\nAll boxed expressions in reasoning_output:")
            
            # Find all \boxed{...} patterns
            boxed_patterns = re.finditer(r'\\boxed\{([^{}]+)\}', reasoning_output)
            for match in boxed_patterns:
                start, end = match.span()
                content = match.group(1)
                context_start = max(0, start - 20)
                context_end = min(len(reasoning_output), end + 20)
                context = reasoning_output[context_start:context_end]
                print(f"  Found: '{content}' in context: '...{context}...'")
        
        # Show a preview of the reasoning output
        if reasoning_output:
            print("\nReasoning output preview (first 150 chars):")
            print(reasoning_output[:150] + "..." if len(reasoning_output) > 150 else reasoning_output)
            
            print("\nReasoning output end (last 150 chars):")
            print("..." + reasoning_output[-150:] if len(reasoning_output) > 150 else reasoning_output)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_math_extractor.py <results.json> <problem_id>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    problem_id = sys.argv[2]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found")
        sys.exit(1)
    
    inspect_answer_extraction(results_file, problem_id) 