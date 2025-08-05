#!/usr/bin/env python3
"""Re-evaluate countdown results with the updated evaluation logic."""

import json
import argparse
import os
import sys
from pathlib import Path
import logging
import ast
import re

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.countdown_check import check_one_countdown_answer, validate_equation, evaluate_equation
from src.reasoning.extractor import extract_answer_with_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reevaluate_countdown_results(results_file: str, output_file: str = None):
    """Re-evaluate countdown results with proper validation."""
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check if this is a countdown experiment
    config = results.get('config', {})
    if config.get('answer_extractor') != 'countdown':
        logger.warning("This doesn't appear to be a countdown experiment. Proceeding anyway...")
    
    # Re-evaluate each problem
    problems_fixed = 0
    total_problems = 0
    
    for problem in results.get('results', []):
        total_problems += 1
        problem_id = problem.get('problem_id', 'unknown')
        
        # Get the ground truth
        correct_answer = problem.get('correct_answer')
        if not correct_answer:
            logger.warning(f"Problem {problem_id} missing correct_answer")
            continue
            
        # Try to get nums from the problem
        nums = None
        
        # First check if it's already in processed_gt_answer
        processed_gt = problem.get('processed_gt_answer')
        if isinstance(processed_gt, dict) and 'nums' in processed_gt:
            nums = processed_gt['nums']
            target = processed_gt['target']
        else:
            # Try to extract nums from the question
            question = problem.get('question', '')
            nums_match = re.search(r'\[([^\]]+)\]', question)
            if nums_match:
                try:
                    nums = ast.literal_eval('[' + nums_match.group(1) + ']')
                    target = int(correct_answer)
                except:
                    logger.warning(f"Problem {problem_id}: Failed to parse nums from question")
                    continue
            else:
                logger.warning(f"Problem {problem_id}: No nums found")
                continue
        
        if not nums:
            logger.warning(f"Problem {problem_id}: Cannot determine available numbers")
            continue
        
        # Re-evaluate all iterations
        for iteration in problem.get('iterations', []):
            # Get the reasoning text to re-extract answer
            reasoning_text = iteration.get('reasoning', '')
            
            # Re-extract the answer from reasoning using countdown extractor
            from src.reasoning.extractor import extract_countdown_answer
            model_answer = extract_countdown_answer(reasoning_text)
            
            # Update the stored answer
            old_answer = iteration.get('answer')
            iteration['answer'] = model_answer
            
            if old_answer != model_answer:
                logger.info(f"Problem {problem_id}, iteration {iteration.get('iteration', 0)}: "
                           f"Re-extracted answer from '{old_answer}' to '{model_answer}'")
            
            # Re-evaluate using countdown checker
            old_correct = iteration.get('correct', False)
            
            # Skip if no answer was extracted
            if not model_answer:
                new_correct = False
                check_result = {
                    'is_correct': False,
                    'error': 'No answer extracted from reasoning'
                }
            else:
                # Check if equation uses all numbers correctly
                is_valid = validate_equation(model_answer, nums)
            
                
                if is_valid:
                    # Evaluate the equation
                    result = evaluate_equation(model_answer)
                    new_correct = result is not None and abs(result - target) < 1e-5
                    check_result = {
                        'is_correct': new_correct,
                        'evaluated_result': result
                    }
                    if not new_correct and result is not None:
                        check_result['error'] = f'Equation evaluates to {result}, not {target}'
                else:
                    new_correct = False
                    check_result = {
                        'is_correct': False,
                        'error': 'Invalid equation - numbers not used correctly'
                    }
            
            # Update the result
            iteration['correct'] = new_correct
            
            # Log if the evaluation changed
            if old_correct != new_correct:
                logger.info(f"Problem {problem_id}, iteration {iteration.get('iteration', 0)}: "
                           f"Changed from {old_correct} to {new_correct}")
                logger.info(f"  Answer: {model_answer}")
                if 'error' in check_result:
                    logger.info(f"  Reason: {check_result['error']}")
                problems_fixed += 1
            
            # Also update summary results if present  
            if 'summary' in iteration:
                summary_text = iteration.get('summary', '')
                
                # Re-extract answer from summary
                summary_answer = extract_countdown_answer(summary_text)
                
                # Update the stored summary answer
                old_summary_answer = iteration.get('summary_answer')
                iteration['summary_answer'] = summary_answer
                
                if old_summary_answer != summary_answer:
                    logger.info(f"Problem {problem_id}, iteration {iteration.get('iteration', 0)} summary: "
                               f"Re-extracted answer from '{old_summary_answer}' to '{summary_answer}'")
                
                if summary_answer:
                    # Evaluate summary answer directly
                    is_valid_summary = validate_equation(summary_answer, nums)
                    
                    if is_valid_summary:
                        result_summary = evaluate_equation(summary_answer)
                        new_summary_correct = result_summary is not None and abs(result_summary - target) < 1e-5
                    else:
                        new_summary_correct = False
                else:
                    new_summary_correct = False
                
                old_summary_correct = iteration.get('summary_correct', False)
                iteration['summary_correct'] = new_summary_correct
                
                if old_summary_correct != new_summary_correct:
                    logger.info(f"Problem {problem_id}, iteration {iteration.get('iteration', 0)} summary: "
                               f"Changed from {old_summary_correct} to {new_summary_correct}")
        
        # Update the backward compatibility fields for iteration 0
        if problem.get('iterations'):
            iter0 = problem['iterations'][0]
            problem['initial_correct'] = iter0.get('correct', False)
            if 'summary_correct' in iter0:
                problem['initial_summary_correct'] = iter0['summary_correct']
    
    # Update timestamp
    import time
    results['reevaluation_timestamp'] = time.time()
    results['reevaluation_note'] = "Re-evaluated with proper countdown validation"
    
    # Save the updated results
    if output_file is None:
        # Create a new filename with _reevaluated suffix
        base = os.path.splitext(results_file)[0]
        output_file = f"{base}_reevaluated.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nRe-evaluation complete!")
    logger.info(f"Total problems: {total_problems}")
    logger.info(f"Evaluations changed: {problems_fixed}")
    logger.info(f"Results saved to: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate countdown results with updated validation')
    parser.add_argument('results_file', help='Path to the results.json file')
    parser.add_argument('--output', '-o', help='Output file path (default: adds _reevaluated suffix)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    output_file = reevaluate_countdown_results(args.results_file, args.output)
    
    print(f"\nYou can now view the re-evaluated results with:")
    print(f"python view_results.py {output_file}")

if __name__ == "__main__":
    main()