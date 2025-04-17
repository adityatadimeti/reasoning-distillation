import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_first_truncated_solution(results_file: str, output_file: str = None):
    """
    Find the first solution with finish_reason "length" and process it.
    
    Args:
        results_file: Path to the input JSON file with results
        output_file: Path to save the processed result (if None, will print to console)
    
    Returns:
        The processed solution or None if no truncated solutions found
    """
    # Load the results from the JSON file
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Check if results is in expected format
        if "results" not in data:
            logger.error("Invalid results file format: 'results' key not found")
            return None
            
        results = data["results"]
        logger.info(f"Loaded {len(results)} results from {results_file}")
    except Exception as e:
        logger.error(f"Error loading results file: {str(e)}")
        return None
    
    # Get the experiment configuration
    config = data.get("config", {})
    reasoning_prompt_template = config.get("reasoning_prompt_template", "")
    
    if not reasoning_prompt_template:
        logger.warning("No reasoning_prompt_template found in config, using empty string")
    
    # Find the first problem with a truncated solution
    truncated_problem = None
    truncated_solution = None
    
    for problem in results:
        problem_id = problem.get("problem_id", "unknown")
        question = problem.get("question", "")
        
        if "solutions" in problem:
            for solution in problem["solutions"]:
                if solution.get("finish_reason") == "length":
                    truncated_problem = problem
                    truncated_solution = solution
                    logger.info(f"Found truncated solution {solution.get('solution_id')} for problem {problem_id}")
                    break
        
        if truncated_problem:
            break
    
    if not truncated_problem:
        logger.info("No truncated solutions found")
        return None
    
    # Process the truncated solution
    problem_id = truncated_problem.get("problem_id", "unknown")
    question = truncated_problem.get("question", "")
    solution_id = truncated_solution.get("solution_id", 0)
    
    # Create modified prompt with <think> tag
    modified_prompt = reasoning_prompt_template.replace("{question}", question)
    modified_prompt += "\n\n<think>\n" + truncated_solution.get("reasoning", "")
    
    logger.info(f"Created modified prompt for problem {problem_id}, solution {solution_id}")
    
    # Here you would normally call the model to continue the solution
    # Since we're just demonstrating the preparation, we'll create a simulated result
    
    # Simulate generating a new solution (this is where you'd call your model)
    simulated_result = {
        "problem_id": problem_id,
        "solution_id": solution_id,
        "original_prompt": reasoning_prompt_template,
        "modified_prompt": modified_prompt,
        "original_reasoning": truncated_solution.get("reasoning", ""),
        "finish_reason": truncated_solution.get("finish_reason", "length"),
        # This would normally be filled with the new model output
        "continued_reasoning": "[This would be the new model output]"
    }
    
    # Save or print the result
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(simulated_result, f, indent=2)
        logger.info(f"Saved processed solution to {output_file}")
    else:
        print(json.dumps(simulated_result, indent=2))
    
    return simulated_result

async def process_and_update_solution(results_file: str, model_client=None):
    """
    Find the first solution with finish_reason "length", process it, and update it
    with a continuation from the model.
    
    Args:
        results_file: Path to the input JSON file with results
        model_client: The model client to use for continuation (must have generate_response_async method)
    
    Returns:
        The updated solution or None if no truncated solutions found
    """
    # Load the results from the JSON file
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Check if results is in expected format
        if "results" not in data:
            logger.error("Invalid results file format: 'results' key not found")
            return None
            
        results = data["results"]
        logger.info(f"Loaded {len(results)} results from {results_file}")
    except Exception as e:
        logger.error(f"Error loading results file: {str(e)}")
        return None
    
    # Get the experiment configuration
    config = data.get("config", {})
    reasoning_prompt_template = config.get("reasoning_prompt_template", "")
    
    if not reasoning_prompt_template:
        logger.warning("No reasoning_prompt_template found in config, using empty string")
    
    # Find the first problem with a truncated solution
    truncated_problem_index = -1
    truncated_solution_index = -1
    truncated_problem = None
    truncated_solution = None
    
    for i, problem in enumerate(results):
        problem_id = problem.get("problem_id", "unknown")
        
        if "solutions" in problem:
            for j, solution in enumerate(problem["solutions"]):
                if solution.get("finish_reason") == "length":
                    truncated_problem_index = i
                    truncated_solution_index = j
                    truncated_problem = problem
                    truncated_solution = solution
                    logger.info(f"Found truncated solution {solution.get('solution_id')} for problem {problem_id}")
                    break
        
        if truncated_problem:
            break
    
    if not truncated_problem:
        logger.info("No truncated solutions found")
        return None
    
    # Process the truncated solution
    problem_id = truncated_problem.get("problem_id", "unknown")
    question = truncated_problem.get("question", "")
    correct_answer = truncated_problem.get("correct_answer", "")
    solution_id = truncated_solution.get("solution_id", 0)
    
    # Create modified prompt with <think> tag
    modified_prompt = reasoning_prompt_template.replace("{question}", question)
    modified_prompt += "\n\n<think>\n" + truncated_solution.get("reasoning", "")
    
    logger.info(f"Created modified prompt for problem {problem_id}, solution {solution_id}")
    
    # If we have a model client, call it to continue the reasoning
    if model_client:
        try:
            # Call model with the modified prompt to continue reasoning
            response = await model_client.generate_response_async(
                modified_prompt,
                max_tokens=config.get("max_tokens", 32768),
                temperature=config.get("temperature", 0.6),
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 40),
                presence_penalty=config.get("presence_penalty", 0.0),
                frequency_penalty=config.get("frequency_penalty", 0.0)
            )
            
            # Process the response
            if isinstance(response, tuple):
                if len(response) >= 5:  # FireworksModelClient returns 5 elements
                    new_solution, finish_reason, token_usage, cost_info, _ = response
                elif len(response) >= 4:  # Handle 4-element tuple
                    new_solution, finish_reason, token_usage, cost_info = response
                elif len(response) >= 3:
                    new_solution, finish_reason, token_usage = response
                    cost_info = None
                elif len(response) >= 2:
                    new_solution, finish_reason = response
                    token_usage = None
                    cost_info = None
                else:
                    new_solution = response[0]
                    finish_reason = "unknown"
                    token_usage = None
                    cost_info = None
            else:
                new_solution = response
                finish_reason = "unknown"
                token_usage = None
                cost_info = None
            
            # Extract answer from the new solution
            from src.reasoning.extractor import extract_answer
            answer = extract_answer(new_solution)
            
            # Check if answer is correct
            is_correct = False
            if answer is not None:
                is_correct = answer.strip() == correct_answer.strip()
            
            # Update the solution in the original data
            updated_solution = {
                "solution_id": solution_id,
                "reasoning": new_solution,
                "answer": answer,
                "correct": is_correct,
                "finish_reason": finish_reason,
                "continued": True  # Mark this as a continued solution
            }
            
            # Update the solution in the original data
            results[truncated_problem_index]["solutions"][truncated_solution_index] = updated_solution
            
            # Recalculate consensus information for the problem
            solutions = results[truncated_problem_index]["solutions"]
            
            # Find consensus answer (most common answer)
            answers = [s["answer"] for s in solutions if s["answer"] is not None]
            consensus_answer, consensus_count = find_consensus(answers) if answers else (None, 0)
            
            # Check if consensus answer is correct
            consensus_correct = False
            if consensus_answer is not None:
                consensus_correct = consensus_answer.strip() == correct_answer.strip()
            
            # Calculate pass@k metrics
            num_correct = sum(1 for s in solutions if s["correct"])
            pass_at_k = num_correct > 0
            
            # Update the problem with new consensus information
            results[truncated_problem_index].update({
                "consensus_answer": consensus_answer,
                "consensus_correct": consensus_correct,
                "consensus_count": consensus_count,
                "pass_at_k": pass_at_k,
                "num_correct": num_correct,
                "continuation_processed": True
            })
            
            # Save the updated results
            with open(results_file, 'w') as f:
                json.dump({"config": config, "results": results}, f, indent=2)
            
            logger.info(f"Updated solution {solution_id} for problem {problem_id} and saved results")
            
            return updated_solution
            
        except Exception as e:
            logger.error(f"Error generating continuation: {str(e)}")
            return None
    else:
        logger.warning("No model client provided, can't generate continuation")
        return {
            "problem_id": problem_id,
            "solution_id": solution_id,
            "modified_prompt": modified_prompt,
            "status": "ready_for_continuation"
        }

def find_consensus(answers: List[str]) -> Tuple[Optional[str], int]:
    """Find the most common answer from a list of answers."""
    if not answers:
        return None, 0
        
    # Normalize answers by stripping whitespace
    normalized_answers = [a.strip() for a in answers if a is not None]
    
    # Count occurrences
    answer_counts = {}
    for answer in normalized_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    # Find most common answer
    if not answer_counts:
        return None, 0
        
    most_common = max(answer_counts.items(), key=lambda x: x[1])
    return most_common[0], most_common[1]  # Return tuple of (answer, count)

# Main function to demonstrate usage
async def main():
    results_file = "results/aime_2024_consensus_update_test_7b/aime_2024_pass_consensus_update_test_7b_20250415_140805/results.json"  # Update this path
    
    # Option 1: Just identify and prepare the first truncated solution
    result = await process_first_truncated_solution(results_file)
    
    # Option 2: Process and update with actual model (uncomment to use)
    # from src.llm.model_factory import create_model_client
    # model_client = create_model_client(
    #     "your_model_name",
    #     provider="your_provider"  # Optional
    # )
    # result = await process_and_update_solution(results_file, model_client)
    
    print("Processed result:", result)

if __name__ == "__main__":
    asyncio.run(main())