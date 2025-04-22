from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import traceback
from threading import Lock
from dataclasses import asdict

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer_with_config
from src.llm.tokenization import format_chat_for_completions

logger = logging.getLogger(__name__)

# Helper function to add token usage and cost
def _add_usage_cost(total_usage, usage, total_cost, cost):
    if usage:
        total_usage.prompt_tokens += usage.prompt_tokens
        total_usage.completion_tokens += usage.completion_tokens
        total_usage.total_tokens += usage.total_tokens
    if cost:
        total_cost.prompt_cost += cost.prompt_cost
        total_cost.completion_cost += cost.completion_cost
        total_cost.total_cost += cost.total_cost

class ContinuationExperiment(BaseExperiment):
    """Experiment for testing reasoning improvement through controlled continuation."""

    def __init__(
        self,
        experiment_name: str = "test_continuation",
        config: Dict[str, Any] = None,
        dashboard: Optional[Any] = None, # Keep dashboard arg for compatibility, but ignore it
        verbose: bool = False
    ):
        """Initialize the continuation experiment."""
        super().__init__(experiment_name, config, dashboard=None) # Explicitly set dashboard to None

        self.verbose = verbose

        # Validate required parameters
        required_params = [
            "reasoning_model", "max_tokens", "temperature",
            "top_p", "top_k", "presence_penalty", "frequency_penalty",
            "reasoning_prompt_template", "max_iterations"
        ]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in configuration")

        # Initialize reasoning model
        reasoning_provider = self.config.get("reasoning_model_provider", None)
        self.reasoning_model = create_model_client(
            self.config["reasoning_model"],
            provider=reasoning_provider
        )

        # Add lock for thread safety when updating results
        self.results_lock = Lock()

    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous run is not implemented for ContinuationExperiment. Use run_parallel."""
        raise NotImplementedError("Synchronous execution is not supported. Please use run_parallel.")

    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Run the continuation experiment on a list of problems in parallel.

        Args:
            problems: List of problem dictionaries
            max_concurrency: Maximum number of problems to process concurrently

        Returns:
            List of results for all problems
        """
        total_problems = len(problems)
        logger.info(f"Processing {total_problems} problems with max concurrency of {max_concurrency}")

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        # Create a list to store tasks
        tasks = []

        # Process each problem asynchronously
        for i, problem in enumerate(problems):
            # Create a task for this problem
            task = asyncio.create_task(self._process_problem_with_semaphore(semaphore, problem, i, total_problems))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Sort results by problem ID or index for consistent output
        sorted_results = sorted(self.results, key=lambda x: x.get("problem_id", ""))

        # Save final results
        self.results = sorted_results
        self.save_results()

        return self.results

    async def _process_problem_with_semaphore(self, semaphore: asyncio.Semaphore, problem: Dict[str, Any],
                                             index: int, total: int) -> Dict[str, Any]:
        """Process a problem with semaphore for concurrency control."""
        async with semaphore:
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(index+1)))

            logger.info(f"[{index+1}/{total}] Starting problem {problem_id}")

            try:
                # Process the problem asynchronously
                result = await self._process_problem_async(problem)

                # Thread-safe update of results
                with self.results_lock:
                    self.results.append(result)

                logger.info(f"[{index+1}/{total}] Completed problem {problem_id}")

                # Save intermediate results
                if self.config.get("save_intermediate", True):
                    self.save_results()

                return result

            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error processing problem {problem_id}: {str(e)}\n{error_traceback}")

                # Add error to results in a thread-safe way
                error_result = {
                    "problem_id": problem_id,
                    "question": problem.get("question", "N/A"),
                    "correct_answer": problem.get("answer", problem.get("correct_answer", "N/A")),
                    "error": str(e),
                    "traceback": error_traceback,
                    "status": "error"
                }

                with self.results_lock:
                    self.results.append(error_result)

                # Save intermediate results
                if self.config.get("save_intermediate", True):
                    self.save_results()

                return error_result

    async def _process_problem_async(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the continuation pipeline asynchronously.

        Args:
            problem: Problem dictionary with 'question' and 'answer' keys

        Returns:
            Result dictionary with reasoning iterations
        """
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]

        # Get initial reasoning prompt template
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")

        # Prepare initial messages for formatting
        initial_messages = [
            {"role": "user", "content": reasoning_template.replace("{question}", question)}
        ]

        # Format the initial prompt using the client's expected format
        try:
            formatted_initial_prompt = format_chat_for_completions(
                initial_messages,
                self.config["reasoning_model"]
            )
            # # Ensure the base prompt ends correctly for assistant generation
            # if not formatted_initial_prompt.strip().endswith(("<|assistant|>", "<|Assistant|>", "ASSISTANT:")):
            #      # Add the standard assistant marker if not present
            #      # Heuristic: Add based on common patterns, might need refinement for other models
            #      if "<|" in formatted_initial_prompt:
            #          formatted_initial_prompt += "<|Assistant|>"
            #      else:
            #          formatted_initial_prompt += "\nASSISTANT:"
            #      logger.warning(f"Appended assistant marker to formatted prompt for {self.config['reasoning_model']}")

        except Exception as e:
            logger.error(f"Failed to format initial prompt for model {self.config['reasoning_model']}: {e}")
            raise ValueError(f"Could not format initial prompt: {e}")

        # Initialize result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "iterations": [],
            "timestamp": time.time(),
            "status": "in-progress"
        }

        # Configuration for sub-iterations
        max_sub_iterations = self.config.get("max_sub_iterations", 4) # Max calls within one logical iteration
        max_tokens_per_sub_call = self.config["max_tokens"]

        # --- Function to handle generation with sub-iterations ---
        async def generate_with_sub_iterations(base_prompt: str, log_iter_num: int):
            full_reasoning = ""
            current_prompt_for_sub_call = base_prompt
            final_finish_reason = "error"
            accumulated_usage = TokenUsage(0, 0, 0)
            accumulated_cost = CostInfo(0, 0, 0)
            detailed_sub_calls = []

            for sub_iter in range(max_sub_iterations):
                logger.info(f"Problem {problem_id}, iteration {log_iter_num}, sub-iteration {sub_iter}")
                try:
                    sub_response = await self.reasoning_model.generate_response_async(
                        current_prompt_for_sub_call,
                        max_tokens=max_tokens_per_sub_call,
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                        presence_penalty=self.config["presence_penalty"],
                        frequency_penalty=self.config["frequency_penalty"],
                        verbose=self.verbose,
                        is_preformatted=True,
                        enable_continuation=False, # Ensure client's internal continuation is off
                        max_continuations=0,
                        track_token_callback=self.track_token_usage_and_cost, # Callback still useful for base tracking
                        track_token_callback_args={
                            "problem_id": problem_id,
                            "iteration": log_iter_num, # Pass the main iteration number
                            "step": f"reasoning_sub_{sub_iter}" # Add sub-iteration info
                        }
                    )

                    partial_reasoning, finish_reason, usage, cost, details = sub_response
                    final_finish_reason = finish_reason # Update final reason each time

                    # Accumulate results
                    full_reasoning += partial_reasoning
                    _add_usage_cost(accumulated_usage, usage, accumulated_cost, cost)
                    # Store details of this specific sub-call API interaction
                    # details is a list, we usually expect one item unless client did internal continuation (which is disabled)
                    sub_call_detail = details[0] if details else {}
                    sub_call_detail["sub_iteration"] = sub_iter
                    detailed_sub_calls.append(sub_call_detail)

                    if finish_reason == "stop":
                        logger.info(f"Problem {problem_id}, iteration {log_iter_num}, sub-iteration {sub_iter} finished with reason: stop")
                        break # Finished generation for this logical iteration
                    elif finish_reason == "length":
                        logger.info(f"Problem {problem_id}, iteration {log_iter_num}, sub-iteration {sub_iter} finished with reason: length. Continuing...")
                        # Prepare prompt for the next sub-iteration
                        current_prompt_for_sub_call += partial_reasoning
                        # Optional: Add a check for total accumulated tokens if needed
                        if sub_iter == max_sub_iterations - 1:
                             logger.warning(f"Problem {problem_id}, iteration {log_iter_num} reached max_sub_iterations ({max_sub_iterations}) and was still truncated.")
                    else:
                        # Handle other potential finish reasons if necessary
                        logger.warning(f"Problem {problem_id}, iteration {log_iter_num}, sub-iteration {sub_iter} finished with unexpected reason: {finish_reason}")
                        break

                except Exception as sub_error:
                    error_traceback = traceback.format_exc()
                    logger.error(f"Error during sub-iteration {sub_iter} for problem {problem_id}, iteration {log_iter_num}: {str(sub_error)}\n{error_traceback}")
                    # Store error info for this sub-call
                    detailed_sub_calls.append({"sub_iteration": sub_iter, "error": str(sub_error), "traceback": error_traceback})
                    final_finish_reason = "error"
                    # Stop processing sub-iterations for this logical iteration on error
                    break

            return full_reasoning, final_finish_reason, accumulated_usage, accumulated_cost, detailed_sub_calls

        # --- Iteration 0 --- 
        iteration_num = 0
        logger.info(f"Processing problem {problem_id}, iteration {iteration_num} (with sub-iterations)")
        base_prompt_for_iter0 = formatted_initial_prompt
        try:
            # Call the helper function for generation
            current_reasoning, finish_reason, usage, cost, detailed_sub_calls = await generate_with_sub_iterations(
                base_prompt_for_iter0, iteration_num
            )

            if finish_reason == "error": # Check if generation failed
                 raise ValueError("Generation failed during sub-iterations")

            logger.info(f"Problem {problem_id}, iteration {iteration_num} final finish reason: {finish_reason}")

            # Extract answer from the FULL reasoning for the iteration
            current_answer = extract_answer_with_config(current_reasoning, self.config)

            # Check correctness
            current_correct = False
            if current_answer is not None:
                current_correct = current_answer.strip() == correct_answer.strip()

            # Store iteration results
            result["iterations"].append({
                "iteration": iteration_num,
                "prompt": base_prompt_for_iter0, # Save the initial formatted prompt
                "reasoning": current_reasoning, # The full reasoning accumulated across sub-iterations
                "answer": current_answer,
                "correct": current_correct,
                "final_finish_reason": finish_reason, # The reason from the last sub-call
                "sub_calls": detailed_sub_calls, # Store details of each sub-call
                "accumulated_token_usage": asdict(usage) if usage else None,
                "accumulated_cost_info": asdict(cost) if cost else None
            })

            found_correct_answer = current_correct

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error during iteration {iteration_num} processing for problem {problem_id}: {str(e)}\n{error_traceback}")
            # Ensure iteration entry exists even on error, potentially with partial sub_calls
            if not any(d['iteration'] == iteration_num for d in result['iterations']):
                 result["iterations"].append({
                     "iteration": iteration_num,
                     "prompt": base_prompt_for_iter0,
                     "error": str(e),
                     "traceback": error_traceback,
                     "sub_calls": detailed_sub_calls if 'detailed_sub_calls' in locals() else [] # Add partial details if available
                 })
            else: # If entry exists, add error info
                result["iterations"][-1]["error"] = str(e)
                result["iterations"][-1]["traceback"] = error_traceback

            result["status"] = "error"
            return result # Stop processing if initial iteration fails

        # --- Subsequent Iterations ---
        max_iterations = self.config.get("max_iterations", 1)
        think_end_tag = "</think>"
        continuation_suffix = "Wait"
        # Initialize the cumulative prompt with the result of iteration 0
        cumulative_prompt_so_far = base_prompt_for_iter0 # Start with formatted initial prompt
        # Append the truncated reasoning from iteration 0
        think_end_index_0 = current_reasoning.rfind(think_end_tag)
        if think_end_index_0 != -1:
            cumulative_prompt_so_far += current_reasoning[:think_end_index_0] + continuation_suffix
        else:
            logger.warning(f"Could not find '{think_end_tag}' in iteration 0 reasoning for problem {problem_id}. Appending full output to cumulative prompt.")
            cumulative_prompt_so_far += current_reasoning + continuation_suffix

        for next_iteration_num in range(1, max_iterations):
            detailed_sub_calls = [] # Reset for each new logical iteration
            accumulated_usage = TokenUsage(0, 0, 0)
            accumulated_cost = CostInfo(0, 0, 0)
            
            if not self.config.get("continue_after_correct", False) and found_correct_answer:
                logger.info(f"Stopping early for problem {problem_id} after finding correct answer in iteration {next_iteration_num - 1}")
                break

            logger.info(f"Processing problem {problem_id}, iteration {next_iteration_num} (with sub-iterations)")

            try:
                # The prompt for this iteration IS the cumulative prompt built so far
                base_prompt_for_this_iter = cumulative_prompt_so_far
                logger.debug(f"Base prompt for iter {next_iteration_num} (last 200 chars):\n...{base_prompt_for_this_iter[-200:]}")

                # Generate next iteration reasoning using the sub-iteration helper
                # The helper will handle potential 'length' finish reasons internally
                next_reasoning, finish_reason, usage, cost, detailed_sub_calls = await generate_with_sub_iterations(
                    base_prompt_for_this_iter, next_iteration_num
                )

                if finish_reason == "error": # Check if generation failed
                     raise ValueError("Generation failed during sub-iterations")

                logger.info(f"Problem {problem_id}, iteration {next_iteration_num} final finish reason: {finish_reason}")

                # Extract answer from the FULL reasoning for the iteration
                next_answer = extract_answer_with_config(next_reasoning, self.config)

                # Check correctness
                next_correct = False
                if next_answer is not None:
                    next_correct = next_answer.strip() == correct_answer.strip()
                    found_correct_answer = found_correct_answer or next_correct

                # Store iteration results
                result["iterations"].append({
                    "iteration": next_iteration_num,
                    "prompt": base_prompt_for_this_iter, # Save the base pre-formatted prompt used for this iteration
                    "reasoning": next_reasoning, # Full reasoning from sub-iterations
                    "answer": next_answer,
                    "correct": next_correct,
                    "final_finish_reason": finish_reason, # Final reason from last sub-call
                    "sub_calls": detailed_sub_calls,
                    "accumulated_token_usage": asdict(usage) if usage else None,
                    "accumulated_cost_info": asdict(cost) if cost else None
                })

                # Update current reasoning for the next loop (if any)
                current_reasoning = next_reasoning
                current_answer = next_answer
                current_correct = next_correct

                # --- IMPORTANT: Update cumulative prompt for the *next* iteration --- 
                think_end_index_next = next_reasoning.rfind(think_end_tag)
                if think_end_index_next != -1:
                    cumulative_prompt_so_far += next_reasoning[:think_end_index_next] + continuation_suffix
                else:
                     logger.warning(f"Could not find '{think_end_tag}' in iteration {next_iteration_num} reasoning for problem {problem_id}. Appending full output to cumulative prompt.")
                     cumulative_prompt_so_far += next_reasoning + continuation_suffix
                # ---------------------------------------------------------------------

            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error during iteration {next_iteration_num} processing for problem {problem_id}: {str(e)}\n{error_traceback}")
                 # Ensure iteration entry exists even on error, potentially with partial sub_calls
                if not any(d['iteration'] == next_iteration_num for d in result['iterations']):
                    result["iterations"].append({
                        "iteration": next_iteration_num,
                        "prompt": base_prompt_for_this_iter if 'base_prompt_for_this_iter' in locals() else None,
                        "error": str(e),
                        "traceback": error_traceback,
                        "sub_calls": detailed_sub_calls # Add partial details if available
                    })
                else: # If entry exists, add error info
                    result["iterations"][-1]["error"] = str(e)
                    result["iterations"][-1]["traceback"] = error_traceback
                    if detailed_sub_calls: # Add sub-call details if available
                         result["iterations"][-1]["sub_calls"] = detailed_sub_calls

                result["status"] = "error"
                # Decide whether to break or continue to next iteration on error
                # For now, let's break to avoid potential cascading errors
                break

        # --- Finalization ---
        if result["status"] != "error":
            result["status"] = "completed"

        # Set final answer based on the last successfully completed iteration
        if result["iterations"]:
            last_successful_iteration = None
            for i in range(len(result["iterations"]) - 1, -1, -1):
                 if "error" not in result["iterations"][i]:
                     last_successful_iteration = result["iterations"][i]
                     break

            if last_successful_iteration:
                 result["final_answer"] = last_successful_iteration.get("answer")
                 result["final_correct"] = last_successful_iteration.get("correct", False)
                 logger.info(f"Final answer for problem {problem_id} (from iter {last_successful_iteration.get('iteration')}): {result['final_answer']}")
                 logger.info(f"Final answer correct: {result['final_correct']}")
            else:
                 # Handle case where even iteration 0 failed
                 result["final_answer"] = None
                 result["final_correct"] = False
                 logger.warning(f"No successful iterations found for problem {problem_id}. Final answer set to None.")
        else:
            # Should not happen if iteration 0 ran, but handle defensively
             result["final_answer"] = None
             result["final_correct"] = False
             logger.error(f"No iterations recorded for problem {problem_id}. Setting final answer to None.")


        return result 