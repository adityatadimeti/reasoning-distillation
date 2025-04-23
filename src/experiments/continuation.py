from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import traceback
from threading import Lock

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer_with_config
from src.llm.tokenization import format_chat_for_completions

logger = logging.getLogger(__name__)

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
            "detailed_metrics": {},
            "timestamp": time.time(),
            "status": "in-progress"
        }

        # --- Iteration 0 ---
        iteration_num = 0
        logger.info(f"Processing problem {problem_id}, iteration {iteration_num}")
        # Store the FORMATTED base prompt for reuse in continuations
        base_prompt_for_continuations = formatted_initial_prompt
        try:
            response = await self.reasoning_model.generate_response_async(
                formatted_initial_prompt, # Use the FORMATTED prompt
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose,
                is_preformatted=True,
                enable_continuation=self.config.get("enable_continuation", False), # Disable internal client continuation
                max_total_tokens=self.config.get("max_total_tokens", None), # Limit total tokens per call if needed
                max_continuations=0, # Disable internal client continuation
                track_token_callback=self.track_token_usage_and_cost,
                track_token_callback_args={
                    "problem_id": problem_id,
                    "iteration": iteration_num,
                    "step": "reasoning"
                }
            )

            # Unpack the response
            current_reasoning, finish_reason, token_usage, cost_info, detailed_api_calls = response

            # Store detailed metrics
            result["detailed_metrics"][f"iteration_{iteration_num}_reasoning"] = detailed_api_calls

            # Track token usage and cost
            self.track_token_usage_and_cost(problem_id, token_usage, cost_info, iteration_num, "reasoning")

            logger.info(f"Problem {problem_id}, iteration {iteration_num} finish reason: {finish_reason}")

            # Extract answer
            current_answer = extract_answer_with_config(current_reasoning, self.config)

            # Check correctness
            current_correct = False
            if current_answer is not None:
                current_correct = current_answer.strip() == correct_answer.strip()

            # Store iteration results
            result["iterations"].append({
                "iteration": iteration_num,
                "prompt": formatted_initial_prompt, # Save the formatted prompt
                "reasoning": current_reasoning,
                "answer": current_answer,
                "correct": current_correct,
                "finish_reason": finish_reason
            })

            found_correct_answer = current_correct

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error during iteration {iteration_num} for problem {problem_id}: {str(e)}\n{error_traceback}")
            result["iterations"].append({
                "iteration": iteration_num,
                "error": str(e),
                "traceback": error_traceback
            })
            result["status"] = "error"
            return result # Stop processing if initial iteration fails


        # --- Subsequent Iterations ---
        max_iterations = self.config.get("max_iterations", 1)
        think_end_tag = "</think>"
        continuation_suffix = "Wait"

        for next_iteration_num in range(1, max_iterations):
            if not self.config.get("continue_after_correct", False) and found_correct_answer:
                logger.info(f"Stopping early for problem {problem_id} after finding correct answer in iteration {next_iteration_num - 1}")
                break

            logger.info(f"Processing problem {problem_id}, iteration {next_iteration_num}")

            try:
                # Prepare continuation prompt by appending to the FORMATTED base prompt
                prev_reasoning_output = current_reasoning # Reasoning from iter N-1
                think_end_index = prev_reasoning_output.rfind(think_end_tag) # Use rfind to get the last occurrence

                if think_end_index != -1:
                    # Truncate the previous reasoning output before the end tag
                    truncated_prev_reasoning = prev_reasoning_output[:think_end_index]
                else:
                    logger.warning(f"Could not find '{think_end_tag}' in iteration {next_iteration_num - 1} reasoning for problem {problem_id}. Using full previous output for continuation prompt base.")
                    truncated_prev_reasoning = prev_reasoning_output

                # Construct the full, already formatted prompt for this iteration
                # Append the RAW truncated reasoning and suffix to the FORMATTED base prompt
                continuation_prompt = base_prompt_for_continuations + truncated_prev_reasoning + continuation_suffix
                logger.debug(f"Continuation prompt for iter {next_iteration_num} (last 200 chars):\n...{continuation_prompt[-200:]}") # Log end of prompt

                # Generate next iteration reasoning
                response = await self.reasoning_model.generate_response_async(
                    continuation_prompt, # Send the fully constructed, pre-formatted prompt
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose,
                    is_preformatted=True,
                    enable_continuation=self.config.get("enable_continuation", False), # Disable internal client continuation
                    max_total_tokens=self.config.get("max_total_tokens", None),
                    max_continuations=0, # Disable internal client continuation
                    track_token_callback=self.track_token_usage_and_cost,
                    track_token_callback_args={
                        "problem_id": problem_id,
                        "iteration": next_iteration_num,
                        "step": "reasoning"
                    }
                )

                # Unpack the response
                next_reasoning, finish_reason, token_usage, cost_info, detailed_api_calls = response

                # Store detailed metrics
                result["detailed_metrics"][f"iteration_{next_iteration_num}_reasoning"] = detailed_api_calls

                # Track token usage and cost
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, next_iteration_num, "reasoning")

                logger.info(f"Problem {problem_id}, iteration {next_iteration_num} finish reason: {finish_reason}")

                # Extract answer from the *new* reasoning
                next_answer = extract_answer_with_config(next_reasoning, self.config)

                # Check correctness
                next_correct = False
                if next_answer is not None:
                    next_correct = next_answer.strip() == correct_answer.strip()
                    found_correct_answer = found_correct_answer or next_correct

                # Store iteration results
                result["iterations"].append({
                    "iteration": next_iteration_num,
                    "prompt": continuation_prompt, # Save the pre-formatted prompt used
                    "reasoning": next_reasoning,
                    "answer": next_answer,
                    "correct": next_correct,
                    "finish_reason": finish_reason
                })

                # Update current reasoning for the next loop
                current_reasoning = next_reasoning
                current_answer = next_answer
                current_correct = next_correct

            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error during iteration {next_iteration_num} for problem {problem_id}: {str(e)}\n{error_traceback}")
                result["iterations"].append({
                    "iteration": next_iteration_num,
                    "error": str(e),
                    "traceback": error_traceback
                })
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