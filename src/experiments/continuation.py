from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import traceback
from threading import Lock
from dataclasses import asdict, is_dataclass

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer_with_config
from src.llm.tokenization import format_chat_for_completions # Keep for initial formatting

logger = logging.getLogger(__name__)

# Helper function to add token usage and cost (can be removed if not used elsewhere)
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
    """
    Experiment for testing reasoning improvement through controlled continuation,
    forcing the model to continue thinking from a specific point by appending "Wait".
    This experiment relies on modifications to the underlying ModelClient
    (e.g., FireworksModelClient) to accept preformatted prompts and disable
    internal continuation logic for subsequent iterations.
    """

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
            "top_p", #"top_k", # top_k might not be universal
            "presence_penalty", "frequency_penalty",
            "reasoning_prompt_template", "max_iterations"
        ]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in configuration")
        if "top_k" not in self.config:
             logger.warning("Parameter 'top_k' not found in config, proceeding without it.")


        # Initialize reasoning model
        reasoning_provider = self.config.get("reasoning_model_provider", None)
        self.reasoning_model = create_model_client(
            self.config["reasoning_model"],
            provider=reasoning_provider
        )

        # Add lock for thread safety when updating results
        self.results_lock = Lock()
        
        # Check if the model client supports preformatted prompts (heuristic check)
        if not hasattr(self.reasoning_model, 'generate_response_async') or \
           'preformatted_prompt' not in self.reasoning_model.generate_response_async.__code__.co_varnames:
            logger.warning(f"The configured reasoning model client ({type(self.reasoning_model).__name__}) "
                           "might not support the 'preformatted_prompt' argument required by ContinuationExperiment. "
                           "Ensure the client is updated.")

        # Flag to control behavior when think tag is missing
        self.append_wait_if_tag_missing = self.config.get("append_wait_if_tag_missing", False)
        if self.append_wait_if_tag_missing:
            logger.info("Configuration enables appending 'Wait' suffix even if think end tag is missing.")

    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous run is not implemented for ContinuationExperiment. Use run_parallel."""
        raise NotImplementedError("Synchronous execution is not supported. Please use run_parallel.")

    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 4) -> List[Dict[str, Any]]:
        """
        Run the continuation experiment on a list of problems in parallel.

        Args:
            problems: List of problem dictionaries
            max_concurrency: Maximum number of problems to process concurrently

        Returns:
            List of results for all problems
        """
        total_problems = len(problems)
        # Adjust concurrency if needed, ensuring it's at least 1
        max_concurrency = max(1, self.config.get("max_concurrency", max_concurrency))
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
        try:
            # Attempt to sort numerically if IDs are numbers, otherwise sort as strings
            sorted_results = sorted(self.results, key=lambda x: int(x.get("problem_id", 0)) if str(x.get("problem_id", "")).isdigit() else float('inf'))
        except ValueError:
            # Fallback to string sorting if conversion fails
             sorted_results = sorted(self.results, key=lambda x: str(x.get("problem_id", "")))


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

        # Initialize result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "iterations": [],
            "timestamp": time.time(),
            "status": "in-progress"
        }

        load_iter0 = hasattr(self, 'reusing_initial_reasoning') and self.reusing_initial_reasoning
        if load_iter0 and problem_id not in self.initial_reasoning_map:
            raise ValueError(f"Problem {problem_id} not found in initial reasoning map but reusing_initial_reasoning=True")
        load_iter0 = load_iter0 and problem_id in self.initial_reasoning_map
        
        if load_iter0:
            logger.info(f"Using preloaded initial reasoning for problem {problem_id}")
            initial_data = self.initial_reasoning_map[problem_id]
            
            # Retrieve necessary data
            formatted_prompt_iter_0 = initial_data["prompt"]
            reasoning_0_output = initial_data["reasoning_output"]
            answer_0 = initial_data["answer"]
            correct_0 = initial_data["correct"]
            finish_reason_0 = initial_data["finish_reason"]
            details_0 = initial_data["api_calls"]
            
            # Construct the full text for continuation
            current_full_text = formatted_prompt_iter_0 + reasoning_0_output

            # Store the loaded iteration 0 data
            result["iterations"].append({
                "iteration": 0,
                "prompt": formatted_prompt_iter_0,
                "reasoning_output": reasoning_0_output,
                "reasoning_full_for_extraction": current_full_text,
                "answer": answer_0,
                "correct": correct_0,
                "final_finish_reason": finish_reason_0,
                "api_calls": details_0, 
                "loaded_from_source": True # Add flag indicating data was loaded
            })
            
            found_correct_answer = correct_0
            # Skip the API call section for iteration 0

        else: # Generate Iteration 0 if not loaded
            # --- Iteration 0 ---
            iteration_num = 0
            initial_user_prompt = reasoning_template.replace("{question}", question)
            logger.info(f"Processing problem {problem_id}, iteration {iteration_num}")

            try:
                # 1. Format the initial prompt explicitly using the tokenizer's chat template
                initial_messages = [{"role": "user", "content": initial_user_prompt}]
                try:
                    formatted_prompt_iter_0 = format_chat_for_completions(
                        initial_messages,
                        self.config["reasoning_model"]
                    )
                except Exception as format_error:
                    logger.error(f"Failed to format initial prompt for model {self.config['reasoning_model']}: {format_error}")
                    raise ValueError(f"Could not format initial prompt: {format_error}")

                # 2. Initial call using the preformatted prompt
                response_0 = await self.reasoning_model.generate_response_async(
                    prompt=None, # Pass None as we provide the preformatted prompt
                    preformatted_prompt=formatted_prompt_iter_0,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k"),
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose,
                    enable_continuation=False, # Disable internal continuation for this call too
                    # max_total_tokens and max_continuations are ignored when enable_continuation is False
                    track_token_callback=self.track_token_usage_and_cost,
                    track_token_callback_args={
                        "problem_id": problem_id,
                        "iteration": iteration_num,
                        "step": "reasoning"
                    }
                )
                # Since enable_continuation=False, details_0 should contain info for one API call
                reasoning_0_output, finish_reason_0, usage_0, cost_0, details_0 = response_0

                # 3. Construct the full text for this iteration
                # The full text includes the formatted prompt AND the model's output
                current_full_text = formatted_prompt_iter_0 + reasoning_0_output

                # 4. Extract answer from the full reasoning output of this iteration
                answer_0 = extract_answer_with_config(reasoning_0_output, self.config)

                # 5. Check correctness
                correct_0 = False
                if answer_0 is not None:
                    correct_0 = answer_0.strip() == correct_answer.strip()

                # 6. Store results
                result["iterations"].append({
                    "iteration": iteration_num,
                    "prompt": formatted_prompt_iter_0, # Store the formatted prompt used
                    "reasoning_output": reasoning_0_output, # Just the output from the model
                    "reasoning_full_for_extraction": current_full_text, # Prompt + Output
                    "answer": answer_0,
                    "correct": correct_0,
                    "final_finish_reason": finish_reason_0,
                    "api_calls": details_0,
                })

                found_correct_answer = correct_0

            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error during iteration {iteration_num} processing for problem {problem_id}: {str(e)}\n{error_traceback}")
                result["iterations"].append({
                    "iteration": iteration_num,
                    "prompt": formatted_prompt_iter_0 if 'formatted_prompt_iter_0' in locals() else initial_user_prompt, # Store formatted if available
                    "error": str(e),
                    "traceback": error_traceback,
                })
                result["status"] = "error"
                return result # Stop processing if initial iteration fails

        # --- Subsequent Iterations (1 to max_iterations - 1) ---
        max_iterations = self.config.get("max_iterations", 1)
        think_end_tag = self.config.get("think_end_tag", "</think>")
        continuation_suffix = self.config.get("continuation_suffix", "Wait")

        for next_iteration_num in range(1, max_iterations + 1):
            if not self.config.get("continue_after_correct", False) and found_correct_answer:
                logger.info(f"Stopping early for problem {problem_id} after finding correct answer in iteration {next_iteration_num - 1}")
                break

            logger.info(f"Processing problem {problem_id}, iteration {next_iteration_num}")

            # --- Prepare Prompt for this iteration ---
            # Start with the *full accumulated text* from the previous iteration
            base_text_for_truncation = current_full_text 
            
            think_end_index = base_text_for_truncation.rfind(think_end_tag)
            if think_end_index != -1:
                # Truncate *before* the tag and append suffix
                # Ensure we truncate the correct base text
                prompt_for_this_iter = base_text_for_truncation[:think_end_index] + continuation_suffix
            else:
                # Check the flag to decide behavior
                if self.append_wait_if_tag_missing:
                    logger.warning(f"Could not find think_end_tag ('{think_end_tag}') in iteration {next_iteration_num-1} "
                                     f"full text for problem {problem_id}. Appending 'Wait' suffix as per config.")
                    prompt_for_this_iter = base_text_for_truncation + continuation_suffix
                else:
                    # Default behavior: Raise error
                    raise ValueError(f"Could not find think_end_tag ('{think_end_tag}') in iteration {next_iteration_num-1} "
                                     f"full text for problem {problem_id}. Cannot prepare continuation prompt. "
                                     f"(Set 'append_wait_if_tag_missing: true' in config to append 'Wait' anyway)")

            try:
                # --- Generate next part ---
                # Call using the *preformatted* prompt, disabling internal continuation
                response_k = await self.reasoning_model.generate_response_async(
                    prompt=None, # Pass None for the standard 'prompt' arg
                    preformatted_prompt=prompt_for_this_iter, # Provide the exact string
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k"),
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose,
                    enable_continuation=False, # CRUCIAL: Disable client's internal continuation
                    track_token_callback=self.track_token_usage_and_cost,
                    track_token_callback_args={
                        "problem_id": problem_id,
                        "iteration": next_iteration_num,
                        "step": "reasoning"
                    }
                )
                # Since enable_continuation=False, client makes only ONE API call.
                reasoning_k_output, finish_reason_k, usage_k, cost_k, details_k = response_k

                # --- Process & Store ---
                # Reconstruct the full reasoning text for this iteration for answer extraction purposes
                # Replace the "Wait" suffix with the actual output received
                reasoning_k_full_for_extraction = prompt_for_this_iter + reasoning_k_output

                # Update the cumulative full reasoning text for the *next* iteration's truncation base
                current_full_text = reasoning_k_full_for_extraction 

                answer_k = extract_answer_with_config(reasoning_k_output, self.config) # Extract from the *new* output only
                correct_k = False
                if answer_k is not None:
                    correct_k = answer_k.strip() == correct_answer.strip()
                    found_correct_answer = found_correct_answer or correct_k

                result["iterations"].append({
                    "iteration": next_iteration_num,
                    "prompt": prompt_for_this_iter, # The prompt ending in "Wait"
                    "reasoning_output": reasoning_k_output, # Just the new part generated
                    "reasoning_full_for_extraction": reasoning_k_full_for_extraction, # Reconstructed full text including prompt
                    "answer": answer_k,
                    "correct": correct_k,
                    "final_finish_reason": finish_reason_k, # Finish reason for this specific API call
                    "api_calls": details_k, # Details for the single API call made in this iteration
                })

                # Log if this specific call was truncated
                if finish_reason_k == "length":
                     logger.warning(f"Problem {problem_id}, iteration {next_iteration_num} API call finished due to length. Proceeding to next iteration if applicable.")

            except Exception as e:
                 error_traceback = traceback.format_exc()
                 logger.error(f"Error during iteration {next_iteration_num} processing for problem {problem_id}: {str(e)}\n{error_traceback}")
                 result["iterations"].append({
                     "iteration": next_iteration_num,
                     "prompt": prompt_for_this_iter if 'prompt_for_this_iter' in locals() else None,
                     "error": str(e),
                     "traceback": error_traceback,
                 })
                 result["status"] = "error"
                 break # Stop processing further iterations on error

        # --- Finalization ---
        if result["status"] != "error":
            result["status"] = "completed"

        # Set final answer based on the last successfully completed iteration
        if result["iterations"]:
            last_successful_iteration_data = None
            for i in range(len(result["iterations"]) - 1, -1, -1):
                 if "error" not in result["iterations"][i]:
                     last_successful_iteration_data = result["iterations"][i]
                     break

            if last_successful_iteration_data:
                 result["final_answer"] = last_successful_iteration_data.get("answer")
                 result["final_correct"] = last_successful_iteration_data.get("correct", False)
                 logger.info(f"Final answer for problem {problem_id} (from iter {last_successful_iteration_data.get('iteration')}): {result['final_answer']}")
                 logger.info(f"Final answer correct: {result['final_correct']}")
            else:
                 # Handle case where even iteration 0 might have failed but wasn't caught above
                 result["final_answer"] = None
                 result["final_correct"] = False
                 logger.warning(f"No successful iterations found for problem {problem_id}. Final answer set to None.")
        else:
             result["final_answer"] = None
             result["final_correct"] = False
             logger.error(f"No iterations recorded for problem {problem_id}. Setting final answer to None.")

        # Clean up results before returning (optional: convert dataclasses)
        # for iteration_data in result.get("iterations", []):
        #     if "api_calls" in iteration_data:
        #          # Convert usage/cost dataclasses inside api_calls if they exist
        #         for call_detail in iteration_data["api_calls"]:
        #             if "token_usage" in call_detail and is_dataclass(call_detail["token_usage"]):
        #                 call_detail["token_usage"] = asdict(call_detail["token_usage"])
        #             if "cost_info" in call_detail and is_dataclass(call_detail["cost_info"]):
        #                 call_detail["cost_info"] = asdict(call_detail["cost_info"])


        return result

    def load_initial_reasoning(self, source_results_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load initial reasoning (iteration 0) from a previous experiment's results file.

        Args:
            source_results_path: Path to the source results file (results.json)

        Returns:
            Dictionary mapping problem_id to initial reasoning data
        """
        import os
        import json
        
        if not os.path.exists(source_results_path):
            raise FileNotFoundError(f"Source results file not found: {source_results_path}")

        with open(source_results_path, "r", encoding="utf-8") as f:
            source_data = json.load(f)

        # Allow loading from either the top-level list or a dict with a 'results' key
        if isinstance(source_data, dict) and 'results' in source_data:
            source_results = source_data['results']
        elif isinstance(source_data, list):
            source_results = source_data
        else:
            raise ValueError(f"Unexpected format in {source_results_path}. Expected a list or a dict with a 'results' key.")

        initial_reasoning_map = {}
        for result in source_results:
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dictionary result: {result}")
                continue

            problem_id = result.get("problem_id")
            if problem_id and "iterations" in result and len(result["iterations"]) > 0:
                iter0_data = result["iterations"][0]
                # Ensure we have the necessary fields from iteration 0
                required_keys = ["prompt", "reasoning_output", "answer", "correct", "final_finish_reason", "api_calls"]
                if all(key in iter0_data for key in required_keys):
                    initial_reasoning_map[problem_id] = {
                        "prompt": iter0_data["prompt"], # This should be the formatted prompt
                        "reasoning_output": iter0_data["reasoning_output"],
                        "answer": iter0_data["answer"],
                        "correct": iter0_data["correct"],
                        "finish_reason": iter0_data["final_finish_reason"],
                        "api_calls": iter0_data["api_calls"],
                        # Include original question/answer for potential checks
                        "question": result.get("question"), 
                        "correct_answer": result.get("correct_answer")
                    }
                    logger.debug(f"Loaded initial reasoning for problem {problem_id}")
                else:
                    logger.warning(f"Skipping problem {problem_id} from {source_results_path}: Iteration 0 data missing required keys ({required_keys}). Found: {list(iter0_data.keys())}")
            else:
                 logger.warning(f"Skipping result from {source_results_path}: Missing problem_id or iteration 0 data.")


        logger.info(f"Loaded initial reasoning for {len(initial_reasoning_map)} problems from {source_results_path}")
        return initial_reasoning_map

    def initialize_with_previous_results(self, source_results_path: str) -> None:
        """
        Initialize the experiment with initial reasoning from a previous run.

        Args:
            source_results_path: Path to the source results file (results.json)
        """
        if not source_results_path:
            self.initial_reasoning_map = {}
            self.reusing_initial_reasoning = False
            logger.info("No source results path provided, generating all initial reasoning.")
            return
            
        self.initial_reasoning_map = self.load_initial_reasoning(source_results_path)
        self.reusing_initial_reasoning = len(self.initial_reasoning_map) > 0
        logger.info(f"Initialized experiment with {len(self.initial_reasoning_map)} preloaded reasonings. Reusing: {self.reusing_initial_reasoning}")
        
        # NOTE: If reusing, cost/token tracking needs careful handling.
        # The current BaseExperiment tracking won't automatically account for reused tokens/cost.
        # For simplicity, we might reset tracking or add a separate mechanism if needed.
        # For now, reused iterations won't contribute to the new run's cost/token count.

        