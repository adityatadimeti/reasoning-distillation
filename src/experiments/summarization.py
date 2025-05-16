from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import copy
import os
import json
import random

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_reasoning_trace, extract_answer_with_config, extract_post_think_content
from src.reasoning.summarizer import summarize_reasoning, summarize_reasoning_async
from src.dashboard.server import DashboardServer
from src.llm.tokenization import load_tokenizer, get_hf_model_name

logger = logging.getLogger(__name__)

class SummarizationExperiment(BaseExperiment):
    """Experiment for testing reasoning improvement through summarization."""
    
    def __init__(
        self, 
        experiment_name: str = "test_summarization", 
        config: Dict[str, Any] = None,
        dashboard: Optional[DashboardServer] = None,
        verbose: bool = False
    ):
        """Initialize the summarization experiment."""
        super().__init__(experiment_name, config, dashboard)
        
        # Store verbose flag
        self.verbose = verbose
        
        # Validate required parameters
        required_params = [
            "reasoning_model", "max_tokens", "temperature", 
            "top_p", "top_k", "presence_penalty", "frequency_penalty"
        ]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in configuration")
        
        # Initialize reasoning model with provider information if available
        reasoning_provider = self.config.get("reasoning_model_provider", None)
        self.reasoning_model = create_model_client(
            self.config["reasoning_model"],
            provider=reasoning_provider
        )
        
        # Initialize summarizer model (could be the same model or a different one)
        summarizer_type = self.config.get("summarizer_type", "self")
        if summarizer_type == "self":
            self.summarizer = self.reasoning_model
            # Auto-enable post-think extraction for self-summarization if not explicitly set
            if "extract_post_think_summary" not in self.config:
                self.config["extract_post_think_summary"] = True
        else:
            if "summarizer_model" not in self.config:
                raise ValueError("summarizer_model must be specified when summarizer_type is not 'self'")
            
            # Use provider information if available
            summarizer_provider = self.config.get("summarizer_model_provider", None)
            self.summarizer = create_model_client(
                self.config["summarizer_model"],
                provider=summarizer_provider
            )
        
        # Add lock for thread safety when updating results
        self.results_lock = Lock()
    
    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the summarization experiment on a list of problems."""
        total_problems = len(problems)
        for i, problem in enumerate(problems):
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            # Get the question text
            question = problem.get("question", "")
            
            logger.info(f"Processing problem {problem_id} ({i+1}/{total_problems})")
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.update_problem_status(problem_id, "in-progress", question)
                self.dashboard.update_experiment_status({
                    "total": total_problems,
                    "completed": i,
                    "status": f"Processing problem {problem_id}",
                    "config": self.config  # Include config with every status update
                })
            
            try:
                result = self._process_problem(problem)
                self.results.append(result)
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "completed", question)
            
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Error processing problem {problem_id}: {str(e)}\n{error_traceback}")
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "error", question)
                
                # Check if we have partial results to include
                partial_result = {}
                if hasattr(self, "_current_problem_result") and self._current_problem_result:
                    partial_result = self._current_problem_result
                
                # Add error to results but preserve question information and any partial results
                error_result = {
                    "problem_id": problem_id,
                    "question": question,  # Preserve the question text
                    "correct_answer": problem.get("answer", problem.get("correct_answer", "")),  # Preserve the correct answer
                    "error": str(e),
                    "status": "error"
                }
                
                # Merge with any partial results we might have
                error_result.update({k: v for k, v in partial_result.items() if k not in error_result})
                
                self.results.append(error_result)
            
            # Save intermediate results
            if self.config.get("save_intermediate", True):
                self.save_results()
                
        return self.results
    
    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Run the summarization experiment on a list of problems in parallel.
        
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
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
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
            question = problem.get("question", "")
            
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
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Error processing problem {problem_id}: {str(e)}\n{error_traceback}")
                
                # Add error to results in a thread-safe way
                error_result = {
                    "problem_id": problem_id,
                    "error": str(e),
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
        Process a single problem through the summarization pipeline asynchronously.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with reasoning iterations
        """
        import traceback
        import copy
        
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Check if we have a full checkpoint for this problem
        if hasattr(self, 'checkpoint_map') and problem_id in self.checkpoint_map:
            # Use checkpoint data
            checkpoint_data = self.checkpoint_map[problem_id]
            
            # Initialize result with the checkpoint data
            result = copy.deepcopy(checkpoint_data)
            
            # Find the current iteration (highest iteration in checkpoint)
            max_iteration = max(iter_data["iteration"] for iter_data in checkpoint_data["iterations"])
            current_iteration = max_iteration
            
            # Get the current reasoning from the last iteration
            current_reasoning = next(
                (iter_data["reasoning"] for iter_data in checkpoint_data["iterations"] 
                 if iter_data["iteration"] == current_iteration),
                None
            )
            
            # Initialize the list of all summaries from the checkpoint
            all_summaries = []
            for i in range(current_iteration):
                iter_data = next(
                    (data for data in checkpoint_data["iterations"] if data["iteration"] == i), 
                    None
                )
                if iter_data and "summary" in iter_data:
                    all_summaries.append({
                        "iteration": i,
                        "summary": iter_data["summary"],
                        "post_think_summary": iter_data.get("post_think_summary", iter_data["summary"]),
                        "finish_reason": iter_data.get("summary_finish_reason", "unknown")
                    })
            
            # Track if we've found the correct answer in any iteration
            found_correct_answer = any(iter_data.get("correct", False) for iter_data in checkpoint_data["iterations"])
            
            logger.info(f"Using checkpoint with {len(checkpoint_data['iterations'])} iterations for problem {problem_id}")
            logger.info(f"Starting from iteration {current_iteration}")
            
            # Use dummy token usage for tracking purposes since we're reusing existing iterations
            token_usage = TokenUsage(0, 0, 0)
            cost_info = CostInfo(0.0, 0.0, 0.0)
            
            # Create placeholder for detailed API calls with serializable dictionaries instead of objects
            detailed_api_calls = [{
                "reused": True, 
                "tokens": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }, 
                "cost": {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                }
            }]
            
            # Update timestamp to current time
            result["timestamp"] = time.time()
            
        # Check if we have preloaded initial reasoning (the original implementation)
        elif hasattr(self, 'initial_reasoning_map') and problem_id in self.initial_reasoning_map:
            # Use preloaded initial reasoning
            initial_data = self.initial_reasoning_map[problem_id]
            
            # Ensure we have the correct question and answer from the problem data
            if question != initial_data.get("question"):
                logger.warning(f"Question mismatch for problem {problem_id}. Using current question.")
            
            if correct_answer != initial_data.get("correct_answer"):
                logger.warning(f"Answer mismatch for problem {problem_id}. Using current answer.")
            
            # Use the reasoning, answer, and finish reason from the preloaded data
            iter0_reasoning = initial_data["reasoning"]
            iter0_answer = initial_data["answer"]
            iter0_correct = initial_data["correct"]
            iter0_finish_reason = initial_data["finish_reason"]
            
            # Set dummy token usage for tracking purposes
            token_usage = TokenUsage(0, 0, 0)
            cost_info = CostInfo(0.0, 0.0, 0.0)  # prompt_cost, completion_cost, total_cost
            
            logger.info(f"Using preloaded initial reasoning for problem {problem_id}")
            
            # Create placeholder for detailed API calls with serializable dictionaries instead of objects
            detailed_api_calls = [{
                "reused": True, 
                "tokens": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }, 
                "cost": {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                }
            }]
            
            # Construct initial result dictionary
            result = {
                "problem_id": problem_id,
                "question": question,
                "correct_answer": correct_answer,
                "iterations": [
                    {
                        "iteration": 0,
                        "reasoning": iter0_reasoning,
                        "answer": iter0_answer,
                        "correct": iter0_correct,
                        "finish_reason": iter0_finish_reason
                    }
                ],
                "timestamp": time.time()
            }
            
            # NOTE: The following fields are duplicates of data already in iterations[0]
            # They are kept for backward compatibility with existing dashboard code
            result["initial_reasoning"] = iter0_reasoning
            result["initial_answer"] = iter0_answer
            result["initial_correct"] = iter0_correct
            result["initial_finish_reason"] = iter0_finish_reason
            
            # Now that the result dictionary is initialized, store the detailed API call information
            result["detailed_metrics"] = {}
            result["detailed_metrics"]["iteration_0_reasoning"] = detailed_api_calls
            
            # Initialize for the iterations loop
            current_iteration = 0
            current_reasoning = iter0_reasoning
            all_summaries = []
            found_correct_answer = iter0_correct
        else:
            # Create initial reasoning prompt (iteration 0)
            reasoning_template = self.config.get("reasoning_prompt_template")
            if not reasoning_template:
                raise ValueError("reasoning_prompt_template must be specified in configuration")
            
            initial_prompt = reasoning_template.replace("{question}", question)
            
            # Generate iteration 0 reasoning asynchronously with detailed tracking
            response = await self.reasoning_model.generate_response_async(
                initial_prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose,
                enable_continuation=self.config.get("enable_continuation", True),
                max_total_tokens=self.config.get("max_total_tokens", 24576),
                max_continuations=self.config.get("max_continuations", 3),
                track_token_callback=self.track_token_usage_and_cost,
                track_token_callback_args={
                    "problem_id": problem_id,
                    "iteration": 0,
                    "step": "reasoning"
                }
            )
            
            # Unpack the tuple (content, finish_reason, token_usage, cost_info, detailed_api_calls)
            iter0_reasoning, iter0_finish_reason, token_usage, cost_info, detailed_api_calls = response
            
            # Track token usage and cost for the initial reasoning
            self.track_token_usage_and_cost(problem_id, token_usage, cost_info, 0, "reasoning")
            
            # Log the finish reason
            logger.info(f"Problem {problem_id}, iteration 0 finish reason: {iter0_finish_reason}")
            
            # Extract answer from iteration 0 reasoning using the configured extractor
            iter0_answer = extract_answer_with_config(iter0_reasoning, self.config)
            
            # Check if answer is correct (simple string comparison)
            iter0_correct = False
            if iter0_answer is not None:
                iter0_correct = iter0_answer.strip() == correct_answer.strip()
                
            # Construct initial result dictionary
            result = {
                "problem_id": problem_id,
                "question": question,
                "correct_answer": correct_answer,
                "iterations": [
                    {
                        "iteration": 0,
                        "reasoning": iter0_reasoning,
                        "answer": iter0_answer,
                        "correct": iter0_correct,
                        "finish_reason": iter0_finish_reason
                    }
                ],
                "timestamp": time.time()
            }
            
            # NOTE: The following fields are duplicates of data already in iterations[0]
            # They are kept for backward compatibility with existing dashboard code
            result["initial_reasoning"] = iter0_reasoning
            result["initial_answer"] = iter0_answer
            result["initial_correct"] = iter0_correct
            result["initial_finish_reason"] = iter0_finish_reason
            
            # Now that the result dictionary is initialized, store the detailed API call information
            result["detailed_metrics"] = {}
            result["detailed_metrics"]["iteration_0_reasoning"] = detailed_api_calls
            
            # Initialize for the iterations loop
            current_iteration = 0
            current_reasoning = iter0_reasoning
            all_summaries = []
            found_correct_answer = iter0_correct
        
        # Maximum number of iterations to perform
        max_iterations = self.config.get("max_iterations", 1)
        
        # Perform additional iterations if enabled and we haven't found a correct answer yet
        while (
            current_iteration < max_iterations and 
            self.config.get("enable_summarization", True) and
            (not found_correct_answer or self.config.get("continue_after_correct", False))
        ):
            # Check if this iteration already has a summary (from checkpoint)
            has_summary_in_checkpoint = False
            if hasattr(self, 'checkpoint_map') and problem_id in self.checkpoint_map:
                # Find this iteration in the checkpoint data
                iter_data = next(
                    (data for data in result["iterations"] if data["iteration"] == current_iteration),
                    None
                )
                if iter_data and "summary" in iter_data:
                    has_summary_in_checkpoint = True
                    summary = iter_data["summary"]
                    post_think_summary = iter_data.get("post_think_summary", iter_data["summary"])
                    summary_finish_reason = iter_data.get("summary_finish_reason", "unknown")
                    
                    logger.info(f"Using checkpoint summary for problem {problem_id}, iteration {current_iteration}")
                    
                    # Add to all_summaries if not already there
                    if not any(s["iteration"] == current_iteration for s in all_summaries):
                        all_summaries.append({
                            "iteration": current_iteration,
                            "summary": summary,
                            "post_think_summary": post_think_summary,
                            "finish_reason": summary_finish_reason
                        })
            
            # Only generate summary if not found in checkpoint
            if not has_summary_in_checkpoint:
                # Get the summarization prompt template
                summarize_template = self.config.get("summarize_prompt_template")
                if not summarize_template:
                    raise ValueError("summarize_prompt_template must be specified in configuration")
                
                # Generate summary of the current reasoning
                logger.info(f"Generating summary for problem {problem_id}, iteration {current_iteration}")

                # Extract reasoning trace from the full reasoning
                reasoning_trace = extract_reasoning_trace(
                    current_reasoning, 
                    allow_fallback=self.config.get("allow_fallback", False)
                )
                # If extraction failed, raise an error
                if reasoning_trace is None:
                    raise ValueError(f"Could not extract reasoning trace for problem {problem_id}. Make sure the model output contains <think> tags.")
                
                # Initialize summary variables to None to ensure they must be filled
                summary = None
                summary_finish_reason = None
                token_usage = None
                cost_info = None
                
                summary_method = self.config.get("summary_method", "single_llm_prompt") # default is using summarizer LLM prompt

                if summary_method == "single_llm_prompt":
                    breakpoint()
                    # Maximum number of retries for summary generation
                    max_retries = 3
                    retry_count = 0
                    
                    while retry_count < max_retries:
                        try:
                            # Extract continuation parameters for summary generation
                            enable_continuation = self.config.get("enable_continuation", True)
                            max_total_tokens = self.config.get("max_total_tokens", 131072)
                            max_continuations = self.config.get("max_continuations", 16)
                            
                            # For summaries, use summary-specific total tokens if specified
                            summary_total_tokens = self.config.get("summary_max_total_tokens", max_total_tokens)
                            summary_continuations = self.config.get("summary_max_continuations", max_continuations)
                            
                            # Format the summarization prompt
                            summarize_prompt = summarize_template.replace("{reasoning}", reasoning_trace).replace("{question}", question)

                            # Use the async model interface
                            summary_response = await self.summarizer.generate_response_async(
                                summarize_prompt,
                                max_tokens=self.config.get("summary_max_tokens", self.config["max_tokens"]),
                                temperature=self.config.get("summary_temperature", self.config["temperature"]),
                                top_p=self.config.get("summary_top_p", self.config["top_p"]),
                                top_k=self.config.get("summary_top_k", self.config["top_k"]) if hasattr(self.summarizer, "top_k") else None,
                                presence_penalty=self.config.get("summary_presence_penalty", self.config["presence_penalty"]),
                                frequency_penalty=self.config.get("summary_frequency_penalty", self.config["frequency_penalty"]),
                                enable_continuation=enable_continuation,
                                max_total_tokens=summary_total_tokens,
                                max_continuations=summary_continuations,
                                verbose=self.verbose,
                                track_token_callback=self.track_token_usage_and_cost,
                                track_token_callback_args={
                                    "problem_id": problem_id,
                                    "iteration": current_iteration,
                                    "step": "summary"
                                }
                            )
                            
                            # Unpack the tuple (content, finish_reason, token_usage, cost_info, detailed_api_calls)
                            summary, summary_finish_reason, token_usage, cost_info, detailed_api_calls = summary_response
                            
                            # Track token usage and cost for summary
                            self.track_token_usage_and_cost(problem_id, token_usage, cost_info, current_iteration, "summary")
                            
                            # Store the detailed API calls for the summary
                            result["detailed_metrics"][f"iteration_{current_iteration}_summary"] = detailed_api_calls
                            
                            # Successfully generated summary, exit the retry loop
                            break
                            
                        except Exception as e:
                            retry_count += 1
                            logger.warning(f"Error generating summary for problem {problem_id}, iteration {current_iteration}, attempt {retry_count}: {e}")
                            logger.warning(traceback.format_exc())
                            
                            # Calculate truncation factor - each retry we reduce by an additional 20%
                            # truncation_factor = max(0.3, 1.0 - (retry_count * 0.2))
                            
                            # logger.warning(f"Prompt too long error, will retry with truncation factor: {truncation_factor:.1%}")
                            raise Exception(f"Prompt too long error, please decrease summary_max_total_tokens")
                        
                    # For other errors, proceed with normal retry logic
                    retry_count += 1
                    logger.error(f"Error generating summary (attempt {retry_count}/{max_retries}): {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    
                    if retry_count < max_retries:
                        wait_time = 1
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # If all retries fail, raise an exception to halt processing
                        raise Exception(f"Failed to generate summary after {max_retries} attempts. Halting run.")
                        
                elif summary_method == "last_k":
                    print(f"Summarizing using last {self.config['last_k_tokens']} tokens")
                    assert "last_k_tokens" in self.config, "last_k_tokens must be specified in configuration"
                    last_k_tokens = self.config["last_k_tokens"]

                    if last_k_tokens < len(reasoning_trace):
                        hf_model_name = get_hf_model_name(self.config["summarizer_model"])
                        
                        tokenizer = load_tokenizer(hf_model_name)
                        reasoning_tokens = tokenizer.encode(reasoning_trace)
                        
                        # get last k tokens, then convert back to text
                        last_k_tokens_slice = reasoning_tokens[-last_k_tokens:]
                        summary = tokenizer.decode(last_k_tokens_slice)
                        summary = "..." + summary
                    else:
                        print(f"k ({last_k_tokens}) is greater than or equal to reasoning trace length ({len(reasoning_trace)}). Using entire reasoning trace.")
                        summary = reasoning_trace

                    if self.verbose:           
                        print(f"Reasoning tokens: {reasoning_tokens}")
                        print(f"Total tokens: {len(reasoning_tokens)}")
                        print(f"Last {last_k_tokens} tokens summary: {summary}")
                elif summary_method == "first_k":
                    print(f"Summarizing using first {self.config['first_k_tokens']} tokens")
                    assert "first_k_tokens" in self.config, "first_k_tokens must be specified in configuration"
                    first_k_tokens = self.config["first_k_tokens"]

                    if first_k_tokens < len(reasoning_trace):
                        hf_model_name = get_hf_model_name(self.config["summarizer_model"])
                        
                        tokenizer = load_tokenizer(hf_model_name)
                        reasoning_tokens = tokenizer.encode(reasoning_trace)
                        first_k_tokens_slice = reasoning_tokens[:first_k_tokens]
                        summary = tokenizer.decode(first_k_tokens_slice)
                        summary = summary + "..."

                        if self.verbose:
                            print(f"Reasoning tokens: {reasoning_tokens}")
                            print(f"Total tokens: {len(reasoning_tokens)}")
                        print(f"First {first_k_tokens} tokens summary: {summary}")
                    else:
                        print(f"k ({first_k_tokens}) is greater than or equal to reasoning trace length ({len(reasoning_trace)}). Using entire reasoning trace.")
                        summary = reasoning_trace
                elif summary_method == "random_k":
                    print(f"Summarizing using random {self.config['random_k_tokens']} tokens")
                    assert "random_k_tokens" in self.config, "random_k_tokens must be specified in configuration"
                    random_k_tokens = self.config["random_k_tokens"]

                    if random_k_tokens < len(reasoning_trace):
                        hf_model_name = get_hf_model_name(self.config["summarizer_model"])
                        tokenizer = load_tokenizer(hf_model_name)
                        reasoning_tokens = tokenizer.encode(reasoning_trace)
                        selected_indices = sorted(random.sample(range(len(reasoning_tokens)), random_k_tokens))
                        random_k_tokens_slice = [reasoning_tokens[i] for i in selected_indices]
                        
                        summary = tokenizer.decode(random_k_tokens_slice)
                    else:
                        print(f"k ({random_k_tokens}) is greater than or equal to reasoning trace length ({len(reasoning_trace)}). Using entire reasoning trace.")
                        summary = reasoning_trace
                elif summary_method == "post_think":
                    print(f"Summarizing using post-think summary")
                    assert "last_k_tokens" in self.config, "last_k_tokens must be specified in configuration"
                    last_k_tokens = self.config["last_k_tokens"]
                    summary = extract_post_think_content(current_reasoning)
                    if summary is None:
                        # use last k tokens
                        hf_model_name = get_hf_model_name(self.config["summarizer_model"])
                        tokenizer = load_tokenizer(hf_model_name)
                        reasoning_tokens = tokenizer.encode(reasoning_trace)
                        summary = tokenizer.decode(reasoning_tokens[-last_k_tokens:])
                        summary = "..." + summary
                    
            #     elif summary_method == "multiple_llm_prompts":
            #         print(f"Summarizing using multiple LLM prompts with chunk size: {self.config.get('multiple_prompt_chunk_words', 200)} words")
            #         assert "multiple_prompt_chunk_words" in self.config, "multiple_prompt_chunk_words must be specified in configuration"
            #         chunk_size = self.config.get("multiple_prompt_chunk_words")
            #         assert "multiple_prompt_overlap_words" in self.config, "multiple_prompt_overlap_words must be specified in configuration"
            #         overlap_size = self.config.get("multiple_prompt_overlap_words")
            #         assert "summarize_prompt_template" in self.config, "summarize_prompt_template must be specified in configuration"
            #         summarize_chunk_template = self.config.get("summarize_prompt_template")

            #         # Split the reasoning_trace into chunks
            #         words = reasoning_trace.split()
            #         chunks = []
            #         i = 0
            #         while i < len(words):
            #             chunk_text = " ".join(words[i : i + chunk_size])
            #             chunks.append(chunk_text)
            #             i += chunk_size - overlap_size # Move with overlap

            #         async def summarize_chunk(chunk_idx: int, chunk_content: str):
            #             retry_count = 0
            #             max_retries = 3

            #             while retry_count < max_retries:
            #                 assert "{reasoning}" in summarize_chunk_template, "summarize_chunk_template must contain {reasoning}"
            #                 assert "{question}" in summarize_chunk_template, "summarize_chunk_template must contain {question}"
            #                 try:
            #                     chunk_prompt = summarize_chunk_template.replace("{reasoning}", chunk_content).replace("{question}", question)

            #                     chunk_response = await self.summarizer.generate_response_async(
            #                         chunk_prompt,
            #                         max_tokens=self.config.get("summary_max_tokens", self.config["max_tokens"]),
            #                         temperature=self.config.get("summary_temperature", self.config["temperature"]),
            #                         top_p=self.config.get("summary_top_p", self.config["top_p"]),
            #                         top_k=self.config.get("summary_top_k", self.config["top_k"]) if hasattr(self.summarizer, "top_k") else None,
            #                         presence_penalty=self.config.get("summary_presence_penalty", self.config["presence_penalty"]),
            #                         frequency_penalty=self.config.get("summary_frequency_penalty", self.config["frequency_penalty"]),
            #                         verbose=self.verbose,
            #                         track_token_callback=self.track_token_usage_and_cost,
            #                         track_token_callback_args={
            #                             "problem_id": problem_id,
            #                             "iteration": current_iteration,
            #                             "step": f"summary_chunk_{chunk_idx+1}"
            #                         }
            #                     )
            #                     print(f"Chunk {chunk_idx+1} prompt: {chunk_prompt}")
            #                     print(f"Chunk {chunk_idx+1} response: {chunk_response}")
            #                     print("="*100)

            #                     try:
            #                         c_summary, c_finish_reason, c_token_usage, c_cost_info, c_detailed_api_calls = chunk_response
            #                     except ValueError:
            #                         logger.warning(f"Chunk {chunk_idx+1} summary response doesn't include detailed metrics, falling back to old format")
            #                         c_summary, c_finish_reason, c_token_usage, c_cost_info = chunk_response
            #                         c_detailed_api_calls = []
                                
            #                     self.track_token_usage_and_cost(
            #                         problem_id,
            #                         c_token_usage,
            #                         c_cost_info,
            #                         current_iteration,
            #                         f"summary_chunk_{chunk_idx+1}"
            #                     )
            #                     return c_summary, c_finish_reason, c_token_usage, c_cost_info, c_detailed_api_calls

            #                 except Exception as e_chunk:
            #                     error_str_chunk = str(e_chunk).lower()
            #                     if "prompt is too long" in error_str_chunk or "maximum context length" in error_str_chunk:
            #                         retry_count += 1
            #                         logger.error(f"Prompt too long for summary chunk {chunk_idx+1}. Consider reducing 'multiple_prompt_chunk_size'.")
            #                         if retry_count >= max_retries:
            #                             raise Exception(f"Prompt too long for summary chunk {chunk_idx+1} after {max_retries} attempts.")
            #                     else:
            #                         retry_count += 1
            #                         logger.error(f"Error generating summary for chunk {chunk_idx+1} (attempt {retry_count}/{max_retries}): {str(e_chunk)}")
            #                         logger.error(f"Stack trace: {traceback.format_exc()}")
            #                         if retry_count >= max_retries:
            #                             raise Exception(f"Failed to generate summary for chunk {chunk_idx+1} after {max_retries} attempts.")
                                
            #                     if retry_count < max_retries:
            #                         wait_time = 1
            #                         logger.info(f"Retrying chunk {chunk_idx+1} in {wait_time} seconds...")
            #                         await asyncio.sleep(wait_time)
                
            #     tasks = [summarize_chunk(idx, chunk_content) for idx, chunk_content in enumerate(chunks)]
                
            #     try:
            #         # Run all chunk summarizations in parallel
            #         parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            #     except Exception as gather_exc:
            #         logger.error(f"An error occurred during asyncio.gather for chunk summarization: {gather_exc}")
            #         logger.error(f"Stack trace: {traceback.format_exc()}")
            #         raise Exception(f"Failed to process chunks in parallel: {gather_exc}")


            #     chunk_summaries = []
            #     total_token_usage = TokenUsage(0,0,0) # Initialize with TokenUsage object
            #     total_cost_info = CostInfo(0.0,0.0,0.0) # Initialize with CostInfo object
            #     all_detailed_api_calls = []

            #     for idx, res in enumerate(parallel_results):
            #         if isinstance(res, Exception):
            #             # Handle exceptions from individual tasks if necessary, or re-raise
            #             logger.error(f"Error processing chunk {idx+1}: {res}")
            #             # Potentially append a placeholder or skip this chunk's result
            #             # For now, we'll re-raise to halt if any chunk fails criticaly
            #             raise Exception(f"Failed to process chunk {idx+1}: {res}")
                    
            #         c_summary, c_finish_reason, c_token_usage, c_cost_info, c_detailed_api_calls = res
            #         chunk_summaries.append(c_summary)
                    
            #         # Accumulate token usage and cost info
            #         if c_token_usage: # Ensure it's not None
            #             total_token_usage.prompt_tokens += c_token_usage.prompt_tokens
            #             total_token_usage.completion_tokens += c_token_usage.completion_tokens
            #             total_token_usage.total_tokens += c_token_usage.total_tokens
            #         if c_cost_info: # Ensure it's not None
            #             total_cost_info.prompt_cost += c_cost_info.prompt_cost
            #             total_cost_info.completion_cost += c_cost_info.completion_cost
            #             total_cost_info.total_cost += c_cost_info.total_cost
                    
            #         if c_detailed_api_calls: # Ensure it's not None or empty list
            #             all_detailed_api_calls.extend(c_detailed_api_calls)

            #     # Combine the chunk summaries
            #     summary = "\n".join(chunk_summaries) # Use escaped newline for string literal
            #     summary_finish_reason = "multiple_chunks_parallel"
            #     token_usage = total_token_usage # This is now an object
            #     cost_info = total_cost_info # This is now an object
            #     # summary_detailed_api_calls is already assigned as all_detailed_api_calls above

            #     # Store detailed API call information in the result
            #     result["detailed_metrics"][f"iteration_{current_iteration}_summary"] = all_detailed_api_calls
            # elif summary_method == "answer_only":
            #     # Get the answer from the latest reasoning iteration
            #     latest_iter = result["iterations"][current_iteration]
            #     latest_answer = latest_iter["answer"]
                
            #     # Check if summary is still None after all attempts
            #     if summary is None:
            #         raise ValueError(f"Failed to generate summary for problem {problem_id}, iteration {current_iteration}")
                
            #     # Log the finish reason
            #     logger.info(f"Problem {problem_id}, summary {current_iteration} finish reason: {summary_finish_reason}")
                
            #     # Extract post-think content from summary if enabled
            #     # post_think_summary = summary
            #     # if self.config.get("extract_post_think_summary", False):
            #     #     extracted = extract_post_think_content(summary)
            #     #     if extracted is not None:
            #     #         post_think_summary = extracted
            #     #         logger.info(f"Extracted post-think content from summary for problem {problem_id}, iteration {current_iteration}")
            #     #     else:
            #     #         logger.info(f"No post-think content found in summary for problem {problem_id}, iteration {current_iteration}. Using full summary.")
                
            #     # Add summary to the collection with iteration number
            #     all_summaries.append({
            #         "iteration": current_iteration,
            #         "summary": summary,
            #         # "post_think_summary": post_think_summary,
            #         "finish_reason": summary_finish_reason
            #     })
                
            #     # Update the current iteration with the summary before moving to the next iteration
            #     result["iterations"][current_iteration]["summary"] = summary
            #     result["iterations"][current_iteration]["post_think_summary"] = post_think_summary
            #     result["iterations"][current_iteration]["summary_finish_reason"] = summary_finish_reason
            
            # Prepare for next iteration
            next_iteration = current_iteration + 1
            
            # Get improved reasoning prompt template
            improved_template = self.config.get("improved_prompt_template")
            if not improved_template:
                raise ValueError("improved_prompt_template must be specified for additional iterations")
            
            # Build accumulated summaries text
            accumulated_summaries = ""
            for i, summary_item in enumerate(all_summaries):
                # Use post_think_summary if available and extract_post_think_summary is enabled
                summary_text = summary_item.get("post_think_summary", summary_item["summary"]) if self.config.get("extract_post_think_summary", False) else summary_item["summary"]
                accumulated_summaries += f"\n\nATTEMPT {summary_item['iteration']} SUMMARY:\n{summary_text}"
            
            # Create prompt for next iteration using accumulated summaries
            improved_prompt = improved_template.replace("{question}", question).replace("{summaries}", accumulated_summaries)
            
            # Check if the next iteration is available in the checkpoint
            next_reasoning_from_checkpoint = None
            if hasattr(self, 'checkpoint_map') and problem_id in self.checkpoint_map:
                # Find the next iteration in the checkpoint data
                next_iter_data = next(
                    (data for data in result["iterations"] if data["iteration"] == next_iteration),
                    None
                )
                if next_iter_data:
                    next_reasoning_from_checkpoint = next_iter_data["reasoning"]
                    next_finish_reason = next_iter_data.get("finish_reason", "unknown")
                    next_answer = next_iter_data.get("answer")
                    next_correct = next_iter_data.get("correct", False)
                    
                    logger.info(f"Using checkpoint reasoning for problem {problem_id}, iteration {next_iteration}")
                    
                    # Use dummy token usage for tracking purposes
                    token_usage = TokenUsage(0, 0, 0)
                    cost_info = CostInfo(0.0, 0.0, 0.0)
                    
                    # Create placeholder for detailed API calls
                    detailed_api_calls = [{
                        "reused": True, 
                        "tokens": {
                            "prompt_tokens": token_usage.prompt_tokens,
                            "completion_tokens": token_usage.completion_tokens,
                            "total_tokens": token_usage.total_tokens
                        }, 
                        "cost": {
                            "prompt_cost": cost_info.prompt_cost,
                            "completion_cost": cost_info.completion_cost,
                            "total_cost": cost_info.total_cost
                        }
                    }]
                    
                    # Store the detailed API calls for this iteration
                    result["detailed_metrics"][f"iteration_{next_iteration}_reasoning"] = detailed_api_calls
            
            # Generate reasoning for next iteration if not in checkpoint
            if next_reasoning_from_checkpoint is None:
                # Generate reasoning for next iteration
                next_finish_reason = "unknown"
                
                # Use the async model interface for the improved reasoning
                next_response = await self.reasoning_model.generate_response_async(
                    improved_prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    enable_continuation=self.config.get("enable_continuation", True),
                    max_total_tokens=self.config.get("max_total_tokens", 24576),
                    max_continuations=self.config.get("max_continuations", 3),
                    verbose=self.verbose,
                    track_token_callback=self.track_token_usage_and_cost,
                    track_token_callback_args={
                        "problem_id": problem_id,
                        "iteration": next_iteration,
                        "step": "reasoning"
                    }
                )
                
                # Unpack the tuple (content, finish_reason, token_usage, cost_info, detailed_api_calls)
                next_reasoning, next_finish_reason, token_usage, cost_info, detailed_api_calls = next_response
                
                # Track token usage and cost
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, next_iteration, "reasoning")
                
                # Store the detailed API calls for this iteration
                result["detailed_metrics"][f"iteration_{next_iteration}_reasoning"] = detailed_api_calls
                
                # Log the finish reason
                logger.info(f"Problem {problem_id}, iteration {next_iteration} finish reason: {next_finish_reason}")
                
                # Extract answer from next iteration reasoning using the configured extractor
                next_answer = extract_answer_with_config(next_reasoning, self.config)
                
                # Check if answer is correct
                next_correct = False
                if next_answer is not None:
                    next_correct = next_answer.strip() == correct_answer.strip()
            else:
                # Use the reasoning from checkpoint
                next_reasoning = next_reasoning_from_checkpoint
            
            # Update for tracking correct answers
            found_correct_answer = found_correct_answer or next_correct
            
            # Add the next iteration to the results - WITHOUT including a summary yet
            # The summary will be added when it's generated in the next loop iteration
            if next_iteration >= len(result["iterations"]):
                result["iterations"].append({
                    "iteration": next_iteration,
                    "reasoning": next_reasoning,
                    "answer": next_answer,
                    "correct": next_correct,
                    "finish_reason": next_finish_reason
                })
            
            
            # Update for next potential iteration
            current_iteration = next_iteration
            current_reasoning = next_reasoning
        
        # Ensure final_answer and final_correct are always set in the results
        if "final_answer" not in result:
            # If we didn't set a final answer in the last iteration (due to missing summarization or other issues),
            # use the answer from the last iteration
            last_iteration_idx = len(result["iterations"]) - 1
            last_iteration = result["iterations"][last_iteration_idx]
            result["final_answer"] = last_iteration["answer"]
            result["final_correct"] = last_iteration["correct"]
            logger.info(f"Setting final answer from last iteration for problem {problem_id}")
        
        # Log the final answer and whether it was correct
        logger.info(f"Final answer for problem {problem_id}: {result['final_answer']}")
        logger.info(f"Final answer correct: {result['final_correct']}")
        
        return result
    
    def _process_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the summarization pipeline.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with reasoning iterations
        """
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Initialize current problem result to track partial progress
        self._current_problem_result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "iterations": []
        }
        
        # Check if we have preloaded initial reasoning for this problem
        if hasattr(self, 'initial_reasoning_map') and problem_id in self.initial_reasoning_map:
            # Use preloaded initial reasoning
            initial_data = self.initial_reasoning_map[problem_id]
            
            # Ensure we have the correct question and answer from the problem data
            if question != initial_data.get("question"):
                logger.warning(f"Question mismatch for problem {problem_id}. Using current question.")
            
            if correct_answer != initial_data.get("correct_answer"):
                logger.warning(f"Answer mismatch for problem {problem_id}. Using current answer.")
            
            # Use the reasoning, answer, and finish reason from the preloaded data
            iter0_reasoning = initial_data["reasoning"]
            iter0_answer = initial_data["answer"]
            iter0_correct = initial_data["correct"]
            iter0_finish_reason = initial_data["finish_reason"]
            
            # Set dummy token usage for tracking purposes
            token_usage = TokenUsage(0, 0, 0)
            cost_info = CostInfo(0.0, 0.0, 0.0)  # prompt_cost, completion_cost, total_cost
            
            logger.info(f"Using preloaded initial reasoning for problem {problem_id}")
            
            # Create placeholder for detailed API calls with serializable dictionaries instead of objects
            detailed_api_calls = [{
                "reused": True, 
                "tokens": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }, 
                "cost": {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                }
            }]
        else:
            # Create initial reasoning prompt (iteration 0)
            reasoning_template = self.config.get("reasoning_prompt_template")
            if not reasoning_template:
                raise ValueError("reasoning_prompt_template must be specified in configuration")
            
            initial_prompt = reasoning_template.replace("{question}", question)
            
            # Generate iteration 0 reasoning with streaming
            if self.dashboard:
                # Use streaming for dashboard updates
                iter0_reasoning, iter0_finish_reason, token_usage, cost_info = self._stream_model_output(problem_id, initial_prompt, iteration=0)
            else:
                # Without dashboard, just get the full response
                response = self.reasoning_model.generate_response(
                    initial_prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                )
                
                # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                iter0_reasoning, iter0_finish_reason, token_usage, cost_info = response
                
                # Track token usage and cost
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, 0, "reasoning")
            
            # Log the finish reason
            logger.info(f"Problem {problem_id}, iteration 0 finish reason: {iter0_finish_reason}")
            
            # Extract answer from iteration 0 reasoning using the configured extractor
            iter0_answer = extract_answer_with_config(iter0_reasoning, self.config)
            
            # Check if answer is correct (simple string comparison)
            iter0_correct = False
            if iter0_answer is not None:
                iter0_correct = iter0_answer.strip() == correct_answer.strip()
        
        # Construct initial result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "iterations": [
                {
                    "iteration": 0,
                    "reasoning": iter0_reasoning,
                    "answer": iter0_answer,
                    "correct": iter0_correct,
                    "finish_reason": iter0_finish_reason
                }
            ],
            "timestamp": time.time()
        }
        
        # NOTE: The following fields are duplicates of data already in iterations[0]
        # They are kept for backward compatibility with existing dashboard code
        # TODO: For a future refactor, remove these redundant fields and update dashboard
        # to use iterations[0] directly
        result["initial_reasoning"] = iter0_reasoning
        result["initial_answer"] = iter0_answer
        result["initial_correct"] = iter0_correct
        result["initial_finish_reason"] = iter0_finish_reason
        
        # Update dashboard with answer information
        if self.dashboard:
            self.dashboard.update_problem_status(
                problem_id, 
                "correct" if iter0_correct else "incorrect"
            )
            
            # Send answer information to the dashboard
            self.dashboard.update_answer_info(
                problem_id,
                iter0_answer or "No answer extracted",
                correct_answer,
                iter0_correct,
                iteration=0
            )
        
        # Maximum number of iterations to perform
        max_iterations = self.config.get("max_iterations", 1)
        
        # Track if we've found the correct answer in any iteration
        found_correct_answer = iter0_correct
        
        # Store all summaries to accumulate them across iterations
        all_summaries = []
        
        # Perform additional iterations if enabled and we haven't found a correct answer yet
        current_iteration = 0
        current_reasoning = iter0_reasoning
        
        while (
            current_iteration < max_iterations and 
            self.config.get("enable_summarization", True) and
            (not found_correct_answer or self.config.get("continue_after_correct", False))
        ):
            # Get the summarization prompt template
            summarize_template = self.config.get("summarize_prompt_template")
            if not summarize_template:
                raise ValueError("summarize_prompt_template must be specified in configuration")
            
            # Generate summary of the current reasoning
            logger.info(f"Generating summary for problem {problem_id}, iteration {current_iteration}")
            # Ensure top_k is included in the config for FireworksModelClient
            if not hasattr(self.summarizer, "top_k"):
                assert "top_k" in self.config, "top_k must be specified in config if using FireworksModelClient"
                
            # Stream the summary if we have a dashboard, otherwise generate normally
            summary_finish_reason = "unknown"
            if self.dashboard:
                summary, summary_finish_reason, token_usage, cost_info = self._stream_summary_generation(
                    problem_id, 
                    question,  # Pass the question directly
                    current_reasoning, 
                    summarize_template, 
                    iteration=current_iteration
                )
                # The summary has already been streamed to the dashboard, 
                # but we need to send a final update to indicate it's complete
                self.dashboard.update_summary(problem_id, summary, iteration=current_iteration, finish_reason=summary_finish_reason)
            else:
                # Extract reasoning trace from the full reasoning
                reasoning_trace = extract_reasoning_trace(
                    current_reasoning, 
                    allow_fallback=self.config.get("allow_fallback", False)
                )
                # If extraction failed, raise an error
                if reasoning_trace is None:
                    raise ValueError(f"Could not extract reasoning trace for problem {problem_id}. Make sure the model output contains <think> tags.")
                
                summary_response = summarize_reasoning(
                    question,
                    reasoning_trace,  # Use extracted reasoning trace instead of full reasoning
                    self.summarizer,
                    summarize_template,
                    max_tokens=self.config.get("summary_max_tokens"),
                    temperature=self.config.get("summary_temperature"),
                    top_p=self.config.get("summary_top_p"),
                    top_k=self.config.get("summary_top_k"),
                    presence_penalty=self.config.get("summary_presence_penalty"),
                    frequency_penalty=self.config.get("summary_frequency_penalty"),
                    verbose=self.verbose
                )
                
                # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                summary, summary_finish_reason, token_usage, cost_info = summary_response
                
                # Track token usage and cost for the summary
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, current_iteration, "summary")
            
            # Log the finish reason for the summary
            logger.info(f"Problem {problem_id}, summary {current_iteration} finish reason: {summary_finish_reason}")
            
            # Extract post-think content from summary if enabled
            post_think_summary = summary
            if self.config.get("extract_post_think_summary", False):
                extracted = extract_post_think_content(summary)
                if extracted is not None:
                    post_think_summary = extracted
                    logger.info(f"Extracted post-think content from summary for problem {problem_id}, iteration {current_iteration}")
                else:
                    logger.info(f"No post-think content found in summary for problem {problem_id}, iteration {current_iteration}. Using full summary.")
            
            # Add summary to the collection with iteration number
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "post_think_summary": post_think_summary,
                "finish_reason": summary_finish_reason
            })
            
            # *** FIX: Update the current iteration with the summary before moving to the next iteration ***
            # This way the summary is associated with the reasoning it summarized
            result["iterations"][current_iteration]["summary"] = summary
            result["iterations"][current_iteration]["post_think_summary"] = post_think_summary
            result["iterations"][current_iteration]["summary_finish_reason"] = summary_finish_reason
            
            # Prepare for next iteration
            next_iteration = current_iteration + 1
            
            # Get improved reasoning prompt template
            improved_template = self.config.get("improved_prompt_template")
            if not improved_template:
                raise ValueError("improved_prompt_template must be specified for additional iterations")
            
            # Build accumulated summaries text
            accumulated_summaries = ""
            for i, summary_item in enumerate(all_summaries):
                # Use post_think_summary if available and extract_post_think_summary is enabled
                summary_text = summary_item.get("post_think_summary", summary_item["summary"]) if self.config.get("extract_post_think_summary", False) else summary_item["summary"]
                accumulated_summaries += f"\n\nATTEMPT {summary_item['iteration']} SUMMARY:\n{summary_text}"
            
            # Create prompt for next iteration using accumulated summaries
            improved_prompt = improved_template.replace("{question}", question).replace("{summaries}", accumulated_summaries)
            
            # Generate reasoning for next iteration
            next_finish_reason = "unknown"
            if self.dashboard:
                # Use streaming for dashboard updates
                next_reasoning, next_finish_reason, token_usage, cost_info = self._stream_model_output(problem_id, improved_prompt, iteration=next_iteration)
            else:
                # Without dashboard, just get the full response
                next_response = self.reasoning_model.generate_response(
                    improved_prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                )
                
                # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                next_reasoning, next_finish_reason, token_usage, cost_info = next_response
                
                # Track token usage and cost
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, next_iteration, "reasoning")
            
            # Log the finish reason
            logger.info(f"Problem {problem_id}, iteration {next_iteration} finish reason: {next_finish_reason}")
            
            # Extract answer from next iteration reasoning using the configured extractor
            next_answer = extract_answer_with_config(next_reasoning, self.config)
            
            # Check if answer is correct
            next_correct = False
            if next_answer is not None:
                next_correct = next_answer.strip() == correct_answer.strip()
                found_correct_answer = found_correct_answer or next_correct
            
            # Add the next iteration to the results - WITHOUT including a summary yet
            # The summary will be added when it's generated in the next loop iteration
            result["iterations"].append({
                "iteration": next_iteration,
                "reasoning": next_reasoning,
                "answer": next_answer,
                "correct": next_correct,
                "finish_reason": next_finish_reason
            })
            
            # NOTE: These are redundant fields duplicating data already in iterations[1]
            # They are kept for backward compatibility with existing dashboard code
            # TODO: For a future refactor, remove these redundant fields and update dashboard
            # to use iterations[1] directly
            result["summary"] = summary  # This is still needed for backwards compatibility
            result["summary_finish_reason"] = summary_finish_reason
            result["improved_reasoning"] = next_reasoning
            result["improved_answer"] = next_answer
            result["improved_correct"] = next_correct
            result["improved_finish_reason"] = next_finish_reason
            
            # Update for next potential iteration
            current_iteration = next_iteration
            current_reasoning = next_reasoning
        
        # Update the problem status based on iteration
        if self.dashboard:
            status = f"iter{current_iteration}-completed"
            self.dashboard.update_problem_status(problem_id, status, question)
        
        # Ensure final_answer and final_correct are always set in the results
        if "final_answer" not in result:
            # If we didn't set a final answer in the last iteration (due to missing summarization or other issues),
            # use the answer from the last iteration
            last_iteration_idx = len(result["iterations"]) - 1
            last_iteration = result["iterations"][last_iteration_idx]
            result["final_answer"] = last_iteration["answer"]
            result["final_correct"] = last_iteration["correct"]
            logger.info(f"Setting final answer from last iteration for problem {problem_id}")
        
        # Log the final answer and whether it was correct
        logger.info(f"Final answer for problem {problem_id}: {result['final_answer']}")
        logger.info(f"Final answer correct: {result['final_correct']}")
        
        return result
    
    def load_initial_reasoning(self, source_results_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load initial reasoning from a previous experiment's results file.
        
        Args:
            source_results_path: Path to the source results file
            
        Returns:
            Dictionary mapping problem_id to initial reasoning data
        """
        if not os.path.exists(source_results_path):
            raise FileNotFoundError(f"Source results file not found: {source_results_path}")
        
        with open(source_results_path, "r", encoding="utf-8") as f:
            source_results = json.load(f)
        
        # Create a dictionary mapping problem_id to initial reasoning data
        initial_reasoning_map = {}
        
        # Check if source_results is a list or a dictionary with a 'results' key
        if isinstance(source_results, dict) and 'results' in source_results:
            source_results = source_results['results']
        
        # Ensure source_results is a list
        if not isinstance(source_results, list):
            raise ValueError(f"Unexpected format in {source_results_path}. Expected a list of results or a dictionary with a 'results' key.")
        
        for result in source_results:
            # Skip if result is not a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dictionary result: {result}")
                continue
                
            problem_id = result.get("problem_id")
            if problem_id and "iterations" in result and len(result["iterations"]) > 0:
                iter0 = result["iterations"][0]
                initial_reasoning_map[problem_id] = {
                    "reasoning": iter0.get("reasoning"),
                    "answer": iter0.get("answer"),
                    "correct": iter0.get("correct"),
                    "finish_reason": iter0.get("finish_reason"),
                    "question": result.get("question"),
                    "correct_answer": result.get("correct_answer")
                }
                logger.info(f"Loaded initial reasoning for problem {problem_id}")
            
        logger.info(f"Loaded initial reasoning for {len(initial_reasoning_map)} problems from {source_results_path}")
        return initial_reasoning_map

    def initialize_with_previous_results(self, source_results_path: str) -> None:
        """
        Initialize the experiment with initial reasoning from a previous run.
        
        Args:
            source_results_path: Path to the source results file
        """
        self.initial_reasoning_map = self.load_initial_reasoning(source_results_path)
        logger.info(f"Initialized experiment with {len(self.initial_reasoning_map)} preloaded reasonings")
        
        # Flag to indicate we're reusing initial reasoning
        self.reusing_initial_reasoning = True
        
        # We'll need to override metrics too
        self.preloaded_metrics = {"token_usage": {}, "cost_info": {}}

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load full checkpoint data from a previous experiment's results file.
        
        Args:
            checkpoint_path: Path to the checkpoint results file
            
        Returns:
            Dictionary mapping problem_id to complete checkpoint data
        """
        # Load the results file
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_results = json.load(f)
        
        # Handle different result formats (list or dict with 'results' key)
        if isinstance(checkpoint_results, dict) and 'results' in checkpoint_results:
            checkpoint_results = checkpoint_results['results']
        
        # Create a dictionary mapping problem_id to complete problem data
        checkpoint_map = {}
        
        for result in checkpoint_results:
            if not isinstance(result, dict):
                continue
            
            problem_id = result.get("problem_id")
            if problem_id and "iterations" in result:
                # Store the entire result object with all iterations
                checkpoint_map[problem_id] = result
                logger.info(f"Loaded checkpoint with {len(result['iterations'])} iterations for problem {problem_id}")
        
        return checkpoint_map
        
    def initialize_with_checkpoint(self, checkpoint_path: str) -> None:
        """
        Initialize the experiment with full checkpoint data from a previous run.
        
        Args:
            checkpoint_path: Path to the checkpoint results file
        """
        self.checkpoint_map = self.load_checkpoint(checkpoint_path)
        logger.info(f"Initialized experiment with checkpoints for {len(self.checkpoint_map)} problems")
        
        # Flag to indicate we're using checkpoints
        self.using_checkpoints = True
        
        # We'll need to override metrics for the preloaded iterations
        self.preloaded_metrics = {"token_usage": {}, "cost_info": {}}