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

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_reasoning_trace, extract_answer_with_config, extract_post_think_content
from src.reasoning.summarizer import summarize_reasoning, summarize_reasoning_async
from src.dashboard.server import DashboardServer
from src.eval.latex_answer_check import get_gt_answer, check_one_latex_answer
from src.eval.countdown_check import check_one_countdown_answer

logger = logging.getLogger(__name__)

def get_answer_checker(config: Dict[str, Any]):
    """Get the appropriate answer checker based on configuration."""
    answer_extractor = config.get("answer_extractor", "default")
    
    if answer_extractor == "countdown":
        def checker(model_answer, gt_answer, **kwargs):
            # For countdown, gt_answer should be a dict with 'target' and 'nums'
            if isinstance(gt_answer, dict) and 'target' in gt_answer and 'nums' in gt_answer:
                return check_one_countdown_answer(
                    model_answer,
                    gt_answer['nums'],
                    gt_answer['target'],
                    debug=kwargs.get('debug', False)
                )
            else:
                # Fallback if gt_answer is not in expected format
                logger.warning("Countdown checker called with non-dict gt_answer, falling back to latex checker")
                return check_one_latex_answer(model_answer, gt_answer, **kwargs)
    else:
        # Default to LaTeX answer checker
        def checker(model_answer, gt_answer, **kwargs):
            return check_one_latex_answer(model_answer, gt_answer, **kwargs)
    
    return checker

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
        vllm_config = self.config.get("vllm_config", None)
        self.reasoning_model = create_model_client(
            self.config["reasoning_model"],
            provider=reasoning_provider,
            vllm_config=vllm_config
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
            
            # Use separate vLLM config for summarizer if available and provider is vllm
            if summarizer_provider == "vllm":
                summarizer_vllm_config = self.config.get("summarizer_vllm_config", self.config.get("vllm_config", None))
            else:
                summarizer_vllm_config = self.config.get("vllm_config", None)
                
            self.summarizer = create_model_client(
                self.config["summarizer_model"],
                provider=summarizer_provider,
                vllm_config=summarizer_vllm_config
            )
        
        # Add lock for thread safety when updating results
        self.results_lock = Lock()
        
        # Initialize preloaded metrics
        self.preloaded_metrics = {"token_usage": {}, "cost_info": {}}
    
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
    
    async def _generate_final_summary(self, result: Dict[str, Any], problem_id: str, question: str, correct_answer: str) -> bool:
        """Generate a summary for the final reasoning iteration and update the result.
        
        Args:
            result: The result dictionary to update
            problem_id: The ID of the problem
            question: The problem question
            correct_answer: The correct answer to compare against
            
        Returns:
            bool: True if a final summary was generated and had a valid answer, False otherwise
        """
        try:
            # Get the last iteration
            last_iteration_idx = len(result["iterations"]) - 1
            last_iteration = result["iterations"][last_iteration_idx]
            last_reasoning = last_iteration["reasoning"]
            
            logger.info(f"Generating final summary for problem {problem_id}, last iteration {last_iteration_idx}")
            
            # Process ground truth answer using get_gt_answer
            extract_policy = self.config.get("extract_policy", "flex")
            eval_policy = self.config.get("eval_policy", "aggressive")
            gt_answer = result.get("processed_gt_answer", get_gt_answer(correct_answer, extract_policy=extract_policy))
            
            # Extract reasoning trace if needed
            reasoning_trace = extract_reasoning_trace(last_reasoning, self.config) \
                if self.config.get("extract_reasoning_trace", False) else last_reasoning
            
            # Get summarization prompt template
            summarize_template = self.config.get("summarize_prompt_template")
            if not summarize_template:
                raise ValueError("summarize_prompt_template must be specified in configuration")
            
            # Format the summarization prompt
            summarize_prompt = summarize_template.replace("{reasoning}", reasoning_trace).replace("{question}", question)
            
            # Generate final summary
            final_summary_response = await summarize_reasoning_async(
                self.summarizer,
                summarize_prompt,
                question,
                config=self.config,
                verbose=self.verbose,
                enable_continuation=self.config.get("summary_enable_continuation", self.config.get("enable_continuation", True)),
                max_total_tokens=self.config.get("summary_max_total_tokens", self.config.get("max_total_tokens", 24576)),
                max_continuations=self.config.get("summary_max_continuations", self.config.get("max_continuations", 3))
            )
            
            # Process the response with detailed metrics if available
            try:
                # Try to unpack with 5 elements (new format with detailed metrics)
                final_summary, final_summary_finish_reason, token_usage, cost_info, final_summary_api_calls = final_summary_response
                
                # Store detailed API call information
                result["detailed_metrics"][f"iteration_{last_iteration_idx}_final_summary"] = final_summary_api_calls
                
                # Track token usage and cost
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, last_iteration_idx, "final_summary")
            except ValueError:
                # Fall back to old format without detailed metrics
                final_summary, final_summary_finish_reason, token_usage, cost_info = final_summary_response
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, last_iteration_idx, "final_summary")
            
            # Extract answer from the final summary
            final_summary_answer = extract_answer_with_config(final_summary, self.config)
            
            # Check if the summary answer is correct using the LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            final_summary_check_result = answer_checker(
                final_summary_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            final_summary_correct = final_summary_check_result["is_correct"]
            
            # Log the final summary information
            logger.info(f"Problem {problem_id}, final summary finish reason: {final_summary_finish_reason}")
            logger.info(f"Problem {problem_id}, final summary answer: {final_summary_answer}")
            logger.info(f"Problem {problem_id}, final summary correct: {final_summary_correct}")
            
            # Add the final summary to the result
            result["final_summary"] = final_summary
            result["final_summary_answer"] = final_summary_answer
            result["final_summary_correct"] = final_summary_correct
            
            # Add the final summary to the last iteration too
            last_iteration["final_summary"] = final_summary
            last_iteration["final_summary_answer"] = final_summary_answer
            last_iteration["final_summary_correct"] = final_summary_correct
            
            # Update final answer if final summary has a valid answer
            if final_summary_answer is not None:
                result["final_answer"] = final_summary_answer
                result["final_correct"] = final_summary_correct
                logger.info(f"Updated final answer from final summary for problem {problem_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error generating final summary for problem {problem_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    async def _process_problem_async(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the summarization pipeline asynchronously.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with reasoning iterations
        """
        import traceback
        
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Process ground truth answer based on answer type
        extract_policy = self.config.get("extract_policy", "flex")
        eval_policy = self.config.get("eval_policy", "aggressive")
        
        # Check if this is a countdown problem
        if self.config.get("answer_extractor") == "countdown":
            # For countdown, prepare the ground truth as a dict with target and nums
            if "nums" in problem:
                # Parse nums if it's a string
                import ast
                nums = problem["nums"]
                if isinstance(nums, str):
                    nums = ast.literal_eval(nums)
                gt_answer = {
                    "target": int(correct_answer),
                    "nums": nums
                }
            else:
                # Fallback to regular processing if nums not available
                logger.warning(f"Countdown problem {problem_id} missing 'nums' field, falling back to latex processing")
                gt_answer = get_gt_answer(correct_answer, extract_policy=extract_policy)
        else:
            # Use regular LaTeX answer processing
            gt_answer = get_gt_answer(correct_answer, extract_policy=extract_policy)
        
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
            
            # Check if answer is correct using the new LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            iter0_check_result = answer_checker(
                iter0_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            iter0_correct = iter0_check_result["is_correct"]
        
        # Construct initial result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "processed_gt_answer": gt_answer,
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
            
            # Extract reasoning trace from the full reasoning
            reasoning_trace = extract_reasoning_trace(
                current_reasoning, 
                allow_fallback=self.config.get("allow_fallback", False)
            )
            # If extraction failed, raise an error
            if reasoning_trace is None:
                raise ValueError(f"Could not extract reasoning trace for problem {problem_id}. Make sure the model output contains <think> tags.")
            
            # Generate the summary asynchronously
            # Initialize default values in case of exception
            summary = "Error generating summary"
            summary_finish_reason = "error"
            token_usage = None
            cost_info = None
            
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
                    verbose=self.verbose,
                    enable_continuation=enable_continuation,
                    max_total_tokens=summary_total_tokens,
                    max_continuations=summary_continuations,
                    track_token_callback=self.track_token_usage_and_cost,
                    track_token_callback_args={
                        "problem_id": problem_id,
                        "iteration": current_iteration,
                        "step": "summary"
                    }
                )
                
                # Handle the response
                try:
                    # Try to unpack with 5 elements (new format with detailed metrics)
                    summary, summary_finish_reason, token_usage, cost_info, summary_detailed_api_calls = summary_response
                    
                    # Store detailed API call information in the result
                    result["detailed_metrics"][f"iteration_{current_iteration}_summary"] = summary_detailed_api_calls
                except ValueError:
                    # Fall back to the old format with 4 elements if needed
                    logger.warning(f"Summary response doesn't include detailed metrics, falling back to old format")
                    summary, summary_finish_reason, token_usage, cost_info = summary_response
            except Exception as e:
                error_str = str(e).lower()
                
                if "prompt is too long" in error_str or "maximum context length" in error_str:
                    # Calculate truncation factor - each retry we reduce by an additional 20%
                    # truncation_factor = max(0.3, 1.0 - (retry_count * 0.2))
                    # logger.warning(f"Prompt too long error, will retry with truncation factor: {truncation_factor:.1%}")
                    raise Exception(f"Prompt too long error, please decrease summary_max_total_tokens")
                
                
                # Continue with default values set above
            
            # Track token usage and cost for the summary only if we have valid data
            if token_usage and cost_info:
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
            
            # Extract answer from the summary using the same extractor as reasoning
            summary_answer = extract_answer_with_config(summary, self.config)
            
            # Check if the summary answer is correct using the LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            summary_check_result = answer_checker(
                summary_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            summary_correct = summary_check_result["is_correct"]
            
            # Log the summary answer
            logger.info(f"Problem {problem_id}, iteration {current_iteration} summary answer: {summary_answer}")
            logger.info(f"Problem {problem_id}, iteration {current_iteration} summary correct: {summary_correct}")
            
            # Get the reasoning answer from the current iteration
            current_iter = result["iterations"][current_iteration]
            reasoning_answer = current_iter["answer"]
            reasoning_correct = current_iter["correct"]
            
            # Log for debugging
            logger.info(f"Retrieved reasoning answer for iteration {current_iteration}: {reasoning_answer}")
            
            # Use summary answer as primary, with fallback to reasoning answer if summary has no answer
            final_answer = summary_answer if summary_answer is not None else reasoning_answer
            final_correct = summary_correct if summary_answer is not None else reasoning_correct
            
            # Add summary to the collection with iteration number and extracted answer
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "post_think_summary": post_think_summary,
                "finish_reason": summary_finish_reason,
                "summary_answer": summary_answer,
                "summary_correct": summary_correct,
                "final_answer": final_answer,
                "final_correct": final_correct
            })
            
            # This way the summary is associated with the reasoning it summarized
            result["iterations"][current_iteration]["summary"] = summary
            result["iterations"][current_iteration]["post_think_summary"] = post_think_summary
            result["iterations"][current_iteration]["summary_finish_reason"] = summary_finish_reason
            result["iterations"][current_iteration]["summary_answer"] = summary_answer
            result["iterations"][current_iteration]["summary_correct"] = summary_correct
            result["iterations"][current_iteration]["final_answer"] = final_answer
            result["iterations"][current_iteration]["final_correct"] = final_correct
            
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
            
            # Generate reasoning for next iteration asynchronously with enhanced metrics tracking
            next_response = await self.reasoning_model.generate_response_async(
                improved_prompt,
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
                    "iteration": next_iteration,
                    "step": "reasoning"
                }
            )
            
            # Handle both tuple and string responses for backward compatibility
            if isinstance(next_response, tuple):
                try:
                    # Try to unpack with 5 elements (new format with detailed metrics)
                    next_reasoning, next_finish_reason, token_usage, cost_info, next_detailed_api_calls = next_response
                    
                    # Store detailed API call information in the result
                    result["detailed_metrics"][f"iteration_{next_iteration}_reasoning"] = next_detailed_api_calls
                    
                    # Track token usage and cost
                    self.track_token_usage_and_cost(problem_id, token_usage, cost_info, next_iteration, "reasoning")
                except ValueError:
                    # Fall back to the old format with 4 elements
                    logger.warning(f"Next reasoning response doesn't include detailed metrics, falling back to old format")
                    next_reasoning, next_finish_reason, token_usage, cost_info = next_response
                    
                    # Track token usage and cost
                    self.track_token_usage_and_cost(problem_id, token_usage, cost_info, next_iteration, "reasoning")
            else:
                next_reasoning = next_response
                next_finish_reason = "unknown"
            
            # Log the finish reason
            logger.info(f"Problem {problem_id}, iteration {next_iteration} finish reason: {next_finish_reason}")
            
            # Extract answer from next iteration reasoning using the configured extractor
            next_answer = extract_answer_with_config(next_reasoning, self.config)
            
            # Check if answer is correct using the LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            next_check_result = answer_checker(
                next_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            next_correct = next_check_result["is_correct"]
            
            # Update found_correct_answer flag
            found_correct_answer = found_correct_answer or next_correct
            
            # Add the next iteration to the results - WITHOUT including a summary yet
            # The summary will be added when it's generated in the next loop iteration
            result["iterations"].append({
                "iteration": next_iteration,
                "reasoning": next_reasoning,
                "answer": next_answer,  # Use consistent key name matching the first iteration
                "correct": next_correct,  # Use consistent key name matching the first iteration
                "finish_reason": next_finish_reason
            })
            
            # NOTE: These are redundant fields duplicating data already in iterations[1]
            # They are kept for backward compatibility with existing dashboard code
            result["summary"] = summary  # This is still needed for backwards compatibility
            result["summary_finish_reason"] = summary_finish_reason
            result["summary_answer"] = summary_answer
            result["summary_correct"] = summary_correct
            result["improved_reasoning"] = next_reasoning
            result["improved_answer"] = next_answer
            result["improved_correct"] = next_correct
            result["improved_finish_reason"] = next_finish_reason
            
            # This is the answer and correctness we'll report in the final results
            # We prioritize the summary answer if available, falling back to the reasoning answer
            if self.config.get("enable_summarization", True) and current_iteration == max_iterations - 1:
                result["final_answer"] = final_answer
                result["final_correct"] = final_correct
            
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
        
        # Process ground truth answer based on answer type
        extract_policy = self.config.get("extract_policy", "flex")
        eval_policy = self.config.get("eval_policy", "aggressive")
        
        # Check if this is a countdown problem
        if self.config.get("answer_extractor") == "countdown":
            # For countdown, prepare the ground truth as a dict with target and nums
            if "nums" in problem:
                # Parse nums if it's a string
                import ast
                nums = problem["nums"]
                if isinstance(nums, str):
                    nums = ast.literal_eval(nums)
                gt_answer = {
                    "target": int(correct_answer),
                    "nums": nums
                }
            else:
                # Fallback to regular processing if nums not available
                logger.warning(f"Countdown problem {problem_id} missing 'nums' field, falling back to latex processing")
                gt_answer = get_gt_answer(correct_answer, extract_policy=extract_policy)
        else:
            # Use regular LaTeX answer processing
            gt_answer = get_gt_answer(correct_answer, extract_policy=extract_policy)
        
        # Initialize current problem result to track partial progress
        self._current_problem_result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "processed_gt_answer": gt_answer,
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
            
            # Check if answer is correct using the new LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            iter0_check_result = answer_checker(
                iter0_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            iter0_correct = iter0_check_result["is_correct"]
        
        # Construct initial result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "processed_gt_answer": gt_answer,
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
            
            # Extract answer from the summary 
            summary_answer = extract_answer_with_config(summary, self.config)
            
            # Check if the summary answer is correct using the LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            summary_check_result = answer_checker(
                summary_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            summary_correct = summary_check_result["is_correct"]
            
            # Log the summary answer
            logger.info(f"Problem {problem_id}, iteration {current_iteration} summary answer: {summary_answer}")
            logger.info(f"Problem {problem_id}, iteration {current_iteration} summary correct: {summary_correct}")
            
            # Use reasoning answer as fallback
            reasoning_answer = result["iterations"][current_iteration]["answer"]
            reasoning_correct = result["iterations"][current_iteration]["correct"]
            
            # Use summary answer as primary, with fallback to reasoning answer if summary has no answer
            final_answer = summary_answer if summary_answer is not None else reasoning_answer
            final_correct = summary_correct if summary_answer is not None else reasoning_correct
            
            # Add summary to the collection with iteration number
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "post_think_summary": post_think_summary,
                "finish_reason": summary_finish_reason,
                "summary_answer": summary_answer,
                "summary_correct": summary_correct,
                "final_answer": final_answer,
                "final_correct": final_correct
            })
            
            # *** FIX: Update the current iteration with the summary before moving to the next iteration ***
            # This way the summary is associated with the reasoning it summarized
            result["iterations"][current_iteration]["summary"] = summary
            result["iterations"][current_iteration]["post_think_summary"] = post_think_summary
            result["iterations"][current_iteration]["summary_finish_reason"] = summary_finish_reason
            result["iterations"][current_iteration]["summary_answer"] = summary_answer
            result["iterations"][current_iteration]["summary_correct"] = summary_correct
            result["iterations"][current_iteration]["final_answer"] = final_answer
            result["iterations"][current_iteration]["final_correct"] = final_correct
            
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
            
            # Check if answer is correct using the LaTeX answer check
            # Get the appropriate answer checker
            answer_checker = get_answer_checker(self.config)
            
            next_check_result = answer_checker(
                next_answer,
                gt_answer,
                extract_policy="none",  # Skip extraction since we already extracted the answer
                eval_policy=eval_policy,
                debug=False
            )
            next_correct = next_check_result["is_correct"]
            
            # Update found_correct_answer flag
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
            
            # Update dashboard with answer information
            if self.dashboard:
                self.dashboard.update_answer_info(
                    problem_id,
                    next_answer or "No answer extracted",
                    correct_answer,
                    next_correct,
                    iteration=next_iteration
                )
            
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