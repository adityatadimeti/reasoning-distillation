from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import copy

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer, extract_reasoning_trace, extract_answer_with_config
from src.reasoning.summarizer import summarize_reasoning, summarize_reasoning_async
from src.dashboard.server import DashboardServer

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
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
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
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
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
        
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create initial reasoning prompt (iteration 0)
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        initial_prompt = reasoning_template.replace("{question}", question)
        
        # Generate iteration 0 reasoning asynchronously
        response = await self.reasoning_model.generate_response_async(
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
                summary_response = await summarize_reasoning_async(
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
                
                # Print the type and value of summary_response for debugging
                logger.info(f"DEBUG: summary_response type: {type(summary_response)}, value: {summary_response}")
                
                # Handle the response from summarize_reasoning_async
                # The response is a tuple of (content, finish_reason, token_usage, cost_info)
                summary, summary_finish_reason, token_usage, cost_info = summary_response
            except Exception as e:
                logger.error(f"Error unpacking summary_response: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                # Continue with default values set above
            
            # Track token usage and cost for the summary only if we have valid data
            if token_usage and cost_info:
                self.track_token_usage_and_cost(problem_id, token_usage, cost_info, current_iteration, "summary")
            
            # Log the finish reason for the summary
            logger.info(f"Problem {problem_id}, summary {current_iteration} finish reason: {summary_finish_reason}")
            
            # Add summary to the collection with iteration number
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "finish_reason": summary_finish_reason
            })
            
            # This way the summary is associated with the reasoning it summarized
            result["iterations"][current_iteration]["summary"] = summary
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
                accumulated_summaries += f"\n\nATTEMPT {summary_item['iteration']} SUMMARY:\n{summary_item['summary']}"
            
            # Create prompt for next iteration using accumulated summaries
            improved_prompt = improved_template.replace("{question}", question).replace("{summaries}", accumulated_summaries)
            
            # Generate reasoning for next iteration asynchronously
            next_response = await self.reasoning_model.generate_response_async(
                improved_prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose
            )
            
            # Handle both tuple and string responses for backward compatibility
            if isinstance(next_response, tuple):
                # Unpack the tuple (content, finish_reason, token_usage, cost_info)
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
        
        return result
    
    def _stream_summary_generation(self, problem_id: str, question: str, reasoning: str, prompt_template: str, iteration: int = 0) -> Tuple[str, str, TokenUsage, CostInfo]:
        """
        Stream the summary generation for a problem and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            question: The question text for the problem
            reasoning: The reasoning to summarize
            prompt_template: The template to use for summarization
            iteration: Iteration number
            
        Returns:
            Tuple of (full_summary, finish_reason, token_usage, cost_info)
        """
        full_summary = ""
        buffered_chunks = []
        finish_reason = "unknown"  # Default if we can't extract it
        
        # Add debug logging
        logger.debug(f"Streaming summary for iteration {iteration}, problem ID: {problem_id}")
        
        # Update the problem status to show it's summarizing
        if self.dashboard:
            status = f"iter{iteration}-summarizing"
            self.dashboard.update_problem_status(problem_id, status)
        
        # Extract the reasoning trace from within <think> tags
        reasoning_trace = extract_reasoning_trace(reasoning, allow_fallback=self.config.get("allow_fallback", False))
        if reasoning_trace is None:
            raise ValueError(f"Could not extract reasoning trace for problem {problem_id}. Make sure the model output contains <think> tags.")
        
        try:
            # Stream the summary using the summarizer model
            summary_stream = summarize_reasoning(
                question,
                reasoning_trace,  # Use extracted reasoning trace instead of full reasoning
                self.summarizer,
                prompt_template,
                max_tokens=self.config.get("summary_max_tokens"),
                temperature=self.config.get("summary_temperature"),
                top_p=self.config.get("summary_top_p"),
                top_k=self.config.get("summary_top_k"),
                presence_penalty=self.config.get("summary_presence_penalty"),
                frequency_penalty=self.config.get("summary_frequency_penalty"),
                verbose=self.verbose,
                stream=True
            )
            
            # Process the streaming output
            for chunk in summary_stream:
                # Add to full response
                full_summary += chunk
                buffered_chunks.append(chunk)
                
                # Stream to dashboard if available
                if self.dashboard and len(buffered_chunks) >= 1:
                    combined_chunk = "".join(buffered_chunks)
                    self.dashboard.stream_summary_chunk(problem_id, combined_chunk, iteration)
                    buffered_chunks = []
            
            # Send any remaining buffered chunks
            if self.dashboard and buffered_chunks:
                combined_chunk = "".join(buffered_chunks)
                self.dashboard.stream_summary_chunk(problem_id, combined_chunk, iteration)
            
            # Get finish_reason for the summary and track token usage and cost
            # Make a non-streaming call with the same parameters
            token_usage = None
            cost_info = None
            
            if hasattr(self.summarizer, 'generate_completion'):
                try:
                    # Create the prompt
                    summary_prompt = prompt_template.replace("{reasoning}", reasoning_trace)
                    if "{question}" in prompt_template:
                        summary_prompt = summary_prompt.replace("{question}", question)
                        
                    # Make a non-streaming API call to get finish_reason and token usage
                    logger.debug(f"Making non-streaming call to get summary finish_reason and token usage for problem {problem_id}, iteration {iteration}")
                    summary_response = self.summarizer.generate_response(
                        summary_prompt,
                        stream=False,
                        max_tokens=self.config.get("summary_max_tokens"),
                        temperature=self.config.get("summary_temperature"),
                        top_p=self.config.get("summary_top_p"),
                        top_k=self.config.get("summary_top_k"),
                        presence_penalty=self.config.get("summary_presence_penalty"),
                        frequency_penalty=self.config.get("summary_frequency_penalty"),
                        verbose=False  # Don't log this auxiliary call
                    )
                    
                    # Extract the finish_reason, token usage, and cost info from the response
                    # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                    _, finish_reason, token_usage, cost_info = summary_response
                    logger.debug(f"Got summary finish_reason '{finish_reason}' for problem {problem_id}, iteration {iteration}")
                    logger.info(f"Summary token usage for problem {problem_id}, iteration {iteration}: {token_usage}")
                    logger.info(f"Summary cost info for problem {problem_id}, iteration {iteration}: {cost_info}")
                    
                    # Track token usage and cost for the summary
                    self.track_token_usage_and_cost(problem_id, token_usage, cost_info, iteration, "summary")
                except Exception as e:
                    logger.warning(f"Error getting summary finish_reason and token usage: {str(e)}. Using 'unknown' instead.")
            
            # Send a final empty chunk with the finish_reason
            if self.dashboard:
                self.dashboard.stream_summary_chunk(problem_id, "", iteration=iteration, finish_reason=finish_reason)
                
                # Send the final complete summary with finish_reason
                self.dashboard.update_summary(problem_id, full_summary, iteration=iteration, finish_reason=finish_reason)
            
            return full_summary, finish_reason, token_usage, cost_info
            
        except Exception as e:
            logger.error(f"Error streaming summary: {str(e)}")
            raise
    
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
            
            # Add summary to the collection with iteration number
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "finish_reason": summary_finish_reason
            })
            
            # *** FIX: Update the current iteration with the summary before moving to the next iteration ***
            # This way the summary is associated with the reasoning it summarized
            result["iterations"][current_iteration]["summary"] = summary
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
                accumulated_summaries += f"\n\nATTEMPT {summary_item['iteration']} SUMMARY:\n{summary_item['summary']}"
            
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
        
        return result
    
    def _stream_model_output(self, problem_id: str, prompt: str, iteration: int = 0, step: str = "reasoning") -> Tuple[str, str, TokenUsage, CostInfo]:
        """
        Generic method to stream model output for any iteration and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            prompt: The prompt to send to the model
            iteration: Iteration number (0 = initial, 1 = first improvement, etc.)
            step: Step name (e.g., "reasoning", "summary")
            
        Returns:
            Tuple of (full_response, finish_reason, token_usage, cost_info)
        """
        full_response = ""
        buffered_chunks = []
        finish_reason = "unknown"  # Default value if we can't extract it
        
        # Add debug logging
        logger.debug(f"Streaming iteration {iteration} for problem ID: {problem_id}")
        
        # Get the question for this problem from the current results
        question = None
        for result in self.results:
            if result.get("problem_id") == problem_id:
                question = result.get("question", "")
                break
        
        # Update the problem status to show it's processing
        if self.dashboard:
            status = f"iter{iteration}-in-progress"
            self.dashboard.update_problem_status(problem_id, status, question)
        
        # Get streaming response with appropriate parameters
        stream = self.reasoning_model.generate_response(
            prompt,
            stream=True,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
            presence_penalty=self.config["presence_penalty"],
            frequency_penalty=self.config["frequency_penalty"],
            verbose=self.verbose
        )
        
        # Process each chunk as it comes in
        last_chunk = None
        for chunk in stream:
            # Get the content from the chunk
            full_response += chunk
            buffered_chunks.append(chunk)
            
            # Send chunk to dashboard with debug
            if self.dashboard:
                logger.debug(f"Streaming iteration {iteration} chunk to problem ID: {problem_id}")
                self.dashboard.stream_model_output(problem_id, chunk, iteration=iteration)
        
        # For streaming responses, we need to make a non-streaming call to get token usage and cost info
        token_usage = None
        cost_info = None
        
        # If using FireworksModelClient, make an API call to get the finish_reason and token usage
        # This is more reliable than trying to extract it from streaming chunks
        if hasattr(self.reasoning_model, 'generate_completion') and 'fireworks' in str(self.reasoning_model.__class__).lower():
            try:
                # Make a non-streaming API call with the same parameters
                # We want to get the finish_reason and token usage information
                logger.debug(f"Making non-streaming call to get finish_reason and token usage for problem {problem_id}, iteration {iteration}")
                response = self.reasoning_model.generate_response(
                    prompt,
                    stream=False,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=False  # Don't log this auxiliary call
                )
                
                # Extract the finish_reason and token usage from the response
                if isinstance(response, tuple) and len(response) >= 4:
                    # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                    _, finish_reason, token_usage, cost_info = response
                    logger.debug(f"Got finish_reason '{finish_reason}' for problem {problem_id}, iteration {iteration}")
                    logger.info(f"Token usage for problem {problem_id}, iteration {iteration}: {token_usage}")
                    logger.info(f"Cost info for problem {problem_id}, iteration {iteration}: {cost_info}")
                    
                    # Track token usage and cost
                    self.track_token_usage_and_cost(problem_id, token_usage, cost_info, iteration, step)
                
            except Exception as e:
                logger.warning(f"Error getting finish_reason and token usage: {str(e)}. Using 'unknown' instead.")
                finish_reason = "unknown"
        else:
            # For non-Fireworks models, use 'streaming' as the finish_reason
            # and try to get token usage information if available
            finish_reason = "streaming"
            try:
                # Make a non-streaming call to get token usage
                response = self.reasoning_model.generate_response(
                    prompt,
                    stream=False,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=False
                )
                
                if isinstance(response, tuple) and len(response) >= 4:
                    # Unpack the tuple (content, finish_reason, token_usage, cost_info)
                    _, _, token_usage, cost_info = response
                    logger.info(f"Token usage for problem {problem_id}, iteration {iteration}: {token_usage}")
                    logger.info(f"Cost info for problem {problem_id}, iteration {iteration}: {cost_info}")
                    
                    # Track token usage and cost
                    self.track_token_usage_and_cost(problem_id, token_usage, cost_info, iteration, step)
            except Exception as e:
                logger.warning(f"Error getting token usage: {str(e)}")
                
        # If the client wasn't ready, ensure all chunks are sent now
        if self.dashboard and hasattr(self.dashboard, 'client_ready') and self.dashboard.client_ready:
            # Check if we need to resend all chunks 
            if buffered_chunks and not self.dashboard.client_ready:
                logger.info(f"Client is now ready, sending all buffered chunks for iteration {iteration} on problem {problem_id}")
                # Send the complete output as one chunk
                self.dashboard.stream_model_output(problem_id, ''.join(buffered_chunks), iteration=iteration)
        
        # Send a final empty chunk with the finish_reason
        if self.dashboard:
            self.dashboard.stream_model_output(problem_id, "", iteration=iteration, finish_reason=finish_reason)
                
        # Update the problem status based on iteration
        if self.dashboard:
            status = f"iter{iteration}-completed"
            self.dashboard.update_problem_status(problem_id, status, question)
        
        return full_response, finish_reason, token_usage, cost_info