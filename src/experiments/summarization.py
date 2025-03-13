from typing import Dict, Any, List, Optional, Tuple
import time
import logging

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer, extract_reasoning_trace
from src.reasoning.summarizer import summarize_reasoning
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
        
        # Initialize reasoning model
        self.reasoning_model = create_model_client(self.config["reasoning_model"])
        
        # Initialize summarizer model (could be the same model or a different one)
        summarizer_type = self.config.get("summarizer_type", "self")
        if summarizer_type == "self":
            self.summarizer = self.reasoning_model
        else:
            if "summarizer_model" not in self.config:
                raise ValueError("summarizer_model must be specified when summarizer_type is not 'self'")
            self.summarizer = create_model_client(self.config["summarizer_model"])
    
    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the summarization experiment on a list of problems."""
        total_problems = len(problems)
        for i, problem in enumerate(problems):
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
            logger.info(f"Processing problem {problem_id} ({i+1}/{total_problems})")
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.update_problem_status(problem_id, "in-progress")
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
                    self.dashboard.update_problem_status(problem_id, "completed")
            
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "error")
                
                # Add error to results
                self.results.append({
                    "problem_id": problem_id,
                    "error": str(e),
                    "status": "error"
                })
            
            # Save intermediate results
            if self.config.get("save_intermediate", True):
                self.save_results()
                
        return self.results
    
    def _stream_summary_generation(self, problem_id: str, question: str, reasoning: str, prompt_template: str, iteration: int = 0) -> Tuple[str, str]:
        """
        Stream the summary generation for a problem and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            question: The question that the reasoning is for
            reasoning: The reasoning to summarize
            prompt_template: The template to use for summarization
            iteration: Iteration number
            
        Returns:
            Tuple of (full_summary, finish_reason)
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
            
            # Get finish_reason for the summary
            # Make a non-streaming call with the same parameters to get finish_reason
            if hasattr(self.summarizer, 'generate_completion') and 'fireworks' in str(self.summarizer.__class__).lower():
                try:
                    # Create the prompt
                    summary_prompt = prompt_template.replace("{reasoning}", reasoning_trace)
                    if "{question}" in prompt_template:
                        summary_prompt = summary_prompt.replace("{question}", question)
                        
                    # Make a non-streaming API call to get finish_reason
                    logger.debug(f"Making non-streaming call to get summary finish_reason for problem {problem_id}, iteration {iteration}")
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
                    
                    # Extract the finish_reason from the response
                    if isinstance(summary_response, tuple):
                        _, finish_reason = summary_response
                        logger.debug(f"Got summary finish_reason '{finish_reason}' for problem {problem_id}, iteration {iteration}")
                except Exception as e:
                    logger.warning(f"Error getting summary finish_reason: {str(e)}. Using 'unknown' instead.")
            
            # Send a final empty chunk with the finish_reason
            if self.dashboard:
                self.dashboard.stream_summary_chunk(problem_id, "", iteration=iteration, finish_reason=finish_reason)
                
                # Send the final complete summary with finish_reason
                self.dashboard.update_summary(problem_id, full_summary, iteration=iteration, finish_reason=finish_reason)
            
            return full_summary, finish_reason
            
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
        
        # Create initial reasoning prompt (iteration 0)
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        initial_prompt = reasoning_template.replace("{question}", question)
        
        # Generate iteration 0 reasoning with streaming
        if self.dashboard:
            # Use streaming for dashboard updates
            iter0_reasoning, iter0_finish_reason = self._stream_model_output(problem_id, initial_prompt, iteration=0)
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
            
            # Handle both tuple and string responses for backward compatibility
            if isinstance(response, tuple):
                iter0_reasoning, iter0_finish_reason = response
            else:
                iter0_reasoning = response
                iter0_finish_reason = "unknown"  # Default if finish_reason is not available
        
        # Log the finish reason
        logger.info(f"Problem {problem_id}, iteration 0 finish reason: {iter0_finish_reason}")
        
        # Extract answer from iteration 0 reasoning
        iter0_answer = extract_answer(iter0_reasoning)
        
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
                summary, summary_finish_reason = self._stream_summary_generation(
                    problem_id, 
                    question,
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
                
                # Handle both tuple and string responses for backward compatibility
                if isinstance(summary_response, tuple):
                    summary, summary_finish_reason = summary_response
                else:
                    summary = summary_response
            
            # Log the finish reason for the summary
            logger.info(f"Problem {problem_id}, summary {current_iteration} finish reason: {summary_finish_reason}")
            
            # Add summary to the collection with iteration number
            all_summaries.append({
                "iteration": current_iteration,
                "summary": summary,
                "finish_reason": summary_finish_reason
            })
            
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
                next_reasoning, next_finish_reason = self._stream_model_output(problem_id, improved_prompt, iteration=next_iteration)
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
                
                # Handle both tuple and string responses for backward compatibility
                if isinstance(next_response, tuple):
                    next_reasoning, next_finish_reason = next_response
                else:
                    next_reasoning = next_response
            
            # Log the finish reason
            logger.info(f"Problem {problem_id}, iteration {next_iteration} finish reason: {next_finish_reason}")
            
            # Extract answer from next iteration reasoning
            next_answer = extract_answer(next_reasoning)
            
            # Check if answer is correct
            next_correct = False
            if next_answer is not None:
                next_correct = next_answer.strip() == correct_answer.strip()
                found_correct_answer = found_correct_answer or next_correct
            
            # Add this iteration to the results
            result["iterations"].append({
                "iteration": next_iteration,
                "summary": summary,
                "summary_finish_reason": summary_finish_reason,
                "reasoning": next_reasoning,
                "answer": next_answer,
                "correct": next_correct,
                "finish_reason": next_finish_reason
            })
            
            # NOTE: These are redundant fields duplicating data already in iterations[1]
            # They are kept for backward compatibility with existing dashboard code
            # TODO: For a future refactor, remove these redundant fields and update dashboard
            # to use iterations[1] directly
            result["summary"] = summary
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
        
        return result
    
    def _stream_model_output(self, problem_id: str, prompt: str, iteration: int = 0) -> Tuple[str, str]:
        """
        Generic method to stream model output for any iteration and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            prompt: The prompt to send to the model
            iteration: Iteration number (0 = initial, 1 = first improvement, etc.)
            
        Returns:
            Tuple of (full_response, finish_reason)
        """
        full_response = ""
        buffered_chunks = []
        finish_reason = "unknown"  # Default value if we can't extract it
        
        # Add debug logging
        logger.debug(f"Streaming iteration {iteration} for problem ID: {problem_id}")
        
        # Update the problem status to show it's processing
        if self.dashboard:
            status = f"iter{iteration}-in-progress"
            self.dashboard.update_problem_status(problem_id, status)
        
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
        
        # If using FireworksModelClient, make an API call to get the finish_reason
        # This is more reliable than trying to extract it from streaming chunks
        if hasattr(self.reasoning_model, 'generate_completion') and 'fireworks' in str(self.reasoning_model.__class__).lower():
            try:
                # Make a non-streaming API call with the same parameters but very small max_tokens
                # We just want to get the finish_reason, not the full content again
                logger.debug(f"Making non-streaming call to get finish_reason for problem {problem_id}, iteration {iteration}")
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
                
                # Extract the finish_reason from the response
                if isinstance(response, tuple):
                    _, finish_reason = response
                    logger.debug(f"Got finish_reason '{finish_reason}' for problem {problem_id}, iteration {iteration}")
                
            except Exception as e:
                logger.warning(f"Error getting finish_reason: {str(e)}. Using 'unknown' instead.")
                finish_reason = "unknown"
        else:
            # For non-Fireworks models, use 'streaming' as the finish_reason
            finish_reason = "streaming"
                
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
            self.dashboard.update_problem_status(problem_id, status)
        
        return full_response, finish_reason