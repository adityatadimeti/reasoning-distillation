from typing import Dict, Any, List, Optional
import time
import logging

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer
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
            iter0_reasoning = self._stream_model_output(problem_id, initial_prompt, iteration=0)
        else:
            # Without dashboard, just get the full response
            iter0_reasoning = self.reasoning_model.generate_response(
                initial_prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose
            )
        
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
                    "correct": iter0_correct
                }
            ],
            "timestamp": time.time()
        }
        
        # Add convenience fields for backward compatibility
        result["initial_reasoning"] = iter0_reasoning
        result["initial_answer"] = iter0_answer
        result["initial_correct"] = iter0_correct
        
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
                
            summary = summarize_reasoning(
                current_reasoning,
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
            
            # Update dashboard with summary
            if self.dashboard:
                self.dashboard.update_summary(problem_id, summary, iteration=current_iteration)
            
            # Prepare for next iteration
            next_iteration = current_iteration + 1
            
            # Get improved reasoning prompt template
            improved_template = self.config.get("improved_prompt_template")
            if not improved_template:
                raise ValueError("improved_prompt_template must be specified for additional iterations")
            
            # Create prompt for next iteration
            improved_prompt = improved_template.format(
                question=question,
                summary=summary
            )
            
            # Generate reasoning for next iteration
            if self.dashboard:
                # Use streaming for dashboard updates
                next_reasoning = self._stream_model_output(problem_id, improved_prompt, iteration=next_iteration)
            else:
                # Without dashboard, just get the full response
                next_reasoning = self.reasoning_model.generate_response(
                    improved_prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                )
            
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
                "reasoning": next_reasoning,
                "answer": next_answer,
                "correct": next_correct
            })
            
            # Add convenience fields for the latest iteration
            if next_iteration == 1:
                result["summary"] = summary
                result["improved_reasoning"] = next_reasoning
                result["improved_answer"] = next_answer
                result["improved_correct"] = next_correct
            
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
    
    def _stream_model_output(self, problem_id: str, prompt: str, iteration: int = 0) -> str:
        """
        Generic method to stream model output for any iteration and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            prompt: The prompt to send to the model
            iteration: Iteration number (0 = initial, 1 = first improvement, etc.)
            
        Returns:
            The full generated output
        """
        full_response = ""
        buffered_chunks = []
        
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
        for chunk in stream:
            full_response += chunk
            buffered_chunks.append(chunk)
            
            # Send chunk to dashboard with debug
            if self.dashboard:
                logger.debug(f"Streaming iteration {iteration} chunk to problem ID: {problem_id}")
                self.dashboard.stream_model_output(problem_id, chunk, iteration=iteration)
                
        # If the client wasn't ready, ensure all chunks are sent now
        if self.dashboard and hasattr(self.dashboard, 'client_ready') and self.dashboard.client_ready:
            # Check if we need to resend all chunks 
            if buffered_chunks and not self.dashboard.client_ready:
                logger.info(f"Client is now ready, sending all buffered chunks for iteration {iteration} on problem {problem_id}")
                # Send the complete output as one chunk
                self.dashboard.stream_model_output(problem_id, ''.join(buffered_chunks), iteration=iteration)
                
        # Update the problem status based on iteration
        if self.dashboard:
            status = f"iter{iteration}-completed"
            self.dashboard.update_problem_status(problem_id, status)
        
        return full_response