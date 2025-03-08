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
            Result dictionary with original reasoning
        """
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create initial reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        initial_prompt = reasoning_template.replace("{question}", question)
        
        # Generate initial reasoning with streaming
        if self.dashboard:
            # Use streaming for dashboard updates
            initial_reasoning = self._stream_reasoning(problem_id, initial_prompt)
        else:
            # Without dashboard, just get the full response
            initial_reasoning = self.reasoning_model.generate_response(
                initial_prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"] if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose
            )
        
        # Extract answer from reasoning
        initial_answer = extract_answer(initial_reasoning)
        
        # Check if answer is correct (simple string comparison)
        initial_correct = False
        if initial_answer is not None:
            initial_correct = initial_answer.strip() == correct_answer.strip()
        
        # Construct result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "initial_reasoning": initial_reasoning,
            "initial_answer": initial_answer,
            "initial_correct": initial_correct,
            "timestamp": time.time()
        }
        
        # Update dashboard with answer information
        if self.dashboard:
            self.dashboard.update_problem_status(
                problem_id, 
                "correct" if initial_correct else "incorrect"
            )
            
            # Send answer information to the dashboard
            self.dashboard.update_answer_info(
                problem_id,
                initial_answer or "No answer extracted",
                correct_answer,
                initial_correct
            )
        
        # If the initial answer is incorrect and summarization is enabled, try summarization
        # if not initial_correct and self.config.get("enable_summarization", True):
        if self.config.get("enable_summarization", True): # FOR TESTING: ALWAYS SUMMARIZE
            # Get the summarization prompt template
            summarize_template = self.config.get("summarize_prompt_template")
            if not summarize_template:
                raise ValueError("summarize_prompt_template must be specified in configuration")
            
            # Generate summary of the reasoning
            logger.info(f"Generating summary for problem {problem_id}")
            
            # Ensure top_k is included in the config for FireworksModelClient
            if not hasattr(self.summarizer, "top_k"):
                assert "top_k" in self.config, "top_k must be specified in config if using FireworksModelClient"
                
            summary = summarize_reasoning(
                initial_reasoning,
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
            
            # Add summary to result
            result["summary"] = summary
            
            # Update dashboard with summary
            if self.dashboard:
                self.dashboard.update_summary(problem_id, summary)
        
        return result
    
    def _stream_reasoning(self, problem_id: str, prompt: str) -> str:
        """
        Stream reasoning generation and update dashboard in real-time.
        
        Args:
            problem_id: ID of the problem
            prompt: The prompt to send to the model
            
        Returns:
            The full generated reasoning
        """
        full_response = ""
        buffered_chunks = []
        
        # Add debug logging
        logger.debug(f"Streaming for problem ID: {problem_id}")
        
        # Update the problem status to show it's processing
        if self.dashboard:
            self.dashboard.update_problem_status(problem_id, "in-progress")
        
        # Get streaming response
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
                logger.debug(f"Streaming chunk to problem ID: {problem_id}")
                self.dashboard.stream_model_output(problem_id, chunk)
                
        # If the client wasn't ready, ensure all chunks are sent now
        if self.dashboard and hasattr(self.dashboard, 'client_ready') and self.dashboard.client_ready:
            # Check if we need to resend all chunks 
            if buffered_chunks and not self.dashboard.client_ready:
                logger.info(f"Client is now ready, sending all buffered chunks for problem {problem_id}")
                # Send the complete output as one chunk
                self.dashboard.stream_model_output(problem_id, ''.join(buffered_chunks))
                
        # Update the problem status to completed
        if self.dashboard:
            self.dashboard.update_problem_status(problem_id, "completed")
            
        return full_response