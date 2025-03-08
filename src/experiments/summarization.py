from typing import Dict, Any, List, Optional
import time
import logging

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
# from src.reasoning.extractor import extract_answer
# from src.reasoning.summarizer import summarize_reasoning
from src.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)

class SummarizationExperiment(BaseExperiment):
    """Experiment for testing reasoning improvement through summarization."""
    
    def __init__(
        self, 
        experiment_name: str = "test_summarization", 
        config: Dict[str, Any] = None,
        dashboard: Optional[DashboardServer] = None
    ):
        """Initialize the summarization experiment."""
        super().__init__(experiment_name, config, dashboard)
        
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
            problem_id = problem.get("id", str(i+1))
            
            logger.info(f"Processing problem {problem_id} ({i+1}/{total_problems})")
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.update_problem_status(problem_id, "in-progress")
                self.dashboard.update_experiment_status({
                    "total": total_problems,
                    "completed": i,
                    "status": f"Processing problem {problem_id}"
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
        problem_id = problem.get("id", "unknown")
        
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create initial reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        initial_prompt = reasoning_template.format(question=question)
        
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
            )
        
        # For now, we're only implementing the first step, so return the result
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "initial_reasoning": initial_reasoning,
            "timestamp": time.time()
        }
        
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
        )
        
        # Process each chunk as it comes in
        for chunk in stream:
            full_response += chunk
            
            # Send chunk to dashboard
            if self.dashboard:
                self.dashboard.stream_model_output(problem_id, chunk)
                
                # Add a small delay to ensure chunks are rendered properly
                # time.sleep(0.01)
        
        # Update the problem status to completed
        if self.dashboard:
            self.dashboard.update_problem_status(problem_id, "completed")
            
        return full_response
    
    # def _create_reasoning_prompt(self, question: str) -> str:
    #     """Create a prompt for generating reasoning."""
    #     template = self.config.get("reasoning_prompt_template", 
    #         "Solve the following math problem step by step:\n\n{question}"
    #     )
    #     return template.format(question=question)
    
    # def _create_improved_reasoning_prompt(self, question: str, summary: str) -> str:
    #     """Create a prompt for generating improved reasoning based on summary."""
    #     template = self.config.get("improved_reasoning_prompt_template",
    #         "Solve the following math problem step by step. "
    #         "Here's a summary of a previous attempt that had errors:\n\n"
    #         "SUMMARY: {summary}\n\n"
    #         "PROBLEM: {question}\n\n"
    #         "Let's solve this correctly:"
    #     )
    #     return template.format(question=question, summary=summary)
    
    # def _check_answer(self, predicted_answer: str, correct_answer: str) -> bool:
    #     """Check if the predicted answer matches the correct answer."""
    #     # Simple string comparison (you might want something more sophisticated)
    #     return predicted_answer.strip() == correct_answer.strip()