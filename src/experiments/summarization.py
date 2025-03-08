from typing import Dict, Any, List
import time

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
# from src.reasoning.extractor import extract_answer
# from src.reasoning.summarizer import summarize_reasoning

class SummarizationExperiment(BaseExperiment):
    """Experiment for testing reasoning improvement through summarization."""
    
    def __init__(self, experiment_name: str = "summarization", **kwargs):
        """Initialize the summarization experiment."""
        super().__init__(experiment_name, **kwargs)
        
        # Initialize reasoning model
        reasoning_model_name = self.config.get("reasoning_model", "accounts/fireworks/models/qwq-32b")
        self.reasoning_model = create_model_client(reasoning_model_name)
        
        # Initialize summarizer model (could be the same model or a different one)
        summarizer_type = self.config.get("summarizer_type", "self")
        if summarizer_type == "self":
            self.summarizer = self.reasoning_model
        else:
            summarizer_model_name = self.config.get("summarizer_model", "gpt-4o")
            self.summarizer = create_model_client(summarizer_model_name)
    
    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the summarization experiment on a list of problems."""
        for problem in problems:
            result = self._process_problem(problem)
            self.results.append(result)
            
            # Optional: save intermediate results
            if self.config.get("save_intermediate", True):
                self.save_results()
                
        return self.results
    
    def _process_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the summarization pipeline.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with original and improved reasoning
        """
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Step 1: Generate initial reasoning
        initial_prompt = self._create_reasoning_prompt(question)

        assert self.config.get("max_tokens") is not None, "max_tokens must be provided"
        assert self.config.get("temperature") is not None, "temperature must be provided"
        assert self.config.get("top_p") is not None, "top_p must be provided"
        assert self.config.get("top_k") is not None, "top_k must be provided"
        assert self.config.get("presence_penalty") is not None, "presence_penalty must be provided"
        assert self.config.get("frequency_penalty") is not None, "frequency_penalty must be provided"

        initial_reasoning = self.reasoning_model.generate_response(
            initial_prompt,
            max_tokens=self.config.get("max_tokens"),
            temperature=self.config.get("temperature"),
            top_p=self.config.get("top_p"),
            top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
            presence_penalty=self.config.get("presence_penalty"),
            frequency_penalty=self.config.get("frequency_penalty"),
        )
        
        # # Step 2: Extract answer from initial reasoning
        # initial_answer = extract_answer(initial_reasoning)
        # initial_correct = self._check_answer(initial_answer, correct_answer)
        
        # result = {
        #     "problem_id": problem.get("id", "unknown"),
        #     "question": question,
        #     "correct_answer": correct_answer,
        #     "initial_reasoning": initial_reasoning,
        #     "initial_answer": initial_answer,
        #     "initial_correct": initial_correct,
        #     "timestamp": time.time()
        # }
        
        # # If initial answer is correct, we're done
        # if initial_correct:
        #     result["correct"] = True
        #     return result
        
        # # Step 3: Summarize the reasoning
        # summary = summarize_reasoning(
        #     initial_reasoning, 
        #     self.summarizer,
        #     self.config.get("summarization_prompt_template")
        # )
        
        # # Step 4: Generate improved reasoning based on summary
        # improved_prompt = self._create_improved_reasoning_prompt(question, summary)
        # improved_reasoning = self.reasoning_model.generate_response(
        #     improved_prompt,
        #     max_tokens=self.config.get("max_tokens", 16384),
        #     temperature=self.config.get("temperature", 0.6),
        #     top_p=self.config.get("top_p", 1.0),
        #     top_k=self.config.get("top_k", 40) if hasattr(self.reasoning_model, "top_k") else None,
        #     presence_penalty=self.config.get("presence_penalty", 0.0),
        #     frequency_penalty=self.config.get("frequency_penalty", 0.0),
        # )
        
        # # Step 5: Extract answer from improved reasoning
        # improved_answer = extract_answer(improved_reasoning)
        # improved_correct = self._check_answer(improved_answer, correct_answer)
        
        # # Update result
        # result.update({
        #     "summary": summary,
        #     "improved_reasoning": improved_reasoning,
        #     "improved_answer": improved_answer,
        #     "improved_correct": improved_correct,
        #     "correct": improved_correct
        # })
        
        # return result
    
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