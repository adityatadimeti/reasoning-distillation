from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import copy
from collections import Counter

from src.llm.base_client import TokenUsage, CostInfo

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_reasoning_trace, extract_answer_with_config
from src.dashboard.server import DashboardServer
from src.eval.latex_answer_check import get_gt_answer, check_one_latex_answer

logger = logging.getLogger(__name__)

class PassKExperiment(BaseExperiment):
    """Experiment for testing pass@k and consensus@k metrics, using LaTeX-aware grading."""
    
    def __init__(
        self, 
        experiment_name: str = "test_passk", 
        config: Dict[str, Any] = None,
        dashboard: Optional[DashboardServer] = None,
        verbose: bool = False
    ):
        """Initialize the pass@k experiment."""
        super().__init__(experiment_name, config, dashboard)
        
        # Store verbose flag
        self.verbose = verbose
        
        # Validate required parameters
        required_params = [
            "reasoning_model", "max_tokens", "temperature", 
            "top_p", "top_k", "presence_penalty", "frequency_penalty",
            "pass_k_iterations"
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
        
        # Add lock for thread safety when updating results
        self.results_lock = Lock()
    
    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the pass@k experiment on a list of problems."""
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
                    self.dashboard.update_experiment_status({
                        "total": total_problems,
                        "completed": i + 1,
                        "status": f"Completed problem {problem_id}",
                        "config": self.config
                    })
                
                # Save intermediate results if configured
                if self.config.get("save_intermediate", False):
                    self.save_results()
                    
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "error", question)
        
        return self.results
    
    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Run the pass@k experiment on a list of problems in parallel.
        
        Args:
            problems: List of problem dictionaries
            max_concurrency: Maximum number of problems to process concurrently
            
        Returns:
            List of results for all problems
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Process all problems in parallel with the semaphore
        tasks = []
        for i, problem in enumerate(problems):
            task = asyncio.create_task(
                self._process_problem_with_semaphore(semaphore, problem, i, len(problems))
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Update self.results with the gathered results
        self.results = results
        
        return self.results
    
    async def _process_problem_with_semaphore(self, semaphore: asyncio.Semaphore, problem: Dict[str, Any], 
                                             index: int, total: int) -> Dict[str, Any]:
        """Process a problem with semaphore for concurrency control."""
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", str(index+1)))
        # Get the question text
        question = problem.get("question", "")
        
        logger.info(f"Processing problem {problem_id} ({index+1}/{total})")
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_problem_status(problem_id, "in-progress", question)
            self.dashboard.update_experiment_status({
                "total": total,
                "completed": index,
                "status": f"Processing problem {problem_id}",
                "config": self.config
            })
        
        # Acquire the semaphore before processing
        async with semaphore:
            try:
                # Process the problem
                result = await self._process_problem_async(problem)
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "completed", question)
                    
                # Save intermediate results if configured
                if self.config.get("save_intermediate", False):
                    with self.results_lock:
                        # Add the result to self.results
                        self.results.append(result)
                        # Save the current results
                        self.save_results()
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                if self.dashboard:
                    self.dashboard.update_problem_status(problem_id, "error", question)
                
                # Return a minimal result with error information
                return {
                    "problem_id": problem_id,
                    "question": question,
                    "error": str(e),
                    "status": "error"
                }
    
    async def _process_problem_async(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the pass@k pipeline asynchronously.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with solutions
        """
        # Extract problem information
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem.get("question", "")
        correct_answer = problem.get("answer", "").strip()
        
        # Process ground truth answer using get_gt_answer
        extract_policy = self.config.get("extract_policy", "flex")
        eval_policy = self.config.get("eval_policy", "aggressive")
        gt_answer = get_gt_answer(correct_answer, extract_policy=extract_policy)
        
        # Initialize result dictionary
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "processed_gt_answer": gt_answer,  # Store the processed GT answer
            "solutions": [],
            "token_usage": {},
            "cost_info": {}
        }
        
        # Get the number of iterations (k)
        k = self.config.get("pass_k_iterations", 5)
        
        # Get the reasoning prompt template
        prompt_type = self.config.get("prompts", {}).get("reasoning", "v0")
        reasoning_prompt_template = self.config.get("reasoning_prompt_template", "")
        
        if not reasoning_prompt_template:
            logger.warning(f"No reasoning_prompt_template found in config for problem {problem_id}, using empty string")
        
        # Generate k solutions
        for i in range(k):
            solution_id = i + 1
            logger.info(f"Generating solution {solution_id}/{k} for problem {problem_id}")
            
            # Format the prompt with the question
            prompt = reasoning_prompt_template.replace("{question}", question)
            
            # Generate a response
            response = await self.reasoning_model.generate_response_async(
                prompt=prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config["top_k"],
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                enable_continuation=self.config.get("enable_continuation", False),
                max_total_tokens=self.config.get("max_total_tokens", self.config["max_tokens"]),
                max_continuations=self.config.get("max_continuations", 0),
                verbose=self.verbose
            )
            
            # Unpack the tuple (content, finish_reason, token_usage, cost_info)
            reasoning, finish_reason, token_usage, cost_info, _ = response
            
            # Track token usage and cost
            self.track_token_usage_and_cost(
                problem_id, 
                token_usage, 
                cost_info, 
                iteration=0, 
                step=f"solution_{solution_id}"
            )
            
            # Extract answer from reasoning using the configured extractor
            answer = extract_answer_with_config(reasoning, self.config)
            
            # Check if answer is correct using the LaTeX answer check
            is_correct = False
            if answer is not None:
                # Use check_one_latex_answer with the extracted answer and processed GT answer
                check_result = check_one_latex_answer(
                    answer,
                    gt_answer,
                    extract_policy="none",  # Answer is already extracted
                    eval_policy=eval_policy,
                    debug=False
                )
                is_correct = check_result["is_correct"]
            
            # Add the solution to the results
            solution = {
                "solution_id": solution_id,
                "reasoning": reasoning,
                "answer": answer,
                "correct": is_correct,
                "finish_reason": finish_reason
            }
            
            result["solutions"].append(solution)
            
            # Update dashboard with answer information
            if self.dashboard:
                self.dashboard.update_answer_info(
                    problem_id,
                    answer or "No answer extracted",
                    correct_answer,
                    is_correct,
                    iteration=solution_id
                )
        
        # Calculate pass@k (at least one correct answer)
        num_correct = sum(1 for s in result["solutions"] if s["correct"])
        result["pass_at_k"] = num_correct > 0
        result["num_correct"] = num_correct
        
        # Calculate consensus answer (most common answer)
        answers = [s["answer"] for s in result["solutions"] if s["answer"] is not None]
        consensus_answer, consensus_count = self._find_consensus(answers)
        
        # Check if consensus answer is correct using LaTeX answer check
        consensus_correct = False
        if consensus_answer is not None:
            check_result = check_one_latex_answer(
                consensus_answer,
                gt_answer,
                extract_policy="none", # Consensus answer is already extracted
                eval_policy=eval_policy,
                debug=False
            )
            consensus_correct = check_result["is_correct"]
        
        result["consensus_answer"] = consensus_answer
        result["consensus_correct"] = consensus_correct
        result["consensus_count"] = consensus_count
        
        # Copy detailed token usage and cost info into the result dict
        for r in self.results:
            if r.get("problem_id") == problem_id:
                result["token_usage"] = r.get("token_usage", {})
                result["cost_info"] = r.get("cost_info", {})
                break
        else:
            # If not found in self.results (e.g., async/parallel), pull from self.token_usage/cost_info
            result["token_usage"] = self.token_usage["problems"].get(problem_id, {})
            result["cost_info"] = self.cost_info["problems"].get(problem_id, {})

        # Log the results
        logger.info(f"Problem {problem_id}: pass@{k}={result['pass_at_k']}, consensus@{k}={consensus_correct} (count={consensus_count}/{k})")
        
        return result
    
    def _process_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single problem through the pass@k pipeline.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            
        Returns:
            Result dictionary with solutions
        """
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async version of the function
        return loop.run_until_complete(self._process_problem_async(problem))
    
    def _find_consensus(self, answers: List[str]) -> Tuple[Optional[str], int]:
        """Find the most common answer from a list of answers."""
        if not answers:
            return None, 0
            
        # Normalize answers by stripping whitespace
        normalized_answers = [a.strip() for a in answers if a is not None]
        
        # Count occurrences
        answer_counts = Counter(normalized_answers)
        
        # Find most common answer
        if not answer_counts:
            return None, 0
            
        most_common = answer_counts.most_common(1)[0]
        return most_common[0], most_common[1]  # Return tuple of (answer, count)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics for the experiment.
        
        Returns:
            Dictionary of metrics
        """
        # Get base metrics from parent class
        metrics = super().calculate_metrics()
        
        # Add pass@k and consensus@k metrics
        k = self.config.get("pass_k_iterations", 5)
        total_problems = len(self.results)
        
        # Skip if no results
        if total_problems == 0:
            return metrics
        
        pass_at_k_count = sum(1 for r in self.results if r.get("pass_at_k", False))
        consensus_correct_count = sum(1 for r in self.results if r.get("consensus_correct", False))
        
        metrics["pass_at_k"] = {
            "k": k,
            "count": pass_at_k_count,
            "percentage": (pass_at_k_count / total_problems) * 100 if total_problems > 0 else 0
        }
        
        metrics["consensus_at_k"] = {
            "k": k,
            "count": consensus_correct_count,
            "percentage": (consensus_correct_count / total_problems) * 100 if total_problems > 0 else 0
        }
        
        return metrics
