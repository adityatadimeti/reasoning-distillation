import asyncio
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import random
import concurrent.futures
import threading
from functools import partial
from threading import Lock
import os
import json

import asyncio

from src.experiments.base import BaseExperiment
from src.llm.model_factory import create_model_client
from src.reasoning.extractor import extract_answer, extract_reasoning_trace
from src.dashboard.server import DashboardServer


logger = logging.getLogger(__name__)

# Thread-safe locks for dashboard updates
dashboard_locks = {}

class PassExperiment(BaseExperiment):
    """Experiment for evaluating pass@k performance of reasoning models."""
    
    def __init__(
        self, 
        experiment_name: str, 
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
            "top_p", "presence_penalty", "frequency_penalty"
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
        
        # Set the parallelism settings
        self.max_workers = min(self.config.get("max_parallel_workers", 10), 10)  # Default 10, max 10
        self.rate_limit = self.config.get("api_rate_limit", 60)  # Calls per minute
        
        # Thread-safe results list
        self.results_lock = Lock()

    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the pass@k experiment on a list of problems with parallelism."""
        total_problems = len(problems)
        
        # Initialize dashboard with overall status
        if self.dashboard:
            self.dashboard.update_experiment_status({
                "total": total_problems,
                "completed": 0,
                "status": "Running with parallel processing",
                "config": self.config
            })
        
        # Process problems in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all problems for processing
            futures = []
            for i, problem in enumerate(problems):
                problem_id = problem.get("id", problem.get("ID", str(i+1)))
                
                # Create a lock for this problem's dashboard updates
                dashboard_locks[problem_id] = threading.Lock()
                
                # Log the submission
                logger.info(f"Submitting problem {problem_id} ({i+1}/{total_problems}) for processing")
                
                # Update dashboard to show the problem is queued
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.update_problem_status(
                            problem_id, "queued", problem.get("question", "")
                        )
                
                # Submit the problem for processing
                future = executor.submit(
                    self._process_problem_wrapper,
                    problem,
                    self.config.get("pass_k_iterations", 8),
                    i,
                    total_problems
                )
                futures.append(future)
            
            # Wait for all problems to complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with self.results_lock:
                            self.results.append(result)
                    
                    # Update completed count
                    completed += 1
                    
                    # Update dashboard with progress
                    if self.dashboard:
                        self.dashboard.update_experiment_status({
                            "total": total_problems,
                            "completed": completed,
                            "status": f"Completed {completed}/{total_problems} problems"
                        })
                    
                    # Save intermediate results
                    if self.config.get("save_intermediate", True) and completed % 5 == 0:
                        self.save_results()
                        
                except Exception as e:
                    logger.error(f"Error in future: {str(e)}")
        
        # Save final results
        self.save_results()
        
        # Sort results by problem_id to maintain consistent order
        with self.results_lock:
            self.results.sort(key=lambda x: x.get("problem_id", ""))
            
        return self.results
    
    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Run the pass@k experiment on a list of problems asynchronously.
        Problems run in parallel, but passes within each problem run sequentially.
        
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
        
        # Initialize dashboard with overall status
        if self.dashboard:
            self.dashboard.update_experiment_status({
                "total": total_problems,
                "completed": 0,
                "status": "Running with async processing",
                "config": self.config
            })
        
        # Process each problem asynchronously
        for i, problem in enumerate(problems):
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
            # Create a lock for this problem's dashboard updates if using the dashboard
            if self.dashboard and problem_id not in dashboard_locks:
                dashboard_locks[problem_id] = threading.Lock()
            
            # Update dashboard to show the problem is queued
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(
                        problem_id, "queued", problem.get("question", "")
                    )
            
            # Create a task for this problem
            task = asyncio.create_task(
                self._process_problem_with_semaphore(
                    semaphore, 
                    problem, 
                    self.config.get("pass_k_iterations", 8),
                    i, 
                    total_problems
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Sort results by problem ID or index for consistent output
        with self.results_lock:
            sorted_results = sorted(self.results, key=lambda x: x.get("problem_id", ""))
            self.results = sorted_results
        
        # Save final results
        self.save_results()
        
        return self.results
    
    async def _process_problem_with_semaphore(self, semaphore: asyncio.Semaphore, problem: Dict[str, Any], 
                                             k_value: int, index: int, total: int) -> Dict[str, Any]:
        """Process a problem with semaphore for concurrency control."""
        async with semaphore:
            # Handle different case variations of 'id' field
            problem_id = problem.get("id", problem.get("ID", str(index+1)))
            question = problem.get("question", "")
            
            logger.info(f"[{index+1}/{total}] Starting problem {problem_id}")
            
            # Update dashboard to show problem is in progress
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(problem_id, "in-progress", question)
                    self.dashboard.update_experiment_status({
                        "total": total,
                        "status": f"Processing problem {problem_id}"
                    })
            
            try:
                # Process the problem using the MODIFIED method that runs passes sequentially
                result = await self._process_problem_sequential_passes(problem, k_value)
                
                # Thread-safe update of results
                with self.results_lock:
                    self.results.append(result)
                
                # Update dashboard to show completion
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        status = "correct" if result.get("consensus_correct") else "incorrect"
                        self.dashboard.update_problem_status(problem_id, status, question)
                
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
                    "question": question,
                    "error": str(e),
                    "status": "error"
                }
                
                with self.results_lock:
                    self.results.append(error_result)
                
                # Update dashboard with error status
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.update_problem_status(problem_id, "error", question)
                
                # Save intermediate results
                if self.config.get("save_intermediate", True):
                    self.save_results()
                
                return error_result
    
    async def _process_problem_sequential_passes(self, problem: Dict[str, Any], k_value: int) -> Dict[str, Any]:
        """
        Process a single problem by generating k solutions SEQUENTIALLY.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            k_value: Number of solutions to generate
            
        Returns:
            Result dictionary with solutions and consensus information
        """
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        prompt = reasoning_template.replace("{question}", question)
        
        # Generate solutions sequentially
        solutions = []
        for solution_id in range(k_value):
            logger.info(f"Starting solution {solution_id} for problem {problem_id}")
            
            # Update dashboard to show solution is in progress
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = f"solution{solution_id}-in-progress"
                    self.dashboard.update_problem_status(problem_id, status)
            
            try:
                # Generate the solution using async method
                solution_data = await self._generate_solution_core(
                    solution_id,
                    problem_id,
                    prompt,
                    correct_answer
                )
                
                if solution_data:
                    solutions.append(solution_data)
                    
            except Exception as e:
                logger.error(f"Error generating solution {solution_id} for problem {problem_id}: {str(e)}")
                # Add an error placeholder
                solutions.append({
                    "solution_id": solution_id,
                    "reasoning": f"ERROR: {str(e)}",
                    "answer": None,
                    "correct": False,
                    "finish_reason": "error"
                })
        
        # Find consensus answer (most common answer)
        answers = [s["answer"] for s in solutions if s["answer"] is not None]
        consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
        
        # Check if consensus answer is correct
        consensus_correct = False
        if consensus_answer is not None:
            consensus_correct = consensus_answer.strip() == correct_answer.strip()
        
        # Calculate pass@k metrics
        num_correct = sum(1 for s in solutions if s["correct"])
        pass_at_k = num_correct > 0
        
        # Construct result
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "solutions": solutions,
            "consensus_answer": consensus_answer,
            "consensus_correct": consensus_correct,
            "consensus_count": consensus_count,
            "pass_at_k": pass_at_k,
            "num_correct": num_correct,
            "timestamp": time.time()
        }
        
        # Update dashboard with consensus information
        if self.dashboard:
            with dashboard_locks[problem_id]:
                # Add consensus information
                self.dashboard.update_answer_info(
                    problem_id,
                    consensus_answer or "No consensus",
                    correct_answer,
                    consensus_correct,
                    iteration=k_value  # Use k_value as special indicator for consensus
                )
        
        return result
    
    async def _process_problem_async(self, problem: Dict[str, Any], k_value: int) -> Dict[str, Any]:
        """
        Process a single problem by generating k solutions asynchronously.
        
        Args:
            problem: Problem dictionary with 'question' and 'answer' keys
            k_value: Number of solutions to generate
            
        Returns:
            Result dictionary with solutions and consensus information
        """
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        prompt = reasoning_template.replace("{question}", question)
        
        # Generate k solutions concurrently but with controlled concurrency
        # Use a semaphore to limit concurrent API calls to avoid rate limits
        api_semaphore = asyncio.Semaphore(min(5, k_value))  # Max 5 concurrent API calls
        
        # Create tasks for all solutions
        solution_tasks = []
        for solution_id in range(k_value):
            task = asyncio.create_task(
                self._generate_solution_async(
                    solution_id,
                    problem_id,
                    prompt,
                    correct_answer,
                    api_semaphore
                )
            )
            solution_tasks.append(task)
        
        # Await all solution tasks
        solutions = await asyncio.gather(*solution_tasks)
        
        # Filter out None results (in case of errors)
        solutions = [s for s in solutions if s is not None]
        
        # Sort solutions by solution_id to maintain order
        solutions.sort(key=lambda x: x["solution_id"])
        
        # Find consensus answer (most common answer)
        answers = [s["answer"] for s in solutions if s["answer"] is not None]
        consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
        
        # Check if consensus answer is correct
        consensus_correct = False
        if consensus_answer is not None:
            consensus_correct = consensus_answer.strip() == correct_answer.strip()
        
        # Calculate pass@k metrics
        num_correct = sum(1 for s in solutions if s["correct"])
        pass_at_k = num_correct > 0
        
        # Construct result
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "solutions": solutions,
            "consensus_answer": consensus_answer,
            "consensus_correct": consensus_correct,
            "consensus_count": consensus_count,
            "pass_at_k": pass_at_k,
            "num_correct": num_correct,
            "timestamp": time.time()
        }
        
        # Update dashboard with consensus information
        if self.dashboard:
            with dashboard_locks[problem_id]:
                # Add consensus information
                self.dashboard.update_answer_info(
                    problem_id,
                    consensus_answer or "No consensus",
                    correct_answer,
                    consensus_correct,
                    iteration=k_value  # Use k_value as special indicator for consensus
                )
        
        return result
    
    async def process_continuations(self, results_file: str = None):
        """
        Process continuations for solutions that were truncated due to length.
        
        Args:
            results_file: Path to results JSON file. If None, will use default path based on config.
        """
        # Determine results file path if not provided
        if results_file is None:
            results_dir = self.config.get("results_dir", "./results")
            experiment_name = self.config.get("experiment_name", "pass_k")
            results_file = os.path.join(results_dir, f"{experiment_name}_results.json")
        
        # Check if results file exists
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            return
        
        # Load existing results
        logger.info(f"Loading existing results from {results_file}")
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                
            # Check if results is in expected format
            if "results" not in data:
                logger.error("Invalid results file format: 'results' key not found")
                return
                
            existing_results = data["results"]
            logger.info(f"Loaded {len(existing_results)} results")
        except Exception as e:
            logger.error(f"Error loading results file: {str(e)}")
            return
        
        # Identify problems with truncated solutions
        problems_to_continue = []
        for result in existing_results:
            problem_id = result.get("problem_id", "unknown")
            
            # Check if any solution has finish_reason="length"
            has_truncated_solution = False
            truncated_solution_ids = []
            
            if "solutions" in result:
                for solution in result["solutions"]:
                    if solution.get("finish_reason") == "length":
                        has_truncated_solution = True
                        truncated_solution_ids.append(solution.get("solution_id", -1))
            
            if has_truncated_solution:
                logger.info(f"Problem {problem_id} has truncated solutions: {truncated_solution_ids}")
                problems_to_continue.append({
                    "problem_id": problem_id,
                    "question": result.get("question", ""),
                    "answer": result.get("correct_answer", ""),
                    "original_result": result,
                    "truncated_solution_ids": truncated_solution_ids
                })
        
        logger.info(f"Found {len(problems_to_continue)} problems with truncated solutions")
        
        if not problems_to_continue:
            logger.info("No truncated solutions found. Nothing to do.")
            return
        
        # Process continuations
        updated_results = []
        
        # Create a semaphore for concurrency control
        semaphore = asyncio.Semaphore(min(5, len(problems_to_continue)))
        
        # Process each problem with truncated solutions
        tasks = []
        for problem in problems_to_continue:
            task = asyncio.create_task(self._process_continuation_with_semaphore(
                semaphore,
                problem
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        updated_problems = await asyncio.gather(*tasks)
        
        # Replace original results with updated ones
        for updated_problem in updated_problems:
            if updated_problem:
                # Find the index of the original result
                for i, result in enumerate(existing_results):
                    if result.get("problem_id") == updated_problem.get("problem_id"):
                        existing_results[i] = updated_problem
                        break
        
        # Save updated results
        with open(results_file, "w") as f:
            json.dump({"results": existing_results}, f, indent=2)
        
        logger.info(f"Updated results saved to {results_file}")
        
        return existing_results

    async def _process_continuation_with_semaphore(self, semaphore, problem):
        """Process a continuation with semaphore for concurrency control."""
        async with semaphore:
            problem_id = problem["problem_id"]
            logger.info(f"Processing continuation for problem {problem_id}")
            
            try:
                # Get the original result
                original_result = problem["original_result"]
                question = problem["question"]
                correct_answer = problem["answer"]
                
                # Get reasoning prompt template
                reasoning_template = self.config.get("reasoning_prompt_template")
                if not reasoning_template:
                    raise ValueError("reasoning_prompt_template must be specified in configuration")
                
                # Replace the solutions that were truncated
                updated_solutions = []
                for solution in original_result["solutions"]:
                    solution_id = solution.get("solution_id", -1)
                    
                    # Check if this solution needs continuation
                    if solution.get("finish_reason") == "length":
                        logger.info(f"Continuing solution {solution_id} for problem {problem_id}")
                        
                        # Prepare prompt according to the specified format
                        base_prompt = reasoning_template.replace("{question}", question)
                        truncated_reasoning = solution.get("reasoning", "")
                        continuation_prompt = f"{base_prompt}\n\n<think>\n{truncated_reasoning}"
                        
                        # Print the entire prompt for debugging
                        logger.info(f"DEBUG - FULL PROMPT FOR PROBLEM {problem_id}, SOLUTION {solution_id}:")
                        logger.info("-" * 80)
                        #logger.info(continuation_prompt)
                        logger.info("-" * 80)
                        
                        # Generate a new solution with continuation enabled
                        try:
                            logger.info(f"Sending continuation request to model for problem {problem_id}, solution {solution_id}")
                            response = await self.reasoning_model.generate_response_async(
                                continuation_prompt,
                                max_tokens=self.config["max_tokens"],
                                temperature=self.config["temperature"],
                                top_p=self.config["top_p"],
                                top_k=self.config.get("top_k", 40),
                                presence_penalty=self.config["presence_penalty"],
                                frequency_penalty=self.config["frequency_penalty"],
                                verbose=self.verbose,
                                # Enable continuation
                                enable_continuation=True,
                                max_total_tokens=self.config.get("max_total_tokens", 24576),
                                max_continuations=self.config.get("max_continuations", 3)
                            )
                            
                            # Handle different response formats
                            if isinstance(response, tuple):
                                if len(response) >= 5:
                                    new_solution, finish_reason, token_usage, cost_info, detailed_api_calls = response
                                    
                                    # Log continuation details
                                    num_continuations = 0
                                    if detailed_api_calls:
                                        for call in detailed_api_calls:
                                            if "num_continuations" in call:
                                                num_continuations = call["num_continuations"]
                                                break
                                    
                                    if num_continuations > 0:
                                        logger.info(f"Solution {solution_id} for problem {problem_id} used {num_continuations} continuations")
                                elif len(response) >= 4:
                                    new_solution, finish_reason, token_usage, cost_info = response
                                else:
                                    new_solution = response[0] if len(response) > 0 else str(response)
                                    finish_reason = response[1] if len(response) > 1 else "unknown"
                            else:
                                new_solution = response
                                finish_reason = "unknown"
                                
                            # Print the model's output for debugging
                            logger.info(f"DEBUG - MODEL OUTPUT FOR PROBLEM {problem_id}, SOLUTION {solution_id}:")
                            logger.info("-" * 80)
                            logger.info(f"Finish reason: {finish_reason}")
                            logger.info(new_solution)
                            logger.info("-" * 80)
                            
                            # Extract answer from new solution
                            new_answer = extract_answer(new_solution)
                            
                            # Check if answer is correct
                            is_correct = False
                            if new_answer is not None:
                                is_correct = new_answer.strip() == correct_answer.strip()
                            
                            combined_solution = truncated_reasoning + new_solution
                            # Create updated solution
                            updated_solution = {
                                "solution_id": solution_id,
                                "reasoning": combined_solution,
                                "answer": new_answer,
                                "correct": is_correct,
                                "finish_reason": finish_reason,
                                "continued": True  # Mark this as a continued solution
                            }
                            
                            updated_solutions.append(updated_solution)
                            logger.info(f"Successfully continued solution {solution_id} for problem {problem_id}")
                            
                        except Exception as e:
                            logger.error(f"Error continuing solution {solution_id} for problem {problem_id}: {str(e)}")
                            # Keep original solution but mark the error
                            solution["continuation_error"] = str(e)
                            updated_solutions.append(solution)
                            
                    else:
                        # Keep original solution
                        updated_solutions.append(solution)
                
                # Find consensus answer (most common answer)
                answers = [s["answer"] for s in updated_solutions if s["answer"] is not None]
                consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
                
                # Check if consensus answer is correct
                consensus_correct = False
                if consensus_answer is not None:
                    consensus_correct = consensus_answer.strip() == correct_answer.strip()
                
                # Calculate pass@k metrics
                num_correct = sum(1 for s in updated_solutions if s["correct"])
                pass_at_k = num_correct > 0
                
                # Create updated result
                updated_result = {
                    "problem_id": problem_id,
                    "question": question,
                    "correct_answer": correct_answer,
                    "solutions": updated_solutions,
                    "consensus_answer": consensus_answer,
                    "consensus_correct": consensus_correct,
                    "consensus_count": consensus_count,
                    "pass_at_k": pass_at_k,
                    "num_correct": num_correct,
                    "timestamp": time.time(),
                    "continuation_processed": True
                }
                
                return updated_result
                
            except Exception as e:
                logger.error(f"Error processing continuation for problem {problem_id}: {str(e)}")
                return None
    
    async def _generate_solution_async(
        self, 
        solution_id: int, 
        problem_id: str, 
        prompt: str, 
        correct_answer: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Generate a single solution for a problem asynchronously."""
        # Use semaphore to control API concurrency
        async with semaphore:
            return await self._generate_solution_core(solution_id, problem_id, prompt, correct_answer)
    
    async def _generate_solution_core(
        self,
        solution_id: int,
        problem_id: str,
        prompt: str,
        correct_answer: str
    ) -> Dict[str, Any]:
        """Core solution generation logic with proper error handling."""
        
        # Add a small delay to stagger requests
        await asyncio.sleep(solution_id * 0.1)
        
        start_time = time.time()
        logger.debug(f"Starting async solution {solution_id} for problem {problem_id}")
        
        # Update dashboard status
        if self.dashboard:
            with dashboard_locks[problem_id]:
                status = f"solution{solution_id}-in-progress"
                self.dashboard.update_problem_status(problem_id, status)
        
        try:
            # Check if the model client supports async generation
            if hasattr(self.reasoning_model, 'generate_response_async'):
                # Use async generation
                response = await self.reasoning_model.generate_response_async(
                    prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k", 40),
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose, 
                    enable_continuation=self.config.get("enable_continuation", True),
                    max_total_tokens=self.config.get("max_total_tokens", 32768),
                    max_continuations=self.config.get("max_continuations", 3)
                )

                # Handle different response formats with robust unpacking
                if isinstance(response, tuple):
                    if len(response) >= 5:  # FireworksModelClient returns 5 elements
                        solution, finish_reason, token_usage, cost_info, _ = response
                    elif len(response) >= 4:  # Handle 4-element tuple
                        solution, finish_reason, token_usage, cost_info = response
                    elif len(response) >= 3:
                        solution, finish_reason, token_usage = response
                        cost_info = None
                    elif len(response) >= 2:
                        solution, finish_reason = response
                        token_usage = None
                        cost_info = None
                    else:
                        solution = response[0]
                        finish_reason = "unknown"
                        token_usage = None
                        cost_info = None
                else:
                    solution = response
                    finish_reason = "unknown"
                    token_usage = None
                    cost_info = None
            
            else:
                # Fallback for non-async models - should not be used with FireworksModelClient
                logger.warning(f"Model does not support async generation - this may cause errors with FireworksModelClient")
                solution = "ERROR: Model does not support async generation"
                finish_reason = "error"
                token_usage = None
                cost_info = None
            
            # Extract answer
            answer = extract_answer(solution)
            
            # Check if answer is correct
            is_correct = False
            if answer is not None:
                is_correct = answer.strip() == correct_answer.strip()
            
            # Update dashboard with answer information
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_answer_info(
                        problem_id,
                        answer or "No answer extracted",
                        correct_answer,
                        is_correct,
                        iteration=solution_id
                    )
                    
                    # Update solution status
                    status = f"solution{solution_id}-completed"
                    self.dashboard.update_problem_status(problem_id, status)
            
            end_time = time.time()
            logger.debug(f"Solution {solution_id} for problem {problem_id} completed in {end_time - start_time:.2f}s")
            
            # Return solution data
            return {
                "solution_id": solution_id,
                "reasoning": solution,
                "answer": answer,
                "correct": is_correct,
                "finish_reason": finish_reason
            }
            
        except Exception as e:
            logger.error(f"Error generating solution {solution_id} for problem {problem_id}: {str(e)}")
            
            # Update dashboard with error status
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = f"solution{solution_id}-error"
                    self.dashboard.update_problem_status(problem_id, status)
            
            # Return error solution
            return {
                "solution_id": solution_id,
                "reasoning": f"ERROR: {str(e)}",
                "answer": None,
                "correct": False,
                "finish_reason": "error"
            }
    
    async def _stream_solution_async(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
        """Stream model output for a solution asynchronously and update dashboard in real-time."""
        # This is the async version of _stream_solution
        full_response = ""
        finish_reason = "unknown"
        
        # Check if the model supports streaming
        if not hasattr(self.reasoning_model, 'generate_response') or not hasattr(self.reasoning_model.generate_response, 'stream'):
            # Fall back to non-streaming async generation
            try:
                if hasattr(self.reasoning_model, 'generate_response_async'):
                    response = await self.reasoning_model.generate_response_async(
                        prompt,
                        max_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
                        presence_penalty=self.config["presence_penalty"],
                        frequency_penalty=self.config["frequency_penalty"],
                        verbose=self.verbose
                    )
                    
                    if isinstance(response, tuple):
                        if len(response) >= 5:  # FireworksModelClient returns 5 elements
                            full_response, finish_reason, _, _, _ = response
                        elif len(response) >= 4:  # Handle 4-element tuple
                            full_response, finish_reason, _, _ = response
                        elif len(response) >= 3:
                            full_response, finish_reason, _ = response
                        elif len(response) >= 2:
                            full_response, finish_reason = response
                        else:
                            full_response = response[0]
                            finish_reason = "unknown"
                    else:
                        full_response = response
                else:
                    # Use sync generation in a thread executor
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.reasoning_model.generate_response(
                            prompt,
                            max_tokens=self.config["max_tokens"],
                            temperature=self.config["temperature"],
                            top_p=self.config["top_p"],
                            top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
                            presence_penalty=self.config["presence_penalty"],
                            frequency_penalty=self.config["frequency_penalty"],
                            verbose=self.verbose
                        )
                    )
                    
                    if isinstance(response, tuple):
                        if len(response) >= 4:
                            full_response, finish_reason, _, _ = response
                        else:
                            full_response = response[0] if isinstance(response, tuple) else response
                            finish_reason = "unknown"
                    else:
                        full_response = response
                
                # Update dashboard with full response at once
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.stream_model_output(problem_id, full_response, iteration=solution_id)
                        self.dashboard.stream_model_output(
                            problem_id, "", 
                            iteration=solution_id, 
                            finish_reason=finish_reason
                        )
            except Exception as e:
                logger.error(f"Error in async API call: {str(e)}")
                full_response = f"ERROR: {str(e)}"
                finish_reason = "error"
            
            return full_response, finish_reason
        
        # Handle streaming mode
        try:
            # Use synchronous streaming in a thread executor since we need to iterate through chunks
            loop = asyncio.get_event_loop()
            
            # Define the function to run in the executor
            def stream_in_thread():
                response_chunks = []
                stream = self.reasoning_model.generate_response(
                    prompt,
                    stream=True,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                )
                
                for chunk in stream:
                    response_chunks.append(chunk)
                    
                    # Stream to dashboard - thread-safe
                    if self.dashboard:
                        with dashboard_locks[problem_id]:
                            self.dashboard.stream_model_output(problem_id, chunk, iteration=solution_id)
                
                return "".join(response_chunks)
            
            # Run the streaming in a thread executor to avoid blocking the event loop
            full_response = await loop.run_in_executor(None, stream_in_thread)
            
            # Send final status update
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.stream_model_output(
                        problem_id, "", 
                        iteration=solution_id, 
                        finish_reason=finish_reason
                    )
            
        except Exception as e:
            logger.error(f"Error in async streaming API call: {str(e)}")
            full_response = f"ERROR: {str(e)}"
            finish_reason = "error"
            
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = f"solution{solution_id}-error"
                    self.dashboard.update_problem_status(problem_id, status)
        
        return full_response, finish_reason
    
    def _process_problem_wrapper(self, problem, k_value, index, total):
        """Thread-safe wrapper for _process_problem."""
        problem_id = problem.get("id", problem.get("ID", str(index+1)))
        question = problem.get("question", "")
        
        try:
            # Update dashboard to show problem is in progress
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(problem_id, "in-progress", question)
                    self.dashboard.update_experiment_status({
                        "total": total,
                        "status": f"Processing problem {problem_id}"
                    })
            
            # Process the problem
            result = self._process_problem(problem, k_value)
            
            # Update dashboard to show completion
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = "correct" if result.get("consensus_correct") else "incorrect"
                    self.dashboard.update_problem_status(problem_id, status, question)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {str(e)}")
            
            # Update dashboard with error status
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(problem_id, "error", question)
            
            # Return error result
            return {
                "problem_id": problem_id,
                "error": str(e),
                "status": "error"
            }
    
    def _find_consensus(self, answers: List[str]) -> Tuple[Optional[str], int]:
        """Find the most common answer from a list of answers."""
        if not answers:
            return None, 0
            
        # Normalize answers by stripping whitespace
        normalized_answers = [a.strip() for a in answers if a is not None]
        
        # Count occurrences
        answer_counts = {}
        for answer in normalized_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Find most common answer
        if not answer_counts:
            return None, 0
            
        most_common = max(answer_counts.items(), key=lambda x: x[1])
        return most_common[0], most_common[1]  # Return tuple of (answer, count)
    
    def _process_problem(self, problem: Dict[str, Any], k_value) -> Dict[str, Any]:
        """Process a single problem by generating k solutions in parallel."""
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Create reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        prompt = reasoning_template.replace("{question}", question)
        
        # Generate k solutions in parallel
        solutions = []
        
        # Create a partial function for solution generation
        generate_solution_func = partial(
            self._generate_solution, 
            problem_id=problem_id, 
            prompt=prompt, 
            correct_answer=correct_answer
        )
        
        # Use ThreadPoolExecutor to generate solutions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(k_value, self.max_workers)) as executor:
            # Submit all k solution tasks
            future_to_solution_id = {
                executor.submit(generate_solution_func, solution_id): solution_id 
                for solution_id in range(k_value)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_solution_id):
                solution_id = future_to_solution_id[future]
                try:
                    solution_data = future.result()
                    if solution_data:
                        solutions.append(solution_data)
                except Exception as e:
                    logger.error(f"Error generating solution {solution_id} for problem {problem_id}: {str(e)}")
                    # Add an error placeholder
                    solutions.append({
                        "solution_id": solution_id,
                        "reasoning": f"ERROR: {str(e)}",
                        "answer": None,
                        "correct": False,
                        "finish_reason": "error"
                    })
        
        # Sort solutions by solution_id to maintain order
        solutions.sort(key=lambda x: x["solution_id"])
        
        # Find consensus answer (most common answer)
        answers = [s["answer"] for s in solutions if s["answer"] is not None]
        consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
        
        # Check if consensus answer is correct
        consensus_correct = False
        if consensus_answer is not None:
            consensus_correct = consensus_answer.strip() == correct_answer.strip()
        
        # Calculate pass@k metrics
        num_correct = sum(1 for s in solutions if s["correct"])
        pass_at_k = num_correct > 0
        
        # Construct result
        result = {
            "problem_id": problem_id,
            "question": question,
            "correct_answer": correct_answer,
            "solutions": solutions,
            "consensus_answer": consensus_answer,
            "consensus_correct": consensus_correct,
            "consensus_count": consensus_count,
            "pass_at_k": pass_at_k,
            "num_correct": num_correct,
            "timestamp": time.time()
        }
        
        # Update dashboard with consensus information
        if self.dashboard:
            with dashboard_locks[problem_id]:
                # Add consensus information
                self.dashboard.update_answer_info(
                    problem_id,
                    consensus_answer or "No consensus",
                    correct_answer,
                    consensus_correct,
                    iteration=k_value  # Use k_value as special indicator for consensus
                )
        
        return result
    
    def _generate_solution(self, solution_id: int, problem_id: str, prompt: str, correct_answer: str) -> Dict[str, Any]:
        """Generate a single solution for a problem."""
        # Rate limiting - sleep a small amount to avoid hitting rate limits
        time.sleep(solution_id * 0.1)  # Stagger requests slightly
        
        start_time = time.time()
        logger.debug(f"Starting solution {solution_id} for problem {problem_id}")
        
        # Generate the solution
        if self.dashboard:
            # Stream solution generation
            with dashboard_locks[problem_id]:
                status = f"solution{solution_id}-in-progress"
                self.dashboard.update_problem_status(problem_id, status)
            
            solution, finish_reason = self._stream_solution(problem_id, prompt, solution_id)
        else:
            # Generate solution without streaming
            try:
                # For FireworksModelClient, we need to use asyncio
                if hasattr(self.reasoning_model, 'generate_response_async') and not hasattr(self.reasoning_model, 'generate_response'):
                    # Run the async method in a synchronous context
                    response = asyncio.run(self.reasoning_model.generate_response_async(
                        prompt,
                        max_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        top_k=self.config.get("top_k", 40),
                        presence_penalty=self.config["presence_penalty"],
                        frequency_penalty=self.config["frequency_penalty"],
                        verbose=self.verbose
                    ))
                else:
                    # Use the synchronous method
                    response = self.reasoning_model.generate_response(
                        prompt,
                        max_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                        top_k=self.config.get("top_k", 40),
                        presence_penalty=self.config["presence_penalty"],
                        frequency_penalty=self.config["frequency_penalty"],
                        verbose=self.verbose
                    )

                # Handle different response formats with robust unpacking
                if isinstance(response, tuple):
                    if len(response) >= 5:  # FireworksModelClient returns 5 elements
                        solution, finish_reason, _, _, _ = response
                    elif len(response) >= 4:  # Handle 4-element tuple
                        solution, finish_reason, _, _ = response
                    elif len(response) >= 3:
                        solution, finish_reason, _ = response
                    elif len(response) >= 2:
                        solution, finish_reason = response
                    else:
                        solution = response[0]
                        finish_reason = "unknown"
                else:
                    solution = response
                    finish_reason = "unknown"
            except Exception as e:
                logger.error(f"Error in API call: {str(e)}")
                solution = f"ERROR: {str(e)}"
                finish_reason = "error"
        
        # Extract answer
        answer = extract_answer(solution)
        
        # Check if answer is correct
        is_correct = False
        if answer is not None:
            is_correct = answer.strip() == correct_answer.strip()
        
        # Update dashboard with answer information
        if self.dashboard:
            with dashboard_locks[problem_id]:
                self.dashboard.update_answer_info(
                    problem_id,
                    answer or "No answer extracted",
                    correct_answer,
                    is_correct,
                    iteration=solution_id
                )
        
        end_time = time.time()
        logger.debug(f"Solution {solution_id} for problem {problem_id} completed in {end_time - start_time:.2f}s")
        
        # Return solution data
        return {
            "solution_id": solution_id,
            "reasoning": solution,
            "answer": answer,
            "correct": is_correct,
            "finish_reason": finish_reason
        }
    
    def _stream_solution(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
        """Stream model output for a solution and update dashboard in real-time."""
        full_response = ""
        finish_reason = "unknown"
        
        # Handle FireworksModelClient which doesn't support streaming
        if hasattr(self.reasoning_model, 'generate_response_async') and not hasattr(self.reasoning_model, 'generate_response'):
            try:
                # Run the async method in a synchronous context
                response = asyncio.run(self.reasoning_model.generate_response_async(
                    prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k", 40),
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                ))
                
                # Handle different response formats
                if isinstance(response, tuple):
                    if len(response) >= 5:  # FireworksModelClient returns 5 elements
                        full_response, finish_reason, _, _, _ = response
                    elif len(response) >= 4:  # Handle 4-element tuple
                        full_response, finish_reason, _, _ = response
                    else:
                        full_response = response[0] if len(response) > 0 else str(response)
                        finish_reason = response[1] if len(response) > 1 else "unknown"
                else:
                    full_response = response
                
                # Update dashboard with full response at once
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.stream_model_output(problem_id, full_response, iteration=solution_id)
                        self.dashboard.stream_model_output(
                            problem_id, "", 
                            iteration=solution_id, 
                            finish_reason=finish_reason
                        )
                        status = f"solution{solution_id}-completed"
                        self.dashboard.update_problem_status(problem_id, status)
                
                return full_response, finish_reason
            except Exception as e:
                logger.error(f"Error in async API call: {str(e)}")
                full_response = f"ERROR: {str(e)}"
                finish_reason = "error"
                
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        status = f"solution{solution_id}-error"
                        self.dashboard.update_problem_status(problem_id, status)
                
                return full_response, finish_reason
        
        # Get streaming response for models that support it
        try:
            stream = self.reasoning_model.generate_response(
                prompt,
                stream=True,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose
            )
            
            # Process each chunk
            for chunk in stream:
                full_response += chunk
                
                # Stream to dashboard - thread-safe
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.stream_model_output(problem_id, chunk, iteration=solution_id)
            
            # Skip the second API call for finish_reason - it's not worth the delay
            
            # Send final status update
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.stream_model_output(
                        problem_id, "", 
                        iteration=solution_id, 
                        finish_reason=finish_reason
                    )
                    status = f"solution{solution_id}-completed"
                    self.dashboard.update_problem_status(problem_id, status)
                    
        except Exception as e:
            logger.error(f"Error in streaming API call: {str(e)}")
            full_response = f"ERROR: {str(e)}"
            finish_reason = "error"
            
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = f"solution{solution_id}-error"
                    self.dashboard.update_problem_status(problem_id, status)
        
        return full_response, finish_reason
    

