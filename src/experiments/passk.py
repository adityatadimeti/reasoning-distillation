# from typing import Dict, Any, List, Optional, Tuple
# import time
# import logging
# import random

# from src.experiments.base import BaseExperiment
# from src.llm.model_factory import create_model_client
# from src.reasoning.extractor import extract_answer, extract_reasoning_trace
# from src.reasoning.summarizer import summarize_reasoning
# from src.dashboard.server import DashboardServer


# logger = logging.getLogger(__name__)

# class PassExperiment(BaseExperiment):
#     """Experiment for evaluating pass@k performance of reasoning models."""
    
#     def __init__(
#         self, 
#         experiment_name: str = "test_pass_at_k", 
#         config: Dict[str, Any] = None,
#         dashboard: Optional[DashboardServer] = None,
#         verbose: bool = False
#     ):
#         """Initialize the pass@k experiment.
        
#         Args:
#             experiment_name: Name of the experiment
#             config: Configuration dictionary
#             dashboard: Optional dashboard server instance
#             verbose: Whether to log verbose output
#         """
#         super().__init__(experiment_name, config, dashboard)
        
#         # Store verbose flag
#         self.verbose = verbose
        
#         # Validate required parameters
#         required_params = [
#             "reasoning_model", "max_tokens", "temperature", 
#             "top_p", "presence_penalty", "frequency_penalty"
#         ]
#         for param in required_params:
#             if param not in self.config:
#                 raise ValueError(f"Required parameter '{param}' not found in configuration")
        
#         # Initialize reasoning model
#         self.reasoning_model = create_model_client(self.config["reasoning_model"])

#     def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Run the pass@k experiment on a list of problems.
        
#         Args:
#             problems: List of problem dictionaries
            
#         Returns:
#             List of result dictionaries
#         """
#         total_problems = len(problems)
#         k_value = self.config.get("pass_k_iterations", 8) # Defaults to 8 pass@k iterations if unspecified
        
#         for i, problem in enumerate(problems):
#             # Handle different case variations of 'id' field
#             problem_id = problem.get("id", problem.get("ID", str(i+1)))
#             question = problem.get("question", "")
            
#             logger.info(f"Processing problem {problem_id} ({i+1}/{total_problems})")
            
#             # Update dashboard
#             if self.dashboard:
#                 self.dashboard.update_problem_status(problem_id, "in-progress", question)
#                 self.dashboard.update_experiment_status({
#                     "total": total_problems,
#                     "completed": i,
#                     "status": f"Processing problem {problem_id}",
#                     "config": self.config
#                 })
            
#             try:
#                 result = self._process_problem(problem, k_value)
#                 self.results.append(result)
                
#                 # Update dashboard
#                 if self.dashboard:
#                     self.dashboard.update_problem_status(problem_id, "completed", question)
            
#             except Exception as e:
#                 logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
#                 # Update dashboard
#                 if self.dashboard:
#                     self.dashboard.update_problem_status(problem_id, "error", question)
                
#                 # Add error to results
#                 self.results.append({
#                     "problem_id": problem_id,
#                     "error": str(e),
#                     "status": "error"
#                 })
            
#             # Save intermediate results
#             if self.config.get("save_intermediate", True):
#                 self.save_results()
                
#         return self.results
    
#     def _process_problem(self, problem: Dict[str, Any], k_value) -> Dict[str, Any]:
#         """Process a single problem by generating k solutions.
        
#         Args:
#             problem: Problem dictionary with 'question' and 'answer' keys
            
#         Returns:
#             Result dictionary with k solutions and consensus
#         """
#         # Handle different case variations of 'id' field
#         problem_id = problem.get("id", problem.get("ID", "unknown"))
        
#         question = problem["question"]
#         correct_answer = problem["answer"]
        
#         # Create reasoning prompt
#         reasoning_template = self.config.get("reasoning_prompt_template")
#         if not reasoning_template:
#             raise ValueError("reasoning_prompt_template must be specified in configuration")
        
#         prompt = reasoning_template.replace("{question}", question)
        
#         solutions = []
#         any_correct = False
        
#         for k in range(k_value):
#             # Generate the solution
#             if self.dashboard:
#                 # Stream solution generation
#                 solution, finish_reason = self._stream_solution(problem_id, prompt, k)
#             else:
#                 # Generate solution without streaming
#                 response = self.reasoning_model.generate_response(
#                     prompt,
#                     max_tokens=self.config["max_tokens"],
#                     temperature=self.config["temperature"],
#                     top_p=self.config["top_p"],
#                     top_k=self.config.get("top_k"),
#                     presence_penalty=self.config["presence_penalty"],
#                     frequency_penalty=self.config["frequency_penalty"],
#                     verbose=self.verbose
#                 )
                
#                 # Handle both tuple and string responses
#                 if isinstance(response, tuple):
#                     solution, finish_reason = response
#                 else:
#                     solution = response
#                     finish_reason = "unknown"
            
#             # Extract answer
#             answer = extract_answer(solution)
            
#             # Check if answer is correct
#             is_correct = False
#             if answer is not None:
#                 is_correct = answer.strip() == correct_answer.strip()
#                 any_correct = any_correct or is_correct
            
#             # Add to solutions
#             solutions.append({
#                 "solution_id": k,
#                 "reasoning": solution,
#                 "answer": answer,
#                 "correct": is_correct,
#                 "finish_reason": finish_reason
#             })
            
#             # Update dashboard with answer information
#             if self.dashboard:
#                 self.dashboard.update_answer_info(
#                     problem_id,
#                     answer or "No answer extracted",
#                     correct_answer,
#                     is_correct,
#                     iteration=k
#                 )
        
#         # Find consensus answer (most common answer)
#         answers = [s["answer"] for s in solutions if s["answer"] is not None]
#         consensus_answer, consensus_count = self._find_consensus(answers) if answers else None
        
#         # Check if consensus answer is correct
#         consensus_correct = False
#         if consensus_answer is not None:
#             consensus_correct = consensus_answer.strip() == correct_answer.strip()
        
#         # Calculate pass@k metrics
#         num_correct = sum(1 for s in solutions if s["correct"])
#         pass_at_k = num_correct > 0
        
#         # Construct result
#         result = {
#             "problem_id": problem_id,
#             "question": question,
#             "correct_answer": correct_answer,
#             "solutions": solutions,
#             "consensus_answer": consensus_answer,
#             "consensus_correct": consensus_correct,
#             "consensus_count": consensus_count,
#             "pass_at_k": pass_at_k,
#             "num_correct": num_correct,
#             "timestamp": time.time()
#         }
        
#         # Update dashboard with consensus information
#         if self.dashboard:
#             final_status = "correct" if consensus_correct else "incorrect"
#             self.dashboard.update_problem_status(problem_id, final_status)
            
#             # Add consensus information
#             self.dashboard.update_answer_info(
#                 problem_id,
#                 consensus_answer or "No consensus",
#                 correct_answer,
#                 consensus_correct,
#                 iteration=k_value  # Use k_value as a special indicator for consensus
#             )
        
#         return result
    
#     def _stream_solution(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
#         """Stream model output for a solution and update dashboard in real-time.
        
#         Args:
#             problem_id: ID of the problem
#             prompt: The prompt to send to the model
#             solution_id: ID of the solution (0 to k-1)
            
#         Returns:
#             Tuple of (full_response, finish_reason)
#         """
#         full_response = ""
#         buffered_chunks = []
#         finish_reason = "unknown"
        
#         # Add debug logging
#         logger.debug(f"Streaming solution {solution_id} for problem ID: {problem_id}")
        
#         # Get the question for this problem
#         question = None
#         for result in self.results:
#             if result.get("problem_id") == problem_id:
#                 question = result.get("question", "")
#                 break
        
#         # Update the problem status
#         if self.dashboard:
#             status = f"solution{solution_id}-in-progress"
#             self.dashboard.update_problem_status(problem_id, status, question)
        
#         # Get streaming response
#         stream = self.reasoning_model.generate_response(
#             prompt,
#             stream=True,
#             max_tokens=self.config["max_tokens"],
#             temperature=self.config["temperature"],
#             top_p=self.config["top_p"],
#             top_k=self.config.get("top_k"),
#             presence_penalty=self.config["presence_penalty"],
#             frequency_penalty=self.config["frequency_penalty"],
#             verbose=self.verbose
#         )
        
#         # Process each chunk
#         for chunk in stream:
#             full_response += chunk
#             buffered_chunks.append(chunk)
            
#             # Stream to dashboard
#             if self.dashboard:
#                 self.dashboard.stream_model_output(problem_id, chunk, iteration=solution_id)
        
#         # Try to get finish_reason if available
#         if hasattr(self.reasoning_model, 'generate_completion') and 'fireworks' in str(self.reasoning_model.__class__).lower():
#             try:
#                 # Make a non-streaming call to get finish_reason
#                 logger.debug(f"Making non-streaming call to get finish_reason for problem {problem_id}, solution {solution_id}")
#                 response = self.reasoning_model.generate_response(
#                     prompt,
#                     stream=False,
#                     max_tokens=self.config["max_tokens"],
#                     temperature=self.config["temperature"],
#                     top_p=self.config["top_p"],
#                     top_k=self.config.get("top_k"),
#                     presence_penalty=self.config["presence_penalty"],
#                     frequency_penalty=self.config["frequency_penalty"],
#                     verbose=False
#                 )
                
#                 # Extract finish_reason
#                 if isinstance(response, tuple):
#                     _, finish_reason = response
#                     logger.debug(f"Got finish_reason '{finish_reason}' for problem {problem_id}, solution {solution_id}")
                
#             except Exception as e:
#                 logger.warning(f"Error getting finish_reason: {str(e)}. Using 'unknown' instead.")
        
#         # Send final status update
#         if self.dashboard:
#             self.dashboard.stream_model_output(problem_id, "", iteration=solution_id, finish_reason=finish_reason)
#             status = f"solution{solution_id}-completed"
#             self.dashboard.update_problem_status(problem_id, status, question)
        
#         return full_response, finish_reason
    
#     def _find_consensus(self, answers: List[str]) -> Optional[str]:
#         """Find the most common answer from a list of answers.
        
#         Args:
#             answers: List of answer strings
            
#         Returns:
#             Most common answer or None if no answers
#         """
#         if not answers:
#             return None
            
#         # Normalize answers by stripping whitespace
#         normalized_answers = [a.strip() for a in answers if a is not None]
        
#         # Count occurrences
#         answer_counts = {}
#         for answer in normalized_answers:
#             answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
#         # Find most common answer
#         if not answer_counts:
#             return None
            
#         most_common = max(answer_counts.items(), key=lambda x: x[1])
#         return most_common[0], max(answer_counts.values()) # also return the count of the most common answer


from typing import Dict, Any, List, Optional, Tuple
import time
import logging
import random
import concurrent.futures
import threading
from functools import partial

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
        experiment_name: str = "test_pass_at_k", 
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
        
        # Initialize reasoning model
        self.reasoning_model = create_model_client(self.config["reasoning_model"])
        
        # Set the parallelism settings
        self.max_workers = min(self.config.get("max_parallel_workers", 10), 10)  # Default 10, max 10
        self.rate_limit = self.config.get("api_rate_limit", 60)  # Calls per minute
        
        # Thread-safe results list
        self.results_lock = threading.Lock()

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
                response = self.reasoning_model.generate_response(
                    prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config.get("top_k"),
                    presence_penalty=self.config["presence_penalty"],
                    frequency_penalty=self.config["frequency_penalty"],
                    verbose=self.verbose
                )
                
                # Handle both tuple and string responses
                if isinstance(response, tuple):
                    solution, finish_reason = response
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
        
        # Get streaming response
        try:
            stream = self.reasoning_model.generate_response(
                prompt,
                stream=True,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config.get("top_k"),
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