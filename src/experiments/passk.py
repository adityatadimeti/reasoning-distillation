# import asyncio
# from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
# import time
# import logging
# import random
# import concurrent.futures
# import threading
# from functools import partial
# from threading import Lock

# from src.experiments.base import BaseExperiment
# from src.llm.model_factory import create_model_client
# from src.reasoning.extractor import extract_answer, extract_reasoning_trace
# from src.dashboard.server import DashboardServer


# logger = logging.getLogger(__name__)

# # Thread-safe locks for dashboard updates
# dashboard_locks = {}

# class PassExperiment(BaseExperiment):
#     """Experiment for evaluating pass@k performance of reasoning models."""
    
#     def __init__(
#         self, 
#         experiment_name: str, 
#         config: Dict[str, Any] = None,
#         dashboard: Optional[DashboardServer] = None,
#         verbose: bool = False
#     ):
#         """Initialize the pass@k experiment."""
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
        
#         # Initialize reasoning model with provider information if available
#         reasoning_provider = self.config.get("reasoning_model_provider", None)
#         self.reasoning_model = create_model_client(
#             self.config["reasoning_model"],
#             provider=reasoning_provider
#         )
        
#         # Set the parallelism settings
#         self.max_workers = min(self.config.get("max_parallel_workers", 10), 10)  # Default 10, max 10
#         self.rate_limit = self.config.get("api_rate_limit", 60)  # Calls per minute
        
#         # Thread-safe results list
#         self.results_lock = Lock()

#     def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Run the pass@k experiment on a list of problems with parallelism."""
#         total_problems = len(problems)
        
#         # Initialize dashboard with overall status
#         if self.dashboard:
#             self.dashboard.update_experiment_status({
#                 "total": total_problems,
#                 "completed": 0,
#                 "status": "Running with parallel processing",
#                 "config": self.config
#             })
        
#         # Process problems in parallel
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             # Submit all problems for processing
#             futures = []
#             for i, problem in enumerate(problems):
#                 problem_id = problem.get("id", problem.get("ID", str(i+1)))
                
#                 # Create a lock for this problem's dashboard updates
#                 dashboard_locks[problem_id] = threading.Lock()
                
#                 # Log the submission
#                 logger.info(f"Submitting problem {problem_id} ({i+1}/{total_problems}) for processing")
                
#                 # Update dashboard to show the problem is queued
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         self.dashboard.update_problem_status(
#                             problem_id, "queued", problem.get("question", "")
#                         )
                
#                 # Submit the problem for processing
#                 future = executor.submit(
#                     self._process_problem_wrapper,
#                     problem,
#                     self.config.get("pass_k_iterations", 8),
#                     i,
#                     total_problems
#                 )
#                 futures.append(future)
            
#             # Wait for all problems to complete
#             completed = 0
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     if result:
#                         with self.results_lock:
#                             self.results.append(result)
                    
#                     # Update completed count
#                     completed += 1
                    
#                     # Update dashboard with progress
#                     if self.dashboard:
#                         self.dashboard.update_experiment_status({
#                             "total": total_problems,
#                             "completed": completed,
#                             "status": f"Completed {completed}/{total_problems} problems"
#                         })
                    
#                     # Save intermediate results
#                     if self.config.get("save_intermediate", True) and completed % 5 == 0:
#                         self.save_results()
                        
#                 except Exception as e:
#                     logger.error(f"Error in future: {str(e)}")
        
#         # Save final results
#         self.save_results()
        
#         # Sort results by problem_id to maintain consistent order
#         with self.results_lock:
#             self.results.sort(key=lambda x: x.get("problem_id", ""))
            
#         return self.results
    
#     async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
#         """
#         Run the pass@k experiment on a list of problems asynchronously.
        
#         Args:
#             problems: List of problem dictionaries
#             max_concurrency: Maximum number of problems to process concurrently
            
#         Returns:
#             List of results for all problems
#         """
#         total_problems = len(problems)
#         logger.info(f"Processing {total_problems} problems with max concurrency of {max_concurrency}")
        
#         # Create a semaphore to limit concurrency
#         semaphore = asyncio.Semaphore(max_concurrency)
        
#         # Create a list to store tasks
#         tasks = []
        
#         # Initialize dashboard with overall status
#         if self.dashboard:
#             self.dashboard.update_experiment_status({
#                 "total": total_problems,
#                 "completed": 0,
#                 "status": "Running with async processing",
#                 "config": self.config
#             })
        
#         # Process each problem asynchronously
#         for i, problem in enumerate(problems):
#             # Handle different case variations of 'id' field
#             problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
#             # Create a lock for this problem's dashboard updates if using the dashboard
#             if self.dashboard and problem_id not in dashboard_locks:
#                 dashboard_locks[problem_id] = threading.Lock()
            
#             # Update dashboard to show the problem is queued
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.update_problem_status(
#                         problem_id, "queued", problem.get("question", "")
#                     )
            
#             # Create a task for this problem
#             task = asyncio.create_task(
#                 self._process_problem_with_semaphore(
#                     semaphore, 
#                     problem, 
#                     self.config.get("pass_k_iterations", 8),
#                     i, 
#                     total_problems
#                 )
#             )
#             tasks.append(task)
        
#         # Wait for all tasks to complete
#         await asyncio.gather(*tasks)
        
#         # Sort results by problem ID or index for consistent output
#         with self.results_lock:
#             sorted_results = sorted(self.results, key=lambda x: x.get("problem_id", ""))
#             self.results = sorted_results
        
#         # Save final results
#         self.save_results()
        
#         return self.results
    
#     async def _process_problem_with_semaphore(self, semaphore: asyncio.Semaphore, problem: Dict[str, Any], 
#                                              k_value: int, index: int, total: int) -> Dict[str, Any]:
#         """Process a problem with semaphore for concurrency control."""
#         async with semaphore:
#             # Handle different case variations of 'id' field
#             problem_id = problem.get("id", problem.get("ID", str(index+1)))
#             question = problem.get("question", "")
            
#             logger.info(f"[{index+1}/{total}] Starting problem {problem_id}")
            
#             # Update dashboard to show problem is in progress
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.update_problem_status(problem_id, "in-progress", question)
#                     self.dashboard.update_experiment_status({
#                         "total": total,
#                         "status": f"Processing problem {problem_id}"
#                     })
            
#             try:
#                 # Process the problem asynchronously
#                 result = await self._process_problem_async(problem, k_value)
                
#                 # Thread-safe update of results
#                 with self.results_lock:
#                     self.results.append(result)
                
#                 # Update dashboard to show completion
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         status = "correct" if result.get("consensus_correct") else "incorrect"
#                         self.dashboard.update_problem_status(problem_id, status, question)
                
#                 logger.info(f"[{index+1}/{total}] Completed problem {problem_id}")
                
#                 # Save intermediate results
#                 if self.config.get("save_intermediate", True):
#                     self.save_results()
                
#                 return result
                
#             except Exception as e:
#                 logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
#                 # Add error to results in a thread-safe way
#                 error_result = {
#                     "problem_id": problem_id,
#                     "question": question,
#                     "error": str(e),
#                     "status": "error"
#                 }
                
#                 with self.results_lock:
#                     self.results.append(error_result)
                
#                 # Update dashboard with error status
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         self.dashboard.update_problem_status(problem_id, "error", question)
                
#                 # Save intermediate results
#                 if self.config.get("save_intermediate", True):
#                     self.save_results()
                
#                 return error_result
    
#     async def _process_problem_async(self, problem: Dict[str, Any], k_value: int) -> Dict[str, Any]:
#         """
#         Process a single problem by generating k solutions asynchronously.
        
#         Args:
#             problem: Problem dictionary with 'question' and 'answer' keys
#             k_value: Number of solutions to generate
            
#         Returns:
#             Result dictionary with solutions and consensus information
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
        
#         # Generate k solutions concurrently but with controlled concurrency
#         # Use a semaphore to limit concurrent API calls to avoid rate limits
#         api_semaphore = asyncio.Semaphore(min(5, k_value))  # Max 5 concurrent API calls
        
#         # Create tasks for all solutions
#         solution_tasks = []
#         for solution_id in range(k_value):
#             task = asyncio.create_task(
#                 self._generate_solution_async(
#                     solution_id,
#                     problem_id,
#                     prompt,
#                     correct_answer,
#                     api_semaphore
#                 )
#             )
#             solution_tasks.append(task)
        
#         # Await all solution tasks
#         solutions = await asyncio.gather(*solution_tasks)
        
#         # Filter out None results (in case of errors)
#         solutions = [s for s in solutions if s is not None]
        
#         # Sort solutions by solution_id to maintain order
#         solutions.sort(key=lambda x: x["solution_id"])
        
#         # Find consensus answer (most common answer)
#         answers = [s["answer"] for s in solutions if s["answer"] is not None]
#         consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
        
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
#             with dashboard_locks[problem_id]:
#                 # Add consensus information
#                 self.dashboard.update_answer_info(
#                     problem_id,
#                     consensus_answer or "No consensus",
#                     correct_answer,
#                     consensus_correct,
#                     iteration=k_value  # Use k_value as special indicator for consensus
#                 )
        
#         return result
    
#     async def _generate_solution_async(
#         self, 
#         solution_id: int, 
#         problem_id: str, 
#         prompt: str, 
#         correct_answer: str,
#         semaphore: asyncio.Semaphore
#     ) -> Dict[str, Any]:
#         """Generate a single solution for a problem asynchronously."""
#         # Use semaphore to control API concurrency
#         async with semaphore:
#             # Add a small delay to stagger requests
#             await asyncio.sleep(solution_id * 0.1)
            
#             start_time = time.time()
#             logger.debug(f"Starting async solution {solution_id} for problem {problem_id}")
            
#             # Update dashboard status
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     status = f"solution{solution_id}-in-progress"
#                     self.dashboard.update_problem_status(problem_id, status)
            
#             try:
#                 # Check if the model client supports async generation
#                 if hasattr(self.reasoning_model, 'generate_response_async'):
#                     # Use async generation
#                     response = await self.reasoning_model.generate_response_async(
#                         prompt,
#                         max_tokens=self.config["max_tokens"],
#                         temperature=self.config["temperature"],
#                         top_p=self.config["top_p"],
#                         top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                         presence_penalty=self.config["presence_penalty"],
#                         frequency_penalty=self.config["frequency_penalty"],
#                         verbose=self.verbose
#                     )

#                     # Log the raw response
#                     logger.debug(f"Raw response type: {type(response)}")
#                     logger.debug(f"Raw response value: {response}")
                    


#                     # Handle both tuple and string responses
#                     if isinstance(response, tuple):
#                         solution, finish_reason, token_usage, cost_info = response
#                     else:
#                         solution = response
#                         finish_reason = "unknown"
#                         token_usage = None
#                         cost_info = None
                
#                 else:
#                     # Use sync generation in a thread executor
#                     loop = asyncio.get_event_loop()
#                     response = await loop.run_in_executor(
#                         None,
#                         lambda: self.reasoning_model.generate_response(
#                             prompt,
#                             max_tokens=self.config["max_tokens"],
#                             temperature=self.config["temperature"],
#                             top_p=self.config["top_p"],
#                             top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                             presence_penalty=self.config["presence_penalty"],
#                             frequency_penalty=self.config["frequency_penalty"],
#                             verbose=self.verbose
#                         )
#                     )

#                     # Log the raw response
#                     logger.debug(f"Raw response type: {type(response)}")
#                     logger.debug(f"Raw response value: {response}")
                    
#                     # Handle both tuple and string responses
#                     if isinstance(response, tuple):
#                         solution, finish_reason, token_usage, cost_info = response
#                     else:
#                         solution = response
#                         finish_reason = "unknown"
#                         token_usage = None
#                         cost_info = None
                
#                 # Extract answer
#                 answer = extract_answer(solution)
                
#                 # Check if answer is correct
#                 is_correct = False
#                 if answer is not None:
#                     is_correct = answer.strip() == correct_answer.strip()
                
#                 # Update dashboard with answer information
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         self.dashboard.update_answer_info(
#                             problem_id,
#                             answer or "No answer extracted",
#                             correct_answer,
#                             is_correct,
#                             iteration=solution_id
#                         )
                        
#                         # Update solution status
#                         status = f"solution{solution_id}-completed"
#                         self.dashboard.update_problem_status(problem_id, status)
                
#                 end_time = time.time()
#                 logger.debug(f"Solution {solution_id} for problem {problem_id} completed in {end_time - start_time:.2f}s")
                
#                 # Return solution data
#                 return {
#                     "solution_id": solution_id,
#                     "reasoning": solution,
#                     "answer": answer,
#                     "correct": is_correct,
#                     "finish_reason": finish_reason
#                 }
                
#             except Exception as e:
#                 logger.error(f"Error generating solution {solution_id} for problem {problem_id}: {str(e)}")
                
#                 # Update dashboard with error status
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         status = f"solution{solution_id}-error"
#                         self.dashboard.update_problem_status(problem_id, status)
                
#                 # Return error solution
#                 return {
#                     "solution_id": solution_id,
#                     "reasoning": f"ERROR: {str(e)}",
#                     "answer": None,
#                     "correct": False,
#                     "finish_reason": "error"
#                 }
    
#     async def _stream_solution_async(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
#         """Stream model output for a solution asynchronously and update dashboard in real-time."""
#         # This is the async version of _stream_solution
#         full_response = ""
#         finish_reason = "unknown"
        
#         # Check if the model supports streaming
#         if not hasattr(self.reasoning_model, 'generate_response') or not hasattr(self.reasoning_model.generate_response, 'stream'):
#             # Fall back to non-streaming async generation
#             try:
#                 if hasattr(self.reasoning_model, 'generate_response_async'):
#                     response = await self.reasoning_model.generate_response_async(
#                         prompt,
#                         max_tokens=self.config["max_tokens"],
#                         temperature=self.config["temperature"],
#                         top_p=self.config["top_p"],
#                         top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                         presence_penalty=self.config["presence_penalty"],
#                         frequency_penalty=self.config["frequency_penalty"],
#                         verbose=self.verbose
#                     )
                    
#                     if isinstance(response, tuple):
#                         full_response, finish_reason, _, _ = response
#                     else:
#                         full_response = response
#                 else:
#                     # Use sync generation in a thread executor
#                     loop = asyncio.get_event_loop()
#                     response = await loop.run_in_executor(
#                         None,
#                         lambda: self.reasoning_model.generate_response(
#                             prompt,
#                             max_tokens=self.config["max_tokens"],
#                             temperature=self.config["temperature"],
#                             top_p=self.config["top_p"],
#                             top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                             presence_penalty=self.config["presence_penalty"],
#                             frequency_penalty=self.config["frequency_penalty"],
#                             verbose=self.verbose
#                         )
#                     )
                    
#                     if isinstance(response, tuple):
#                         full_response, finish_reason, _, _ = response
#                     else:
#                         full_response = response
                
#                 # Update dashboard with full response at once
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         self.dashboard.stream_model_output(problem_id, full_response, iteration=solution_id)
#                         self.dashboard.stream_model_output(
#                             problem_id, "", 
#                             iteration=solution_id, 
#                             finish_reason=finish_reason
#                         )
#             except Exception as e:
#                 logger.error(f"Error in async API call: {str(e)}")
#                 full_response = f"ERROR: {str(e)}"
#                 finish_reason = "error"
            
#             return full_response, finish_reason
        
#         # Handle streaming mode
#         try:
#             # Use synchronous streaming in a thread executor since we need to iterate through chunks
#             loop = asyncio.get_event_loop()
            
#             # Define the function to run in the executor
#             def stream_in_thread():
#                 response_chunks = []
#                 stream = self.reasoning_model.generate_response(
#                     prompt,
#                     stream=True,
#                     max_tokens=self.config["max_tokens"],
#                     temperature=self.config["temperature"],
#                     top_p=self.config["top_p"],
#                     top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                     presence_penalty=self.config["presence_penalty"],
#                     frequency_penalty=self.config["frequency_penalty"],
#                     verbose=self.verbose
#                 )
                
#                 for chunk in stream:
#                     response_chunks.append(chunk)
                    
#                     # Stream to dashboard - thread-safe
#                     if self.dashboard:
#                         with dashboard_locks[problem_id]:
#                             self.dashboard.stream_model_output(problem_id, chunk, iteration=solution_id)
                
#                 return "".join(response_chunks)
            
#             # Run the streaming in a thread executor to avoid blocking the event loop
#             full_response = await loop.run_in_executor(None, stream_in_thread)
            
#             # Send final status update
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.stream_model_output(
#                         problem_id, "", 
#                         iteration=solution_id, 
#                         finish_reason=finish_reason
#                     )
            
#         except Exception as e:
#             logger.error(f"Error in async streaming API call: {str(e)}")
#             full_response = f"ERROR: {str(e)}"
#             finish_reason = "error"
            
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     status = f"solution{solution_id}-error"
#                     self.dashboard.update_problem_status(problem_id, status)
        
#         return full_response, finish_reason
    
#     def _process_problem_wrapper(self, problem, k_value, index, total):
#         """Thread-safe wrapper for _process_problem."""
#         problem_id = problem.get("id", problem.get("ID", str(index+1)))
#         question = problem.get("question", "")
        
#         try:
#             # Update dashboard to show problem is in progress
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.update_problem_status(problem_id, "in-progress", question)
#                     self.dashboard.update_experiment_status({
#                         "total": total,
#                         "status": f"Processing problem {problem_id}"
#                     })
            
#             # Process the problem
#             result = self._process_problem(problem, k_value)
            
#             # Update dashboard to show completion
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     status = "correct" if result.get("consensus_correct") else "incorrect"
#                     self.dashboard.update_problem_status(problem_id, status, question)
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing problem {problem_id}: {str(e)}")
            
#             # Update dashboard with error status
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.update_problem_status(problem_id, "error", question)
            
#             # Return error result
#             return {
#                 "problem_id": problem_id,
#                 "error": str(e),
#                 "status": "error"
#             }
    
#     def _find_consensus(self, answers: List[str]) -> Tuple[Optional[str], int]:
#         """Find the most common answer from a list of answers."""
#         if not answers:
#             return None, 0
            
#         # Normalize answers by stripping whitespace
#         normalized_answers = [a.strip() for a in answers if a is not None]
        
#         # Count occurrences
#         answer_counts = {}
#         for answer in normalized_answers:
#             answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
#         # Find most common answer
#         if not answer_counts:
#             return None, 0
            
#         most_common = max(answer_counts.items(), key=lambda x: x[1])
#         return most_common[0], most_common[1]  # Return tuple of (answer, count)
    
#     def _process_problem(self, problem: Dict[str, Any], k_value) -> Dict[str, Any]:
#         """Process a single problem by generating k solutions in parallel."""
#         # Handle different case variations of 'id' field
#         problem_id = problem.get("id", problem.get("ID", "unknown"))
#         question = problem["question"]
#         correct_answer = problem["answer"]
        
#         # Create reasoning prompt
#         reasoning_template = self.config.get("reasoning_prompt_template")
#         if not reasoning_template:
#             raise ValueError("reasoning_prompt_template must be specified in configuration")
        
#         prompt = reasoning_template.replace("{question}", question)
        
#         # Generate k solutions in parallel
#         solutions = []
        
#         # Create a partial function for solution generation
#         generate_solution_func = partial(
#             self._generate_solution, 
#             problem_id=problem_id, 
#             prompt=prompt, 
#             correct_answer=correct_answer
#         )
        
#         # Use ThreadPoolExecutor to generate solutions in parallel
#         with concurrent.futures.ThreadPoolExecutor(max_workers=min(k_value, self.max_workers)) as executor:
#             # Submit all k solution tasks
#             future_to_solution_id = {
#                 executor.submit(generate_solution_func, solution_id): solution_id 
#                 for solution_id in range(k_value)
#             }
            
#             # Process results as they complete
#             for future in concurrent.futures.as_completed(future_to_solution_id):
#                 solution_id = future_to_solution_id[future]
#                 try:
#                     solution_data = future.result()
#                     if solution_data:
#                         solutions.append(solution_data)
#                 except Exception as e:
#                     logger.error(f"Error generating solution {solution_id} for problem {problem_id}: {str(e)}")
#                     # Add an error placeholder
#                     solutions.append({
#                         "solution_id": solution_id,
#                         "reasoning": f"ERROR: {str(e)}",
#                         "answer": None,
#                         "correct": False,
#                         "finish_reason": "error"
#                     })
        
#         # Sort solutions by solution_id to maintain order
#         solutions.sort(key=lambda x: x["solution_id"])
        
#         # Find consensus answer (most common answer)
#         answers = [s["answer"] for s in solutions if s["answer"] is not None]
#         consensus_answer, consensus_count = self._find_consensus(answers) if answers else (None, 0)
        
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
#             with dashboard_locks[problem_id]:
#                 # Add consensus information
#                 self.dashboard.update_answer_info(
#                     problem_id,
#                     consensus_answer or "No consensus",
#                     correct_answer,
#                     consensus_correct,
#                     iteration=k_value  # Use k_value as special indicator for consensus
#                 )
        
#         return result
    
#     def _generate_solution(self, solution_id: int, problem_id: str, prompt: str, correct_answer: str) -> Dict[str, Any]:
#         """Generate a single solution for a problem."""
#         # Rate limiting - sleep a small amount to avoid hitting rate limits
#         time.sleep(solution_id * 0.1)  # Stagger requests slightly
        
#         start_time = time.time()
#         logger.debug(f"Starting solution {solution_id} for problem {problem_id}")
        
#         # Generate the solution
#         if self.dashboard:
#             # Stream solution generation
#             with dashboard_locks[problem_id]:
#                 status = f"solution{solution_id}-in-progress"
#                 self.dashboard.update_problem_status(problem_id, status)
            
#             solution, finish_reason = self._stream_solution(problem_id, prompt, solution_id)
#         else:
#             # Generate solution without streaming
#             try:
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
#                     # Handle 4-tuple response (solution, finish_reason, token_usage, cost_info)
#                     solution, finish_reason, _, _ = response
#                 else:
#                     solution = response
#                     finish_reason = "unknown"
#             except Exception as e:
#                 logger.error(f"Error in API call: {str(e)}")
#                 solution = f"ERROR: {str(e)}"
#                 finish_reason = "error"
        
#         # Extract answer
#         answer = extract_answer(solution)
        
#         # Check if answer is correct
#         is_correct = False
#         if answer is not None:
#             is_correct = answer.strip() == correct_answer.strip()
        
#         # Update dashboard with answer information
#         if self.dashboard:
#             with dashboard_locks[problem_id]:
#                 self.dashboard.update_answer_info(
#                     problem_id,
#                     answer or "No answer extracted",
#                     correct_answer,
#                     is_correct,
#                     iteration=solution_id
#                 )
        
#         end_time = time.time()
#         logger.debug(f"Solution {solution_id} for problem {problem_id} completed in {end_time - start_time:.2f}s")
        
#         # Return solution data
#         return {
#             "solution_id": solution_id,
#             "reasoning": solution,
#             "answer": answer,
#             "correct": is_correct,
#             "finish_reason": finish_reason
#         }
    
#     def _stream_solution(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
#         """Stream model output for a solution and update dashboard in real-time."""
#         full_response = ""
#         finish_reason = "unknown"
        
#         # Get streaming response
#         try:
#             stream = self.reasoning_model.generate_response(
#                 prompt,
#                 stream=True,
#                 max_tokens=self.config["max_tokens"],
#                 temperature=self.config["temperature"],
#                 top_p=self.config["top_p"],
#                 top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
#                 presence_penalty=self.config["presence_penalty"],
#                 frequency_penalty=self.config["frequency_penalty"],
#                 verbose=self.verbose
#             )
            
#             # Process each chunk
#             for chunk in stream:
#                 full_response += chunk
                
#                 # Stream to dashboard - thread-safe
#                 if self.dashboard:
#                     with dashboard_locks[problem_id]:
#                         self.dashboard.stream_model_output(problem_id, chunk, iteration=solution_id)
            
#             # Skip the second API call for finish_reason - it's not worth the delay
            
#             # Send final status update
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     self.dashboard.stream_model_output(
#                         problem_id, "", 
#                         iteration=solution_id, 
#                         finish_reason=finish_reason
#                     )
#                     status = f"solution{solution_id}-completed"
#                     self.dashboard.update_problem_status(problem_id, status)
                    
#         except Exception as e:
#             logger.error(f"Error in streaming API call: {str(e)}")
#             full_response = f"ERROR: {str(e)}"
#             finish_reason = "error"
            
#             if self.dashboard:
#                 with dashboard_locks[problem_id]:
#                     status = f"solution{solution_id}-error"
#                     self.dashboard.update_problem_status(problem_id, status)
        
#         return full_response, finish_reason
import asyncio
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import time
import logging
import threading
from threading import Lock

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
        
        # Store verbose flag - force to True to enable all logging
        self.verbose = True  # Always enable verbose logging
        
        # Configure logging to ensure everything gets output
        self._setup_logging()
        
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
        
        # Print experiment configuration
        print("\n===== PASS@K EXPERIMENT CONFIGURATION =====")
        for key, value in self.config.items():
            if key != "reasoning_prompt_template":  # Skip printing the full prompt template
                print(f"{key}: {value}")
        print("==========================================\n")
    
    def _setup_logging(self):
        """Set up logging to ensure all output is captured."""
        # Configure the root logger to output everything
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Make sure console handler exists and is configured properly
        console_handler_exists = False
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)
                console_handler_exists = True
        
        # Add a console handler if none exists
        if not console_handler_exists:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Ensure our logger is properly configured
        logger.setLevel(logging.INFO)
        
        logger.info("Logging configured for PassExperiment")

    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the pass@k experiment on a list of problems sequentially."""
        total_problems = len(problems)
        
        print("\n========== STARTING PASS@K EXPERIMENT ==========")
        print(f"Total problems to process: {total_problems}")
        print(f"Pass@k value: {self.config.get('pass_k_iterations', 8)}")
        print("==============================================\n")
        
        # Initialize dashboard with overall status
        if self.dashboard:
            self.dashboard.update_experiment_status({
                "total": total_problems,
                "completed": 0,
                "status": "Running with sequential processing",
                "config": self.config
            })
        
        # Process problems sequentially
        for i, problem in enumerate(problems):
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
            # Create a lock for this problem's dashboard updates
            dashboard_locks[problem_id] = threading.Lock()
            
            # Print problem information to console 
            print(f"\n\n========== PROBLEM {i+1}/{total_problems} (ID: {problem_id}) ==========")
            print(f"Question: {problem.get('question', '')}")
            print(f"Expected answer: {problem.get('answer', '')}")
            print("==============================================\n")
            
            # Log the processing
            logger.info(f"Processing problem {problem_id} ({i+1}/{total_problems})")
            
            # Update dashboard to show the problem is in progress
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(
                        problem_id, "in-progress", problem.get("question", "")
                    )
            
            # Process the problem
            try:
                result = self._process_problem(
                    problem,
                    self.config.get("pass_k_iterations", 8),
                    i,
                    total_problems
                )
                
                if result:
                    self.results.append(result)
                    
                    # Print consensus results to console
                    print(f"\n========== PROBLEM {problem_id} RESULTS ==========")
                    print(f"Consensus answer: {result.get('consensus_answer') or 'No consensus'}")
                    print(f"Consensus correct: {result.get('consensus_correct')}")
                    print(f"Correct solutions: {result.get('num_correct')}/{self.config.get('pass_k_iterations', 8)}")
                    print(f"Pass@{self.config.get('pass_k_iterations', 8)}: {result.get('pass_at_k')}")
                    print("==============================================\n")
                    
                    # Print a summary of all solutions
                    print("\nSolution summary:")
                    for s in result.get('solutions', []):
                        solution_id = s.get('solution_id', 'unknown')
                        answer = s.get('answer', 'No answer')
                        correct = "✓" if s.get('correct', False) else "✗"
                        print(f"  Solution {solution_id}: {answer} {correct}")
                    print()
                    
                    # Save results after each problem is completed
                    self.save_results()
                    print(f"Saved intermediate results to {self.results_file} after problem {problem_id}")
                    logger.info(f"Saved intermediate results after problem {problem_id}")
                
                # Update dashboard with progress
                if self.dashboard:
                    self.dashboard.update_experiment_status({
                        "total": total_problems,
                        "completed": i + 1,
                        "status": f"Completed {i + 1}/{total_problems} problems"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
                # Add error to results
                error_result = {
                    "problem_id": problem_id,
                    "question": problem.get("question", ""),
                    "error": str(e),
                    "status": "error"
                }
                
                self.results.append(error_result)
                
                # Update dashboard with error status
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.update_problem_status(problem_id, "error", problem.get("question", ""))
        
        # Print the overall results
        num_problems = len(self.results)
        num_passed = sum(1 for r in self.results if r.get('pass_at_k', False))
        pass_at_k_rate = num_passed / num_problems if num_problems > 0 else 0
        
        print("\n========== OVERALL EXPERIMENT RESULTS ==========")
        print(f"Total problems processed: {num_problems}")
        print(f"Problems passed at k={self.config.get('pass_k_iterations', 8)}: {num_passed}/{num_problems}")
        print(f"Pass@{self.config.get('pass_k_iterations', 8)} rate: {pass_at_k_rate:.2%}")
        print("==============================================\n")
        
        # Save final results
        self.save_results()
        print(f"Final results saved to {self.results_file}")
        
        # Sort results by problem_id to maintain consistent order
        self.results.sort(key=lambda x: x.get("problem_id", ""))
            
        return self.results
    
    async def run_parallel(self, problems: List[Dict[str, Any]], max_concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Run the pass@k experiment on a list of problems with parallel processing.
        Each problem is processed sequentially, but multiple problems are processed in parallel.
        
        Args:
            problems: List of problem dictionaries
            max_concurrency: Maximum number of problems to process concurrently
            
        Returns:
            List of results for all problems
        """
        total_problems = len(problems)
        logger.info(f"Processing {total_problems} problems with max concurrency of {max_concurrency}")
        
        print("\n========== STARTING PARALLEL PASS@K EXPERIMENT ==========")
        print(f"Total problems to process: {total_problems}")
        print(f"Pass@k value: {self.config.get('pass_k_iterations', 8)}")
        print(f"Max concurrency: {max_concurrency}")
        print("==============================================\n")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Create tasks for each problem
        tasks = []
        for i, problem in enumerate(problems):
            problem_id = problem.get("id", problem.get("ID", str(i+1)))
            
            # Create a lock for this problem's dashboard updates
            dashboard_locks[problem_id] = threading.Lock()
            
            # Create a task that processes the problem
            task = asyncio.create_task(self._process_problem_parallel(
                semaphore,
                problem,
                self.config.get("pass_k_iterations", 8),
                i,
                total_problems
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and add to the results list
        with self.results_lock:
            for result in results:
                if result:
                    self.results.append(result)
            
            # Sort results by problem_id to maintain consistent order
            self.results.sort(key=lambda x: x.get("problem_id", ""))
        
        # Print the overall results
        num_problems = len(self.results)
        num_passed = sum(1 for r in self.results if r.get('pass_at_k', False))
        pass_at_k_rate = num_passed / num_problems if num_problems > 0 else 0
        
        print("\n========== OVERALL EXPERIMENT RESULTS ==========")
        print(f"Total problems processed: {num_problems}")
        print(f"Problems passed at k={self.config.get('pass_k_iterations', 8)}: {num_passed}/{num_problems}")
        print(f"Pass@{self.config.get('pass_k_iterations', 8)} rate: {pass_at_k_rate:.2%}")
        print("==============================================\n")
        
        # Save final results
        self.save_results()
        print(f"Final results saved to {self.results_file}")
        
        return self.results
        
    async def _process_problem_parallel(
        self, 
        semaphore: asyncio.Semaphore, 
        problem: Dict[str, Any], 
        k_value: int, 
        index: int, 
        total: int
    ) -> Dict[str, Any]:
        """
        Process a single problem with semaphore for concurrency control.
        This method is called by run_parallel to process a problem while
        respecting the concurrency limits.
        """
        # Wait for the semaphore to be available
        async with semaphore:
            problem_id = problem.get("id", problem.get("ID", str(index+1)))
            
            # Print problem information to console 
            print(f"\n\n========== PROBLEM {index+1}/{total} (ID: {problem_id}) ==========")
            print(f"Question: {problem.get('question', '')}")
            print(f"Expected answer: {problem.get('answer', '')}")
            print(f"Processing on worker thread with semaphore value: {semaphore._value}")
            print("==============================================\n")
            
            # Log the processing
            logger.info(f"Processing problem {problem_id} ({index+1}/{total})")
            
            # Update dashboard to show the problem is in progress
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_problem_status(
                        problem_id, "in-progress", problem.get("question", "")
                    )
            
            try:
                # Process the problem using the sequential _process_problem method
                # We run this in an executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._process_problem,
                    problem,
                    k_value,
                    index,
                    total
                )
                
                if result:
                    # Print consensus results to console
                    print(f"\n========== PROBLEM {problem_id} RESULTS ==========")
                    print(f"Consensus answer: {result.get('consensus_answer') or 'No consensus'}")
                    print(f"Consensus correct: {result.get('consensus_correct')}")
                    print(f"Correct solutions: {result.get('num_correct')}/{k_value}")
                    print(f"Pass@{k_value}: {result.get('pass_at_k')}")
                    print("==============================================\n")
                    
                    # Print a summary of all solutions
                    print("\nSolution summary:")
                    for s in result.get('solutions', []):
                        solution_id = s.get('solution_id', 'unknown')
                        answer = s.get('answer', 'No answer')
                        correct = "✓" if s.get('correct', False) else "✗"
                        print(f"  Solution {solution_id}: {answer} {correct}")
                    print()
                    
                    # Save intermediate results after each problem
                    with self.results_lock:
                        self.results.append(result)
                        self.save_results()
                    print(f"Saved intermediate results to {self.results_file} after problem {problem_id}")
                    logger.info(f"Saved intermediate results after problem {problem_id}")
                
                # Update dashboard with progress
                if self.dashboard:
                    self.dashboard.update_experiment_status({
                        "total": total,
                        "status": f"Processed problem {problem_id}"
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
                # Add error to results
                error_result = {
                    "problem_id": problem_id,
                    "question": problem.get("question", ""),
                    "error": str(e),
                    "status": "error"
                }
                
                # Update dashboard with error status
                if self.dashboard:
                    with dashboard_locks[problem_id]:
                        self.dashboard.update_problem_status(problem_id, "error", problem.get("question", ""))
                
                return error_result
    
    def _process_problem(self, problem: Dict[str, Any], k_value: int, index: int, total: int) -> Dict[str, Any]:
        """Process a single problem by generating k solutions sequentially."""
        # Handle different case variations of 'id' field
        problem_id = problem.get("id", problem.get("ID", "unknown"))
        question = problem["question"]
        correct_answer = problem["answer"]
        
        # Log problem details
        logger.info(f"Processing problem {problem_id} ({index+1}/{total})")
        logger.info(f"Problem question: {question}")
        logger.info(f"Expected answer: {correct_answer}")
        
        # Create reasoning prompt
        reasoning_template = self.config.get("reasoning_prompt_template")
        if not reasoning_template:
            raise ValueError("reasoning_prompt_template must be specified in configuration")
        
        prompt = reasoning_template.replace("{question}", question)
        
        # Generate k solutions sequentially
        solutions = []
        
        logger.info(f"Generating {k_value} solutions for problem {problem_id}")
        
        for solution_id in range(k_value):
            logger.info(f"Starting solution {solution_id+1}/{k_value} for problem {problem_id}")
            
            # Generate each solution sequentially
            solution_data = self._generate_solution(
                solution_id,
                problem_id, 
                prompt, 
                correct_answer
            )
            
            if solution_data:
                solutions.append(solution_data)
                logger.info(f"Solution {solution_id+1}/{k_value} completed with answer: {solution_data.get('answer')}")
            else:
                # Add an error placeholder if generation failed
                logger.error(f"Solution {solution_id+1}/{k_value} failed to generate")
                solutions.append({
                    "solution_id": solution_id,
                    "reasoning": f"ERROR: Solution generation failed",
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
        
        # Log consensus and metrics information
        logger.info(f"Problem {problem_id} consensus results:")
        logger.info(f"  Consensus answer: {consensus_answer or 'No consensus'}")
        logger.info(f"  Consensus count: {consensus_count}/{k_value}")
        logger.info(f"  Correct answers: {num_correct}/{k_value}")
        logger.info(f"  Consensus correct: {consensus_correct}")
        logger.info(f"  Pass@{k_value}: {pass_at_k}")
        
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
        start_time = time.time()
        
        print(f"\n===== STARTING SOLUTION {solution_id}/{self.config.get('pass_k_iterations', 8)} FOR PROBLEM {problem_id} =====")
        question_extract = prompt.replace('{question}', '')
        question_preview = question_extract[:100] + "..." if len(question_extract) > 100 else question_extract
        print(f"Question: {question_preview}")
        print(f"Expected answer: {correct_answer}")
        
        logger.info(f"Starting solution {solution_id} for problem {problem_id}")
        
        # Show a progress indicator
        print(f"Generating solution... ", end="", flush=True)
        
        # Update dashboard to show solution is in progress
        if self.dashboard:
            with dashboard_locks[problem_id]:
                status = f"solution{solution_id}-in-progress"
                self.dashboard.update_problem_status(problem_id, status)
        
        # Generate the solution
        solution, finish_reason = self._stream_solution(problem_id, prompt, solution_id)
        
        # Extract answer
        answer = extract_answer(solution)
        
        # Check if answer is correct
        is_correct = False
        if answer is not None:
            is_correct = answer.strip() == correct_answer.strip()
        
        # Log the answer information
        correctness_str = "CORRECT" if is_correct else "INCORRECT"
        
        print(f"\n===== SOLUTION {solution_id} RESULT =====")
        print(f"Extracted answer: {answer}")
        print(f"Correctness: {correctness_str}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        print("=====================================\n")
        
        logger.info(f"Solution {solution_id} for problem {problem_id} answer: {answer} - {correctness_str}")
        
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
        logger.info(f"Solution {solution_id} for problem {problem_id} completed in {end_time - start_time:.2f}s")
        
        # Return solution data
        return {
            "solution_id": solution_id,
            "reasoning": solution,
            "answer": answer,
            "correct": is_correct,
            "finish_reason": finish_reason
        }
    
    def _stream_solution(self, problem_id: str, prompt: str, solution_id: int) -> Tuple[str, str]:
        """Generate a solution with progress indicators instead of token-by-token streaming."""
        full_response = ""
        finish_reason = "unknown"
        
        print(f"\n----- GENERATING SOLUTION FOR PROBLEM {problem_id}, SOLUTION {solution_id} -----")
        logger.info(f"Starting solution generation for solution {solution_id} of problem {problem_id}")
        
        # Track generation start time
        generation_start = time.time()
        
        # Get response from model
        try:
            # Don't stream token by token, just get full response
            response = self.reasoning_model.generate_response(
                prompt,
                stream=False,  # No streaming
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                top_k=self.config.get("top_k") if hasattr(self.reasoning_model, "top_k") else None,
                presence_penalty=self.config["presence_penalty"],
                frequency_penalty=self.config["frequency_penalty"],
                verbose=self.verbose
            )
            
            # Handle both tuple and string responses
            if isinstance(response, tuple):
                full_response, finish_reason, token_usage, cost_info = response
                if token_usage:
                    logger.info(f"Solution {solution_id} token usage: {token_usage}")
            else:
                full_response = response
                finish_reason = "unknown"
            
            # Print first few lines of the response to show progress
            print("\nResponse received. Preview of first few lines:")
            preview_lines = full_response.split('\n')[:5]  # Get first 5 lines
            for line in preview_lines:
                if line.strip():  # Only print non-empty lines
                    print(f"> {line}")
            if len(preview_lines) < len(full_response.split('\n')):
                print("> ...")  # Indicate there's more content
            
            # Send to dashboard if available
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    self.dashboard.update_model_output(problem_id, full_response, iteration=solution_id)
                    
        except Exception as e:
            logger.error(f"Error in API call for solution {solution_id} of problem {problem_id}: {str(e)}")
            full_response = f"ERROR: {str(e)}"
            finish_reason = "error"
            
            if self.dashboard:
                with dashboard_locks[problem_id]:
                    status = f"solution{solution_id}-error"
                    self.dashboard.update_problem_status(problem_id, status)
        
        # Print completion message with timing
        generation_time = time.time() - generation_start
        print(f"\n----- RESPONSE GENERATED ({generation_time:.2f}s) -----")
        
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