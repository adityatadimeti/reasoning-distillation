"""
Recursive pipeline for iterative reasoning refinement based on summaries.
"""
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import re
import hashlib

from src.utils.config import Config
from src.models.base import Model
from src.models.factory import create_reasoning_model, create_summarization_model
from src.data.dataset import Dataset
from src.pipeline.base_pipeline import BasePipeline
from src.pipeline.summary_pipeline import SummaryReasoningPipeline
import src.reasoning.generation as generation
import src.reasoning.extraction as extraction
from src.summarization.summarizers import get_summarizer

logger = logging.getLogger(__name__)

class RecursiveReasoningPipeline(BasePipeline):
    """
    Pipeline that recursively refines reasoning through targeted summarization and error correction.
    
    This pipeline builds on the summary pipeline but adds a more sophisticated evaluation mechanism
    that decides whether to continue refining the reasoning based on error detection and confidence.
    """
    def __init__(self, config: Config):
        """
        Initialize the recursive pipeline.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Set name if not already set
        if self.name == "unknown_pipeline":
            self.name = "recursive"
        
        # Initialize models
        self.reasoning_model = create_reasoning_model(config)
        self.summarization_model = create_summarization_model(config)
        
        # Initialize summarizer
        self.summarizer = get_summarizer(config, self.reasoning_model, self.summarization_model)
        
        # Initialize evaluator
        self.evaluator_model = self.summarization_model or self.reasoning_model
        
        # Pipeline parameters
        pipeline_config = config.get("pipeline", {})
        self.batch_size = pipeline_config.get("batch_size", 4)
        self.max_iterations = pipeline_config.get("max_iterations", 3)
        self.early_stopping = pipeline_config.get("early_stopping", True)
        self.confidence_threshold = pipeline_config.get("confidence_threshold", 0.8)
        
        # Prompts for different steps
        self.prompts = {
            "continuation": pipeline_config.get("continuation_prompt", 
                "Based on the summary of your previous reasoning, please continue solving the problem. "
                "Focus specifically on addressing the issues identified in the summary."
            ),
            "evaluation": pipeline_config.get("evaluation_prompt",
                "Evaluate the above reasoning and answer. Is it correct? If there are errors, "
                "what are they? Should the reasoning be refined further?"
            ),
            "synthesis": pipeline_config.get("synthesis_prompt",
                "Based on all the iterations of reasoning so far, synthesize a final answer to the problem."
            )
        }
        
        # Summarization parameters
        summarization_config = config.get("summarization", {})
        self.method = summarization_config.get("method", "self")
        
        # Recursive pipeline specific settings
        recursive_config = config.get("recursive", {})
        self.use_think_tags = recursive_config.get("use_think_tags", True)
        self.use_summarize_tags = recursive_config.get("use_summarize_tags", False)
        self.track_confidence = recursive_config.get("track_confidence", True)
        self.refine_errors_only = recursive_config.get("refine_errors_only", True)
        
        # Initialize dashboard logging if enabled
        self.dashboard_enabled = os.environ.get("ENABLE_EXPERIMENT_DASHBOARD") == "1"
        if self.dashboard_enabled:
            self.dashboard_data_path = os.environ.get("DASHBOARD_DATA_PATH", "dashboard/experiment_data.json")
            self._init_dashboard_data()
            logger.info(f"Dashboard logging enabled. Data will be saved to {self.dashboard_data_path}")
        
        logger.info(f"Initialized {self.name} pipeline with reasoning model {self.reasoning_model.model_name}")
        logger.info(f"Summarization method: {self.method}")
        logger.info(f"Max iterations: {self.max_iterations}, Early stopping: {self.early_stopping}")
        
        if self.summarization_model:
            logger.info(f"Using external summarization model: {self.summarization_model.model_name}")
        else:
            logger.info("Using self-summarization")
    
    def _init_dashboard_data(self):
        """Initialize the dashboard data file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.dashboard_data_path), exist_ok=True)
        
        # Initialize dashboard data
        dashboard_data = {
            "status": "initialized",
            "problems": {},
            "current_problem": None,
            "api_calls": [],
            "start_time": time.time(),
            "last_update": time.time()
        }
        
        # Save to file
        self._save_dashboard_data(dashboard_data)
    
    def _save_dashboard_data(self, data=None):
        """Save dashboard data to file."""
        if not self.dashboard_enabled:
            return
            
        try:
            # If data is not provided, read from file first
            if data is None:
                if os.path.exists(self.dashboard_data_path):
                    with open(self.dashboard_data_path, 'r') as f:
                        data = json.load(f)
                else:
                    # Initialize with default data
                    data = {
                        "status": "initialized",
                        "problems": {},
                        "current_problem": None,
                        "api_calls": [],
                        "start_time": time.time(),
                        "last_update": time.time()
                    }
            
            # Update last_update timestamp
            data["last_update"] = time.time()
            
            # Save to file
            with open(self.dashboard_data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dashboard data: {str(e)}")
    
    def _update_dashboard_status(self, status):
        """Update the status in the dashboard data."""
        if not self.dashboard_enabled:
            return
            
        try:
            # Read current data
            if os.path.exists(self.dashboard_data_path):
                with open(self.dashboard_data_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "status": "initialized",
                    "problems": {},
                    "current_problem": None,
                    "api_calls": [],
                    "start_time": time.time(),
                    "last_update": time.time()
                }
            
            # Update status
            data["status"] = status
            
            # Save updated data
            self._save_dashboard_data(data)
        except Exception as e:
            logger.error(f"Error updating dashboard status: {str(e)}")
    
    def _log_problem_update(self, problem_id, status=None, question=None, iteration=None, iteration_data=None):
        """Log problem update to dashboard."""
        if not self.dashboard_enabled:
            return
            
        try:
            # Read current data
            if os.path.exists(self.dashboard_data_path):
                with open(self.dashboard_data_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "status": "initialized",
                    "problems": {},
                    "current_problem": None,
                    "api_calls": [],
                    "start_time": time.time(),
                    "last_update": time.time()
                }
            
            # Initialize problem if it doesn't exist
            if problem_id not in data["problems"]:
                data["problems"][problem_id] = {
                    "status": "waiting",
                    "iterations": {},
                    "start_time": time.time()
                }
            
            # Update problem data
            if status:
                data["problems"][problem_id]["status"] = status
            
            if question:
                data["problems"][problem_id]["question"] = question
            
            if iteration is not None and iteration_data:
                data["problems"][problem_id]["iterations"][str(iteration)] = iteration_data
            
            # Update current problem
            data["current_problem"] = problem_id
            
            # Save updated data
            self._save_dashboard_data(data)
        except Exception as e:
            logger.error(f"Error logging problem update: {str(e)}")
    
    def _log_api_call(self, endpoint, payload, response):
        """Log API call to dashboard."""
        if not self.dashboard_enabled:
            return
            
        try:
            # Read current data
            if os.path.exists(self.dashboard_data_path):
                with open(self.dashboard_data_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    "status": "initialized",
                    "problems": {},
                    "current_problem": None,
                    "api_calls": [],
                    "start_time": time.time(),
                    "last_update": time.time()
                }
            
            # Add API call
            data["api_calls"].append({
                "timestamp": time.time(),
                "endpoint": endpoint,
                "payload": payload,
                "response": response
            })
            
            # Save updated data
            self._save_dashboard_data(data)
        except Exception as e:
            logger.error(f"Error logging API call: {str(e)}")
    
    def run(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the recursive pipeline on a dataset.
        
        Args:
            dataset: Dataset to process
            **kwargs: Additional arguments
                - split: Data split to use (train, test, all)
                - max_problems: Maximum number of problems to process
                - problem_ids: List of specific problem IDs to process
                - max_iterations: Maximum number of iterations
                
        Returns:
            Dictionary with pipeline results
        """
        # Update dashboard status
        if self.dashboard_enabled:
            self._update_dashboard_status("running")
            
        # Get parameters
        split = kwargs.get("split", "test")
        max_problems = kwargs.get("max_problems", None)
        problem_ids = kwargs.get("problem_ids", None)
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        
        # Get problems to process
        if problem_ids:
            # If specific problem IDs are provided, get those problems
            if isinstance(problem_ids, str):
                # Handle single problem ID as a string
                problem_ids = [problem_ids]
                
            problems = []
            for pid in problem_ids:
                try:
                    problem = dataset.get_problem_by_id(pid)
                    problems.append(problem)
                except ValueError as e:
                    logger.warning(f"Problem ID {pid} not found: {str(e)}")
            
            logger.info(f"Processing {len(problems)} specific problems by ID")
        else:
            # Otherwise, get all problems from the specified split
            problems = dataset.get_problems(split=split)
            
            # Limit the number of problems if specified
            if max_problems is not None and len(problems) > max_problems:
                logger.info(f"Processing {max_problems} problems from {split} split")
                problems = problems[:max_problems]
            else:
                logger.info(f"Processing {len(problems)} problems from {split} split")
        
        # Process each problem
        results = []
        start_time = time.time()
        
        for i, problem in enumerate(problems):
            problem_id = problem["id"]
            question = problem["question"]
            expected_answer = problem.get("ground_truth", "")
            
            logger.info(f"Processing problem {i+1}/{len(problems)}: {problem_id}")
            
            try:
                # Process the problem
                result = self._process_problem(question, max_iterations)
                
                # Add problem metadata
                result["problem_id"] = problem_id
                result["expected_answer"] = expected_answer
                
                # Add to results
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing problem {problem_id}: {str(e)}")
                
                # Log to dashboard
                if self.dashboard_enabled:
                    self._log_problem_update(
                        problem_id=problem_id,
                        status="error",
                        question=question
                    )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile pipeline results
        pipeline_results = {
            "pipeline": self.name,
            "timestamp": int(time.time()),
            "num_problems": len(results),
            "total_time": total_time,
            "average_time_per_problem": total_time / len(results) if results else 0,
            "max_iterations": max_iterations,
            "summarization_method": self.method,
            "results": results
        }
        
        # Update dashboard status
        if self.dashboard_enabled:
            self._update_dashboard_status("completed")
        
        return pipeline_results
    
    def _process_problem(self, question: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Process a single problem using recursive reasoning.
        
        Args:
            question: The question to solve
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary with problem results
        """
        start_time = time.time()
        problem_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        # Log problem start to dashboard
        if self.dashboard_enabled:
            self._log_problem_update(
                problem_id=problem_id,
                status="processing",
                question=question
            )
        
        # Initial reasoning generation
        logger.info("Generating initial reasoning")
        initial_result = self.reasoning_model.generate_reasoning(question=question)
        
        # Log API call to dashboard
        if self.dashboard_enabled:
            self._log_api_call(
                endpoint="generate_reasoning",
                payload={"question": question},
                response=initial_result
            )
        
        # Extract initial answer
        initial_answer = extraction.extract_answer_from_reasoning_result(initial_result)
        initial_normalized = extraction.normalize_answer(initial_answer)
        
        # Store all iterations
        iterations = [
            {
                "iteration": 0,
                "reasoning": initial_result["reasoning"],
                "answer": initial_answer,
                "extracted_answer": initial_answer,
                "normalized_answer": initial_normalized,
                "confidence": None,
                "errors_detected": None,
                "should_continue": None
            }
        ]
        
        # Log initial iteration to dashboard
        if self.dashboard_enabled:
            self._log_problem_update(
                problem_id=problem_id,
                iteration=0,
                iteration_data={
                    "reasoning": initial_result["reasoning"],
                    "answer": initial_answer,
                    "timestamp": time.time()
                }
            )
        
        # Track the current reasoning and answer
        current_reasoning = initial_result["reasoning"]
        current_answer = initial_answer
        current_normalized = initial_normalized
        
        # Track if we should stop early
        should_stop = False
        
        # Continue iterating with summarized reasoning
        for i in range(max_iterations):
            if should_stop:
                logger.info(f"Early stopping at iteration {i}")
                break
                
            logger.info(f"Iteration {i+1}: Evaluating reasoning")
            
            # Evaluate the current reasoning and answer
            evaluation_result = self._evaluate_reasoning(question, current_reasoning, current_answer)
            
            # Log API call to dashboard
            if self.dashboard_enabled:
                self._log_api_call(
                    endpoint="evaluate_reasoning",
                    payload={
                        "question": question,
                        "reasoning": current_reasoning,
                        "answer": current_answer
                    },
                    response=evaluation_result
                )
            
            # Extract evaluation metrics
            confidence = evaluation_result.get("confidence", 0.0)
            errors_detected = evaluation_result.get("errors_detected", True)
            should_continue = evaluation_result.get("should_continue", True)
            
            # Check if we should stop refining
            if self.early_stopping and (confidence >= self.confidence_threshold or not should_continue):
                should_stop = True
                logger.info(f"Early stopping triggered: confidence={confidence}, should_continue={should_continue}")
                # Still summarize the reasoning to provide a final summary
            
            # Summarize the current reasoning with focus on errors
            logger.info(f"Iteration {i+1}: Summarizing reasoning")
            summary = self.summarizer.summarize(current_reasoning, strategy="error_focused")
            
            # Log API call to dashboard
            if self.dashboard_enabled:
                self._log_api_call(
                    endpoint="summarize",
                    payload={
                        "reasoning": current_reasoning,
                        "strategy": "error_focused"
                    },
                    response={"summary": summary}
                )
            
            # If no errors and refine_errors_only is True, skip to next iteration
            if not errors_detected and self.refine_errors_only and not should_stop:
                logger.info(f"No errors detected, skipping iteration {i+1}")
                
                # Store this iteration but don't generate new reasoning
                iterations.append({
                    "iteration": i + 1,
                    "summary": summary,
                    "reasoning": current_reasoning,  # Same as previous
                    "answer": current_answer,
                    "extracted_answer": current_answer,
                    "normalized_answer": current_normalized,
                    "confidence": confidence,
                    "errors_detected": errors_detected,
                    "should_continue": should_continue
                })
                
                # Log iteration to dashboard
                if self.dashboard_enabled:
                    self._log_problem_update(
                        problem_id=problem_id,
                        iteration=i+1,
                        iteration_data={
                            "summary": summary,
                            "reasoning": current_reasoning,
                            "answer": current_answer,
                            "confidence": confidence,
                            "errors_detected": errors_detected,
                            "should_continue": should_continue,
                            "timestamp": time.time()
                        }
                    )
                
                continue
            
            # If should_stop is True, we've already decided to stop iterating
            if not should_stop:
                # Generate new reasoning based on the summary and evaluation
                logger.info(f"Iteration {i+1}: Generating new reasoning based on summary")
                
                # Format the question with summary for continuation
                if self.use_summarize_tags:
                    # Use experimental summarize tags
                    continuation_prompt = f"{question}\n\n<think>\n{current_reasoning}\n\n<summarize>{summary}</summarize>\n\n{self.prompts['continuation']}"
                else:
                    # Use standard format
                    continuation_prompt = f"{question}\n\nHere is a summary of your previous reasoning:\n{summary}\n\n{self.prompts['continuation']}"
                
                # Generate new reasoning based on the summary
                new_result = self.reasoning_model.generate_reasoning(question=continuation_prompt)
                
                # Log API call to dashboard
                if self.dashboard_enabled:
                    self._log_api_call(
                        endpoint="generate_reasoning",
                        payload={"question": continuation_prompt},
                        response=new_result
                    )
                
                # Extract new answer
                new_answer = extraction.extract_answer_from_reasoning_result(new_result)
                new_normalized = extraction.normalize_answer(new_answer)
                
                # Update current reasoning and answer
                current_reasoning = new_result["reasoning"]
                current_answer = new_answer
                current_normalized = new_normalized
            
            # Store this iteration
            iterations.append({
                "iteration": i + 1,
                "summary": summary,
                "reasoning": current_reasoning,
                "answer": current_answer,
                "extracted_answer": current_answer,
                "normalized_answer": current_normalized,
                "confidence": confidence,
                "errors_detected": errors_detected,
                "should_continue": should_continue
            })
            
            # Log iteration to dashboard
            if self.dashboard_enabled:
                self._log_problem_update(
                    problem_id=problem_id,
                    iteration=i+1,
                    iteration_data={
                        "summary": summary,
                        "reasoning": current_reasoning,
                        "answer": current_answer,
                        "confidence": confidence,
                        "errors_detected": errors_detected,
                        "should_continue": should_continue,
                        "timestamp": time.time()
                    }
                )
        
        # Final synthesis to determine best answer
        logger.info("Synthesizing final answer")
        final_result = self._synthesize_final_answer(question, iterations)
        
        # Log API call to dashboard
        if self.dashboard_enabled:
            self._log_api_call(
                endpoint="synthesize_final_answer",
                payload={
                    "question": question,
                    "iterations": iterations
                },
                response=final_result
            )
        
        # Use synthesized answer if available, otherwise use the last iteration's answer
        final_answer = final_result.get("answer") or current_answer
        
        # Build and return the result
        processing_time = time.time() - start_time
        
        result = {
            "question": question,
            "iterations": iterations,
            "final_reasoning": final_result.get("reasoning", current_reasoning),
            "synthesized_answer": final_result.get("answer"),
            "extracted_answer": final_answer,
            "processing_time": processing_time,
            "iteration_count": len(iterations) - 1  # Exclude initial iteration
        }
        
        # Log problem completion to dashboard
        if self.dashboard_enabled:
            self._log_problem_update(
                problem_id=problem_id,
                status="completed",
                iteration_data={
                    "final_answer": final_answer,
                    "processing_time": processing_time,
                    "iteration_count": len(iterations) - 1
                }
            )
        
        return result
    
    def _evaluate_reasoning(self, question: str, reasoning: str, answer: str) -> Dict[str, Any]:
        """
        Evaluate the quality of the reasoning and answer.
        
        Args:
            question: The original question
            reasoning: The reasoning trace
            answer: The extracted answer
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.track_confidence:
            # Simple evaluation that always continues
            return {
                "confidence": 0.5,
                "errors_detected": True,
                "should_continue": True,
                "evaluation": "Continuing to next iteration"
            }
        
        try:
            # Create evaluation prompt
            evaluation_prompt = f"""Question: {question}

Reasoning:
{reasoning}

Answer: {answer}

{self.prompts['evaluation']}

In your response, include the following:
1. CONFIDENCE: A number between 0 and 1 indicating your confidence in the answer (0=completely wrong, 1=definitely correct)
2. ERRORS: Yes/No - Are there any errors in the reasoning?
3. CONTINUE: Yes/No - Should the reasoning process continue with another iteration?
4. EXPLANATION: A brief explanation of your assessment
"""
            
            # Use chat completion for evaluation
            evaluation_messages = [
                {"role": "system", "content": "You are an expert mathematical reasoning evaluator who critically assesses reasoning traces and answers."},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            evaluation_response = self.evaluator_model.chat_completion(
                messages=evaluation_messages,
                temperature=0.3  # Low temperature for consistent evaluation
            )
            
            evaluation_text = evaluation_response.text
            
            # Extract confidence score
            confidence_match = re.search(r"CONFIDENCE:\s*(0?\.\d+|[01])", evaluation_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract errors detected
            errors_match = re.search(r"ERRORS:\s*(Yes|No)", evaluation_text, re.IGNORECASE)
            errors_detected = errors_match.group(1).lower() == "yes" if errors_match else True
            
            # Extract should continue
            continue_match = re.search(r"CONTINUE:\s*(Yes|No)", evaluation_text, re.IGNORECASE)
            should_continue = continue_match.group(1).lower() == "yes" if continue_match else True
            
            return {
                "confidence": confidence,
                "errors_detected": errors_detected,
                "should_continue": should_continue,
                "evaluation": evaluation_text
            }
            
        except Exception as e:
            logger.error(f"Error during reasoning evaluation: {str(e)}")
            # Default to continuing with medium confidence
            return {
                "confidence": 0.5,
                "errors_detected": True,
                "should_continue": True,
                "evaluation": f"Error during evaluation: {str(e)}"
            }
    
    def _synthesize_final_answer(self, question: str, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize a final answer from all iterations.
        
        Args:
            question: The original question
            iterations: List of all reasoning iterations
            
        Returns:
            Dictionary with synthesized answer
        """
        try:
            # Create a synthesis prompt that includes all iterations
            synthesis_prompt = f"Question: {question}\n\n"
            
            for i, iteration in enumerate(iterations):
                synthesis_prompt += f"Iteration {i}:\n"
                
                if i > 0 and "summary" in iteration:
                    synthesis_prompt += f"Summary: {iteration['summary']}\n"
                
                # Include a truncated version of the reasoning to keep the prompt manageable
                reasoning = iteration["reasoning"]
                if len(reasoning) > 1000:
                    reasoning = reasoning[:500] + "..." + reasoning[-500:]
                synthesis_prompt += f"Reasoning: {reasoning}\n"
                
                synthesis_prompt += f"Answer: {iteration['answer']}\n"
                
                if "confidence" in iteration and iteration["confidence"] is not None:
                    synthesis_prompt += f"Confidence: {iteration['confidence']}\n"
                
                synthesis_prompt += "\n"
            
            synthesis_prompt += f"\n{self.prompts['synthesis']}\n"
            synthesis_prompt += "After considering all iterations, what is the final answer to the problem?"
            
            # Use chat completion for synthesis
            synthesis_messages = [
                {"role": "system", "content": "You are an expert mathematical reasoning synthesizer who can analyze multiple reasoning attempts and determine the best final answer."},
                {"role": "user", "content": synthesis_prompt}
            ]
            
            synthesis_response = self.evaluator_model.chat_completion(
                messages=synthesis_messages,
                temperature=0.3  # Low temperature for consistent synthesis
            )
            
            synthesis_text = synthesis_response.text
            
            # Extract the answer from the synthesis
            final_answer = extraction.extract_answer_from_text(synthesis_text)
            
            return {
                "reasoning": synthesis_text,
                "answer": final_answer
            }
            
        except Exception as e:
            logger.error(f"Error during final synthesis: {str(e)}")
            # Just return the last iteration's answer
            return {}
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of the recursive pipeline.
        
        Args:
            results: Pipeline results
            
        Returns:
            Dictionary with evaluation metrics
        """
        problem_results = results.get("results", [])
        
        if not problem_results:
            return {"error": "No results to evaluate"}
        
        # Count correct answers
        num_correct = 0
        num_with_answer = 0
        
        # Track accuracy per iteration for problems that have iterations
        iteration_correct_counts = {}
        
        # Track improvements from iterations
        iteration_transitions = {
            "wrong_to_right": 0,
            "right_to_wrong": 0,
            "wrong_to_wrong": 0,
            "right_to_right": 0
        }
        
        for result in problem_results:
            # Skip problems with errors
            if "error" in result:
                continue
                
            if "normalized_extracted" in result and "normalized_ground_truth" in result:
                # Skip problems without a ground truth
                if not result["normalized_ground_truth"]:
                    continue
                
                num_with_answer += 1
                ground_truth = result["normalized_ground_truth"]
                
                # Check final answer correctness
                final_correct = result["normalized_extracted"] == ground_truth
                if final_correct:
                    num_correct += 1
                
                # Check iterations if available
                if "iterations" in result:
                    iterations = result["iterations"]
                    
                    # Track initial and final correctness to measure improvement
                    if len(iterations) > 0:
                        initial_correct = (
                            extraction.normalize_answer(iterations[0].get("normalized_answer", "")) 
                            == ground_truth
                        )
                        
                        # Record transition
                        if initial_correct and final_correct:
                            iteration_transitions["right_to_right"] += 1
                        elif initial_correct and not final_correct:
                            iteration_transitions["right_to_wrong"] += 1
                        elif not initial_correct and final_correct:
                            iteration_transitions["wrong_to_right"] += 1
                        else:
                            iteration_transitions["wrong_to_wrong"] += 1
                    
                    # Calculate per-iteration accuracy
                    for iteration in iterations:
                        iter_num = iteration["iteration"]
                        
                        if iter_num not in iteration_correct_counts:
                            iteration_correct_counts[iter_num] = {"correct": 0, "total": 0}
                        
                        iteration_correct_counts[iter_num]["total"] += 1
                        
                        iter_answer = extraction.normalize_answer(iteration.get("normalized_answer", ""))
                        if iter_answer == ground_truth:
                            iteration_correct_counts[iter_num]["correct"] += 1
        
        # Calculate accuracy
        accuracy = num_correct / num_with_answer if num_with_answer > 0 else 0
        
        # Calculate accuracy per iteration
        iteration_accuracy = {}
        for iter_num, counts in iteration_correct_counts.items():
            iteration_accuracy[f"iteration_{iter_num}_accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            iteration_accuracy[f"iteration_{iter_num}_correct"] = counts["correct"]
            iteration_accuracy[f"iteration_{iter_num}_total"] = counts["total"]
        
        # Calculate improvement metrics
        improvement_metrics = {}
        if 0 in iteration_correct_counts and len(iteration_correct_counts) > 1:
            initial_acc = iteration_correct_counts[0]["correct"] / iteration_correct_counts[0]["total"] if iteration_correct_counts[0]["total"] > 0 else 0
            max_iter = max(iteration_correct_counts.keys())
            final_acc = iteration_correct_counts[max_iter]["correct"] / iteration_correct_counts[max_iter]["total"] if iteration_correct_counts[max_iter]["total"] > 0 else 0
            
            improvement_metrics = {
                "initial_accuracy": initial_acc,
                "final_accuracy": final_acc,
                "absolute_improvement": final_acc - initial_acc,
                "relative_improvement": (final_acc - initial_acc) / initial_acc if initial_acc > 0 else 0,
                "wrong_to_right_count": iteration_transitions["wrong_to_right"],
                "wrong_to_right_percentage": 100 * iteration_transitions["wrong_to_right"] / (iteration_transitions["wrong_to_right"] + iteration_transitions["wrong_to_wrong"]) if (iteration_transitions["wrong_to_right"] + iteration_transitions["wrong_to_wrong"]) > 0 else 0,
                "right_to_wrong_count": iteration_transitions["right_to_wrong"],
                "right_to_wrong_percentage": 100 * iteration_transitions["right_to_wrong"] / (iteration_transitions["right_to_right"] + iteration_transitions["right_to_wrong"]) if (iteration_transitions["right_to_right"] + iteration_transitions["right_to_wrong"]) > 0 else 0
            }
        
        # Calculate average number of iterations actually used
        iteration_counts = [len(r.get("iterations", [])) - 1 for r in problem_results if "iterations" in r]  # -1 to exclude initial iteration
        avg_iterations = sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0
        
        # Compile metrics
        metrics = {
            "accuracy": accuracy,
            "correct_count": num_correct,
            "total_evaluated": num_with_answer,
            "total_problems": len(problem_results),
            "average_iterations_used": avg_iterations,
            "max_allowed_iterations": results.get("max_iterations", 0),
            "average_time_per_problem": results.get("average_time_per_problem", 0),
            "summarization_method": results.get("summarization_method", self.method),
            **iteration_accuracy,
            **improvement_metrics
        }
        
        # Log the metrics
        self.log_results(metrics)
        
        return metrics