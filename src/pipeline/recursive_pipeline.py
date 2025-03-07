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

from src.utils.config import Config
from src.models.base import Model
from src.models.factory import create_reasoning_model, create_summarization_model
from src.data.dataset import Dataset
from src.pipeline.base_pipeline import BasePipeline
from src.pipeline.summary_pipeline import SummaryReasoningPipeline
import src.reasoning.generation as generation
import src.reasoning.extraction as extraction
from src.summarization.summarizers import get_summarizer

# Import experiment monitor
try:
    from src.utils.experiment_monitor import (
        initialize_monitor, 
        start_monitor_server, 
        update_problem_status, 
        add_reasoning_trace, 
        complete_problem,
        complete_experiment
    )
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

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
        
        logger.info(f"Initialized {self.name} pipeline with reasoning model {self.reasoning_model.model_name}")
        logger.info(f"Summarization method: {self.method}")
        logger.info(f"Max iterations: {self.max_iterations}, Early stopping: {self.early_stopping}")
        
        if self.summarization_model:
            logger.info(f"Using external summarization model: {self.summarization_model.model_name}")
        else:
            logger.info("Using self-summarization")
    
    def run(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the recursive pipeline on the dataset.
        
        Args:
            dataset: Dataset instance
            **kwargs: Additional arguments
                - split: Data split to use ('train', 'test', or 'all')
                - problem_ids: List of specific problem IDs to process
                - max_problems: Maximum number of problems to process
                - max_iterations: Maximum number of reasoning iterations (overrides config)
                - monitor: Whether to start the experiment monitor (default: False)
                - monitor_port: Port for the experiment monitor (default: 5000)
        
        Returns:
            Dictionary with pipeline results
        """
        # Get parameters, allowing overrides
        split = kwargs.get("split", "test")
        problem_ids = kwargs.get("problem_ids", None)
        max_problems = kwargs.get("max_problems", None)
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        start_monitor = kwargs.get("monitor", False)
        monitor_port = kwargs.get("monitor_port", 5000)
        
        if problem_ids:
            # Process specific problems by ID
            problems = [dataset.get_problem_by_id(pid) for pid in problem_ids]
            logger.info(f"Processing {len(problems)} specific problems")
        else:
            # Process all problems in the specified split
            problems = dataset.get_problems(split=split)
            logger.info(f"Processing {len(problems)} problems from {split} split")
            
        # Limit the number of problems if specified
        if max_problems and max_problems > 0:
            problems = problems[:max_problems]
            logger.info(f"Limited to {max_problems} problems")
        
        # Start the experiment monitor if requested
        if start_monitor and MONITOR_AVAILABLE:
            # Initialize the monitor with experiment info
            initialize_monitor(self.name, len(problems), max_iterations)
            # Start the monitor server
            monitor_started = start_monitor_server(port=monitor_port)
            if monitor_started:
                logger.info(f"Experiment monitor started at http://localhost:{monitor_port}")
            else:
                logger.warning("Failed to start experiment monitor. Continuing without monitoring.")
        
        # Process each problem
        start_time = time.time()
        results = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Processing problem {i+1}/{len(problems)}: {problem['id']}")
            
            try:
                # Process the problem
                result = self._process_problem(problem["question"], max_iterations)
                
                # Add problem metadata
                result["problem_id"] = problem["id"]
                result["question"] = problem["question"]
                result["ground_truth"] = problem.get("answer", "")
                
                # Normalize the ground truth for comparison
                result["normalized_ground_truth"] = extraction.normalize_answer(result["ground_truth"])
                
                # Add to results
                results.append(result)
                
                # Update the monitor if available
                if MONITOR_AVAILABLE:
                    complete_problem(
                        problem_id=problem["id"],
                        final_answer=result.get("extracted_answer", ""),
                        processing_time=result.get("processing_time"),
                        iteration_count=result.get("iteration_count", 0)
                    )
                
            except Exception as e:
                logger.error(f"Error processing problem {problem['id']}: {str(e)}")
                # Continue with the next problem
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Create the experiment summary
        experiment_summary = {
            "name": self.name,
            "model": self.reasoning_model.model_name,
            "split": split,
            "problem_count": len(results),
            "max_iterations": max_iterations,
            "total_time": total_time,
            "metrics": metrics
        }
        
        # Mark the experiment as complete in the monitor
        if MONITOR_AVAILABLE:
            complete_experiment(metrics)
            
        return {
            "results": results,
            "metrics": metrics,
            "summary": experiment_summary
        }
    
    def _process_problem(self, question: str, max_iterations: int) -> Dict[str, Any]:
        """
        Process a single problem using recursive reasoning.
        
        Args:
            question: The question to solve
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary with problem results
        """
        try:
            start_time = time.time()
            
            # Extract problem ID from the question if possible
            problem_id = None
            id_match = re.search(r'(\d{4}-[I|II]-\d+)', question)
            if id_match:
                problem_id = id_match.group(1)
            else:
                # Generate a timestamp-based ID if no ID found
                problem_id = f"problem_{int(time.time())}"
            
            # Update monitor with current problem
            if MONITOR_AVAILABLE:
                update_problem_status(problem_id, question, 0)
            
            # Initial reasoning generation
            logger.info("Generating initial reasoning")
            initial_result = self.reasoning_model.generate_reasoning(
                question=question,
                problem_id=problem_id
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
            
            # Add reasoning trace to monitor
            if MONITOR_AVAILABLE:
                add_reasoning_trace(
                    problem_id=problem_id,
                    iteration=0,
                    reasoning=initial_result["reasoning"],
                    answer=initial_answer,
                    confidence=None,
                    errors_detected=None,
                    should_continue=None
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
                
                # Update monitor with iteration data
                if MONITOR_AVAILABLE:
                    add_reasoning_trace(
                        problem_id=problem_id,
                        iteration=i + 1,
                        reasoning=current_reasoning,
                        answer=current_answer,
                        confidence=confidence,
                        errors_detected=errors_detected,
                        should_continue=should_continue
                    )
            
            # Final synthesis to determine best answer
            logger.info("Synthesizing final answer")
            final_result = self._synthesize_final_answer(question, iterations)
            
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
            
            # Mark problem as complete in the monitor
            if MONITOR_AVAILABLE:
                complete_problem(
                    problem_id=problem_id,
                    final_answer=final_answer,
                    processing_time=processing_time,
                    iteration_count=len(iterations) - 1
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing problem: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a minimal result with error information
            return {
                "question": question,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "extracted_answer": "ERROR",
                "processing_time": 0,
                "iteration_count": 0
            }
    
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
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for the experiment results.
        
        Args:
            results: List of problem results
            
        Returns:
            Dictionary with calculated metrics
        """
        if not results:
            return {
                "accuracy": 0.0,
                "average_iterations": 0.0,
                "average_time_per_problem": 0.0
            }
        
        # Calculate accuracy
        correct_count = 0
        total_iterations = 0
        total_time = 0.0
        
        for result in results:
            # Check if the normalized answer matches the normalized ground truth
            if result.get("normalized_ground_truth") and result.get("normalized_answer"):
                if result["normalized_answer"] == result["normalized_ground_truth"]:
                    correct_count += 1
            
            # Sum iterations and time
            total_iterations += result.get("iteration_count", 0)
            total_time += result.get("processing_time", 0.0)
        
        # Calculate metrics
        accuracy = correct_count / len(results) if results else 0.0
        average_iterations = total_iterations / len(results) if results else 0.0
        average_time = total_time / len(results) if results else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_problems": len(results),
            "average_iterations": average_iterations,
            "total_iterations": total_iterations,
            "average_time_per_problem": average_time,
            "total_processing_time": total_time
        }