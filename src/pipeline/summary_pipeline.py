"""
Pipeline for enhancing reasoning through summarization.
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
from src.pipeline.baseline_pipeline import BaselineReasoningPipeline
import src.reasoning.generation as generation
import src.reasoning.extraction as extraction
from src.summarization.summarizers import get_summarizer

logger = logging.getLogger(__name__)

class SummaryReasoningPipeline(BasePipeline):
    """
    Pipeline that enhances reasoning through summarization.
    """
    def __init__(self, config: Config):
        """
        Initialize the summary pipeline.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Set name if not already set
        if self.name == "unknown_pipeline":
            self.name = "summary"
        
        # Initialize models
        self.reasoning_model = create_reasoning_model(config)
        self.summarization_model = create_summarization_model(config)
        
        # Initialize summarizer
        self.summarizer = get_summarizer(config, self.reasoning_model, self.summarization_model)
        
        # Pipeline parameters
        pipeline_config = config.get("pipeline", {})
        self.batch_size = pipeline_config.get("batch_size", 4)
        self.max_iterations = pipeline_config.get("max_iterations", 2)
        self.continuation_prompt = pipeline_config.get("continuation_prompt", 
            "Based on the summary of your previous reasoning, continue solving the problem."
        )
        
        # Summarization parameters
        summarization_config = config.get("summarization", {})
        self.method = summarization_config.get("method", "self")
        
        logger.info(f"Initialized {self.name} pipeline with reasoning model {self.reasoning_model.model_name}")
        logger.info(f"Summarization method: {self.method}")
        
        if self.summarization_model:
            logger.info(f"Using external summarization model: {self.summarization_model.model_name}")
        else:
            logger.info("Using self-summarization")
    
    def run(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the summary pipeline on the dataset.
        
        Args:
            dataset: Dataset instance
            **kwargs: Additional arguments
                - split: Data split to use ('train', 'test', or 'all')
                - problem_ids: List of specific problem IDs to process
                - max_problems: Maximum number of problems to process
                - max_iterations: Maximum number of reasoning iterations (overrides config)
        
        Returns:
            Dictionary with pipeline results
        """
        # Get parameters, allowing overrides
        split = kwargs.get("split", "test")
        problem_ids = kwargs.get("problem_ids", None)
        max_problems = kwargs.get("max_problems", None)
        max_iterations = kwargs.get("max_iterations", self.max_iterations)
        
        if problem_ids:
            # Process specific problems by ID
            problems = [dataset.get_problem_by_id(pid) for pid in problem_ids]
            logger.info(f"Processing {len(problems)} specific problems")
        else:
            # Process all problems in the specified split
            problems = dataset.get_problems(split=split)
            logger.info(f"Processing {len(problems)} problems from {split} split")
            
            # Limit the number of problems if specified
            if max_problems and len(problems) > max_problems:
                problems = problems[:max_problems]
                logger.info(f"Limited to {max_problems} problems")
        
        start_time = time.time()
        
        # Process each problem
        results = []
        for i, problem in enumerate(problems):
            logger.info(f"Processing problem {i+1}/{len(problems)}: {problem['id']}")
            
            try:
                # Process the problem with summary-based reasoning
                problem_result = self._process_problem(
                    problem["question"],
                    max_iterations=max_iterations
                )
                
                # Add problem metadata
                problem_result["problem_id"] = problem.get("id", "")
                problem_result["ground_truth"] = problem.get("ground_truth", "")
                
                # Normalize answers for comparison
                problem_result["normalized_extracted"] = extraction.normalize_answer(
                    problem_result.get("extracted_answer", "")
                )
                problem_result["normalized_ground_truth"] = extraction.normalize_answer(
                    problem.get("ground_truth", "")
                )
                
                results.append(problem_result)
                
            except Exception as e:
                logger.error(f"Error processing problem {problem['id']}: {str(e)}")
                # Add a failed result
                results.append({
                    "problem_id": problem.get("id", ""),
                    "question": problem.get("question", ""),
                    "ground_truth": problem.get("ground_truth", ""),
                    "error": str(e),
                    "success": False
                })
        
        total_time = time.time() - start_time
        
        # Compile pipeline results
        pipeline_results = {
            "pipeline": self.name,
            "timestamp": int(time.time()),
            "num_problems": len(problems),
            "total_time": total_time,
            "average_time_per_problem": total_time / len(problems) if problems else 0,
            "max_iterations": max_iterations,
            "summarization_method": self.method,
            "results": results
        }
        
        return pipeline_results
    
    def _process_problem(self, question: str, max_iterations: int) -> Dict[str, Any]:
        """
        Process a single problem using summary-based reasoning.
        
        Args:
            question: The question to solve
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            Dictionary with problem results
        """
        start_time = time.time()
        
        # Initial reasoning generation
        logger.info("Generating initial reasoning")
        initial_result = self.reasoning_model.generate_reasoning(question=question)
        
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
                "normalized_answer": initial_normalized
            }
        ]
        
        # Track the current reasoning and answer
        current_reasoning = initial_result["reasoning"]
        current_answer = initial_answer
        current_normalized = initial_normalized
        
        # Continue iterating with summarized reasoning
        for i in range(max_iterations):
            logger.info(f"Iteration {i+1}: Summarizing reasoning")
            
            # Summarize the current reasoning
            summary = self.summarizer.summarize(current_reasoning)
            
            # Create prompt with the summary for continuation
            continuation_prompt = f"{question}\n\nHere is a summary of your previous reasoning:\n{summary}\n\n{self.continuation_prompt}"
            
            logger.info(f"Iteration {i+1}: Generating new reasoning based on summary")
            
            # Generate new reasoning based on the summary
            new_result = self.reasoning_model.generate_reasoning(question=continuation_prompt)
            
            # Extract new answer
            new_answer = extraction.extract_answer_from_reasoning_result(new_result)
            new_normalized = extraction.normalize_answer(new_answer)
            
            # Store this iteration
            iterations.append({
                "iteration": i + 1,
                "summary": summary,
                "reasoning": new_result["reasoning"],
                "answer": new_answer,
                "extracted_answer": new_answer,
                "normalized_answer": new_normalized
            })
            
            # Update current reasoning and answer
            current_reasoning = new_result["reasoning"]
            current_answer = new_answer
            current_normalized = new_normalized
        
        # Determine the final answer from all iterations
        # For now, just use the answer from the last iteration
        final_answer = current_answer
        
        # Build and return the result
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "iterations": iterations,
            "summary": summary if 'summary' in locals() else "",
            "final_reasoning": current_reasoning,
            "extracted_answer": final_answer,
            "processing_time": processing_time,
            "iteration_count": max_iterations
        }
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of the pipeline.
        
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
        
        for result in problem_results:
            # Skip problems with errors
            if "error" in result:
                continue
                
            if "normalized_extracted" in result and "normalized_ground_truth" in result:
                # Skip problems without a ground truth
                if not result["normalized_ground_truth"]:
                    continue
                
                num_with_answer += 1
                
                if result["normalized_extracted"] == result["normalized_ground_truth"]:
                    num_correct += 1
                
                # Check iterations if available
                if "iterations" in result:
                    for iteration in result["iterations"]:
                        iter_num = iteration["iteration"]
                        
                        if iter_num not in iteration_correct_counts:
                            iteration_correct_counts[iter_num] = {"correct": 0, "total": 0}
                        
                        iteration_correct_counts[iter_num]["total"] += 1
                        
                        if extraction.normalize_answer(iteration["normalized_answer"]) == result["normalized_ground_truth"]:
                            iteration_correct_counts[iter_num]["correct"] += 1
        
        # Calculate accuracy
        accuracy = num_correct / num_with_answer if num_with_answer > 0 else 0
        
        # Calculate accuracy per iteration
        iteration_accuracy = {}
        for iter_num, counts in iteration_correct_counts.items():
            iteration_accuracy[f"iteration_{iter_num}_accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            iteration_accuracy[f"iteration_{iter_num}_correct"] = counts["correct"]
            iteration_accuracy[f"iteration_{iter_num}_total"] = counts["total"]
        
        # Calculate improvement over iterations
        improvement = {}
        if 0 in iteration_correct_counts and max(iteration_correct_counts.keys()) > 0:
            baseline_acc = iteration_correct_counts[0]["correct"] / iteration_correct_counts[0]["total"] if iteration_correct_counts[0]["total"] > 0 else 0
            final_iter = max(iteration_correct_counts.keys())
            final_acc = iteration_correct_counts[final_iter]["correct"] / iteration_correct_counts[final_iter]["total"] if iteration_correct_counts[final_iter]["total"] > 0 else 0
            
            improvement = {
                "absolute_improvement": final_acc - baseline_acc,
                "relative_improvement": (final_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0
            }
        
        # Compile metrics
        metrics = {
            "accuracy": accuracy,
            "correct_count": num_correct,
            "total_evaluated": num_with_answer,
            "total_problems": len(problem_results),
            "average_time_per_problem": results.get("average_time_per_problem", 0),
            "max_iterations": results.get("max_iterations", 0),
            "summarization_method": results.get("summarization_method", self.method),
            **iteration_accuracy,
            **improvement
        }
        
        # Log the metrics
        self.log_results(metrics)
        
        return metrics