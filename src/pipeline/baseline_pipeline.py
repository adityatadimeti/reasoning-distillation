"""
Baseline pipeline implementation for reasoning evaluation.
"""
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from src.utils.config import Config
from src.models.base import Model
from src.models.fireworks import FireworksModel
from src.data.dataset import Dataset
from src.pipeline.base_pipeline import BasePipeline
import src.reasoning.generation as generation
import src.reasoning.extraction as extraction

logger = logging.getLogger(__name__)

class BaselineReasoningPipeline(BasePipeline):
    """
    Baseline pipeline that generates reasoning traces and evaluates performance.
    """
    def __init__(self, config: Config):
        """
        Initialize the baseline pipeline.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Set name if not already set
        if self.name == "unknown_pipeline":
            self.name = "baseline"
        
        # Initialize model
        model_config = config.get("model", {})
        model_type = model_config.get("api_type", "fireworks")
        
        if model_type == "fireworks":
            self.model = FireworksModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Pipeline parameters
        pipeline_config = config.get("pipeline", {})
        self.batch_size = pipeline_config.get("batch_size", 8)
        
        logger.info(f"Initialized {self.name} pipeline with model {self.model.model_name}")
    
    def run(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the baseline pipeline on the dataset.
        
        Args:
            dataset: Dataset instance
            **kwargs: Additional arguments
                - split: Data split to use ('train', 'test', or 'all')
                - problem_ids: List of specific problem IDs to process
                - max_problems: Maximum number of problems to process
        
        Returns:
            Dictionary with pipeline results
        """
        # Get the problems to process
        split = kwargs.get("split", "test")
        problem_ids = kwargs.get("problem_ids", None)
        max_problems = kwargs.get("max_problems", None)
        
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
        
        # Extract questions for generation
        questions = [problem["question"] for problem in problems]
        
        # Generate reasoning traces
        start_time = time.time()
        
        reasoning_results = generation.batch_generate_reasoning_traces(
            questions=questions,
            model=self.model,
            config=self.config,
            batch_size=self.batch_size
        )
        
        total_time = time.time() - start_time
        
        # Extract answers from reasoning
        for i, result in enumerate(reasoning_results):
            answer = extraction.extract_answer_from_reasoning_result(result)
            normalized_answer = extraction.normalize_answer(answer)
            
            result["extracted_answer"] = answer
            result["normalized_extracted"] = normalized_answer
            
            # Add ground truth for evaluation
            result["ground_truth"] = problems[i].get("ground_truth", "")
            result["normalized_ground_truth"] = extraction.normalize_answer(
                problems[i].get("ground_truth", "")
            )
            
            # Add problem ID
            result["problem_id"] = problems[i].get("id", "")
        
        # Compile pipeline results
        results = {
            "pipeline": self.name,
            "timestamp": int(time.time()),
            "num_problems": len(problems),
            "total_time": total_time,
            "average_time_per_problem": total_time / len(problems) if problems else 0,
            "results": reasoning_results
        }
        
        return results
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of the pipeline.
        
        Args:
            results: Pipeline results
            
        Returns:
            Dictionary with evaluation metrics
        """
        reasoning_results = results.get("results", [])
        
        if not reasoning_results:
            return {"error": "No results to evaluate"}
        
        # Count correct answers
        num_correct = 0
        num_with_answer = 0
        
        for result in reasoning_results:
            if "normalized_extracted" in result and "normalized_ground_truth" in result:
                # Skip problems without a ground truth
                if not result["normalized_ground_truth"]:
                    continue
                
                num_with_answer += 1
                
                if result["normalized_extracted"] == result["normalized_ground_truth"]:
                    num_correct += 1
        
        # Calculate accuracy
        accuracy = num_correct / num_with_answer if num_with_answer > 0 else 0
        
        # Calculate token efficiency
        total_tokens = sum(result.get("estimated_token_count", 0) for result in reasoning_results)
        average_tokens = total_tokens / len(reasoning_results) if reasoning_results else 0
        
        # Compile metrics
        metrics = {
            "accuracy": accuracy,
            "correct_count": num_correct,
            "total_evaluated": num_with_answer,
            "total_problems": len(reasoning_results),
            "average_tokens_per_problem": average_tokens,
            "total_time": results.get("total_time", 0),
            "average_time_per_problem": results.get("average_time_per_problem", 0)
        }
        
        # Log the metrics
        self.log_results(metrics)
        
        return metrics

class BaselinePipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the baseline pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = FireworksModel(config)
        
    def run(
        self,
        problem_id: str,
        reasoning_trace: str,
        summarization_mode: str = "append",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a reasoning trace.
        
        Args:
            problem_id: ID of the problem
            reasoning_trace: The reasoning trace to process
            summarization_mode: Mode of summarization ('append' or 'prepend')
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing pipeline output
        """
        # Generate summary
        summary = self.model.summarize_reasoning(
            reasoning_trace,
            mode=summarization_mode,
            **kwargs
        )
        
        # Combine reasoning and summary based on mode
        if summarization_mode == "append":
            combined_reasoning = f"{reasoning_trace}\n\n{summary}"
        else:  # prepend
            combined_reasoning = f"{summary}\n\n{reasoning_trace}"
        
        # Extract answer from combined reasoning
        answer = self.model.extract_answer(combined_reasoning)
        
        return {
            "problem_id": problem_id,
            "original_reasoning": reasoning_trace,
            "summary": summary,
            "combined_reasoning": combined_reasoning,
            "answer": answer,
            "summarization_mode": summarization_mode,
            "timestamp": datetime.now().isoformat()
        }