"""
Functions and utilities for generating reasoning traces.
"""
from typing import Dict, List, Optional, Any
import time
import logging

from src.utils.config import Config
from src.models.base import Model

logger = logging.getLogger(__name__)

def generate_reasoning_trace(
    question: str,
    model: Model,
    config: Config,
    max_extensions: Optional[int] = None,
    target_token_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a reasoning trace for a given question.
    
    Args:
        question: The question to generate reasoning for
        model: Model instance to use for generation
        config: Configuration object
        max_extensions: Maximum number of reasoning extensions (overrides config)
        target_token_count: Target token count for reasoning (overrides config)
    
    Returns:
        Dictionary containing the generated reasoning, answer, and metadata
    """
    reasoning_config = config.get("reasoning", {})
    max_extensions = max_extensions or reasoning_config.get("max_extensions", 10)
    target_token_count = target_token_count or reasoning_config.get("target_token_count", 2000)
    
    logger.info(f"Generating reasoning trace for question: {question[:100]}...")
    
    start_time = time.time()
    result = model.generate_reasoning(
        question=question,
        max_extensions=max_extensions,
        target_token_count=target_token_count
    )
    generation_time = time.time() - start_time
    
    # Add generation time to result metadata
    result["generation_time"] = generation_time
    
    logger.info(f"Generated reasoning trace in {generation_time:.2f}s")
    logger.info(f"Reasoning length: {len(result['reasoning'])} characters")
    
    return result

def batch_generate_reasoning_traces(
    questions: List[str],
    model: Model,
    config: Config,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate reasoning traces for a batch of questions.
    
    Args:
        questions: List of questions to generate reasoning for
        model: Model instance to use for generation
        config: Configuration object
        batch_size: Batch size for generation (overrides config)
    
    Returns:
        List of dictionaries containing the generated reasoning traces
    """
    pipeline_config = config.get("pipeline", {})
    batch_size = batch_size or pipeline_config.get("batch_size", 8)
    
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}")
        
        batch_results = []
        for question in batch:
            try:
                result = generate_reasoning_trace(
                    question=question, 
                    model=model, 
                    config=config
                )
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error generating reasoning for question: {question[:100]}... - {str(e)}")
                # Add a placeholder result for failed generations
                batch_results.append({
                    "question": question,
                    "reasoning": "",
                    "answer": "",
                    "error": str(e),
                    "success": False
                })
        
        results.extend(batch_results)
        
    return results