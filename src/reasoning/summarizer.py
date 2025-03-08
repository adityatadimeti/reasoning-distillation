import logging
from typing import Optional, Dict, Any, Union

from src.llm.base_client import ModelClient

logger = logging.getLogger(__name__)

def summarize_reasoning(
    reasoning: str,
    model,
    prompt_template: str,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    verbose: bool = False
) -> str:
    """
    Generate a summary of the reasoning trace.
    
    Args:
        reasoning: The reasoning to summarize
        model: The model to use for summarization
        prompt_template: The template to use for the summarization prompt
        max_tokens: Maximum number of tokens for the response
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        presence_penalty: Presence penalty parameter
        frequency_penalty: Frequency penalty parameter
        verbose: Whether to log model calls
        
    Returns:
        The summarized reasoning
    """
    logger.info("Generating summary of reasoning trace")
    
    # Create the prompt
    prompt = prompt_template.replace("{reasoning}", reasoning)
    
    # Generate the summary with the model
    # Build generation parameters dict based on model type
    if hasattr(model, 'generate_completion') and 'fireworks' in str(model.__class__).lower():
        # FireworksModelClient requires top_k
        if top_k is None:
            raise ValueError("top_k is required for FireworksModelClient")
            
        gen_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "verbose": verbose
        }
    else:
        # Other model clients
        gen_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "verbose": verbose
        }
        # Only add top_k if provided
        if top_k is not None:
            gen_params["top_k"] = top_k
    
    # Remove None values
    gen_params = {k: v for k, v in gen_params.items() if v is not None}
    
    # Generate the summary
    summary = model.generate_response(prompt, **gen_params)
    
    return summary

