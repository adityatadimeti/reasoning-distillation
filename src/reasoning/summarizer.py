import logging
from typing import Optional, Dict, Any, Union, Iterator, Tuple, AsyncIterator

from src.llm.base_client import ModelClient, TokenUsage, CostInfo

logger = logging.getLogger(__name__)

def summarize_reasoning(
    question: str,
    reasoning: str,
    model,
    prompt_template: str,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    verbose: bool = False,
    stream: bool = False
) -> Union[Tuple[str, str, TokenUsage, CostInfo], Iterator[str]]:
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
        stream: Whether to stream the summary in chunks
        
    Returns:
        If stream=False: Tuple of (summary, finish_reason, token_usage, cost_info) where finish_reason indicates why generation stopped
        If stream=True: Iterator yielding summary chunks
    """
    logger.info("Generating summary of reasoning trace")
    
    # Create the prompt
    prompt = prompt_template.replace("{reasoning}", reasoning)
    if "{question}" in prompt_template:
        prompt = prompt.replace("{question}", question)
    
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
            "verbose": verbose,
            "stream": stream
        }
    else:
        # Other model clients
        gen_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "verbose": verbose,
            "stream": stream
        }
        # Only add top_k if provided
        if top_k is not None:
            gen_params["top_k"] = top_k
    
    # Remove None values
    gen_params = {k: v for k, v in gen_params.items() if v is not None}
    
    # Generate the summary
    return model.generate_response(prompt, **gen_params)

async def summarize_reasoning_async(
    question: str,
    reasoning: str,
    model,
    prompt_template: str,
    max_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    verbose: bool = False,
    stream: bool = False
) -> Union[Tuple[str, str, TokenUsage, CostInfo], AsyncIterator[str]]:
    """
    Generate a summary of the reasoning trace asynchronously.
    
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
        stream: Whether to stream the summary in chunks
        
    Returns:
        If stream=False: Tuple of (summary, finish_reason, token_usage, cost_info) where finish_reason indicates why generation stopped
        If stream=True: AsyncIterator yielding summary chunks
    """
    logger.info("Generating summary of reasoning trace asynchronously")
    
    # Create the prompt
    prompt = prompt_template.replace("{reasoning}", reasoning)
    if "{question}" in prompt_template:
        prompt = prompt.replace("{question}", question)
    
    # Generate the summary with the model
    # Build generation parameters dict based on model type
    if hasattr(model, 'generate_completion_async') and 'fireworks' in str(model.__class__).lower():
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
            "verbose": verbose,
            "stream": stream
        }
    else:
        # Other model clients
        gen_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "verbose": verbose,
            "stream": stream
        }
        # Only add top_k if provided
        if top_k is not None:
            gen_params["top_k"] = top_k
    
    # Remove None values
    gen_params = {k: v for k, v in gen_params.items() if v is not None}
    
    # Generate the summary
    return await model.generate_response_async(prompt, **gen_params)

