import os
import json
import time
import random
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncIterator

from src.llm.base_client import ModelClient, TokenUsage, CostInfo
from src.llm.tokenization import format_chat_for_completions, create_continuation_prompt, count_tokens

logger = logging.getLogger(__name__)

class FireworksModelClient(ModelClient):
    """
    Client for making API calls to Fireworks AI models using the completions API with continuation support.
    
    This client implements the asynchronous interface of ModelClient and does not support
    synchronous operations. All generation requests use the Completions API endpoint rather than
    the Chat API to enable better token limit handling and continuation mechanisms.
    """
    
    # Completions API endpoint
    BASE_URL = "https://api.fireworks.ai/inference/v1/completions"
    
    # Known token limit per request
    MAX_TOKENS_PER_REQUEST = 8192
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, input_price_per_million: float = None, output_price_per_million: float = None):
        """
        Initialize the Fireworks API client.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for Fireworks (if None, will use environment variable)
            input_price_per_million: Price per million input tokens (if None, will use default pricing)
            output_price_per_million: Price per million output tokens (if None, will use default pricing)
        """
        # Initialize the base class
        super().__init__(model_name, api_key)
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("No Fireworks API key provided and FIREWORKS_API_KEY not found in environment")
        
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Set pricing based on model or use provided pricing
        self._set_model_pricing(input_price_per_million, output_price_per_million)
    
    def _set_model_pricing(self, input_price_per_million: Optional[float] = None, output_price_per_million: Optional[float] = None):
        """
        Set pricing for the model based on the model name or provided pricing.
        
        Args:
            input_price_per_million: Price per million input tokens
            output_price_per_million: Price per million output tokens
            
        Raises:
            ValueError: If pricing information is not available for the model
        """
        # Use provided pricing if available
        if input_price_per_million is not None and output_price_per_million is not None:
            self.input_price_per_million_tokens = input_price_per_million
            self.output_price_per_million_tokens = output_price_per_million
            return
        
        # Load model info from unified JSON
        info_file = os.path.join(os.path.dirname(__file__), "models", "fireworks_model_info.json")
        try:
            with open(info_file, "r") as f:
                model_info = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading model info: {str(e)}")
        # Ensure model exists
        if self.model_name not in model_info:
            raise ValueError(f"Model info not available for model: {self.model_name}")
        info = model_info[self.model_name]
        # Set pricing
        self.input_price_per_million_tokens = info.get("input_price_per_million")
        self.output_price_per_million_tokens = info.get("output_price_per_million")
        # Set HF mapping if available
        self.hf_model_name = info.get("hf_model_name")
        logger.info(f"Set pricing for {self.model_name}: ${self.input_price_per_million_tokens}/M input tokens, ${self.output_price_per_million_tokens}/M output tokens")
        
    async def _call_completions_api(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS_PER_REQUEST,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_retries: int = 500,
        verbose: bool = False
    ) -> Tuple[Dict[str, Any], TokenUsage, CostInfo]:
        """
        Make a direct call to the Fireworks Completions API
        
        Args:
            prompt: The full formatted prompt string
            max_tokens: Maximum tokens to generate (capped at MAX_TOKENS_PER_REQUEST)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            max_retries: Maximum number of retries for rate limit errors
            verbose: Whether to log verbose output
            
        Returns:
            Tuple of (response_json, token_usage, cost_info)
        """
        # Cap max_tokens at the known limit
        max_tokens = min(max_tokens, self.MAX_TOKENS_PER_REQUEST)
        
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "prompt": prompt,
            "stream": False
        }
        
        if verbose:
            logger.debug(f"Completions API request for {self.model_name}")
            logger.debug(f"Prompt tokens (approx): {count_tokens(prompt)}")
        
        # Initialize retry counter and backoff time
        retry_count = 0
        backoff_time = 1  # Start with 1 second
        
        while True:
            try:
                if verbose:
                    logger.info(f"Making API request to {self.BASE_URL}")
                # Set timeout to prevent infinite waiting
                timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    if verbose:
                        logger.info("Session created, sending request...")
                    async with session.post(
                        self.BASE_URL, 
                        headers=self.headers, 
                        json=payload
                    ) as response:
                        if verbose:
                            logger.info(f"Received response with status {response.status}")
                        # Check for rate limit errors (429) or server errors (5xx)
                        if response.status == 429 or 500 <= response.status < 600:
                            retry_count += 1
                            if retry_count > max_retries:
                                if response.status == 429:
                                    raise ValueError(f"Fireworks API rate limit exceeded after {max_retries} retries")
                                else:
                                    raise ValueError(f"Fireworks API server error ({response.status}) persisted after {max_retries} retries")
                            
                            # Get retry-after header or use exponential backoff with jitter
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                base_sleep_time = float(retry_after)
                                sleep_time = base_sleep_time + random.uniform(0, 3)  # Add 0-3 seconds of jitter
                            else:
                                # Exponential backoff with jitter, capped at 60 seconds
                                sleep_time = backoff_time + random.uniform(0, 1)
                                backoff_time = min(backoff_time * 2, 60)
                            
                            error_type = "Rate limit" if response.status == 429 else f"Server error {response.status}"
                            logger.warning(f"{error_type} encountered, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                            await asyncio.sleep(sleep_time)
                            continue
                        
                        # Raise exception for other HTTP errors
                        response.raise_for_status()
                        
                        # Parse response
                        response_json = await response.json()
                        
                        # Extract token usage and cost information
                        token_usage = await self.get_token_usage(response_json)
                        cost_info = self.calculate_cost(token_usage)
                        
                        if verbose:
                            logger.info(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
                            logger.info(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")
                            
                            # Log if response was truncated
                            finish_reason = response_json["choices"][0]["finish_reason"] if "choices" in response_json else "unknown"
                            if finish_reason == "length":
                                logger.info(f"Response was truncated due to token limit ({max_tokens} tokens)")
                        
                        return response_json, token_usage, cost_info
                        
            except aiohttp.ClientError as e:
                # For connection errors, retry with backoff
                if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientOSError, 
                                 aiohttp.ServerDisconnectedError, asyncio.TimeoutError)):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise ValueError(f"Fireworks API request failed after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff with jitter
                    sleep_time = backoff_time + random.uniform(0, 1)
                    backoff_time *= 2  # Double the backoff time for next retry
                    
                    logger.warning(f"Connection error, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    # For other types of exceptions, raise immediately
                    raise ValueError(f"Fireworks API request failed: {str(e)}")
    
    async def generate_response_async(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        verbose: bool = False,
        enable_continuation: bool = True,
        max_total_tokens: int = 24576,  # Allow up to 3x the single request limit
        max_continuations: int = 3,
        track_token_callback = None,  # Callback function for token usage tracking with additional metadata
        track_token_callback_args = None,  # Extra args for the token tracking callback
        is_preformatted: bool = False, # Add is_preformatted flag
        **kwargs
    ) -> Tuple[str, str, TokenUsage, CostInfo, List[Dict]]:
        """
        Generate a response from the model for a prompt with automatic continuation.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            verbose: Whether to log verbose output
            enable_continuation: Whether to enable automatic continuation for long outputs
            max_total_tokens: Maximum total tokens to generate across all continuations
            max_continuations: Maximum number of continuations to attempt
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Tuple of (content, finish_reason, token_usage, cost_info)
        """
        # Prepare the list of detailed API calls for tracking
        detailed_api_calls = []
        
        # Prepare initial messages list if prompt is not preformatted
        # If preformatted, the prompt string is used directly
        if not is_preformatted:
            initial_messages = [
                {"role": "user", "content": prompt}
            ]
            # Format the initial prompt for the completions API
            formatted_prompt = format_chat_for_completions(initial_messages, self.model_name)
        else:
            # Use the prompt directly as it's already formatted
            formatted_prompt = prompt
        
        if verbose:
            logger.info("Launching response in generate_response_async")
        
        # Initialize tracking variables
        all_text = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_prompt_cost = 0.0
        total_completion_cost = 0.0
        final_finish_reason = "unknown"
        
        # This is the first call, use the initial formatted prompt
        current_prompt = formatted_prompt
        current_max_tokens = min(max_tokens, self.MAX_TOKENS_PER_REQUEST)
        
        # Call the completions API
        response_json, token_usage, cost_info = await self._call_completions_api(
            prompt=current_prompt,
            max_tokens=current_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            verbose=verbose
        )

        print(f"Response: {response_json}")
        
        # Extract text and finish reason
        if "choices" in response_json and response_json["choices"]:
            current_text = response_json["choices"][0]["text"]
            finish_reason = response_json["choices"][0].get("finish_reason", "unknown")
            final_finish_reason = finish_reason
            all_text += current_text
            
            # Update token usage
            total_prompt_tokens += token_usage.prompt_tokens
            total_completion_tokens += token_usage.completion_tokens
            total_prompt_cost += cost_info.prompt_cost
            total_completion_cost += cost_info.completion_cost
            
            # Record detailed metrics for initial API call
            initial_call_info = {
                "call_index": 0,
                "continuation_index": None,
                "token_usage": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                },
                "cost_info": {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                },
                "finish_reason": finish_reason,
                "text_length": len(current_text)
            }
            detailed_api_calls.append(initial_call_info)
            
            # If a token tracking callback is provided, call it with the initial API metrics
            if track_token_callback:
                callback_args = track_token_callback_args or {}
                track_token_callback(
                    problem_id=callback_args.get("problem_id", "unknown"),
                    token_usage=token_usage,
                    cost_info=cost_info,
                    iteration=callback_args.get("iteration", 0),
                    step=callback_args.get("step", "reasoning"),
                    continuation_idx=None,
                    api_call_idx=0
                )
            
            # Continue generating if needed
            iteration = 0
            
            while (enable_continuation and 
                   finish_reason == "length" and 
                   iteration < max_continuations and 
                   total_completion_tokens < max_total_tokens):
                
                # Check if continuation is needed and allowed
                if finish_reason == "length" and enable_continuation and total_tokens_so_far < max_total_tokens and continuation_count < max_continuations:
                    continuation_count += 1
                    if verbose:
                        logger.info(f"Continuation needed (reason: {finish_reason}), attempt {continuation_count}/{max_continuations}")
                    
                    # Create the continuation prompt (using the ORIGINAL raw prompt for context)
                    # Note: If the initial call was preformatted, we don't have the original message structure easily.
                    # We need to adjust how continuation prompts are built in this case, or require original_messages.
                    # For now, we assume continuation is NOT used with is_preformatted=True from the experiment side
                    # as the experiment handles its own continuation logic by modifying the prompt.
                    if is_preformatted:
                         logger.warning("Client-side continuation requested with a pre-formatted prompt. This is not standard usage and might lead to unexpected behavior.")
                         # Attempt a simple continuation by appending to the raw generated text
                         # This might not respect the chat template correctly
                         continuation_prompt_str = current_prompt + current_text 
                    else:
                        # This is the standard continuation logic
                         continuation_prompt_str = create_continuation_prompt(
                             initial_messages, 
                             current_text,
                             self.model_name
                         )
                    
                    # Make the continuation call
                    remaining_total_tokens = max_total_tokens - total_tokens_so_far
                    # Calculate max tokens for this continuation step, capped by request limit
                    continuation_max_tokens = min(remaining_total_tokens, self.MAX_TOKENS_PER_REQUEST)
                    
                    if verbose:
                        logger.info(f"Continuing generation (continuation {iteration+1})")
                    
                    # Call the completions API
                    try:
                        cont_response, cont_token_usage, cont_cost_info = await self._call_completions_api(
                            prompt=continuation_prompt_str,
                            max_tokens=continuation_max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            verbose=verbose
                        )
                        
                        # Extract continuation text and finish reason
                        if "choices" in cont_response and cont_response["choices"]:
                            cont_text = cont_response["choices"][0]["text"]
                            finish_reason = cont_response["choices"][0].get("finish_reason", "unknown")
                            final_finish_reason = finish_reason
                            all_text += cont_text
                            
                            # Update token usage
                            total_prompt_tokens += cont_token_usage.prompt_tokens
                            total_completion_tokens += cont_token_usage.completion_tokens
                            total_prompt_cost += cont_cost_info.prompt_cost
                            total_completion_cost += cont_cost_info.completion_cost
                            
                            # Record detailed metrics for continuation API call
                            continuation_call_info = {
                                "call_index": iteration + 1,  # +1 because the initial call is index 0
                                "continuation_index": iteration,
                                "token_usage": {
                                    "prompt_tokens": cont_token_usage.prompt_tokens,
                                    "completion_tokens": cont_token_usage.completion_tokens,
                                    "total_tokens": cont_token_usage.total_tokens
                                },
                                "cost_info": {
                                    "prompt_cost": cont_cost_info.prompt_cost,
                                    "completion_cost": cont_cost_info.completion_cost,
                                    "total_cost": cont_cost_info.total_cost
                                },
                                "finish_reason": finish_reason,
                                "text_length": len(cont_text)
                            }
                            detailed_api_calls.append(continuation_call_info)
                            
                            # If a token tracking callback is provided, call it with the continuation API metrics
                            if track_token_callback:
                                callback_args = track_token_callback_args or {}
                                track_token_callback(
                                    problem_id=callback_args.get("problem_id", "unknown"),
                                    token_usage=cont_token_usage,
                                    cost_info=cont_cost_info,
                                    iteration=callback_args.get("iteration", 0),
                                    step=callback_args.get("step", "reasoning"),
                                    continuation_idx=iteration,
                                    api_call_idx=iteration + 1
                                )
                            
                            if finish_reason != "length" or total_completion_tokens >= max_total_tokens:
                                break
                        
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Continuation failed: {str(e)}")
                        break
                    
                    # Increment iteration counter
                    iteration += 1
                
                # Check finish reason
                finish_reason = response_json["choices"][0]["finish_reason"]
                generated_text = response_json["choices"][0]["text"]
                
                # Update token usage
                total_prompt_tokens += token_usage.prompt_tokens
                total_completion_tokens += token_usage.completion_tokens
                total_prompt_cost += cost_info.prompt_cost
                total_completion_cost += cost_info.completion_cost
                
                # Record detailed metrics for current API call
                api_call_info = {
                    "call_index": iteration,
                    "continuation_index": None,
                    "token_usage": {
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens
                    },
                    "cost_info": {
                        "prompt_cost": cost_info.prompt_cost,
                        "completion_cost": cost_info.completion_cost,
                        "total_cost": cost_info.total_cost
                    },
                    "finish_reason": finish_reason,
                    "text_length": len(generated_text)
                }
                detailed_api_calls.append(api_call_info)
                
                # Update total tokens so far
                total_tokens_so_far = total_prompt_tokens + total_completion_tokens
                
                # Check if continuation is needed and allowed
                if finish_reason == "length" and enable_continuation and total_tokens_so_far < max_total_tokens and iteration < max_continuations:
                    iteration += 1
                
                # Update current prompt and max tokens
                current_prompt = continuation_prompt_str
                current_max_tokens = continuation_max_tokens
            
            # Create combined token usage and cost info
            combined_token_usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )
            
            combined_cost_info = CostInfo(
                prompt_cost=total_prompt_cost,
                completion_cost=total_completion_cost,
                total_cost=total_prompt_cost + total_completion_cost
            )
            
            # Add summary statistics to detailed API calls
            generation_summary = {
                "num_api_calls": len(detailed_api_calls),
                "num_continuations": sum(1 for call in detailed_api_calls if call.get("continuation_index") is not None),
                "total_token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                },
                "total_cost_info": {
                    "prompt_cost": total_prompt_cost,
                    "completion_cost": total_completion_cost,
                    "total_cost": total_prompt_cost + total_completion_cost
                },
                "total_text_length": len(all_text),
                "final_finish_reason": final_finish_reason
            }
            detailed_api_calls.append(generation_summary)
            
            if verbose:
                continuations = "with continuation" if enable_continuation else "without continuation"
                continuation_info = f" ({generation_summary['num_continuations']} continuations)" if generation_summary['num_continuations'] > 0 else ""
                logger.info(f"Generation complete {continuations}{continuation_info}: {combined_token_usage.total_tokens} total tokens")
            
            return all_text, final_finish_reason, combined_token_usage, combined_cost_info, detailed_api_calls
    
    async def get_token_usage(self, response_json: Dict[str, Any]) -> TokenUsage:
        """Extract token usage from a Fireworks API response."""
        usage = response_json.get("usage", {})
        
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        )
    
    def calculate_cost(self, token_usage: TokenUsage) -> CostInfo:
        """Calculate cost based on token usage and model pricing."""
        prompt_cost = (token_usage.prompt_tokens / 1_000_000) * self.input_price_per_million_tokens
        completion_cost = (token_usage.completion_tokens / 1_000_000) * self.output_price_per_million_tokens
        
        return CostInfo(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=prompt_cost + completion_cost
        )
    
    # Implement the base class interface
    
    def generate_completion(self, *args, **kwargs):
        """Not implemented - this client only supports async operations."""
        raise NotImplementedError("FireworksModelClient only supports asynchronous operations")
    
    def generate_response(self, *args, **kwargs):
        """Not implemented - this client only supports async operations."""
        raise NotImplementedError("FireworksModelClient only supports asynchronous operations")
    
    async def generate_completion_async(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, **kwargs) -> Tuple[Dict[str, Any], TokenUsage, CostInfo]:
        """Generate a completion using the chat format converted to completions API format."""
        # Convert chat messages to a completions API format
        prompt = format_chat_for_completions(messages)
        
        # Call the completions API and return only the response part
        response, token_usage, cost_info = await self._call_completions_api(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response, token_usage, cost_info