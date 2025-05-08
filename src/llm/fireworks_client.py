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
from src.llm.tokenization import format_chat_for_completions, count_tokens

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
        #self._set_model_pricing(input_price_per_million, output_price_per_million)
    
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
                        
                        # Handle 400 Bad Request specifically to get more detailed error info
                        if response.status == 400:
                            try:
                                error_text = await response.text()
                                error_json = json.loads(error_text)
                                error_message = error_json.get('error', {}).get('message', 'Unknown error')
                                logger.error(f"Fireworks API 400 error: {error_message}")
                                logger.error(f"Full error response: {error_text}")
                                # Raise ValueError with the detailed error message
                                raise ValueError(f"Fireworks API bad request: {error_message}")
                            except:
                                error_text = await response.text()
                                logger.error(f"Fireworks API 400 error with unparseable response: {error_text[:1000]}")
                                # Raise ValueError with the error text
                                raise ValueError(f"Fireworks API bad request: {error_text[:500]}")
                        
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
                        
                        # For other status codes, try to get error details
                        if response.status != 200:
                            try:
                                error_text = await response.text()
                                logger.error(f"API error response (status {response.status}): {error_text[:1000]}")
                            except:
                                pass
                        
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
                    # Log full exception details for debugging
                    import traceback
                    logger.error(f"Fireworks API request error: {str(e)}")
                    logger.error(f"Error traceback: {traceback.format_exc()}")
                    
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
        preformatted_prompt: Optional[str] = None, # New argument
        **kwargs
    ) -> Tuple[str, str, TokenUsage, CostInfo, List[Dict]]:
        """
        Generate a response from the model for a prompt with automatic continuation.
        If `preformatted_prompt` is provided, it bypasses chat formatting and internal continuation logic.
        
        Args:
            prompt: The prompt to send to the model (used if preformatted_prompt is None)
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            verbose: Whether to log verbose output
            enable_continuation: Whether to enable automatic continuation for long outputs (ignored if preformatted_prompt is used)
            max_total_tokens: Maximum total tokens to generate across all continuations (ignored if preformatted_prompt is used)
            max_continuations: Maximum number of continuations to attempt (ignored if preformatted_prompt is used)
            track_token_callback: Callback function for token usage tracking
            track_token_callback_args: Extra args for the token tracking callback
            preformatted_prompt: If provided, this string is used directly as the prompt, bypassing chat formatting and internal continuation.
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Tuple of (content, finish_reason, token_usage, cost_info, detailed_api_calls)
        """
        MAX_CONTINUATION_ATTEMPT_RETRIES = 5  # 1 initial attempt + 2 retries = 3 total attempts
        CONTINUATION_RETRY_BACKOFF_BASE = 1  # seconds

        # If preformatted_prompt is provided, use it directly and disable internal continuation
        if preformatted_prompt is not None:
            if verbose:
                logger.info(f"Using preformatted prompt (length: {len(preformatted_prompt)} chars). Internal continuation disabled.")
            
            # print(f"Preformatted prompt: {preformatted_prompt}")
            # breakpoint()

            # Make a single API call using the preformatted prompt
            response, token_usage, cost_info = await self._call_completions_api(
                prompt=preformatted_prompt,
                max_tokens=min(max_tokens, self.MAX_TOKENS_PER_REQUEST),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                verbose=verbose
            )
            
            # Extract results from the single call
            current_text = ""
            finish_reason = "error"
            if "choices" in response and response["choices"]:
                current_text = response["choices"][0]["text"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")

            # Prepare detailed metrics for this single call
            api_call_detail = {
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
            detailed_api_calls = [api_call_detail]

            # Add summary (which is just this call's info)
            summary_info = {
                "num_api_calls": 1,
                "num_continuations": 0,
                "total_token_usage": api_call_detail["token_usage"],
                "total_cost_info": api_call_detail["cost_info"],
                "total_text_length": len(current_text),
                "final_finish_reason": finish_reason
            }
            detailed_api_calls.append(summary_info)

            # If a token tracking callback is provided, call it
            if track_token_callback:
                callback_args = track_token_callback_args or {}
                track_token_callback(
                    problem_id=callback_args.get("problem_id", "unknown"),
                    token_usage=token_usage,
                    cost_info=cost_info,
                    iteration=callback_args.get("iteration", 0),
                    step=callback_args.get("step", "reasoning"),
                    continuation_idx=None, # Not an internal continuation
                    api_call_idx=0
                )

            return current_text, finish_reason, token_usage, cost_info, detailed_api_calls

        # --- Original logic for standard prompts with internal continuation --- 
        # Convert prompt to chat format if not preformatted
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            logger.info(f"Generating response for prompt (length: {len(prompt)} chars)")
        
        # Format the initial prompt for completions API
        formatted_prompt = format_chat_for_completions(messages, self.model_name)
        if verbose: # Changed logger level from info to debug
            logger.debug(f"Initial formatted prompt: {formatted_prompt}")

        # Initialize tracking variables
        all_text = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_prompt_cost = 0.0
        total_completion_cost = 0.0
        final_finish_reason = "unknown"  # Will be updated by the first successful call
        detailed_api_calls = []  # Track individual API calls for detailed metrics
        
        # Keep track of the prompt string used in the last API call
        current_prompt_string = formatted_prompt 
        
        print(f"Current prompt string: {current_prompt_string}")
        # breakpoint()

        # First API call
        response, token_usage, cost_info = await self._call_completions_api(
            prompt=current_prompt_string, # Use current_prompt_string
            max_tokens=min(max_tokens, self.MAX_TOKENS_PER_REQUEST),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            verbose=verbose
        )

        # if verbose:
        #     print(f"Response: {response}")
        
        # Extract text and finish reason
        if "choices" in response and response["choices"]:
            current_text = response["choices"][0]["text"]
            finish_reason = response["choices"][0].get("finish_reason", "unknown")
            final_finish_reason = finish_reason # Initialize with the result of the first call
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
            
            last_continuation_step_error = None # To store error if all retries for a step fail

            while (enable_continuation and 
                   finish_reason == "length" and  # Loop continues if the LAST successful op was 'length'
                   iteration < max_continuations and 
                   total_completion_tokens < max_total_tokens):
                
                # --- Continuation Logic ---
                # The next prompt is simply the previous prompt + the generated text
                next_prompt_string = current_prompt_string + current_text
                current_prompt_string = next_prompt_string # Update for the next potential iteration

                # Calculate remaining tokens allowed
                remaining_tokens = max_total_tokens - total_completion_tokens
                tokens_to_generate = min(remaining_tokens, self.MAX_TOKENS_PER_REQUEST)
                
                if tokens_to_generate <= 0:
                    break
                
                if verbose:
                    logger.info(f"Continuing generation (continuation {iteration+1})")
                    logger.debug(f"Continuation prompt (last few chars): ...{current_prompt_string[-100:]}") # Log snippet
                
                continuation_step_succeeded = False
                last_error_for_this_continuation_step = None

                for attempt in range(MAX_CONTINUATION_ATTEMPT_RETRIES + 1): # +1 for initial attempt
                    try:
                        cont_response, cont_token_usage, cont_cost_info = await self._call_completions_api(
                            prompt=current_prompt_string, # Use the updated prompt string
                            max_tokens=tokens_to_generate,
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
                            # finish_reason for the current successful step
                            finish_reason = cont_response["choices"][0].get("finish_reason", "unknown")
                            # final_finish_reason is updated with this current successful step's reason
                            final_finish_reason = finish_reason 
                            all_text += cont_text
                            current_text = cont_text # Update current_text with the latest generation for the next prompt
                            
                            # Update token usage
                            total_prompt_tokens += cont_token_usage.prompt_tokens
                            total_completion_tokens += cont_token_usage.completion_tokens
                            total_prompt_cost += cont_cost_info.prompt_cost
                            total_completion_cost += cont_cost_info.completion_cost
                            
                            # Record detailed metrics for continuation API call
                            continuation_call_info = {
                                "call_index": len(detailed_api_calls), # Current number of calls is the index for the new one
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
                                    iteration=callback_args.get("iteration", 0), # This should be the outer loop's iteration concept
                                    step=callback_args.get("step", "reasoning"),
                                    continuation_idx=iteration, # Index of the client's continuation logic
                                    api_call_idx=len(detailed_api_calls) -1 # Index within the client's sequence of calls
                                )
                            
                            continuation_step_succeeded = True
                            last_continuation_step_error = None # Clear error on success
                            break # Break from inner retry loop on success
                        else:
                            # API call succeeded but no choices? Treat as an error for this attempt.
                            last_error_for_this_continuation_step = ValueError("API call successful but no choices returned.")
                            logger.warning(f"Continuation attempt {attempt + 1}/{MAX_CONTINUATION_ATTEMPT_RETRIES + 1} for iteration {iteration} succeeded but returned no choices.")
                            # Fall through to retry or final failure
                        
                    except Exception as e:
                        last_error_for_this_continuation_step = e
                        logger.warning(f"Continuation attempt {attempt + 1}/{MAX_CONTINUATION_ATTEMPT_RETRIES + 1} for iteration {iteration} failed: {str(e)}")
                        if attempt == MAX_CONTINUATION_ATTEMPT_RETRIES: # If this was the last attempt
                            logger.error(f"All {MAX_CONTINUATION_ATTEMPT_RETRIES + 1} attempts for continuation iteration {iteration} failed. Last error: {str(e)}")
                            # The loop will exit as continuation_step_succeeded is False
                        else:
                            backoff = CONTINUATION_RETRY_BACKOFF_BASE * (2 ** attempt) # Exponential backoff
                            logger.info(f"Retrying continuation step in {backoff} seconds...")
                            await asyncio.sleep(backoff)
                
                if not continuation_step_succeeded:
                    # All retries for the current continuation step failed.
                    # Update final_finish_reason to reflect this error state.
                    error_msg = str(last_error_for_this_continuation_step).replace('\\n', ' ').replace('"', "'") # Basic sanitization
                    final_finish_reason = f"continuation_failed_iter_{iteration}_after_retries:_{error_msg[:100]}"
                    if verbose:
                        logger.error(f"Stopping continuations for prompt due to persistent error in continuation step {iteration}.")
                    break # Break from outer continuation while loop
                
                # If continuation step was successful, check if we need to stop based on its finish_reason or token limits
                if finish_reason != "length" or total_completion_tokens >= max_total_tokens:
                    break # Break from outer while loop
                
                # Increment iteration counter for the *outer* continuation loop
                iteration += 1
            
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
        
        # Handle potential null values gracefully
        prompt_tokens = usage.get("prompt_tokens") if usage.get("prompt_tokens") is not None else 0
        completion_tokens = usage.get("completion_tokens") if usage.get("completion_tokens") is not None else 0
        total_tokens = usage.get("total_tokens") if usage.get("total_tokens") is not None else 0

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
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