import os
import json
import time
import random
import logging
import requests
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import List, Dict, Any, Iterator, Optional, Union, Tuple, AsyncIterator

logger = logging.getLogger(__name__)

from src.llm.base_client import ModelClient, TokenUsage, CostInfo

class TogetherModelClient(ModelClient):
    """
    Client for making API calls to Together AI models.
    """
    
    BASE_URL = "https://api.together.xyz/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, input_price_per_million: float = None, output_price_per_million: float = None):
        """
        Initialize the Together API client.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for Together (if None, will use environment variable)
            input_price_per_million: Price per million input tokens (if None, will use default pricing)
            output_price_per_million: Price per million output tokens (if None, will use default pricing)
        """
        # Initialize the base class
        super().__init__(model_name, api_key)
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("No Together API key provided and TOGETHER_API_KEY not found in environment")
        
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
        
        # Load pricing from JSON file
        pricing_file = os.path.join(os.path.dirname(__file__), "pricing", "together_prices.json")
        try:
            with open(pricing_file, "r") as f:
                model_pricing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading pricing information: {str(e)}")
        
        # Get pricing for the model
        if self.model_name not in model_pricing:
            raise ValueError(f"Pricing information not available for model: {self.model_name}")
            
        pricing = model_pricing[self.model_name]
        
        self.input_price_per_million_tokens = pricing["input"]
        self.output_price_per_million_tokens = pricing["output"]
        
        logger.info(f"Set pricing for {self.model_name}: ${self.input_price_per_million_tokens}/M input tokens, ${self.output_price_per_million_tokens}/M output tokens")
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        frequency_penalty: float,
        stream: bool = False,   
        max_retries: int = 500,  # Increased from 15 to 50 for better handling of rate limits
        return_usage: bool = False,  # Whether to return token usage and cost information
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]], Tuple[Dict[str, Any], TokenUsage, CostInfo]]:
        """
        Generate a completion from the Together model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            stream: Whether to stream the response
            max_retries: Maximum number of retries for rate limit errors
            **kwargs: Additional parameters
            
        Returns:
            If stream=False: The complete API response
            If stream=True: An iterator yielding response chunks
        """
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "messages": messages,
            "stream": stream
        }
        
        # Initialize retry counter and backoff time
        retry_count = 0
        backoff_time = 1  # Start with 1 second
        
        while True:
            try:
                response = requests.post(
                    self.BASE_URL, 
                    headers=self.headers, 
                    data=json.dumps(payload),
                    stream=stream
                )
                
                # Check for rate limit errors (429) or service unavailable (503)
                if response.status_code in [429, 503]:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise Exception(f"Together API rate limit exceeded after {max_retries} retries")
                    
                    # Get retry-after header or use exponential backoff with jitter
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        # Add jitter to the Retry-After value to avoid synchronized retries
                        # Only add positive jitter to ensure we never go below the server's requested wait time
                        base_sleep_time = float(retry_after)
                        sleep_time = base_sleep_time + random.uniform(0, 3)  # Add 0-3 seconds of jitter
                    else:
                        # Exponential backoff with jitter, capped at 60 seconds (1 minute)
                        sleep_time = backoff_time + random.uniform(0, 1)
                        backoff_time = min(backoff_time * 2, 60)  # Double the backoff time but cap at 60 seconds
                    
                    logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                
                # Raise exception for other HTTP errors
                response.raise_for_status()
                
                if stream:
                    return self._process_stream(response)
                else:
                    response_json = response.json()
                    
                    # If usage tracking is requested, extract and return token usage and cost
                    if return_usage:
                        token_usage = self.get_token_usage(response_json)
                        cost_info = self.calculate_cost(token_usage)
                        logger.info(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
                        logger.info(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")
                        return response_json, token_usage, cost_info
                    
                    return response_json
                    
            except requests.exceptions.RequestException as e:
                # For connection errors, retry with backoff
                if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise Exception(f"Together API request failed after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff with jitter
                    sleep_time = backoff_time + random.uniform(0, 1)
                    backoff_time *= 2  # Double the backoff time for next retry
                    
                    logger.warning(f"Connection error, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    # For other types of exceptions, raise immediately
                    raise Exception(f"Together API request failed: {str(e)}")
    
    async def generate_completion_async(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        presence_penalty: float,
        frequency_penalty: float,
        stream: bool = False,   
        max_retries: int = 500,  # Increased from 50 to 500 for better handling of rate limits
        return_usage: bool = False,  # Whether to return token usage and cost information
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]], Tuple[Dict[str, Any], TokenUsage, CostInfo]]:
        """
        Generate a completion from the Together model asynchronously.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            stream: Whether to stream the response
            max_retries: Maximum number of retries for rate limit errors
            **kwargs: Additional parameters
            
        Returns:
            If stream=False: The complete API response
            If stream=True: An async iterator yielding response chunks
        """
        payload = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "messages": messages,
            "stream": stream
        }
        
        # Initialize retry counter and backoff time
        retry_count = 0
        backoff_time = 1  # Start with 1 second
        
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.BASE_URL, 
                        headers=self.headers, 
                        json=payload
                    ) as response:
                        # Check for rate limit errors (429) or service unavailable (503)
                        if response.status in [429, 503]:
                            retry_count += 1
                            if retry_count > max_retries:
                                raise Exception(f"Together API rate limit exceeded after {max_retries} retries")
                            
                            # Get retry-after header or use exponential backoff with jitter
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                # Add jitter to the Retry-After value to avoid synchronized retries
                                # Only add positive jitter to ensure we never go below the server's requested wait time
                                base_sleep_time = float(retry_after)
                                sleep_time = base_sleep_time + random.uniform(0, 3)  # Add 0-3 seconds of jitter
                            else:
                                # Exponential backoff with jitter, capped at 60 seconds (1 minute)
                                sleep_time = backoff_time + random.uniform(0, 1)
                                backoff_time = min(backoff_time * 2, 60)  # Double the backoff time but cap at 60 seconds
                            
                            logger.warning(f"Rate limit hit, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                            await asyncio.sleep(sleep_time)
                            continue
                        
                        # Raise exception for other HTTP errors
                        response.raise_for_status()
                        
                        if stream:
                            return self._process_stream_async(response)
                        else:
                            response_json = await response.json()
                            
                            # If usage tracking is requested, extract and return token usage and cost
                            if return_usage:
                                token_usage = self.get_token_usage(response_json)
                                cost_info = self.calculate_cost(token_usage)
                                logger.info(f"Token usage: {token_usage.prompt_tokens} prompt, {token_usage.completion_tokens} completion, {token_usage.total_tokens} total")
                                logger.info(f"Cost: ${cost_info.total_cost:.6f} (${cost_info.prompt_cost:.6f} prompt, ${cost_info.completion_cost:.6f} completion)")
                                return response_json, token_usage, cost_info
                            
                            return response_json
                        
            except aiohttp.ClientError as e:
                # For connection errors, retry with backoff
                if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientOSError, 
                                 aiohttp.ServerDisconnectedError, asyncio.TimeoutError)):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise Exception(f"Together API request failed after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff with jitter
                    sleep_time = backoff_time + random.uniform(0, 1)
                    backoff_time *= 2  # Double the backoff time for next retry
                    
                    logger.warning(f"Connection error, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    # For other types of exceptions, raise immediately
                    raise Exception(f"Together API request failed: {str(e)}")
    
    def _process_stream(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Process a streaming response from the API."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() == '[DONE]':
                        break
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from stream: {e}")
    
    async def _process_stream_async(self, response: aiohttp.ClientResponse) -> AsyncIterator[Dict[str, Any]]:
        """Process a streaming response from the API asynchronously."""
        async for line in response.content:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix
                if line.strip() == '[DONE]':
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from stream: {e}")
    
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], Iterator[str]]:
        """
        Get a response from the model for a specific prompt.
        
        Returns:
            If stream=False: A tuple of (content, finish_reason) where finish_reason indicates why generation stopped
            If stream=True: An iterator yielding content chunks
        """
        # Format the prompt as a message
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            print(f"Sending prompt to Together model {self.model_name}: {prompt[:100]}...")
        
        # Generate completion
        response = self.generate_completion(
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        # Extract content based on whether we're streaming
        if stream:
            return self._extract_streaming_content(response)
        else:
            try:
                content = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")
                return content, finish_reason
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from Together response: {str(e)}")
    
    async def generate_response_async(
        self, 
        prompt: str, 
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str], AsyncIterator[str]]:
        """
        Get a response from the model for a specific prompt asynchronously.
        
        Returns:
            If stream=False: A tuple of (content, finish_reason) where finish_reason indicates why generation stopped
            If stream=True: An async iterator yielding content chunks
        """
        # Format the prompt as a message
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            print(f"Sending prompt to Together model {self.model_name}: {prompt[:100]}...")
        
        # Generate completion
        response = await self.generate_completion_async(
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        # Extract content based on whether we're streaming
        if stream:
            return self._extract_streaming_content_async(response)
        else:
            try:
                content = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")
                return content, finish_reason
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from Together response: {str(e)}")
    
    def _extract_streaming_content(self, response_stream: Iterator[Dict[str, Any]]) -> Union[Iterator[str], Tuple[Iterator[str], str]]:
        """
        Extract content from a streaming response.
        
        If stream=True, yields each content piece as it arrives.
        If stream=False, collects all content and returns it along with the finish_reason from the last chunk.
        """
        finish_reason = "unknown"
        last_chunk = None
        
        for chunk in response_stream:
            try:
                # Save the last chunk to extract finish_reason later
                last_chunk = chunk
                
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from chunk: {e}")
                # Don't let errors break the stream, yield an empty string
                yield ""
        
        # Extract finish_reason from the last chunk if available
        if last_chunk:
            try:
                if "choices" in last_chunk and last_chunk["choices"]:
                    finish_reason = last_chunk["choices"][0].get("finish_reason", "unknown")
            except Exception as e:
                print(f"Error extracting finish_reason from last chunk: {e}")
    
    async def _extract_streaming_content_async(self, response_stream: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[str]:
        """
        Extract content from an async streaming response.
        
        Yields each content piece as it arrives.
        """
        finish_reason = "unknown"
        last_chunk = None
        
        async for chunk in response_stream:
            try:
                # Save the last chunk to extract finish_reason later
                last_chunk = chunk
                
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (KeyError, IndexError) as e:
                print(f"Error extracting content from chunk: {e}")
                # Don't let errors break the stream, yield an empty string
                yield ""
    
    def get_content_only(
        self, 
        prompt: str, 
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Get only the content response from the model (for backward compatibility).
        
        Returns:
            If stream=False: Just the content string
            If stream=True: An iterator yielding content chunks
        """
        if stream:
            return self.generate_response(prompt, stream=True, verbose=verbose, **kwargs)
        else:
            content, _ = self.generate_response(prompt, stream=False, verbose=verbose, **kwargs)
            return content
            
    async def get_content_only_async(
        self, 
        prompt: str, 
        stream: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Get only the content response from the model asynchronously (for backward compatibility).
        
        Returns:
            If stream=False: Just the content string
            If stream=True: An async iterator yielding content chunks
        """
        if stream:
            return await self.generate_response_async(prompt, stream=True, verbose=verbose, **kwargs)
        else:
            content, _ = await self.generate_response_async(prompt, stream=False, verbose=verbose, **kwargs)
            return content
