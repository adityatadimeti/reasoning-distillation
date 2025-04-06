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

from src.llm.base_client import ModelClient

class FireworksModelClient(ModelClient):
    """
    Client for making API calls to Fireworks AI models.
    """
    
    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the Fireworks API client.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for Fireworks (if None, will use environment variable)
        """
        # Load environment variables
        load_dotenv()
        
        self.model_name = model_name
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("No Fireworks API key provided and FIREWORKS_API_KEY not found in environment")
        
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
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
        max_retries: int = 50,  # Increased from 15 to 50 for better handling of rate limits
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the Fireworks model.
        
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
                        raise Exception(f"Fireworks API rate limit exceeded after {max_retries} retries")
                    
                    # Get retry-after header or use exponential backoff with jitter
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        sleep_time = float(retry_after)
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
                    return response.json()
                    
            except requests.exceptions.RequestException as e:
                # For connection errors, retry with backoff
                if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise Exception(f"Fireworks API request failed after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff with jitter
                    sleep_time = backoff_time + random.uniform(0, 1)
                    backoff_time *= 2  # Double the backoff time for next retry
                    
                    logger.warning(f"Connection error, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    # For other types of exceptions, raise immediately
                    raise Exception(f"Fireworks API request failed: {str(e)}")
    
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
        max_retries: int = 50,  # Increased from 15 to 50 for better handling of rate limits
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Generate a completion from the Fireworks model asynchronously.
        
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
                                raise Exception(f"Fireworks API rate limit exceeded after {max_retries} retries")
                            
                            # Get retry-after header or use exponential backoff with jitter
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                sleep_time = float(retry_after)
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
                            return await response.json()
                        
            except aiohttp.ClientError as e:
                # For connection errors, retry with backoff
                if isinstance(e, (aiohttp.ClientConnectionError, aiohttp.ClientOSError, 
                                 aiohttp.ServerDisconnectedError, asyncio.TimeoutError)):
                    retry_count += 1
                    if retry_count > max_retries:
                        raise Exception(f"Fireworks API request failed after {max_retries} retries: {str(e)}")
                    
                    # Exponential backoff with jitter
                    sleep_time = backoff_time + random.uniform(0, 1)
                    backoff_time *= 2  # Double the backoff time for next retry
                    
                    logger.warning(f"Connection error, retrying in {sleep_time:.2f} seconds (retry {retry_count}/{max_retries})")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    # For other types of exceptions, raise immediately
                    raise Exception(f"Fireworks API request failed: {str(e)}")
    
    def _process_stream(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Process a streaming response from the API."""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line == '[DONE]':
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, Line: {line}")
    
    async def _process_stream_async(self, response: aiohttp.ClientResponse) -> AsyncIterator[Dict[str, Any]]:
        """Process a streaming response from the API asynchronously."""
        async for line in response.content:
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    line_str = line_str[6:]
                if line_str == '[DONE]':
                    break
                try:
                    yield json.loads(line_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, Line: {line_str}")
    
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
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            print(f"\n[VERBOSE] Fireworks API Request to {self.model_name}")
            print(f"[VERBOSE] Messages: {json.dumps(messages, indent=2)}")
            print(f"[VERBOSE] Parameters: {json.dumps({k: v for k, v in kwargs.items()}, indent=2)}")
        
        if stream:
            response_stream = self.generate_completion(messages, stream=True, **kwargs)
            return self._extract_streaming_content(response_stream)
        else:
            response = self.generate_completion(messages, stream=False, **kwargs)
            try:
                content = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")
                return content, finish_reason
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from Fireworks response: {str(e)}")
    
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
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            print(f"\n[VERBOSE] Fireworks API Request to {self.model_name}")
            print(f"[VERBOSE] Messages: {json.dumps(messages, indent=2)}")
            print(f"[VERBOSE] Parameters: {json.dumps({k: v for k, v in kwargs.items()}, indent=2)}")
        
        if stream:
            response_stream = await self.generate_completion_async(messages, stream=True, **kwargs)
            return self._extract_streaming_content_async(response_stream)
        else:
            response = await self.generate_completion_async(messages, stream=False, **kwargs)
            try:
                content = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "unknown")
                return content, finish_reason
            except (KeyError, IndexError) as e:
                raise Exception(f"Failed to extract content from Fireworks response: {str(e)}")
    
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