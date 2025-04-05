import os
import json
import requests
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import List, Dict, Any, Iterator, Optional, Union, Tuple, AsyncIterator

from src.llm.base_client import ModelClient

class TogetherModelClient(ModelClient):
    """
    Client for making API calls to Together AI models.
    """
    
    BASE_URL = "https://api.together.xyz/v1/chat/completions"
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the Together API client.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for Together (if None, will use environment variable)
        """
        # Load environment variables
        load_dotenv()
        
        self.model_name = model_name
        
        # Get API key from args or environment
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("No Together API key provided and TOGETHER_API_KEY not found in environment")
        
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
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the Together model.
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
        
        try:
            response = requests.post(
                self.BASE_URL, 
                headers=self.headers, 
                data=json.dumps(payload),
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
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
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Generate a completion from the Together model asynchronously.
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
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL, 
                    headers=self.headers, 
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    if stream:
                        return self._process_stream_async(response)
                    else:
                        return await response.json()
                    
        except aiohttp.ClientError as e:
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
