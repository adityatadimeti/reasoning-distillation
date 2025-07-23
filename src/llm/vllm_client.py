import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator, AsyncIterator
import aiohttp
import requests
from dataclasses import dataclass

from src.llm.base_client import ModelClient, TokenUsage, CostInfo

logger = logging.getLogger(__name__)

class VLLMModelClient(ModelClient):
    """
    Client for vLLM (Versatile Large Language Model) server.
    
    vLLM provides a fast and easy-to-use inference server for LLMs.
    This client communicates with a vLLM server running locally or remotely.
    """
    
    def __init__(
        self, 
        model_name: str,
        host: str = "localhost",
        port: int = 8000,
        api_key: Optional[str] = None,
        max_model_len: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize vLLM client.
        
        Args:
            model_name: Name of the model (should match the model loaded in vLLM server)
            host: Host where vLLM server is running
            port: Port where vLLM server is listening
            api_key: Optional API key for authentication
            max_model_len: Maximum model context length
            **kwargs: Additional parameters
        """
        super().__init__(model_name, api_key)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.max_model_len = max_model_len or 32768  # Default context length
        
        # vLLM uses OpenAI-compatible API, so pricing is zero for local inference
        self.input_price_per_million_tokens = 0.0
        self.output_price_per_million_tokens = 0.0
        
        # Verify server is running
        self._verify_server()
    
    def _verify_server(self):
        """Verify that the vLLM server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                logger.warning(f"vLLM server health check returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to vLLM server at {self.base_url}: {e}")
            raise ConnectionError(f"Cannot connect to vLLM server at {self.base_url}. "
                                "Please ensure the server is running with: "
                                f"python -m vllm.entrypoints.openai.api_server --model {self.model_name}")
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[Dict[str, Any], TokenUsage, CostInfo], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the vLLM model (synchronous version).
        
        vLLM uses OpenAI-compatible API endpoints.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }
        
        # Add optional parameters if provided
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            payload["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        try:
            if stream:
                # For streaming, return an iterator
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                return self._stream_response(response)
            else:
                # For non-streaming, return the complete response
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                token_usage = self.get_token_usage(result)
                cost_info = self.calculate_cost(token_usage)
                
                return result, token_usage, cost_info
                
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API request failed: {e}")
            raise
    
    def _stream_response(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Handle streaming response from vLLM."""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response: {data_str}")
    
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str, TokenUsage, CostInfo], Iterator[str]]:
        """
        Get a complete response from the model for a specific prompt (synchronous version).
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Get default parameters from kwargs or use sensible defaults
        max_tokens = kwargs.get("max_tokens", 8192)
        temperature = kwargs.get("temperature", 0.7)
        
        if stream:
            # For streaming, yield content chunks
            response_iter = self.generate_completion(
                messages, max_tokens, temperature, stream=True, **kwargs
            )
            for chunk in response_iter:
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
        else:
            # For non-streaming, return complete response
            response, token_usage, cost_info = self.generate_completion(
                messages, max_tokens, temperature, stream=False, **kwargs
            )
            
            content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0]["finish_reason"]
            
            return content, finish_reason, token_usage, cost_info
    
    async def generate_completion_async(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[Dict[str, Any], TokenUsage, CostInfo], AsyncIterator[Dict[str, Any]]]:
        """
        Generate a completion from the vLLM model (asynchronous version).
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }
        
        # Add optional parameters if provided
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            payload["top_k"] = kwargs["top_k"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        # Set timeout to prevent hanging on long requests
        # With max_tokens=30000, generation can take a long time
        timeout = aiohttp.ClientTimeout(total=3600)  # 60 minute timeout for very long generations
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                if stream:
                    # For streaming, return an async iterator
                    async with session.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        return self._async_stream_response(response)
                else:
                    # For non-streaming, return the complete response
                    async with session.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        token_usage = self.get_token_usage(result)
                        cost_info = self.calculate_cost(token_usage)
                        
                        return result, token_usage, cost_info
                        
            except aiohttp.ClientError as e:
                logger.error(f"vLLM async API request failed: {e}")
                raise
    
    async def _async_stream_response(self, response: aiohttp.ClientResponse) -> AsyncIterator[Dict[str, Any]]:
        """Handle async streaming response from vLLM."""
        async for line in response.content:
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response: {data_str}")
    
    async def generate_response_async(
        self, 
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        enable_continuation: bool = True,
        max_total_tokens: int = 24576,
        **kwargs
    ) -> Tuple[str, str, TokenUsage, CostInfo]:
        """
        Get a complete response from the model for a prompt (asynchronous version).
        
        Supports automatic continuation if the response is truncated.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Track API calls for detailed metrics
        api_calls = []
        total_content = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        finish_reason = "stop"
        continuation_count = 0
        
        while True:
            # Check if we've reached the continuation limit
            if continuation_count >= kwargs.get("max_continuations", 3):
                logger.warning(f"Reached maximum continuation limit ({continuation_count})")
                finish_reason = "max_continuations"
                break
            
            # Calculate remaining tokens
            remaining_tokens = min(max_tokens, max_total_tokens - total_completion_tokens)
            if remaining_tokens <= 0:
                logger.warning("Reached total token limit")
                finish_reason = "length"
                break
            
            # Generate response
            response, token_usage, cost_info = await self.generate_completion_async(
                messages, remaining_tokens, temperature, stream=False, **kwargs
            )
            
            # Extract content and finish reason
            choice = response["choices"][0]
            content = choice["message"]["content"]
            current_finish_reason = choice["finish_reason"]
            
            # Accumulate content and tokens
            total_content += content
            total_prompt_tokens += token_usage.prompt_tokens
            total_completion_tokens += token_usage.completion_tokens
            
            # Track this API call
            api_calls.append({
                "continuation": continuation_count,
                "tokens": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                },
                "cost": {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                },
                "finish_reason": current_finish_reason
            })
            
            # Check if we need to continue
            if not enable_continuation or current_finish_reason != "length":
                finish_reason = current_finish_reason
                break
            
            # Prepare for continuation
            logger.info(f"Continuing generation (continuation {continuation_count + 1})")
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": total_content},
                {"role": "user", "content": "Continue from where you left off."}
            ]
            continuation_count += 1
        
        # Calculate total usage and cost
        total_usage = TokenUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens
        )
        total_cost = self.calculate_cost(total_usage)
        
        # Log token usage if verbose
        if kwargs.get("verbose", False):
            logger.info(f"vLLM token usage - Prompt: {total_prompt_tokens}, "
                       f"Completion: {total_completion_tokens}, Total: {total_usage.total_tokens}")
            logger.info(f"vLLM cost - Total: ${total_cost.total_cost:.4f} (free for local inference)")
        
        # Return with detailed API call information
        return total_content, finish_reason, total_usage, total_cost, api_calls 