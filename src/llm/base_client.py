from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, AsyncIterator, List, Optional, Union, Tuple, Protocol
from dataclasses import dataclass

@dataclass
class TokenUsage:
    """Token usage information from an API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class CostInfo:
    """Cost information for an API call."""
    prompt_cost: float
    completion_cost: float
    total_cost: float

class ModelClient(ABC):
    """
    Abstract base class for all LLM API clients.
    Provides a consistent interface regardless of the underlying provider.
    
    Implementations can choose to implement either the synchronous or asynchronous methods,
    depending on their needs. Typically, newer implementations should favor the async methods
    for better performance with network I/O.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        # Default pricing - should be overridden by subclasses
        self.input_price_per_million_tokens = 0.0
        self.output_price_per_million_tokens = 0.0
        self.model_name = model_name
        self.api_key = api_key
    
    # Synchronous interface
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[Dict[str, Any], TokenUsage, CostInfo], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the model (synchronous version).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            If stream=False: A tuple of (response, token_usage, cost_info)
            If stream=True: An iterator yielding response chunks
            
        Note:
            This method can be implemented directly, or by wrapping the async version
            in an event loop if the implementation primarily uses async/await.
        """
        raise NotImplementedError("This client does not implement the synchronous interface")
    
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[str, str, TokenUsage, CostInfo], Iterator[str]]:
        """
        Get a complete response from the model for a specific prompt (synchronous version).
        
        Args:
            prompt: The prompt to send to the model
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            If stream=False: A tuple of (content, finish_reason, token_usage, cost_info)
            If stream=True: An iterator yielding response content chunks
            
        Note:
            This method can be implemented directly, or by wrapping the async version
            in an event loop if the implementation primarily uses async/await.
        """
        raise NotImplementedError("This client does not implement the synchronous interface")
    
    # Asynchronous interface
    
    async def generate_completion_async(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ) -> Union[Tuple[Dict[str, Any], TokenUsage, CostInfo], AsyncIterator[Dict[str, Any]]]:
        """
        Generate a completion from the model (asynchronous version).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            If stream=False: A tuple of (response, token_usage, cost_info)
            If stream=True: An async iterator yielding response chunks
        """
        raise NotImplementedError("This client does not implement the asynchronous interface")
    
    async def generate_response_async(
        self, 
        prompt: str,
        max_tokens: int = 8192,  # Reasonable default
        temperature: float = 0.7,
        stream: bool = False,
        enable_continuation: bool = True,  # Support for continuation
        max_total_tokens: int = 24576,  # Target for long generations
        **kwargs
    ) -> Tuple[str, str, TokenUsage, CostInfo]:
        """
        Get a complete response from the model for a prompt (asynchronous version).
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature
            stream: Whether to stream the response (not all implementations support this)
            enable_continuation: Whether to continue generation if truncated
            max_total_tokens: Maximum total tokens to generate across all continuations
            **kwargs: Additional parameters for the model
            
        Returns:
            A tuple of (content, finish_reason, token_usage, cost_info)
        """
        raise NotImplementedError("This client does not implement the asynchronous interface")
    
    # Utility methods
    
    def get_token_usage(self, response: Dict[str, Any]) -> TokenUsage:
        """
        Extract token usage information from an API response.
        
        Args:
            response: The API response containing usage information
            
        Returns:
            TokenUsage object with prompt, completion, and total token counts
        """
        usage = response.get('usage', {})
        return TokenUsage(
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0)
        )
    
    def calculate_cost(self, token_usage: TokenUsage) -> CostInfo:
        """
        Calculate the cost of an API call based on token usage.
        
        Args:
            token_usage: TokenUsage object with token counts
            
        Returns:
            CostInfo object with prompt, completion, and total costs
        """
        prompt_cost = (token_usage.prompt_tokens / 1_000_000) * self.input_price_per_million_tokens
        completion_cost = (token_usage.completion_tokens / 1_000_000) * self.output_price_per_million_tokens
        total_cost = prompt_cost + completion_cost
        
        return CostInfo(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost
        )