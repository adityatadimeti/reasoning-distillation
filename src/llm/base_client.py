from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, List, Optional, Union, Tuple
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
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        # Default pricing - should be overridden by subclasses
        self.input_price_per_million_tokens = 0.0
        self.output_price_per_million_tokens = 0.0
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            If stream=False: The complete API response
            If stream=True: An iterator yielding response chunks
        """
        pass
    
    @abstractmethod
    def generate_response(
        self, 
        prompt: str, 
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Get a complete response from the model for a specific prompt.
        
        Args:
            prompt: The prompt to send to the model
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to generate_completion
            
        Returns:
            If stream=False: The model's complete response content as a string
            If stream=True: An iterator yielding response content chunks
        """
        pass
    
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