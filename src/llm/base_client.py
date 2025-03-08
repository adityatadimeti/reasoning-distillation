from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, List, Optional, Union

class ModelClient(ABC):
    """
    Abstract base class for all LLM API clients.
    Provides a consistent interface regardless of the underlying provider.
    """
    
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