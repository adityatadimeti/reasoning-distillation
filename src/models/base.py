"""
Base classes for model interaction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from src.utils.config import Config
import tiktoken

class Model(ABC):
    """
    Abstract base class for all model interactions.
    """
    def __init__(self, config: Config):
        """
        Initialize the model with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.model_name = self.model_config.get("name", "")
        
        # Default model parameters
        self.max_tokens = self.model_config.get("max_tokens", 1000)
        self.temperature = self.model_config.get("temperature", 0.7)
        self.top_p = self.model_config.get("top_p", 1.0)
        self.frequency_penalty = self.model_config.get("frequency_penalty", 0.0)
        self.presence_penalty = self.model_config.get("presence_penalty", 0.0)
        
        # API settings
        self.timeout = self.model_config.get("timeout", 30)
        self.retries = self.model_config.get("retries", 3)
        self.retry_delay = self.model_config.get("retry_delay", 2)
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_tokens: Maximum number of tokens to generate (overrides config)
            temperature: Temperature for generation (overrides config)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing at least the generated text and metadata
        """
        pass
    
    @abstractmethod
    def generate_reasoning(self, 
                          question: str, 
                          max_extensions: Optional[int] = None,
                          target_token_count: Optional[int] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate reasoning trace for a question.
        
        Args:
            question: The input question to reason about
            max_extensions: Maximum number of reasoning extensions (overrides config)
            target_token_count: Target number of tokens for reasoning (overrides config)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing at least the reasoning trace, answer, and metadata
        """
        pass
    
    @abstractmethod
    def summarize_reasoning(self, 
                           reasoning_trace: str, 
                           **kwargs) -> str:
        """
        Summarize a reasoning trace.
        
        Args:
            reasoning_trace: The reasoning trace to summarize
            **kwargs: Additional model-specific parameters
            
        Returns:
            Summarized reasoning trace
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text using tiktoken.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # print(f"Estimating tokens with tiktoken (inaccurate for DeepSeek)")
        try:
            # Get the encoding based on the model name if possible
            # Default to cl100k_base which is used for newer models like GPT-4
            encoding_name = "cl100k_base"
            
            # Get the encoding and count tokens
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)
            return len(tokens)
            
        except ImportError:
            # Fallback to a simple approximation if tiktoken is not installed
            return len(text.split())


class ModelResponse:
    """
    Class to standardize model responses across different APIs.
    """
    def __init__(self, 
                text: str, 
                prompt: str,
                tokens_used: Optional[int] = None,
                model_name: Optional[str] = None,
                raw_response: Any = None):
        """
        Initialize a model response.
        
        Args:
            text: Generated text from the model
            prompt: Original prompt sent to the model
            tokens_used: Total tokens used (prompt + completion)
            model_name: Name of the model used
            raw_response: Raw response from the API
        """
        self.text = text
        self.prompt = prompt
        self.tokens_used = tokens_used
        self.model_name = model_name
        self.raw_response = raw_response
    
    def __str__(self) -> str:
        """String representation of the response"""
        return self.text
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary"""
        return {
            "text": self.text,
            "prompt": self.prompt,
            "tokens_used": self.tokens_used,
            "model_name": self.model_name
        }