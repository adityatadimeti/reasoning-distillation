"""
Summarizers for reasoning traces.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from src.utils.config import Config
from src.models.base import Model

logger = logging.getLogger(__name__)

class Summarizer(ABC):
    """Abstract base class for reasoning trace summarizers."""
    
    def __init__(self, config: Config, model: Model):
        """
        Initialize the summarizer.
        
        Args:
            config: Configuration object
            model: Model to use for summarization
        """
        self.config = config
        self.model = model
        self.summarization_config = config.get("summarization", {})
    
    @abstractmethod
    def summarize(self, reasoning_trace: str, **kwargs) -> str:
        """
        Summarize a reasoning trace.
        
        Args:
            reasoning_trace: The reasoning trace to summarize
            **kwargs: Additional parameters
            
        Returns:
            Summarized reasoning trace
        """
        pass

class PromptBasedSummarizer(Summarizer):
    """
    Summarizer that uses prompt engineering to guide the model.
    """
    
    def __init__(self, config: Config, model: Model):
        """
        Initialize the prompt-based summarizer.
        
        Args:
            config: Configuration object
            model: Model to use for summarization
        """
        super().__init__(config, model)
        
        # Load prompts and strategy
        from src.summarization.prompts import get_summarization_prompt
        from src.summarization.strategies import get_summarization_strategy
        
        self.strategy_name = self.summarization_config.get("strategy", "default")
        self.strategy = get_summarization_strategy(self.strategy_name)
        
        self.prompt_name = self.summarization_config.get("prompt", "default")
        self.prompt_template = get_summarization_prompt(self.prompt_name)
        
        self.max_summary_tokens = self.summarization_config.get("max_tokens", 1000)
        self.temperature = self.summarization_config.get("temperature", 0.5)
    
    def summarize(self, reasoning_trace: str, **kwargs) -> str:
        """
        Summarize a reasoning trace using prompts.
        
        Args:
            reasoning_trace: The reasoning trace to summarize
            **kwargs: Additional parameters
                - max_tokens: Override max tokens config
                - temperature: Override temperature config
                - strategy: Override strategy config
                
        Returns:
            Summarized reasoning trace
        """
        # Get parameters, allowing overrides
        max_tokens = kwargs.get("max_tokens", self.max_summary_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        strategy_name = kwargs.get("strategy", self.strategy_name)
        
        # Get the appropriate strategy if overridden
        if strategy_name != self.strategy_name:
            from src.summarization.strategies import get_summarization_strategy
            strategy = get_summarization_strategy(strategy_name)
        else:
            strategy = self.strategy
        
        # Preprocess the reasoning trace (e.g., truncate if too long)
        processed_trace = strategy.preprocess(reasoning_trace)
        
        # Create the prompt for summarization
        system_content, user_content = self.prompt_template.format(reasoning_trace=processed_trace)
        
        try:
            # Use chat completion for summarization
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            response = self.model.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            summary = response.text
            
            # Post-process the summary
            summary = strategy.postprocess(summary)
            
            logger.debug(f"Generated summary of length {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            # Fall back to a simple extraction-based summary
            return self._fallback_summary(reasoning_trace)
    
    def _fallback_summary(self, reasoning_trace: str) -> str:
        """
        Create a fallback summary if model-based summarization fails.
        
        Args:
            reasoning_trace: The reasoning trace
            
        Returns:
            Simple summary based on extraction
        """
        raise NotImplementedError("Fallback summary not implemented")

class SelfSummarizer(PromptBasedSummarizer):
    """
    Summarizer that uses the same model for reasoning and summarization.
    """
    
    def __init__(self, config: Config, model: Model):
        """
        Initialize the self-summarizer.
        
        Args:
            config: Configuration object
            model: Model to use for both reasoning and summarization
        """
        super().__init__(config, model)
        
        # Specific config for self-summarization
        self.self_summary_config = self.summarization_config.get("self_summary", {})
        
        # May use different prompt or strategy for self-summarization
        self.prompt_name = self.self_summary_config.get("prompt", self.prompt_name)
        self.strategy_name = self.self_summary_config.get("strategy", self.strategy_name)
        
        # Reload prompt and strategy if different
        if self.prompt_name != self.summarization_config.get("prompt", "default"):
            from src.summarization.prompts import get_summarization_prompt
            self.prompt_template = get_summarization_prompt(self.prompt_name)
            
        if self.strategy_name != self.summarization_config.get("strategy", "default"):
            from src.summarization.strategies import get_summarization_strategy
            self.strategy = get_summarization_strategy(self.strategy_name)

class ExternalSummarizer(PromptBasedSummarizer):
    """
    Summarizer that uses a different model for summarization than for reasoning.
    """
    
    def __init__(self, config: Config, reasoning_model: Model, summarization_model: Model):
        """
        Initialize the external summarizer.
        
        Args:
            config: Configuration object
            reasoning_model: Model used for reasoning
            summarization_model: Model to use for summarization
        """
        # Pass the summarization model to the parent constructor
        super().__init__(config, summarization_model)
        
        # Store the reasoning model separately
        self.reasoning_model = reasoning_model
        
        # Specific config for external summarization
        self.external_summary_config = self.summarization_config.get("external_summary", {})
        
        # May use different prompt or strategy for external summarization
        self.prompt_name = self.external_summary_config.get("prompt", self.prompt_name)
        self.strategy_name = self.external_summary_config.get("strategy", self.strategy_name)
        
        # Reload prompt and strategy if different
        if self.prompt_name != self.summarization_config.get("prompt", "default"):
            from src.summarization.prompts import get_summarization_prompt
            self.prompt_template = get_summarization_prompt(self.prompt_name)
            
        if self.strategy_name != self.summarization_config.get("strategy", "default"):
            from src.summarization.strategies import get_summarization_strategy
            self.strategy = get_summarization_strategy(self.strategy_name)

def get_summarizer(config: Config, reasoning_model: Model, summarization_model: Optional[Model] = None) -> Summarizer:
    """
    Factory function to get the appropriate summarizer based on configuration.
    
    Args:
        config: Configuration object
        reasoning_model: Model used for reasoning
        summarization_model: Optional separate model for summarization
        
    Returns:
        Summarizer instance
    """
    summarization_config = config.get("summarization", {})
    method = summarization_config.get("method", "self")
    
    if method == "self" or summarization_model is None:
        logger.info("Using self-summarization")
        return SelfSummarizer(config, reasoning_model)
    else:
        logger.info("Using external summarization")
        return ExternalSummarizer(config, reasoning_model, summarization_model)