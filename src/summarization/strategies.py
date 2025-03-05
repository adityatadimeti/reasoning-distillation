"""
Strategies for summarizing reasoning traces.
"""
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class SummarizationStrategy(ABC):
    """
    Abstract base class for summarization strategies.
    """
    
    @abstractmethod
    def preprocess(self, reasoning_trace: str) -> str:
        """
        Preprocess a reasoning trace before summarization.
        
        Args:
            reasoning_trace: The original reasoning trace
            
        Returns:
            Preprocessed reasoning trace
        """
        pass
    
    @abstractmethod
    def postprocess(self, summary: str) -> str:
        """
        Postprocess a summary after generation.
        
        Args:
            summary: The generated summary
            
        Returns:
            Postprocessed summary
        """
        pass

class DefaultStrategy(SummarizationStrategy):
    """
    Default strategy that preserves all reasoning steps.
    """
    
    def preprocess(self, reasoning_trace: str, truncate_length: int = -1) -> str:
        """
        Preprocess a reasoning trace to prepare for summarization.
        
        Args:
            reasoning_trace: The original reasoning trace
            
        Returns:
            Preprocessed reasoning trace
        """
        # Ensure there's a reasonable input
        if not reasoning_trace or not isinstance(reasoning_trace, str):
            return ""
        
        # Clean up the trace
        cleaned_trace = reasoning_trace.strip()
        
        # If the trace is longer than the truncate length, truncate it
        if truncate_length > 0 and len(cleaned_trace) > truncate_length:
            # Take the first truncate_length//2 and last truncate_length//2 characters
            cleaned_trace = cleaned_trace[:truncate_length//2] + "\n...[middle section omitted for length]...\n" + cleaned_trace[-truncate_length//2:]
        
        return cleaned_trace
    
    def postprocess(self, summary: str) -> str:
        """
        Postprocess a summary to improve its quality.
        
        Args:
            summary: The generated summary
            
        Returns:
            Postprocessed summary
        """
        if not summary:
            return ""
        
        # Clean up the summary
        cleaned_summary = summary.strip()
        
        # Remove any meta-comments the model might have added
        cleaned_summary = re.sub(r'^(Here\'s a summary of the reasoning trace:|Summary:|In summary:)', '', cleaned_summary, flags=re.IGNORECASE)
        
        return cleaned_summary.strip()

class ConciseStrategy(DefaultStrategy):
    """
    Strategy focused on producing a very concise summary.
    """
    
    def preprocess(self, reasoning_trace: str) -> str:
        """
        Preprocess with an emphasis on brevity.
        
        Args:
            reasoning_trace: The original reasoning trace
            
        Returns:
            Preprocessed reasoning trace
        """
        # Use the default preprocessing
        cleaned_trace = super().preprocess(reasoning_trace)
        
        # Add markers to emphasize brevity requirement
        # (This is handled in the prompt, but we can reinforce it here)
        return cleaned_trace
    
    def postprocess(self, summary: str) -> str:
        """
        Ensure the summary is concise.
        
        Args:
            summary: The generated summary
            
        Returns:
            Concise postprocessed summary
        """
        # Start with default postprocessing
        cleaned_summary = super().postprocess(summary)
        
        # Ensure brevity by truncating if necessary (rough limit of 1000 chars)
        if len(cleaned_summary) > 1000:
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_summary)
            result = ""
            for sentence in sentences:
                if len(result) + len(sentence) < 1000:
                    result += sentence + " "
                else:
                    break
            
            return result.strip()
        
        return cleaned_summary

class ErrorFocusedStrategy(DefaultStrategy):
    """
    Strategy that focuses on identifying potential errors in reasoning.
    """
    
    def preprocess(self, reasoning_trace: str) -> str:
        """
        Preprocess with an emphasis on error identification.
        
        Args:
            reasoning_trace: The original reasoning trace
            
        Returns:
            Preprocessed reasoning trace
        """
        # Use the default preprocessing
        cleaned_trace = super().preprocess(reasoning_trace)
        
        # Add markers to emphasize error checking requirement
        # (This is handled in the prompt, but we can reinforce it here)
        return cleaned_trace
    
    def postprocess(self, summary: str) -> str:
        """
        Ensure the summary highlights errors.
        
        Args:
            summary: The generated summary
            
        Returns:
            Error-focused postprocessed summary
        """
        # Start with default postprocessing
        cleaned_summary = super().postprocess(summary)
        
        # If no error-related words are found, add a note
        error_words = ['error', 'mistake', 'incorrect', 'wrong', 'flaw', 'issue']
        if not any(word in cleaned_summary.lower() for word in error_words):
            # Add a note if there appears to be no errors mentioned
            # Only if the summary doesn't already end with error-assessment
            if not re.search(r'(no errors|correct|sound|valid|accurate)\s*\.?$', cleaned_summary.lower()):
                cleaned_summary += "\n\nNo obvious errors were identified in the reasoning."
        
        return cleaned_summary

class KeyStepsStrategy(DefaultStrategy):
    """
    Strategy that extracts and highlights key reasoning steps.
    """
    
    def preprocess(self, reasoning_trace: str) -> str:
        """
        Preprocess with an emphasis on key steps.
        
        Args:
            reasoning_trace: The original reasoning trace
            
        Returns:
            Preprocessed reasoning trace
        """
        # Use the default preprocessing
        cleaned_trace = super().preprocess(reasoning_trace)
        
        # Try to identify key steps through patterns
        # Equations, "therefore" statements, etc.
        # (We'll let the model do most of this work through the prompt)
        return cleaned_trace
    
    def postprocess(self, summary: str) -> str:
        """
        Format the summary to clearly show key steps.
        
        Args:
            summary: The generated summary
            
        Returns:
            Key-steps focused postprocessed summary
        """
        # Start with default postprocessing
        cleaned_summary = super().postprocess(summary)
        
        # Try to format key steps with numbering if not already done
        if not re.search(r'^\d+[\)\.]\s', cleaned_summary, re.MULTILINE):
            steps = re.split(r'\n\s*\n', cleaned_summary)
            if len(steps) > 1:
                # Add numbering to steps
                numbered_steps = []
                for i, step in enumerate(steps, 1):
                    if step.strip():
                        numbered_steps.append(f"{i}. {step.strip()}")
                
                return "\n\n".join(numbered_steps)
        
        return cleaned_summary

# Dictionary mapping strategy names to their classes
STRATEGIES = {
    "default": DefaultStrategy,
    "concise": ConciseStrategy,
    "error_focused": ErrorFocusedStrategy,
    "key_steps": KeyStepsStrategy
}

def get_summarization_strategy(strategy_name: str) -> SummarizationStrategy:
    """
    Get a summarization strategy by name.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown summarization strategy: {strategy_name}")
    
    return STRATEGIES[strategy_name]()