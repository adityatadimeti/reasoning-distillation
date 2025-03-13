import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model output, focusing on LaTeX boxed expressions.
    
    Args:
        text: The full text output from the model
        
    Returns:
        The extracted answer as a string, or None if no answer is found
    """
    # Handle None inputs
    if text is None:
        logger.warning("Input text is None, cannot extract answer")
        return None
    
    # Try to find content in a LaTeX \boxed{} environment
    # This pattern handles nested braces by using a more complex regex
    boxed_pattern = r'\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))+)\}'
    
    # First search for the \boxed pattern
    boxed_matches = re.findall(boxed_pattern, text)
    
    if boxed_matches:
        # Return the content of the last \boxed{} in the text
        # (Usually, the final answer comes last)
        answer = boxed_matches[-1].strip()
        logger.info(f"Extracted answer from boxed expression: {answer}")
        return answer
    
    # If no boxed answer is found, log a warning
    logger.warning("No boxed answer found in the output")
    
    # Return None to indicate no answer was found
    return None

def extract_reasoning_trace(text: str, allow_fallback: bool = False) -> Optional[str]:
    """
    Extract the reasoning trace from model output, contained within <think> tags.
    If multiple <think> tag pairs exist, all content from all pairs will be extracted.
    
    Args:
        text: The full text output from the model
        allow_fallback: If True, return the original text when no <think> tags are found
                        If False, return None when no <think> tags are found
        
    Returns:
        The extracted reasoning trace as a string, or None if no trace is found
    """
    if not text:
        return None
    
    # Look for content between <think> and </think> tags
    pattern = r'<think>(.*?)</think>'
    # Use re.DOTALL to allow matching across multiple lines
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Join all <think> blocks with newlines if there are multiple
        reasoning_trace = "\n\n".join([match.strip() for match in matches])
        logger.info(f"Successfully extracted reasoning trace: found {len(matches)} <think> blocks")
        return reasoning_trace
    
    # If no <think> tags are found, log a warning
    logger.warning("No reasoning trace found between <think> tags")
    
    if allow_fallback:
        # If fallback is allowed, return the original text
        logger.info("Returning original text as reasoning trace (fallback enabled)")
        return text
    else:
        # If fallback is not allowed, return None
        logger.warning("Fallback is disabled, returning None instead of original text")
        return None
