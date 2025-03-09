import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model output, focusing on LaTeX boxed expressions.
    If no boxed answer is found, falls back to finding the last integer in the text.
    
    Args:
        text: The full text output from the model
        
    Returns:
        The extracted answer as a string, or None if no answer is found
    """
    # Try to find content in a LaTeX \boxed{} environment
    boxed_pattern = r'\\boxed\{([^{}]+)\}'
    
    # First search for the \boxed pattern
    boxed_matches = re.findall(boxed_pattern, text)
    
    if boxed_matches:
        # Return the content of the last \boxed{} in the text
        # (Usually, the final answer comes last)
        answer = boxed_matches[-1].strip()
        logger.info(f"Extracted answer from boxed expression: {answer}")
        return answer
    
    # If no boxed answer is found, try to find the last integer in the text
    integer_pattern = r'\b\d+\b'
    integer_matches = re.findall(integer_pattern, text)
    
    if integer_matches:
        answer = integer_matches[-1].strip()
        logger.info(f"No boxed answer found, using last integer as answer: {answer}")
        return answer
    
    # If no answer is found at all, log a warning
    logger.warning("No answer found in the output (neither boxed nor integer)")
    
    # Return None to indicate no answer was found
    return None
