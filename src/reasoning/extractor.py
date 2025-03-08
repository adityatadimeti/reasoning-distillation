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
    
    # If no boxed answer is found, log a warning
    logger.warning("No boxed answer found in the output")
    
    # Return None to indicate no answer was found
    return None
