import re
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

# Dictionary of available extractors
EXTRACTOR_REGISTRY = {}

def register_extractor(name: str):
    """Decorator to register an extractor function"""
    def decorator(func: Callable):
        EXTRACTOR_REGISTRY[name] = func
        return func
    return decorator

@register_extractor('default')
def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model output, focusing on LaTeX boxed expressions.
    If no boxed answer is found, falls back to finding numbers in \(NUMBER\) format.
    If that fails too, falls back to finding the last integer in the text.
    
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
    
    # If no boxed answer is found, try to find numbers in \(NUMBER\) format
    paren_pattern = r'\\\(\s*(\d+)\s*\\\)'
    paren_matches = re.findall(paren_pattern, text)
    
    if paren_matches:
        answer = paren_matches[-1].strip()
        logger.info(f"No boxed answer found, using last number in \\(NUMBER\\) format: {answer}")
        return answer
    
    # If no answer in \(NUMBER\) format is found, try to find the last integer in the text
    integer_pattern = r'\b\d+\b'
    integer_matches = re.findall(integer_pattern, text)
    
    if integer_matches:
        answer = integer_matches[-1].strip()
        logger.info(f"No boxed or parenthesized answer found, using last integer as answer: {answer}")
        return answer
    
    # If no answer is found at all, log a warning
    logger.warning("No answer found in the output (neither boxed, parenthesized, nor integer)")
    
    # Return None to indicate no answer was found
    return None

@register_extractor('gpqa_mc')
def extract_gpqa_mc_answer(text: str) -> Optional[str]:
    """
    Extract the answer from GPQA multiple choice model output.
    Uses regex patterns to find answers in the format specified by the GPQA paper.
    
    Args:
        text: The full text output from the model
        
    Returns:
        The extracted answer as a string (A, B, C, or D), or None if no answer is found
    """
    if text is None:
        logger.warning("Input text is None, cannot extract answer")
        return None
    
    # Regex pattern based on the GPQA paper's answer extraction
    patterns = [
        r'answer is \(([A-D])\)',
        r'Answer: \(([A-D])\)',
        r'answer: \(([A-D])\)',
        r'answer \(([A-D])\)',
        r'The correct answer is \(([A-D])\)',
        r'the correct answer is \(([A-D])\)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip()
            logger.info(f"Extracted GPQA MC answer: {answer}")
            return answer
    
    # If no answer is found with the specific patterns, try a more general approach
    general_pattern = r'[Cc]orrect answer is \(?([A-D])\)?'
    matches = re.findall(general_pattern, text)
    if matches:
        answer = matches[-1].strip()
        logger.info(f"Extracted GPQA MC answer using general pattern: {answer}")
        return answer
    
    # If still no answer, look for simple letter mentions at the end
    last_lines = text.strip().split('\n')[-3:]
    for line in reversed(last_lines):
        if re.search(r'\b[A-D]\b', line):
            matches = re.findall(r'\b([A-D])\b', line)
            if matches:
                answer = matches[-1]
                logger.info(f"Extracted GPQA MC answer from end of text: {answer}")
                return answer
    
    logger.warning("No GPQA MC answer found in the output")
    return None

def get_extractor(extractor_type: str = 'default') -> Callable:
    """
    Get the appropriate answer extractor function based on the specified type.
    
    Args:
        extractor_type: The type of extractor to use (e.g., 'default', 'gpqa_mc')
        
    Returns:
        The extractor function
    """
    if extractor_type not in EXTRACTOR_REGISTRY:
        logger.warning(f"Extractor type '{extractor_type}' not found, using default extractor")
        return EXTRACTOR_REGISTRY['default']
    
    return EXTRACTOR_REGISTRY[extractor_type]

def extract_answer_with_config(text: str, config: Dict[str, Any] = None) -> Optional[str]:
    """
    Extract the answer from model output based on the experiment configuration.
    This is a convenience function that selects the appropriate extractor based on the config.
    
    Args:
        text: The full text output from the model
        config: The experiment configuration dictionary
        
    Returns:
        The extracted answer as a string, or None if no answer is found
    """
    if config is None:
        config = {}
    
    # Get the extractor type from the configuration
    extractor_type = config.get("answer_extractor", "default")
    
    # Get the extractor function
    extractor_fn = get_extractor(extractor_type)
    
    # Extract the answer
    return extractor_fn(text)

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
