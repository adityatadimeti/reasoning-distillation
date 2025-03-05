"""
Functions for extracting answers and key information from reasoning traces.
"""
import re
from typing import Dict, Optional, List, Union
import logging
import os

logger = logging.getLogger(__name__)

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract an answer from LaTeX-style boxed notation.
    
    Args:
        text: Text containing the answer in \\boxed{} format
    
    Returns:
        Extracted answer or None if not found
    """
    if not isinstance(text, str) or not text:
        return None
    
    # Look for \boxed{X} pattern
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[0].strip()
    
    return None

def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract an answer from various formats in text.
    Tries multiple patterns to find the answer.
    
    Args:
        text: Text containing the answer
    
    Returns:
        Extracted answer or None if not found
    """
    logger.debug(f"extract_answer_from_text called with: {text[:50]}...")
    
    if not isinstance(text, str) or not text:
        logger.debug("Text is not a string or is empty, returning None")
        return None
    
    # First try boxed format
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        logger.debug(f"Found boxed answer: {boxed_answer}")
        return boxed_answer
    
    # Try to find "the answer is X" pattern
    answer_patterns = [
        r'[Tt]herefore,?\s+the\s+answer\s+is\s*:?\s*(.*?)(?:\.|$)',
        r'[Tt]he\s+answer\s+is\s*:?\s*(.*?)(?:\.|$)',
        r'[Tt]he\s+result\s+is\s*:?\s*(.*?)(?:\.|$)',
        r'[Tt]he\s+value\s+is\s*:?\s*(.*?)(?:\.|$)',
        r'[Aa]nswer\s*:?\s*(.*?)(?:\.|$)'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches and matches[0].strip():
            logger.debug(f"Found answer using pattern {pattern}: {matches[0].strip()}")
            return matches[0].strip()
    
    # Try to find any numerical result at the end
    lines = text.split('\n')
    for line in reversed(lines):  # Start from the end
        line = line.strip()
        if not line:
            continue
            
        # Check if the line consists mainly of a number
        numeric_match = re.search(r'[-+]?\d*\.?\d+', line)
        if numeric_match and len(numeric_match.group(0)) > len(line) / 2:
            logger.debug(f"Found numeric answer: {numeric_match.group(0)}")
            return numeric_match.group(0)
    
    logger.debug("No answer found, returning None")
    logger.warning(f"Could not extract answer from text: {text[:100]}...")
    return None

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Normalize an extracted answer for consistent comparison.
    
    Args:
        answer: Answer to normalize
    
    Returns:
        Normalized answer
    """
    if answer is None:
        return None
    
    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()
    
    # Remove commas from numbers
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
    
    # Remove dollar signs, percentage signs, etc.
    answer = re.sub(r'[$%]', '', answer)
    
    # Try to convert decimal to integer if it's a whole number
    numeric_match = re.search(r'^[-+]?\d*\.?\d+$', answer)
    if numeric_match:
        try:
            number = float(answer)
            if number.is_integer():
                return str(int(number))
            return answer
        except ValueError:
            pass
    
    return answer

def extract_answer_from_reasoning_result(reasoning_result: Dict, problem_type: str = "general") -> str:
    """
    Extract the answer from a reasoning generation result.
    
    Args:
        reasoning_result: Dictionary containing reasoning generation results
        problem_type: Type of problem ("general" or "aime")
    
    Returns:
        Extracted answer or empty string if not found
    """
    logger.debug("extract_answer_from_reasoning_result called")
    
    # Try to get answer from the dedicated answer field
    if "answer" in reasoning_result and reasoning_result["answer"]:
        logger.debug(f"Trying to extract from answer field: {reasoning_result['answer'][:50]}...")
        answer = extract_answer_from_text(reasoning_result["answer"])
        if answer:
            logger.debug(f"Found answer in answer field: {answer}")
            return answer
    
    # Fall back to extracting from the reasoning trace
    if "reasoning" in reasoning_result and reasoning_result["reasoning"]:
        logger.debug("Trying to extract from reasoning field")
        answer = extract_answer_from_text(reasoning_result["reasoning"])
        if answer:
            logger.debug(f"Found answer in reasoning field: {answer}")
            return answer
    
    # If we have full reasoning with tags, try that
    if "full_reasoning_with_tags" in reasoning_result and reasoning_result["full_reasoning_with_tags"]:
        logger.debug("Trying to extract from full_reasoning_with_tags field")
        answer = extract_answer_from_text(reasoning_result["full_reasoning_with_tags"])
        if answer:
            logger.debug(f"Found answer in full_reasoning_with_tags field: {answer}")
            return answer
    
    # If all extraction methods fail and GPT-4o fallback is enabled, try that
    if os.environ.get("ENABLE_GPT4O_EXTRACTION") == "1":
        logger.info("Regular extraction failed, trying GPT-4o extraction")
        
        # Try GPT-4o on the answer field first
        if "answer" in reasoning_result and reasoning_result["answer"]:
            answer = extract_with_gpt4o(reasoning_result["answer"], problem_type)
            if answer:
                logger.info(f"GPT-4o extracted answer from answer field: {answer}")
                return answer
        
        # Then try on the reasoning field
        if "reasoning" in reasoning_result and reasoning_result["reasoning"]:
            answer = extract_with_gpt4o(reasoning_result["reasoning"], problem_type)
            if answer:
                logger.info(f"GPT-4o extracted answer from reasoning field: {answer}")
                return answer
    
    logger.warning("Could not extract answer from reasoning result")
    return ""

def extract_with_gpt4o(text: str, problem_type: str = "general") -> Optional[str]:
    """
    Use GPT-4o to extract the answer when regex methods fail.
    
    Args:
        text: Text containing the answer
        problem_type: Type of problem ("general" or "aime")
    
    Returns:
        Extracted answer or None if not found
    """
    try:
        # Check if OpenAI is available
        import openai
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI package not installed, cannot use GPT-4o extraction")
        return None
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment, cannot use GPT-4o extraction")
        return None
    
    # Initialize client
    client = OpenAI(api_key=api_key)
    
    # Prepare prompt based on problem type
    if problem_type.lower() == "aime":
        prompt = f"""
        Extract the final numerical answer from this AIME problem solution.
        AIME answers are always integers between 0 and 999.
        
        Important AIME answer patterns:
        1. If the answer is expressed as a fraction m/n in lowest form, the answer is often m+n.
        2. If the answer involves absolute value |x|, return the value without the bars.
        3. For logarithmic expressions, compute the final value.
        
        Return ONLY the integer answer, with no explanation.
        If no clear answer is found, return "NO_ANSWER_FOUND".
        
        TEXT:
        {text}
        """
        system_content = "You are a helpful assistant that extracts AIME competition answers from mathematical solutions."
    else:
        prompt = f"""
        Extract the numerical answer or mathematical expression from the following text. 
        Return ONLY the number, fraction, or mathematical expression, with no explanation.
        
        Guidelines:
        - If there are multiple possible answers, return the final one
        - For equations like "x = 3/4", return just "3/4"
        - For AIME problems, the answer should be an integer between 0 and 999
        - If the answer is a fraction like m/n, return it in simplified form
        - If no clear answer is found, return "NO_ANSWER_FOUND"
        
        TEXT:
        {text}
        """
        system_content = "You are a helpful assistant that extracts numerical answers and mathematical expressions from text."
    
    try:
        # Call GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Extract answer
        extracted = response.choices[0].message.content.strip()
        
        # If GPT-4o couldn't find an answer
        if extracted == "NO_ANSWER_FOUND":
            logger.warning("GPT-4o could not find an answer")
            return None
            
        logger.info(f"GPT-4o extracted: {extracted}")
        return extracted
        
    except Exception as e:
        logger.error(f"Error using GPT-4o for extraction: {e}")
        return None