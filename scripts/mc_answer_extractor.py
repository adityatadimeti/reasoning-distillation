#!/usr/bin/env python3
"""
Updated answer extractor for multiple-choice answers.
Extracts letter choices from model responses in various formats.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_mc_answer(text: str) -> Optional[str]:
    """
    Extract a multiple choice answer (A, B, C, D) from model output.
    Handles various formats:
    - Boxed letter: \boxed{A}
    - Parenthesized letter: (A), (a)
    - Direct letter preceded/followed by "answer", "option", "choice"
    
    Args:
        text: The full text output from the model
        
    Returns:
        The extracted answer letter as a single capital letter: A, B, C, or D, or None if no answer found
    """
    # Handle None inputs
    if text is None:
        logger.warning("Input text is None, cannot extract answer")
        return None
    
    # Pattern 1: Look for \boxed{X} where X is A, B, C, or D
    boxed_pattern = r'\\boxed\{([A-Da-d])\}'
    boxed_matches = re.findall(boxed_pattern, text)
    
    if boxed_matches:
        # Convert to uppercase and return as single letter
        answer_letter = boxed_matches[-1].strip().upper()
        logger.info(f"Extracted answer from boxed format: {answer_letter}")
        return answer_letter
    
    # Pattern 2: Look for (X) where X is A, B, C, or D
    paren_pattern = r'\(([A-Da-d])\)'
    paren_matches = re.findall(paren_pattern, text)
    
    if paren_matches:
        answer_letter = paren_matches[-1].strip().upper()
        logger.info(f"Extracted answer from parenthesized format: {answer_letter}")
        return answer_letter
    
    # Pattern 3: Look for "answer is X" or "choice X" or "option X" etc.
    answer_word_pattern = r'(?:answer|choice|option|select)(?:\s+is)?\s+(?:\()?([A-Da-d])(?:\))?'
    answer_word_matches = re.findall(answer_word_pattern, text, re.IGNORECASE)
    
    if answer_word_matches:
        answer_letter = answer_word_matches[-1].strip().upper()
        logger.info(f"Extracted answer from phrase: {answer_letter}")
        return answer_letter
    
    # Pattern 4: Last resort, look for any standalone A, B, C, D in the last paragraph
    if text:
        # Get the last paragraph
        paragraphs = text.split('\n\n')
        last_paragraph = paragraphs[-1]
        
        # Find all standalone A, B, C, D letters
        standalone_pattern = r'\b([A-Da-d])\b'
        standalone_matches = re.findall(standalone_pattern, last_paragraph)
        
        if standalone_matches:
            answer_letter = standalone_matches[-1].strip().upper()
            logger.info(f"Extracted answer from standalone letter in last paragraph: {answer_letter}")
            return answer_letter
    
    # If no answer is found at all, log a warning
    logger.warning("No multiple-choice answer found in the output")
    
    # Return None to indicate no answer was found
    return None

def test_extractor():
    """Test the multiple-choice answer extractor with various formats."""
    test_cases = [
        (r"The correct answer is \boxed{A}", "A"),
        (r"I select option \boxed{b}", "B"),
        (r"The answer is (C)", "C"),
        (r"After analyzing all options, I choose (d).", "D"),
        (r"Answer: A", "A"),
        (r"The correct choice is B", "B"),
        (r"I select option C", "C"),
        (r"Looking at all possibilities, we can see that D is correct.", "D"),
        (r"Some complicated reasoning...\n\nTherefore A.", "A"),
        (r"This problem has a more nuanced solution, which corresponds to choice B.", "B"),
    ]
    
    print("Testing multiple-choice answer extractor:")
    print("-" * 50)
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = extract_mc_answer(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"Test {i}: {status}")
        print(f"Input: {input_text}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        print("-" * 50)

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_extractor() 