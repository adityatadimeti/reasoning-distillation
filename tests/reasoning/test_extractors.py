import pytest
from src.reasoning.extractor import extract_answer, extract_reasoning_trace

def test_extract_reasoning_trace():
    """Test the extraction of reasoning traces from model outputs."""
    
    # Test case 1: Standard think tags with content
    input_text = """
    <think>
    Let me solve this step by step.
    First, I'll calculate the area of the rectangle.
    Area = length × width = 10 × 5 = 50
    Now I need to find the perimeter.
    Perimeter = 2 × (length + width) = 2 × (10 + 5) = 2 × 15 = 30
    </think>
    
    The area of the rectangle is 50 square units and the perimeter is 30 units.
    Therefore, the answer is \boxed{30}.
    """
    
    expected_output = """Let me solve this step by step.
    First, I'll calculate the area of the rectangle.
    Area = length × width = 10 × 5 = 50
    Now I need to find the perimeter.
    Perimeter = 2 × (length + width) = 2 × (10 + 5) = 2 × 15 = 30"""
    
    result = extract_reasoning_trace(input_text)
    assert result.strip() == expected_output.strip()
    
    # Test case 2: Multiple think tags (should extract all blocks and join them)
    input_text = """
    <think>
    This is the first think block.
    </think>
    
    Some intermediate text.
    
    <think>
    This is the second think block.
    </think>
    """
    
    expected_output = """This is the first think block.

This is the second think block."""
    
    result = extract_reasoning_trace(input_text)
    assert result.strip() == expected_output.strip()
    
    # Test case 3: No think tags with allow_fallback=True
    input_text = """
    This text has no think tags.
    It should be returned as is.
    """
    
    result = extract_reasoning_trace(input_text, allow_fallback=True)
    assert result.strip() == input_text.strip()
    
    # Test case 3b: No think tags with allow_fallback=False (default)
    result = extract_reasoning_trace(input_text)
    assert result is None
    
    # Test case 4: Empty think tags
    input_text = "<think></think>"
    
    result = extract_reasoning_trace(input_text)
    assert result.strip() == ""
    
    # Test case 5: None input
    result = extract_reasoning_trace(None)
    assert result is None
    
    # Test case 6: Empty string input
    result = extract_reasoning_trace("")
    assert result is None
    
    # Test case 7: Multiline complex content
    input_text = """
    <think>
    Let me solve this problem:
    
    Step 1: Identify the variables.
    Let x = number of apples
    Let y = number of oranges
    
    Step 2: Set up equations.
    x + y = 20  (equation 1)
    2x + 3y = 50  (equation 2)
    
    Step 3: Solve the system of equations.
    From equation 1: x = 20 - y
    Substitute into equation 2:
    2(20 - y) + 3y = 50
    40 - 2y + 3y = 50
    40 + y = 50
    y = 10
    
    Therefore, x = 20 - 10 = 10
    
    So we have 10 apples and 10 oranges.
    </think>
    
    I've determined that there are 10 apples and 10 oranges.
    The answer is \boxed{10}.
    """
    
    expected_output = """Let me solve this problem:
    
    Step 1: Identify the variables.
    Let x = number of apples
    Let y = number of oranges
    
    Step 2: Set up equations.
    x + y = 20  (equation 1)
    2x + 3y = 50  (equation 2)
    
    Step 3: Solve the system of equations.
    From equation 1: x = 20 - y
    Substitute into equation 2:
    2(20 - y) + 3y = 50
    40 - 2y + 3y = 50
    40 + y = 50
    y = 10
    
    Therefore, x = 20 - 10 = 10
    
    So we have 10 apples and 10 oranges."""
    
    result = extract_reasoning_trace(input_text)
    assert result.strip() == expected_output.strip()

def test_extract_answer():
    """Test the extraction of answers from model outputs."""
    
    # Test case 1: Standard boxed answer
    input_text = "The answer is \\boxed{42}."
    result = extract_answer(input_text)
    assert result == "42"
    
    # Test case 2: Multiple boxed expressions (should take the last one)
    input_text = "First, we get \\boxed{24}. But after recalculating, the answer is \\boxed{36}."
    result = extract_answer(input_text)
    assert result == "36"
    
    # Test case 3: No boxed answer
    input_text = "The answer is 42."
    result = extract_answer(input_text)
    assert result is None
    
    # Test case 4: Empty string
    result = extract_answer("")
    assert result is None
    
    # Test case 5: None input
    result = extract_answer(None)
    assert result is None
    
    # Test case 6: Complex boxed content
    input_text = "After calculating, I found that \\boxed{x = \\frac{a^2 + b^2}{2}}."
    result = extract_answer(input_text)
    assert result == "x = \\frac{a^2 + b^2}{2}" 