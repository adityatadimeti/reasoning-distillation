#!/usr/bin/env python3
"""Test extraction with malformed answer tags."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.countdown_check import extract_solution

# Test cases with various malformed tags
test_cases = [
    # Standard format
    ("The answer is <answer>(86 - 71) + (30 - 6)</answer>", "(86 - 71) + (30 - 6)"),
    
    # Double closing tags
    ("The answer is </answer>(86 - 71) + (30 - 6)</answer>", "(86 - 71) + (30 - 6)"),
    
    # Double opening tags
    ("The answer is <answer>(86 - 71) + (30 - 6)<answer>", "(86 - 71) + (30 - 6)"),
    
    # Mixed malformed tags
    ("The answer is </answer>(86 - 71) + (30 - 6)<answer>", "(86 - 71) + (30 - 6)"),
    
    # Multiple answers with mixed formats
    ("First try <answer>wrong</answer> then later: </answer>(86 - 71) + (30 - 6)</answer>", "(86 - 71) + (30 - 6)"),
    
    # Boxed format
    ("The answer is \\boxed{(86 - 71) + (30 - 6)}", "(86 - 71) + (30 - 6)"),
    
    # No tags but equation-like content
    ("Let me calculate: >(86 - 71) + (30 - 6)< equals 39", "(86 - 71) + (30 - 6)"),
    
    # Equals sign handling
    ("The answer is <answer>(86 - 71) + (30 - 6) = 39</answer>", "(86 - 71) + (30 - 6)"),
]

print("Testing malformed answer tag extraction:")
print("-" * 60)

for i, (input_text, expected) in enumerate(test_cases):
    result = extract_solution(input_text)
    status = "✓" if result == expected else "✗"
    print(f"\nTest {i+1}: {status}")
    print(f"Input: {input_text[:50]}...")
    print(f"Expected: {expected}")
    print(f"Got: {result}")

print("\n" + "-" * 60)
print("Testing complete!")