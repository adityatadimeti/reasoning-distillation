#!/usr/bin/env python3
"""
This script demonstrates how to update the experiment framework to support
custom answer extractors for multiple-choice questions.

This would need to be integrated into the src/experiments/summarization.py file.
"""

import importlib.util
import sys
import os
from typing import Optional, Callable

def load_custom_extractor(file_path: str, function_name: str) -> Optional[Callable]:
    """
    Dynamically load a custom answer extractor function from a Python file.
    
    Args:
        file_path: Path to the Python file containing the extractor
        function_name: Name of the function to load
        
    Returns:
        The loaded function, or None if loading failed
    """
    try:
        # Make the path absolute if it's not already
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
            
        # Load the module from file
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            print(f"Error: Could not load spec from {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get the function from the module
        if hasattr(module, function_name):
            return getattr(module, function_name)
        else:
            print(f"Error: Function {function_name} not found in {file_path}")
            return None
            
    except Exception as e:
        print(f"Error loading custom extractor: {str(e)}")
        return None

# Example integration into the experimental framework
def example_usage():
    """
    Example of how to integrate custom extractors into the experiment framework.
    This code would need to be integrated into src/experiments/summarization.py.
    """
    # Configuration example
    config = {
        "use_multiple_choice_extractor": True,
        "answer_extractor_path": "mc_answer_extractor.py",
        "answer_extractor_function": "extract_mc_answer",
    }
    
    # Load the custom extractor if specified
    custom_extractor = None
    if config.get("use_multiple_choice_extractor", False):
        extractor_path = config.get("answer_extractor_path")
        extractor_function = config.get("answer_extractor_function")
        
        if extractor_path and extractor_function:
            custom_extractor = load_custom_extractor(extractor_path, extractor_function)
            print(f"Loaded custom answer extractor from {extractor_path}")
        else:
            print("Missing extractor path or function name")
    
    # Example of how to use the custom extractor
    model_output = "After analyzing all options, I choose (B)."
    
    # Use either the custom extractor or the default one
    if custom_extractor:
        answer = custom_extractor(model_output)
        print(f"Using custom extractor: extracted answer '{answer}'")
    else:
        # Use the default extractor
        from src.reasoning.extractor import extract_answer
        answer = extract_answer(model_output)
        print(f"Using default extractor: extracted answer '{answer}'")
    
    # Here, you would compare answer with correct_answer to determine correctness

# Test the custom extractor loading
if __name__ == "__main__":
    print("Testing custom extractor loading...")
    extractor = load_custom_extractor("mc_answer_extractor.py", "extract_mc_answer")
    
    if extractor:
        test_input = "After analyzing all options, I believe (C) is the correct answer."
        extracted = extractor(test_input)
        print(f"Successfully loaded extractor and extracted answer: {extracted}")
    else:
        print("Failed to load custom extractor")
    
    print("\nExample of integration into experiment framework:")
    example_usage() 