#!/usr/bin/env python3
"""
This script patches the src/experiments/summarization.py file to add support
for custom answer extractors.

Run this script before running the multiple-choice experiment.
"""

import os
import re
import sys
import shutil
import importlib.util
from typing import Optional, Callable

def load_custom_extractor(file_path: str, function_name: str) -> Optional[Callable]:
    """
    Dynamically load a custom answer extractor function from a Python file.
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

def patch_summarization_file():
    """
    Patch the summarization.py file to add support for custom extractors.
    """
    summarization_path = "src/experiments/summarization.py"
    
    # Make a backup of the original file
    backup_path = f"{summarization_path}.bak"
    if not os.path.exists(backup_path):
        print(f"Creating backup of {summarization_path} to {backup_path}")
        shutil.copy2(summarization_path, backup_path)
    
    # Read the file
    with open(summarization_path, 'r') as f:
        content = f.read()
    
    # Find where to insert the custom extractor loading function
    # Ideally after the imports but before the class definitions
    import_section_end = re.search(r'import.*?\n\n', content, re.DOTALL)
    
    if not import_section_end:
        print("Could not find a good place to insert the custom extractor function")
        return False
    
    insert_pos = import_section_end.end()
    
    # Define the function to insert
    custom_extractor_function = '''
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
        # Make the path absolute if it\'s not already
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
            
        # Load the module from file
        module_name = os.path.basename(file_path).replace(\'.py\', \'\')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            logger.error(f"Could not load spec from {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get the function from the module
        if hasattr(module, function_name):
            return getattr(module, function_name)
        else:
            logger.error(f"Function {function_name} not found in {file_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading custom extractor: {str(e)}")
        return None
'''
    
    # Insert the function
    new_content = content[:insert_pos] + custom_extractor_function + content[insert_pos:]
    
    # Now find the SummarizationRunner.__init__ method to add custom extractor loading
    init_method = re.search(r'def __init__\(self.*?\).*?self.verbose = verbose', new_content, re.DOTALL)
    
    if not init_method:
        print("Could not find __init__ method in SummarizationRunner")
        return False
    
    init_end_pos = init_method.end()
    
    # Define the code to insert for custom extractor initialization
    custom_extractor_init = '''
        
        # Load custom answer extractor if specified
        self.custom_extractor = None
        if self.config.get("use_custom_extractor", False):
            extractor_path = self.config.get("custom_extractor_path")
            extractor_function = self.config.get("custom_extractor_function")
            
            if extractor_path and extractor_function:
                self.custom_extractor = load_custom_extractor(extractor_path, extractor_function)
                if self.custom_extractor:
                    logger.info(f"Loaded custom answer extractor from {extractor_path}")
                else:
                    logger.warning(f"Failed to load custom answer extractor from {extractor_path}")
            else:
                logger.warning("Missing custom_extractor_path or custom_extractor_function in config")
'''
    
    # Insert the custom extractor initialization
    new_content = new_content[:init_end_pos] + custom_extractor_init + new_content[init_end_pos:]
    
    # Find the locations where extract_answer is called
    # There are at least two places: initial reasoning and subsequent iterations
    
    # First, find where extract_answer is called for the initial reasoning
    iter0_answer_extraction = re.search(r'iter0_answer = extract_answer\(iter0_reasoning\)', new_content)
    
    if iter0_answer_extraction:
        extract_pos = iter0_answer_extraction.start()
        extract_end = iter0_answer_extraction.end()
        
        # Replace with conditional call to either custom or default extractor
        custom_extraction_code = '''iter0_answer = self.custom_extractor(iter0_reasoning) if self.custom_extractor else extract_answer(iter0_reasoning)'''
        
        new_content = new_content[:extract_pos] + custom_extraction_code + new_content[extract_end:]
    
    # Next, find where extract_answer is called for subsequent iterations
    next_answer_extraction = re.search(r'next_answer = extract_answer\(next_reasoning\)', new_content)
    
    if next_answer_extraction:
        extract_pos = next_answer_extraction.start()
        extract_end = next_answer_extraction.end()
        
        # Replace with conditional call to either custom or default extractor
        custom_extraction_code = '''next_answer = self.custom_extractor(next_reasoning) if self.custom_extractor else extract_answer(next_reasoning)'''
        
        new_content = new_content[:extract_pos] + custom_extraction_code + new_content[extract_end:]
    
    # Write the updated file
    with open(summarization_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched {summarization_path} to support custom extractors")
    return True

if __name__ == "__main__":
    print("Patching src/experiments/summarization.py to support custom extractors...")
    if patch_summarization_file():
        print("Patch applied successfully!")
        print("Now you can run the multiple-choice experiment with:")
        print("python run_experiment.py --config config/experiments/gpqa_diamond.yaml")
        sys.exit(0)
    else:
        print("Failed to apply patch")
        sys.exit(1) 