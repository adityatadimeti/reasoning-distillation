"""
Tokenization and chat template formatting utilities for LLM models.
"""

import os
import json
import tiktoken
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

# Load unified model info for mapping and pricing
from os import path
INFO_PATH = "/Users/justinshen/Documents/Code/cocolab/reasoning-distillation/paper_results/fireworks_model_info.json" # path.join(path.dirname(__file__), "models", "fireworks_model_info.json")
with open(INFO_PATH, "r") as f:
    FIREWORKS_MODEL_INFO = json.load(f)

# Derive FIREWORKS_MODEL_MAP from unified info
FIREWORKS_MODEL_MAP = {
    model_name: info.get("hf_model_name")
    for model_name, info in FIREWORKS_MODEL_INFO.items()
    if info.get("hf_model_name")
}

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        encoding_name: The encoding to use (default: cl100k_base for OpenAI models)
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def get_hf_model_name(fireworks_model_name: str) -> str:
    """
    Get the Hugging Face model name corresponding to a Fireworks API model name.
    
    Args:
        fireworks_model_name: The Fireworks API model name
        
    Returns:
        The corresponding Hugging Face model name
        
    Raises:
        ValueError: If the model name is not found in the mapping
    """
    if fireworks_model_name not in FIREWORKS_MODEL_MAP:
        raise ValueError(f"Model {fireworks_model_name} not found in Fireworks model map")
    
    return FIREWORKS_MODEL_MAP[fireworks_model_name]

def load_tokenizer(model_name: str):
    """
    Load the HF tokenizer for a model.
    
    Args:
        model_name: The HF model name
        
    Returns:
        The loaded tokenizer
        
    Raises:
        ImportError: If transformers is not installed
        ValueError: If the tokenizer could not be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for {model_name}: {str(e)}")

def format_chat_for_completions(messages: List[Dict[str, str]], model_name: str) -> str:
    """
    Format chat messages for the completions API using the model's chat template.
    
    Args:
        messages: List of chat messages (as dict with role and content)
        model_name: The Fireworks model name
        
    Returns:
        Formatted prompt string for the completions API
        
    Raises:
        ValueError: If the model name is not found or the tokenizer fails to load
    """
    # Get the corresponding HF model name
    hf_model_name = get_hf_model_name(model_name)
    
    # Load the tokenizer
    tokenizer = load_tokenizer(hf_model_name)
    
    # Apply the chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_prompt