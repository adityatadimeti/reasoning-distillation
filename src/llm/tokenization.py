"""
Tokenization and chat template formatting utilities for LLM models.
"""

import os
import json
import tiktoken
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

# Load model mapping from Fireworks API model names to HF model names
MODEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "models", "fireworks_model_map.json")
with open(MODEL_MAP_PATH, "r") as f:
    FIREWORKS_MODEL_MAP = json.load(f)

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

def create_continuation_prompt(
    original_messages: List[Dict[str, str]], 
    generated_text: str, 
    model_name: str,
    context_tokens: int = 2000
) -> str:
    """
    Create a prompt for continuing generation from where it left off.
    
    Args:
        original_messages: The original chat messages
        generated_text: The text generated so far (that got cut off)
        model_name: The Fireworks model name
        context_tokens: Number of tokens from previous generation to include
        
    Returns:
        A properly formatted continuation prompt string
    """
    # Keep the system message if any
    system_message = next((msg for msg in original_messages if msg["role"] == "system"), None)
    
    # Get the user message
    user_message = next((msg for msg in original_messages if msg["role"] == "user"), {"content": ""})
    
    # Determine the context window size for the continuation
    generated_tokens = count_tokens(generated_text)
    continuation_context_tokens = min(generated_tokens, context_tokens)
    
    # Very rough approximation: 1 token ≈ 4 characters for English text
    char_estimate = continuation_context_tokens * 4
    continuation_context = generated_text[-char_estimate:] if len(generated_text) > char_estimate else generated_text
    
    # Build new continuation messages
    continuation_messages = []
    if system_message:
        continuation_messages.append(system_message)
    
    # Combine the original question with a request to continue from previous output
    continuation_messages.append({
        "role": "user",
        "content": f"{user_message['content']}\n\nHere's what you've written so far, please continue from where you left off:\n\n{continuation_context}"
    })
    
    # Format for completions API
    return format_chat_for_completions(continuation_messages, model_name)
