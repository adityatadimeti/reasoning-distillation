from typing import Optional, Dict, Any

from src.llm.base_client import ModelClient
from src.llm.fireworks_client import FireworksModelClient
from src.llm.openai_client import OpenAIModelClient

def create_model_client(
    model_name: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None
) -> ModelClient:
    """
    Factory function to create the appropriate model client based on model name or provider.
    
    Args:
        model_name: Name of the model to use
        provider: Explicitly specify the provider ('fireworks', 'openai')
        api_key: Optional API key
        
    Returns:
        An instance of the appropriate ModelClient
    """
    # Determine provider from model name if not explicitly provided
    if not provider:
        if "gpt" in model_name.lower():
            provider = "openai"
        elif any(name in model_name.lower() for name in ["qwq", "fireworks", "deepseek", "llama", "qwen"]):
            provider = "fireworks"
        else:
            raise ValueError(f"Could not determine provider for model '{model_name}'. Please specify provider.")
    
    # Create appropriate client
    if provider.lower() == "fireworks":
        return FireworksModelClient(model_name, api_key)
    elif provider.lower() == "openai":
        return OpenAIModelClient(model_name, api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")