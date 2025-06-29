from typing import Optional, Dict, Any

from src.llm.base_client import ModelClient
from src.llm.fireworks_client import FireworksModelClient
from src.llm.openai_client import OpenAIModelClient
from src.llm.together_client import TogetherModelClient
from src.llm.vllm_client import VLLMModelClient

def create_model_client(
    model_name: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    vllm_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ModelClient:
    """
    Create the appropriate model client based on model name or provider.
    
    Args:
        model_name: Name of the model to use
        provider: Explicitly specify the provider ('fireworks', 'openai', 'together', 'vllm')
        api_key: Optional API key
        vllm_config: Optional vLLM-specific configuration (host, port, etc.)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An instance of the appropriate ModelClient
    """
    # Determine provider from model name if not explicitly provided
    if not provider:
        if "gpt" in model_name.lower():
            provider = "openai"
        elif any(name in model_name.lower() for name in ["qwq", "fireworks", "deepseek", "qwen"]):
            provider = "fireworks"
        elif any(name in model_name.lower() for name in ["together", "meta-llama", "mistral", "mixtral"]):
            provider = "together"    
        # Note: llama models can be on both Fireworks and Together, so provider must be specified
        else:
            raise ValueError(f"Could not determine provider for model '{model_name}'. Please specify provider.")
    
    # Create appropriate client
    if provider.lower() == "vllm":
        # Extract vLLM-specific configuration
        vllm_config = vllm_config or {}
        host = vllm_config.get("host", kwargs.pop("host", "localhost"))
        port = vllm_config.get("port", kwargs.pop("port", 8000))
        max_model_len = vllm_config.get("max_model_len", kwargs.pop("max_model_len", None))
        
        return VLLMModelClient(
            model_name=model_name,
            host=host,
            port=port,
            api_key=api_key,
            max_model_len=max_model_len,
            **kwargs
        )
    elif provider.lower() == "fireworks":
        return FireworksModelClient(model_name, api_key, **kwargs)
    elif provider.lower() == "openai":
        return OpenAIModelClient(model_name, api_key, **kwargs)
    elif provider.lower() == "together":
        return TogetherModelClient(model_name, api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")