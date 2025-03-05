"""
Factory for creating model instances based on configuration.
"""
import logging
from typing import Dict, Optional, Any, Type

from src.utils.config import Config
from src.models.base import Model
from src.models.fireworks import FireworksModel

logger = logging.getLogger(__name__)

# Dictionary mapping model API types to their classes
MODEL_TYPES = {
    "fireworks": FireworksModel,
}

def create_model(config: Config, model_config_key: str = "model") -> Model:
    """
    Create a model instance based on configuration.
    
    Args:
        config: Configuration object
        model_config_key: Key to access model-specific config
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_config = config.get(model_config_key, {})
    api_type = model_config.get("api_type", "fireworks").lower()
    
    if api_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model API type: {api_type}")
    
    model_class = MODEL_TYPES[api_type]
    logger.info(f"Creating {api_type} model: {model_config.get('name', 'unknown')}")
    
    return model_class(config, model_config_key)

def create_reasoning_model(config: Config) -> Model:
    """
    Create a model instance for reasoning based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance for reasoning
    """
    return create_model(config, "reasoning_model")

def create_summarization_model(config: Config) -> Optional[Model]:
    """
    Create a model instance for summarization based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance for summarization, or None if using the same model for reasoning
    """
    summarization_config = config.get("summarization", {})
    method = summarization_config.get("method", "self")
    
    if method == "self":
        # Use the reasoning model for summarization
        return None
    else:
        # Create a separate model for summarization
        return create_model(config, "summarization_model")