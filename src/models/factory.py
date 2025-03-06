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
    logger.info(f"Creating model with config key: {model_config_key}")
    logger.info(f"Full config: {config.config_dict}")
    
    model_config = config.get(model_config_key, {})
    logger.info(f"Model config: {model_config}")
    
    api_type = model_config.get("api_type", "fireworks").lower()
    logger.info(f"API type: {api_type}")
    
    if api_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model API type: {api_type}")
    
    model_class = MODEL_TYPES[api_type]
    logger.info(f"Using model class: {model_class.__name__}")
    
    return model_class(config, model_config_key)

def create_reasoning_model(config: Config) -> Model:
    """
    Create a model instance for reasoning based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance for reasoning
    """
    logger.info("Creating reasoning model")
    return create_model(config, "model")

def create_summarization_model(config: Config) -> Optional[Model]:
    """
    Create a model instance for summarization based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance for summarization, or None if using the same model for reasoning
    """
    logger.info("Creating summarization model")
    summarization_config = config.get("summarization", {})
    method = summarization_config.get("method", "self")
    logger.info(f"Summarization method: {method}")
    
    if method == "self":
        # Use the reasoning model for summarization
        logger.info("Using reasoning model for summarization")
        return None
    else:
        # Create a separate model for summarization
        logger.info("Creating separate model for summarization")
        return create_model(config, "summarization_model")