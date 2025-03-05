"""
Tests for utility functions (configuration, logging, etc.)
"""
import os
import pytest
import tempfile
import yaml
from pathlib import Path

from src.utils.config import Config, load_config, PROJECT_ROOT

# Sample configs for testing
SAMPLE_CONFIG = {
    "model": {
        "name": "test_model",
        "parameters": {
            "temperature": 0.8,
            "max_tokens": 1000
        }
    },
    "data": {
        "dataset": "test_dataset"
    }
}

OVERRIDE_CONFIG = {
    "model": {
        "name": "override_model",
        "parameters": {
            "temperature": 0.5
        }
    }
}

def test_config_initialization():
    """Test basic config initialization"""
    config = Config(base_config=SAMPLE_CONFIG)
    assert config.get("model.name") == "test_model"
    assert config.get("model.parameters.temperature") == 0.8
    assert config.get("data.dataset") == "test_dataset"

def test_config_get_with_default():
    """Test getting config values with defaults"""
    config = Config(base_config=SAMPLE_CONFIG)
    assert config.get("nonexistent", "default_value") == "default_value"
    assert config.get("model.parameters.nonexistent", 42) == 42

def test_config_contains():
    """Test checking if config contains a key"""
    config = Config(base_config=SAMPLE_CONFIG)
    assert "model.name" in config
    assert "nonexistent" not in config

def test_config_getitem():
    """Test dictionary-style access to config"""
    config = Config(base_config=SAMPLE_CONFIG)
    assert config["model"]["name"] == "test_model"
    
    with pytest.raises(KeyError):
        _ = config["nonexistent"]

def test_config_loading_from_file():
    """Test loading config from a YAML file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp:
        yaml.dump(SAMPLE_CONFIG, temp)
        temp_path = temp.name
    
    try:
        config = Config(temp_path)
        assert config.get("model.name") == "test_model"
        assert config.get("model.parameters.temperature") == 0.8
    finally:
        os.unlink(temp_path)

def test_config_merging():
    """Test merging of config dictionaries"""
    base_config = Config(base_config=SAMPLE_CONFIG)
    override_config = Config(base_config=OVERRIDE_CONFIG)
    
    base_config._merge_config(override_config.config_dict)
    
    # Check overridden values
    assert base_config.get("model.name") == "override_model"
    assert base_config.get("model.parameters.temperature") == 0.5
    
    # Check retained values
    assert base_config.get("model.parameters.max_tokens") == 1000
    assert base_config.get("data.dataset") == "test_dataset"

def test_load_config():
    """Test the load_config function with default config"""
    # Create a temporary default config
    default_config_dir = PROJECT_ROOT / 'configs'
    default_config_dir.mkdir(exist_ok=True, parents=True)
    default_config_path = default_config_dir / 'default.yaml'
    
    # Create a temporary specific config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_default:
        yaml.dump(SAMPLE_CONFIG, temp_default)
        temp_default_path = temp_default.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_specific:
        yaml.dump(OVERRIDE_CONFIG, temp_specific)
        temp_specific_path = temp_specific.name
    
    try:
        # Move the temporary default config to the expected location
        with open(temp_default_path, 'r') as src, open(default_config_path, 'w') as dst:
            dst.write(src.read())
        
        # Test loading with overrides
        config = load_config(temp_specific_path)
        
        # Check merged config
        assert config.get("model.name") == "override_model"
        assert config.get("model.parameters.temperature") == 0.5
        assert config.get("model.parameters.max_tokens") == 1000
        assert config.get("data.dataset") == "test_dataset"
    finally:
        # Clean up temporary files
        if os.path.exists(temp_default_path):
            os.unlink(temp_default_path)
        if os.path.exists(temp_specific_path):
            os.unlink(temp_specific_path)
        if os.path.exists(default_config_path):
            os.unlink(default_config_path)