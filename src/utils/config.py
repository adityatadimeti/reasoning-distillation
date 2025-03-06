"""
Configuration handling for the reasoning enhancement project.
"""
import os
import yaml
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path

# Set project root to be used for resolving relative paths
PROJECT_ROOT = Path(__file__).absolute().parents[2]

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class that loads and merges YAML config files.
    """
    def __init__(self, config_path: Optional[Union[str, Path]] = None, base_config: Optional[Dict] = None):
        """
        Initialize a configuration object.
        
        Args:
            config_path: Path to a YAML config file (can be absolute or relative to project root)
            base_config: Optional dictionary to use as base configuration
        """
        self.config_dict = base_config or {}
        logger.info(f"Initializing Config with path: {config_path}")
        
        if config_path:
            config_file = self._resolve_path(config_path)
            logger.info(f"Resolved config path to: {config_file}")
            loaded_config = self._load_yaml(config_file)
            logger.info(f"Loaded config: {loaded_config}")
            
            # If this config extends another config, load it first
            if 'base_config' in loaded_config:
                base_config_path = loaded_config.pop('base_config')
                logger.info(f"Loading base config from: {base_config_path}")
                base_config = Config(base_config_path)
                self.config_dict = base_config.config_dict
                logger.info(f"Base config loaded: {self.config_dict}")
            
            # Handle imports if present
            if 'imports' in loaded_config:
                imports = loaded_config.pop('imports')
                logger.info(f"Processing imports: {imports}")
                for import_path in imports:
                    logger.info(f"Loading imported config from: {import_path}")
                    imported_config = Config(import_path)
                    logger.info(f"Imported config: {imported_config.config_dict}")
                    self._merge_config(imported_config.config_dict)
                    logger.info(f"Config after merging import: {self.config_dict}")
            
            # Merge the loaded config with the base config
            logger.info(f"Merging loaded config: {loaded_config}")
            self._merge_config(loaded_config)
            logger.info(f"Final config after merging: {self.config_dict}")
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a path relative to the project root if it's not absolute.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved absolute path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        
        # Try looking in configs/ directory if not specified
        if not str(path_obj).startswith('configs/') and not os.path.exists(PROJECT_ROOT / path_obj):
            config_path = PROJECT_ROOT / 'configs' / path_obj
            if os.path.exists(config_path):
                logger.info(f"Found config in configs/ directory: {config_path}")
                return config_path
        
        resolved_path = PROJECT_ROOT / path_obj
        logger.info(f"Resolved config path to: {resolved_path}")
        return resolved_path
    
    def _load_yaml(self, file_path: Path) -> Dict:
        """
        Load a YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary containing loaded YAML
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        logger.info(f"Loading YAML from: {file_path}")
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f) or {}
            logger.info(f"Loaded YAML content: {config}")
            return config
    
    def _merge_config(self, config: Dict) -> None:
        """
        Recursively merge a config dictionary into the current config.
        
        Args:
            config: Dictionary to merge into current config
        """
        logger.info(f"Merging config: {config}")
        for key, value in config.items():
            if (key in self.config_dict and 
                isinstance(self.config_dict[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge dictionaries
                logger.info(f"Recursively merging key: {key}")
                self._merge_dict(self.config_dict[key], value)
            else:
                # Otherwise just update the value
                logger.info(f"Updating key: {key} with value: {value}")
                self.config_dict[key] = value
    
    def _merge_dict(self, base_dict: Dict, new_dict: Dict) -> None:
        """
        Helper method to recursively merge dictionaries.
        
        Args:
            base_dict: Base dictionary to merge into
            new_dict: Dictionary whose values will be merged into base_dict
        """
        for key, value in new_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key, with optional default.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        # Handle nested keys with dot notation (e.g., "model.parameters.temperature")
        keys = key.split('.')
        value = self.config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If the key doesn't exist
        """
        result = self.get(key)
        if result is None:
            raise KeyError(f"Config key not found: {key}")
        return result
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None
    
    def to_dict(self) -> Dict:
        """
        Convert config to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return self.config_dict.copy()


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load a configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config file (relative to project root or absolute)
        
    Returns:
        Config object containing loaded configuration
    """
    # First load the default config
    default_config_path = PROJECT_ROOT / 'configs' / 'default.yaml'
    
    if os.path.exists(default_config_path):
        config = Config(default_config_path)
    else:
        config = Config()
    
    # Then load and merge the specified config
    specified_config = Config(config_path)
    config._merge_config(specified_config.config_dict)
    
    return config