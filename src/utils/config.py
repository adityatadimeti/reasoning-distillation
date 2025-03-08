import os
import yaml
from typing import Dict, Any

def load_config(experiment_name: str = "baseline") -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    
    Args:
        experiment_name: Name of the experiment config to load
        
    Returns:
        Merged configuration dictionary
    """
    experiment_config_path = f"config/experiments/{experiment_name}.yaml"
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Experiment config not found at {experiment_config_path}")
    
    return config