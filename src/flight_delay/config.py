"""
Configuration loading utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/params.yaml") -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file (relative to project root)
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure the file exists at the specified path."
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return config
