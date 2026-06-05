"""
Configuration utility for loading, resolving, and validating YAML configs.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file. Resolves absolute paths, paths relative
    to current working directory, and paths relative to the project root.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Dictionary with loaded configuration
    """
    input_path = Path(yaml_path).expanduser()
    
    # Resolve the project root (3 levels up from src/my_package/config.py)
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate_paths = []

    if input_path.is_absolute():
        candidate_paths.append(input_path)
    else:
        candidate_paths.append(Path.cwd() / input_path)
        candidate_paths.append(project_root / input_path)

    resolved_path = next((p for p in candidate_paths if p.exists()), None)
    if resolved_path is None:
        searched = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            f"Config file not found: {yaml_path}. "
            f"cwd={Path.cwd()}. searched=[{searched}]"
        )

    config = OmegaConf.load(str(resolved_path))
    return OmegaConf.to_container(config, resolve=True)


def get_component_params(config: Dict[str, Any], component_name: str) -> Dict[str, Any]:
    """
    Generic function to extract parameters of a nested component from the config.

    Args:
        config: Full configuration dictionary
        component_name: Name of the component (e.g., 'Model', 'DataLoader')

    Returns:
        Dictionary of parameters
    """
    component_config = config.get(component_name, {})
    return component_config.get('parameters', {}).copy()


def get_run_dir(config: Dict[str, Any]) -> str:
    """
    Generate or retrieve a robust run directory based on configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Path to output directory for current run
    """
    if config.get('resume_from'):
        return config['resume_from']
    
    base_dir = config.get('base_dir', './exp')
    run_name = config.get('run_name', 'run')
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    
    return os.path.join(base_dir, f"{run_name}_{timestamp}")


def save_config(config: Dict[str, Any], yaml_path: str) -> None:
    """
    Save configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save
        yaml_path: Path to the output YAML file
    """
    conf = OmegaConf.create(config)
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    OmegaConf.save(conf, yaml_path)
