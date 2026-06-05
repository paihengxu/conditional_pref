#!/usr/bin/env python3
"""
General pipeline execution skeleton demonstrating how to load configurations, load data,
and execute your custom project workflows.
"""
import os
import sys
import logging

# Ensure the 'src' directory is in the import search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from my_package.config import load_config, get_run_dir, get_component_params, save_config
from my_package.data.loader import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Pipeline Workflow...")
    
    # 1. Load configuration file
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # 2. Extract component-specific parameters
    loader_params = get_component_params(config, "DataLoader")
    logger.info(f"Data Loader parameters extracted: {loader_params}")
    
    # 3. Resolve and create run/output backup directory
    run_dir = get_run_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created active run output directory at: {run_dir}")
    
    # 4. Load Dataset
    dataset = load_dataset(config.get("dataset", {}))
    logger.info("Dataset loading complete.")
    
    # 5. Core Pipeline Logic / Loop Placeholder
    logger.info("Running custom project workflow pipeline...")
    # TODO: Implement your core project training/evaluation/processing loop here.
        
    # 6. Save backup configuration and run state to the experiment directory
    saved_config_path = os.path.join(run_dir, "run_config.yaml")
    save_config(config, saved_config_path)
    logger.info(f"Saved run configuration backup to {saved_config_path}")
    
    logger.info("Pipeline Workflow Completed successfully!")


if __name__ == "__main__":
    main()
