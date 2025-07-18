"""
Configuration settings for the China stock analysis project.

This module must be imported before any other modules to ensure
proper initialization of environment variables and settings.
"""

import os
from typing import Dict, Any


def configure_environment() -> Dict[str, Any]:
    """
    Configure environment variables and global settings.

    Returns:
        Dict containing the applied configuration settings
    """
    settings = {}

    # Disable tqdm progress bars from dependencies to avoid interference with Rich progress bars
    os.environ["TQDM_DISABLE"] = "1"
    settings["tqdm_disabled"] = True

    # Set other environment variables as needed
    # os.environ["PYTHONPATH"] = os.getcwd()
    # settings["pythonpath_set"] = True

    return settings


# Auto-configure when this module is imported
_config = configure_environment()


def get_config() -> Dict[str, Any]:
    """Get the current configuration settings."""
    return _config.copy()


def is_tqdm_disabled() -> bool:
    """Check if tqdm is disabled."""
    return _config.get("tqdm_disabled", False)
