"""
Centralized logging configuration for the China stock analysis project.

This module provides a unified logging setup with both file and console handlers.
It uses standard Python logging for better compatibility with tqdm progress bars.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "stock_analysis", level: str = "INFO") -> logging.Logger:
    """
    Setup centralized logger for the project

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler - detailed logging
    today = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(log_dir / f"stock_analysis_{today}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Standard console handler - compatible with tqdm progress bars
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show WARNING and above on console
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance for a specific module

    Args:
        name: Module name (optional)

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"stock_analysis.{name}")
    return logger


def set_log_level(level: str) -> None:
    """
    Change the logging level globally

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.setLevel(getattr(logging, level.upper()))
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(getattr(logging, level.upper()))


def set_console_log_level(level: str) -> None:
    """
    Change only the console logging level (useful for progress bars)

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(getattr(logging, level.upper()))
