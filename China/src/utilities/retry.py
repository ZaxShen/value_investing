"""
Retry utilities for API calls and network operations.

This module provides decorators and functions for implementing retry logic
with exponential backoff, configurable parameters, and comprehensive logging.
"""

import time
from functools import wraps
from typing import Any, Callable, Optional, Type, Union

from src.utilities.logger import get_logger

logger = get_logger("retry")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    logger_name: Optional[str] = None,
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        backoff_multiplier: Multiplier for delay on each retry (default: 2.0)
        exceptions: Exception types to catch and retry (default: Exception)
        logger_name: Optional custom logger name for this decorator
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=5, initial_delay=0.5)
        def api_call():
            return requests.get("https://api.example.com/data")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return retry_call(
                func,
                *args,
                max_retries=max_retries,
                initial_delay=initial_delay,
                backoff_multiplier=backoff_multiplier,
                exceptions=exceptions,
                logger_name=logger_name,
                **kwargs
            )
        return wrapper
    return decorator


def retry_call(
    func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    logger_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Retry a function call with exponential backoff.
    
    Args:
        func: The function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        backoff_multiplier: Multiplier for delay on each retry (default: 2.0)
        exceptions: Exception types to catch and retry (default: Exception)
        logger_name: Optional custom logger name
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the successful function call
        
    Raises:
        The last exception encountered if all retries fail
        
    Example:
        result = retry_call(
            requests.get,
            "https://api.example.com/data",
            max_retries=5,
            initial_delay=0.5
        )
    """
    retry_logger = get_logger(logger_name) if logger_name else logger
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                retry_logger.warning(
                    "Function %s failed (attempt %d/%d): %s. Retrying in %.1f seconds...",
                    func.__name__,
                    attempt + 1,
                    max_retries,
                    str(e),
                    delay
                )
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                retry_logger.error(
                    "Function %s failed after %d attempts: %s",
                    func.__name__,
                    max_retries,
                    str(e)
                )
    
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"No attempts were made for function {func.__name__}")


class RetryConfig:
    """Configuration class for retry parameters."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        exceptions: Union[Type[Exception], tuple] = Exception,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_multiplier: Multiplier for delay on each retry
            exceptions: Exception types to catch and retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.exceptions = exceptions
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Retry a function call using this configuration.
        
        Args:
            func: The function to retry
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the successful function call
        """
        return retry_call(
            func,
            *args,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_multiplier=self.backoff_multiplier,
            exceptions=self.exceptions,
            **kwargs
        )


# Pre-configured retry configurations for common use cases
API_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    backoff_multiplier=2.0,
    exceptions=(ConnectionError, TimeoutError, Exception)
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    backoff_multiplier=1.5,
    exceptions=(ConnectionError, TimeoutError)
)

FILE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    initial_delay=0.1,
    backoff_multiplier=2.0,
    exceptions=(OSError, PermissionError)
)