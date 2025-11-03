"""
Retry utilities for API calls and network operations.

This module provides decorators and functions for implementing retry logic
with exponential backoff, timeout handling, configurable parameters, and comprehensive logging.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps
from typing import Any, Callable, Optional, Type, Union

from src.utilities.logger import get_logger

logger = get_logger("retry")


def _get_func_name(func: Callable) -> str:
    """Get function name safely, handling cases where __name__ may not exist."""
    return getattr(func, '__name__', repr(func))


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
    timeout: Optional[float] = 30.0,
    **kwargs
) -> Any:
    """
    Retry a function call with exponential backoff and timeout handling.
    
    Args:
        func: The function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        backoff_multiplier: Multiplier for delay on each retry (default: 2.0)
        exceptions: Exception types to catch and retry (default: Exception)
        logger_name: Optional custom logger name
        timeout: Timeout in seconds for each attempt (default: 30.0)
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
            initial_delay=0.5,
            timeout=60.0
        )
    """
    try:
        retry_logger = get_logger(logger_name) if logger_name else logger
    except Exception:
        # Fallback to default logger if custom logger fails
        retry_logger = logger
    last_exception = None
    delay = initial_delay
    
    def run_with_timeout():
        """Run the function with timeout handling."""
        if timeout is None:
            return func(*args, **kwargs)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                retry_logger.warning(
                    "Function %s timed out after %.1f seconds",
                    _get_func_name(func),
                    timeout
                )
                raise TimeoutError(f"Function {_get_func_name(func)} timed out after {timeout} seconds")
    
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            result = run_with_timeout()
            elapsed = time.time() - start_time
            
            if elapsed > 0.01:  # Log if call took longer than 10 milliseconds
                retry_logger.info(
                    "Function %s completed in %.1f seconds (attempt %d/%d)",
                    _get_func_name(func),
                    elapsed,
                    attempt + 1,
                    max_retries
                )
            
            return result
            
        except (Exception, TimeoutError) as e:
            # Only catch if it matches our configured exceptions or is a TimeoutError
            if isinstance(e, TimeoutError) or isinstance(e, exceptions):
                last_exception = e
                elapsed = time.time() - start_time
                
                if attempt < max_retries - 1:
                    if isinstance(e, TimeoutError):
                        retry_logger.warning(
                            "Function %s timed out (attempt %d/%d) after %.1f seconds. Retrying in %.1f seconds...",
                            _get_func_name(func),
                            attempt + 1,
                            max_retries,
                            elapsed,
                            delay
                        )
                    else:
                        retry_logger.warning(
                            "Function %s failed (attempt %d/%d): %s. Retrying in %.1f seconds...",
                            _get_func_name(func),
                            attempt + 1,
                            max_retries,
                            str(e),
                            delay
                        )
                    time.sleep(delay)
                    delay *= backoff_multiplier
                else:
                    if isinstance(e, TimeoutError):
                        retry_logger.error(
                            "Function %s timed out after %d attempts (total time: %.1f seconds)",
                            _get_func_name(func),
                            max_retries,
                            elapsed
                        )
                    else:
                        retry_logger.error(
                            "Function %s failed after %d attempts: %s",
                            _get_func_name(func),
                            max_retries,
                            str(e)
                        )
            else:
                # Re-raise exceptions we're not configured to handle
                raise
    
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"No attempts were made for function {_get_func_name(func)}")


class RetryConfig:
    """Configuration class for retry parameters."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        exceptions: Union[Type[Exception], tuple] = Exception,
        timeout: Optional[float] = 30.0,
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_multiplier: Multiplier for delay on each retry
            exceptions: Exception types to catch and retry
            timeout: Timeout in seconds for each attempt
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.exceptions = exceptions
        self.timeout = timeout
    
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
            timeout=self.timeout,
            **kwargs
        )


# Pre-configured retry configurations for common use cases
API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    backoff_multiplier=2.0,
    exceptions=(ConnectionError, TimeoutError, Exception),
    timeout=60.0  # 60 second timeout for API calls
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    backoff_multiplier=1.5,
    exceptions=(ConnectionError, TimeoutError),
    timeout=30.0  # 30 second timeout for network operations
)

FILE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    initial_delay=0.1,
    backoff_multiplier=2.0,
    exceptions=(OSError, PermissionError),
    timeout=10.0  # 10 second timeout for file operations
)