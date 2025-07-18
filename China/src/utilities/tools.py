"""
Utility decorators and tools for the China stock analysis project.

This module provides decorators for timing, logging, and debugging functions.
It includes support for both synchronous and asynchronous functions.
"""

import asyncio
import functools
import time
from typing import Callable, Any
from .logger import get_logger


def timer(func: Callable) -> Callable:
    """
    Timer decorator that works with both sync and async functions.
    Logs execution time and function details.
    """
    logger = get_logger("timer")

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.info("Starting async function: %s", func.__name__)
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, str(e))
                raise
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                logger.info("⏱️  Function '%s' runtime: %.4f s", func.__name__, exec_time)

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.info("Starting sync function: %s", func.__name__)
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, str(e))
                raise
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                logger.info("⏱️  Function '%s' runtime: %.4f s", func.__name__, exec_time)

        return sync_wrapper


def logged(func: Callable) -> Callable:
    """
    Logging decorator that tracks function entry, exit, and arguments.
    """
    logger = get_logger("function_tracker")

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Log function entry with arguments
            args_str = ", ".join(
                [str(arg)[:50] + "..." if len(str(arg)) > 50 else str(arg) for arg in args]
            )
            kwargs_str = ", ".join(
                [
                    f"{k}={str(v)[:30] + '...' if len(str(v)) > 30 else str(v)}"
                    for k, v in kwargs.items()
                ]
            )
            logger.debug(
                "→ Entering %s(%s%s)",
                func.__name__,
                args_str,
                ', ' + kwargs_str if kwargs_str else '',
            )

            try:
                result = await func(*args, **kwargs)
                logger.debug("← Exiting %s successfully", func.__name__)
                return result
            except Exception as e:
                logger.error("✗ Exception in %s: %s", func.__name__, str(e))
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Log function entry with arguments
            args_str = ", ".join(
                [str(arg)[:50] + "..." if len(str(arg)) > 50 else str(arg) for arg in args]
            )
            kwargs_str = ", ".join(
                [
                    f"{k}={str(v)[:30] + '...' if len(str(v)) > 30 else str(v)}"
                    for k, v in kwargs.items()
                ]
            )
            logger.debug(
                "→ Entering %s(%s%s)",
                func.__name__,
                args_str,
                ', ' + kwargs_str if kwargs_str else '',
            )

            try:
                result = func(*args, **kwargs)
                logger.debug("← Exiting %s successfully", func.__name__)
                return result
            except Exception as e:
                logger.error("✗ Exception in %s: %s", func.__name__, str(e))
                raise

        return sync_wrapper


def timed_and_logged(func: Callable) -> Callable:
    """
    Combined decorator that provides both timing and detailed logging.
    """
    return timer(logged(func))
