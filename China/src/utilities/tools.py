import asyncio
import functools
import time
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """
    Timer decorator that works with both sync and async functions.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                print(f"⏱️  Function '{func.__name__}' runtime: {exec_time:.4f} s")

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                print(f"⏱️  Function '{func.__name__}' runtime: {exec_time:.4f} s")

        return sync_wrapper
