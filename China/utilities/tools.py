import functools
import time


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        print(f"⏱️  {func.__name__} runtime: {runtime:.4f} s")
        return result

    return wrapper
