#!/usr/bin/env python3
"""
Example demonstrating the logging features in the stock analysis project.

Run with: uv run python logging_example.py
"""

import asyncio
from src.utilities.logger import get_logger, set_log_level
from src.utilities.tools import timer, logged, timed_and_logged


# Get logger for this module
logger = get_logger("example")


@logged
def simple_function(name: str, age: int):
    """Example function with basic logging."""
    logger.info(f"Processing user: {name}, age: {age}")
    return f"Hello {name}, you are {age} years old"


@timer
def timed_function():
    """Example function with timing."""
    import time
    logger.info("Starting time-consuming operation...")
    time.sleep(0.1)  # Simulate work
    logger.info("Operation completed")
    return "Work done!"


@timed_and_logged
async def async_function(data_size: int):
    """Example async function with full logging and timing."""
    logger.info(f"Processing {data_size} items asynchronously")
    
    # Simulate async work
    await asyncio.sleep(0.05)
    
    # Log progress
    for i in range(3):
        logger.debug(f"Processing batch {i+1}/3")
        await asyncio.sleep(0.01)
    
    logger.info("Async processing completed")
    return f"Processed {data_size} items"


def demonstrate_log_levels():
    """Demonstrate different log levels."""
    logger.debug("This is a DEBUG message (detailed info)")
    logger.info("This is an INFO message (general info)")
    logger.warning("This is a WARNING message (something unusual)")
    logger.error("This is an ERROR message (something went wrong)")
    
    try:
        # Simulate an error
        raise ValueError("Example error for demonstration")
    except ValueError as e:
        logger.error(f"Caught exception: {e}")


async def main():
    """Main demonstration function."""
    logger.info("=== Starting Logging Demo ===")
    
    print("🔧 Testing basic logging...")
    result1 = simple_function("Alice", 30)
    print(f"Result: {result1}")
    
    print("\n⏱️  Testing timed function...")
    result2 = timed_function()
    print(f"Result: {result2}")
    
    print("\n🚀 Testing async function with full logging...")
    result3 = await async_function(100)
    print(f"Result: {result3}")
    
    print("\n📊 Testing different log levels...")
    demonstrate_log_levels()
    
    print("\n🔍 Testing verbose logging (DEBUG level)...")
    set_log_level("DEBUG")
    result4 = simple_function("Bob", 25)
    print(f"Result: {result4}")
    
    logger.info("=== Logging Demo Completed ===")
    print("\n✅ Demo completed! Check these files:")
    print("   📁 logs/stock_analysis_YYYYMMDD.log - Detailed log file")
    print("   📺 Console output - Key information")


if __name__ == "__main__":
    asyncio.run(main())