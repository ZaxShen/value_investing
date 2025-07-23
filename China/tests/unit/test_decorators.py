"""
Unit tests for the decorator functionality (timer, logged, timed_and_logged).

This module provides comprehensive unit tests for the custom decorators used
throughout the stock analysis application. The decorators tested include:

- @timer: Measures and reports function execution time
- @logged: Provides detailed function entry/exit logging with argument capture
- @timed_and_logged: Combines both timing and logging functionality

The tests verify that decorators:
- Work correctly with both sync and async functions
- Preserve original function metadata and signatures
- Handle exceptions gracefully while maintaining logging
- Don't introduce significant performance overhead
- Support complex function signatures and argument patterns

These decorators are critical for monitoring and debugging the async
stock analysis pipeline performance and execution flow.
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import patch, MagicMock, call
from io import StringIO

from src.utilities.tools import timer, logged, timed_and_logged


class TestTimerDecorator:
    """
    Test the timer decorator functionality.
    
    The timer decorator measures function execution time and provides
    performance feedback through console output. It supports both
    synchronous and asynchronous functions and is essential for
    monitoring the performance of data-intensive stock analysis operations.
    """

    @pytest.mark.unit
    def test_timer_sync_function(self, capsys):
        """
        Test timer decorator with synchronous function.
        
        Verifies that the timer decorator correctly measures execution time
        for synchronous functions and outputs timing information to stdout.
        This is important for monitoring blocking operations like file I/O
        and data processing in the stock analysis pipeline.
        """
        # Define a test function with artificial delay for measurable timing
        @timer
        def sync_function(x, y):
            time.sleep(0.01)  # Small delay to measure (10ms)
            return x + y

        # Execute the timed function
        result = sync_function(2, 3)

        # Verify function result is unchanged by decorator
        assert result == 5
        
        # Capture and verify timing output
        captured = capsys.readouterr()
        assert "⏱️  Function 'sync_function' runtime:" in captured.out
        assert "s" in captured.out  # Should show seconds unit

    @pytest.mark.unit
    async def test_timer_async_function(self, capsys):
        """
        Test timer decorator with asynchronous function.
        
        Verifies that the timer decorator works correctly with async functions,
        which are extensively used in the stock analysis pipeline for concurrent
        API calls and data processing. The decorator should properly handle
        await expressions and measure total async execution time.
        """
        # Define async test function with artificial delay
        @timer
        async def async_function(x, y):
            await asyncio.sleep(0.01)  # Non-blocking delay (10ms)
            return x * y

        # Execute the timed async function
        result = await async_function(3, 4)

        # Verify function result is preserved
        assert result == 12
        
        # Verify timing output is generated for async functions
        captured = capsys.readouterr()
        assert "⏱️  Function 'async_function' runtime:" in captured.out

    @pytest.mark.unit
    def test_timer_preserves_function_metadata(self):
        """Test that timer decorator preserves original function metadata."""

        @timer
        def documented_function(param1: int, param2: str) -> str:
            """This is a documented function."""
            return f"{param1}: {param2}"

        assert documented_function.__name__ == "documented_function"
        assert "documented function" in documented_function.__doc__

    @pytest.mark.unit
    def test_timer_handles_exceptions_sync(self, capsys):
        """Test timer decorator handles exceptions in sync functions."""

        @timer
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Should still log timing even with exception
        captured = capsys.readouterr()
        assert "⏱️  Function 'failing_function' runtime:" in captured.out

    @pytest.mark.unit
    async def test_timer_handles_exceptions_async(self, capsys):
        """Test timer decorator handles exceptions in async functions."""

        @timer
        async def failing_async_function():
            await asyncio.sleep(0.001)
            raise ValueError("Async test error")

        with pytest.raises(ValueError, match="Async test error"):
            await failing_async_function()

        # Should still log timing even with exception
        captured = capsys.readouterr()
        assert "⏱️  Function 'failing_async_function' runtime:" in captured.out


class TestLoggedDecorator:
    """
    Test the logged decorator functionality.
    
    The logged decorator provides comprehensive function call tracking
    including entry/exit logging, argument capture, and exception handling.
    This is crucial for debugging the complex async stock analysis pipeline
    where functions may be called concurrently and errors need to be traced
    back to specific function calls with their parameters.
    """

    @pytest.mark.unit
    def test_logged_sync_function_entry_exit(self, caplog):
        """Test logged decorator tracks function entry and exit."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @logged
            def simple_function(a, b):
                return a + b

            result = simple_function(1, 2)

            assert result == 3

            # Check log messages
            log_messages = [record.message for record in caplog.records]
            entry_logs = [
                msg for msg in log_messages if "→ Entering simple_function" in msg
            ]
            exit_logs = [
                msg
                for msg in log_messages
                if "← Exiting simple_function successfully" in msg
            ]

            assert len(entry_logs) >= 1
            assert len(exit_logs) >= 1

    @pytest.mark.unit
    async def test_logged_async_function_entry_exit(self, caplog):
        """Test logged decorator tracks async function entry and exit."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @logged
            async def async_simple_function(x):
                await asyncio.sleep(0.001)
                return x * 2

            result = await async_simple_function(5)

            assert result == 10

            # Check log messages
            log_messages = [record.message for record in caplog.records]
            entry_logs = [
                msg for msg in log_messages if "→ Entering async_simple_function" in msg
            ]
            exit_logs = [
                msg
                for msg in log_messages
                if "← Exiting async_simple_function successfully" in msg
            ]

            assert len(entry_logs) >= 1
            assert len(exit_logs) >= 1

    @pytest.mark.unit
    def test_logged_function_with_arguments(self, caplog, sample_test_data):
        """Test logged decorator captures function arguments."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @logged
            def function_with_args(pos_arg, keyword_arg=None):
                return f"{pos_arg}-{keyword_arg}"

            result = function_with_args("test", keyword_arg="value")

            assert result == "test-value"

            # Check that arguments are logged
            log_messages = [record.message for record in caplog.records]
            entry_logs = [
                msg for msg in log_messages if "→ Entering function_with_args" in msg
            ]

            assert len(entry_logs) >= 1
            assert "test" in entry_logs[0]
            assert "keyword_arg=value" in entry_logs[0]

    @pytest.mark.unit
    def test_logged_function_handles_exceptions(self, caplog):
        """Test logged decorator handles and logs exceptions."""
        with caplog.at_level(logging.DEBUG):

            @logged
            def failing_function():
                raise RuntimeError("Function failed")

            with pytest.raises(RuntimeError, match="Function failed"):
                failing_function()

            # Check exception is logged
            log_messages = [record.message for record in caplog.records]
            error_logs = [
                msg for msg in log_messages if "✗ Exception in failing_function" in msg
            ]

            assert len(error_logs) >= 1
            assert "Function failed" in error_logs[0]

    @pytest.mark.unit
    def test_logged_truncates_long_arguments(self, caplog):
        """Test that logged decorator truncates very long arguments."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @logged
            def function_with_long_args(long_arg):
                return len(long_arg)

            long_string = "x" * 100  # 100 character string
            result = function_with_long_args(long_string)

            assert result == 100

            # Check that long argument is truncated
            log_messages = [record.message for record in caplog.records]
            entry_logs = [
                msg
                for msg in log_messages
                if "→ Entering function_with_long_args" in msg
            ]

            assert len(entry_logs) >= 1
            assert "..." in entry_logs[0]  # Should contain truncation indicator


class TestTimedAndLoggedDecorator:
    """
    Test the combined timed_and_logged decorator.
    
    This decorator combines both timing and logging functionality,
    providing comprehensive monitoring for critical functions in the
    stock analysis pipeline. It's particularly useful for monitoring
    the main pipeline functions that need both performance tracking
    and detailed execution logging.
    """

    @pytest.mark.unit
    def test_timed_and_logged_combines_both(self, caplog, capsys):
        """Test that timed_and_logged provides both timing and logging."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @timed_and_logged
            def combined_function(value):
                time.sleep(0.01)
                return value * 2

            result = combined_function(10)

            assert result == 20

            # Check logging output
            log_messages = [record.message for record in caplog.records]
            has_entry = any(
                "→ Entering combined_function" in msg for msg in log_messages
            )
            has_exit = any(
                "← Exiting combined_function successfully" in msg
                for msg in log_messages
            )

            assert has_entry
            assert has_exit

            # Check timing output
            captured = capsys.readouterr()
            assert "⏱️  Function 'combined_function' runtime:" in captured.out

    @pytest.mark.unit
    async def test_timed_and_logged_async_combines_both(self, caplog, capsys):
        """Test timed_and_logged with async functions."""
        # Set logger level explicitly for the function_tracker logger
        logger = logging.getLogger("stock_analysis.function_tracker")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="stock_analysis.function_tracker"):

            @timed_and_logged
            async def async_combined_function(value):
                await asyncio.sleep(0.01)
                return value + 5

            result = await async_combined_function(15)

            assert result == 20

            # Check both logging and timing are present
            log_messages = [record.message for record in caplog.records]
            has_entry = any(
                "→ Entering async_combined_function" in msg for msg in log_messages
            )

            assert has_entry

            captured = capsys.readouterr()
            assert "⏱️  Function 'async_combined_function' runtime:" in captured.out


class TestDecoratorIntegration:
    """
    Integration tests for decorators.
    
    These tests verify that decorators work correctly when used together
    and don't interfere with each other. This is important because the
    stock analysis codebase uses multiple decorators on the same functions
    to provide comprehensive monitoring and debugging capabilities.
    """

    @pytest.mark.integration
    def test_multiple_decorators_on_same_function(self, caplog, capsys):
        """Test applying multiple decorators to the same function."""
        with caplog.at_level(logging.DEBUG):
            # Apply decorators in different order
            @timer
            @logged
            def multi_decorated_function(x, y):
                time.sleep(0.01)
                return x + y

            result = multi_decorated_function(3, 7)

            assert result == 10

            # Should have both timing and logging
            log_messages = [record.message for record in caplog.records]
            has_entry = any("→ Entering" in msg for msg in log_messages)

            assert has_entry

            captured = capsys.readouterr()
            assert "⏱️  Function" in captured.out

    @pytest.mark.integration
    @pytest.mark.slow
    def test_decorator_performance_overhead(self):
        """Test that decorators don't add significant overhead."""

        # Test function without decorator
        def plain_function():
            return sum(range(1000))

        @timed_and_logged
        def decorated_function():
            return sum(range(1000))

        # Time both functions (rough performance check)
        start = time.perf_counter()
        for _ in range(100):
            plain_function()
        plain_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(100):
            decorated_function()
        decorated_time = time.perf_counter() - start

        # Decorated function shouldn't be more than 10x slower
        # (This is a very generous limit to account for logging overhead)
        assert decorated_time < plain_time * 10
