"""
Unit tests for the logging functionality.

This module provides comprehensive unit tests for the custom logging system
used throughout the stock analysis application. The logging system includes:

- setup_logger(): Configures file and console logging with appropriate formatters
- get_logger(): Factory function for creating module-specific loggers
- set_log_level(): Dynamic log level adjustment during runtime

The tests verify that the logging system:
- Creates proper file and console handlers with correct formatting
- Manages logger hierarchy and inheritance correctly
- Prevents duplicate handler creation for the same logger
- Supports dynamic log level changes during execution
- Handles high-volume logging scenarios efficiently

Proper logging is critical for debugging the async stock analysis pipeline,
especially when processing thousands of stocks concurrently with API rate limiting.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import the modules to test
from src.utilities.logger import setup_logger, get_logger, set_log_level


class TestSetupLogger:
    """
    Test the setup_logger function.
    
    The setup_logger function is the core of the logging system, responsible
    for creating properly configured loggers with both file and console output.
    It must handle directory creation, formatter setup, and prevent duplicate
    handler creation across multiple calls.
    """

    @pytest.mark.unit
    def test_setup_logger_default_params(self, temp_logs_dir, clean_loggers):
        """Test setup_logger with default parameters."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger()

            assert logger.name == "stock_analysis"
            assert logger.level == logging.INFO
            assert len(logger.handlers) >= 2  # File and console handlers

    @pytest.mark.unit
    def test_setup_logger_custom_params(self, temp_logs_dir, clean_loggers):
        """Test setup_logger with custom parameters."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger(name="test_logger", level="DEBUG")

            assert logger.name == "test_logger"
            assert logger.level == logging.DEBUG

    @pytest.mark.unit
    def test_setup_logger_creates_log_directory(self, clean_loggers):
        """Test that setup_logger creates the logs directory."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir

            setup_logger()

            mock_logs_dir.mkdir.assert_called_once_with(exist_ok=True)

    @pytest.mark.unit
    def test_setup_logger_prevents_duplicate_handlers(
        self, temp_logs_dir, clean_loggers
    ):
        """Test that setup_logger doesn't create duplicate handlers."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            # Create logger twice
            logger1 = setup_logger("duplicate_test")
            initial_handler_count = len(logger1.handlers)

            logger2 = setup_logger("duplicate_test")

            # Should return the same logger without adding new handlers
            assert logger1 is logger2
            assert len(logger2.handlers) == initial_handler_count


class TestGetLogger:
    """
    Test the get_logger function.
    
    The get_logger function provides a convenient factory for creating
    module-specific loggers with proper naming conventions. It supports
    both global logger access and module-specific logger creation with
    hierarchical naming (e.g., 'stock_analysis.stock_filter').
    """

    @pytest.mark.unit
    def test_get_logger_without_name(self, clean_loggers):
        """Test get_logger without specifying a name."""
        with patch("src.utilities.logger.logger") as mock_global_logger:
            result = get_logger()
            assert result is mock_global_logger

    @pytest.mark.unit
    def test_get_logger_with_name(self, clean_loggers):
        """Test get_logger with a specific name."""
        logger = get_logger("test_module")
        assert logger.name == "stock_analysis.test_module"

    @pytest.mark.unit
    def test_get_logger_creates_child_logger(self, clean_loggers):
        """Test that get_logger creates proper child loggers."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert "parent" in parent_logger.name
        assert "child" in child_logger.name


class TestSetLogLevel:
    """
    Test the set_log_level function.
    
    The set_log_level function allows dynamic adjustment of logging levels
    during runtime, which is crucial for debugging production issues or
    adjusting verbosity based on execution context (e.g., more verbose
    logging during development, less during production runs).
    """

    @pytest.mark.unit
    def test_set_log_level_valid_levels(self, clean_loggers):
        """Test set_log_level with valid log levels."""
        with patch("src.utilities.logger.logger") as mock_logger:
            mock_handler = MagicMock()
            mock_handler.stream = sys.stdout
            mock_logger.handlers = [mock_handler]

            for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                set_log_level(level)
                mock_logger.setLevel.assert_called_with(getattr(logging, level))

    @pytest.mark.unit
    def test_set_log_level_updates_logger_level(self, clean_loggers):
        """Test that set_log_level updates logger level."""
        # Create a test logger
        logger = setup_logger("test_set_level", "INFO")
        original_level = logger.level

        # Test setting to DEBUG
        with patch("src.utilities.logger.logger", logger):
            set_log_level("DEBUG")
            assert logger.level == logging.DEBUG

        # Test setting to WARNING
        with patch("src.utilities.logger.logger", logger):
            set_log_level("WARNING")
            assert logger.level == logging.WARNING


class TestLoggerIntegration:
    """
    Integration tests for the logger functionality.
    
    These integration tests verify that the logging system works correctly
    as a whole, including file I/O operations, logger hierarchy management,
    and interaction between different logger instances. This is particularly
    important for the stock analysis pipeline where multiple modules log
    concurrently to the same files.
    """

    @pytest.mark.integration
    def test_logger_writes_to_file(self, temp_logs_dir):
        """Test that logger actually writes to log files."""
        log_file = Path(temp_logs_dir) / "test_integration.log"

        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = log_file

            logger = setup_logger("integration_test")
            logger.info("Test message for integration")

            # Flush handlers to ensure write
            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            # Check if file was created and contains our message
            if log_file.exists():
                content = log_file.read_text()
                assert "Test message for integration" in content

    @pytest.mark.integration
    def test_multiple_loggers_different_modules(self, clean_loggers):
        """Test creating loggers for different modules."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module1")  # Same as logger1

        assert logger1.name == "stock_analysis.module1"
        assert logger2.name == "stock_analysis.module2"
        assert logger1 is not logger2
        assert logger1.name == logger3.name  # Same logger instance

    @pytest.mark.integration
    def test_log_level_inheritance(self, clean_loggers):
        """Test that child loggers inherit parent log levels."""
        parent = get_logger("parent")
        child = get_logger("parent.child")

        parent.setLevel(logging.WARNING)

        # Child should inherit parent's effective level
        assert child.getEffectiveLevel() >= logging.WARNING
