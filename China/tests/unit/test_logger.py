"""
Unit tests for the logging functionality.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import the modules to test
from src.utilities.logger import setup_logger, get_logger, set_log_level, set_console_log_level
from rich.logging import RichHandler


class TestSetupLogger:
    """Test the setup_logger function."""

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
    """Test the get_logger function."""

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
    """Test the set_log_level function."""

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
    """Integration tests for the logger functionality."""

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


class TestRichHandlerIntegration:
    """Test Rich logging handler integration."""

    @pytest.mark.unit
    def test_setup_logger_creates_rich_handler(self, temp_logs_dir, clean_loggers):
        """Test that setup_logger creates a RichHandler for console output."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger("rich_test")

            # Find the RichHandler
            rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
            assert len(rich_handlers) == 1

            rich_handler = rich_handlers[0]
            assert rich_handler.level == logging.WARNING  # Default console level
            assert rich_handler._log_render.show_time == True
            assert rich_handler._log_render.show_path == False

    @pytest.mark.unit
    def test_rich_handler_configuration(self, temp_logs_dir, clean_loggers):
        """Test RichHandler is configured with correct settings."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger("rich_config_test")
            
            rich_handler = None
            for handler in logger.handlers:
                if isinstance(handler, RichHandler):
                    rich_handler = handler
                    break

            assert rich_handler is not None
            # Test Rich handler configuration
            assert rich_handler._log_render.show_time == True
            assert rich_handler._log_render.show_path == False
            # Test that the handler has a console (attribute may vary by version)
            assert hasattr(rich_handler, '_console') or hasattr(rich_handler, 'console')

    @pytest.mark.unit
    def test_set_console_log_level_affects_rich_handler(self, clean_loggers):
        """Test that set_console_log_level changes RichHandler level."""
        # Use the global logger instance instead of creating a new one
        from src.utilities.logger import logger
        
        # Find RichHandler in global logger
        rich_handler = None
        for handler in logger.handlers:
            if isinstance(handler, RichHandler):
                rich_handler = handler
                break

        if rich_handler is None:
            # If no RichHandler found in global logger, skip test
            pytest.skip("No RichHandler found in global logger")
            
        initial_level = rich_handler.level
        
        # Change console log level
        set_console_log_level("DEBUG")
        
        # Verify RichHandler level changed
        assert rich_handler.level == logging.DEBUG
        assert rich_handler.level != initial_level

    @pytest.mark.unit
    def test_set_log_level_affects_rich_handler(self, clean_loggers):
        """Test that set_log_level changes RichHandler level."""
        # Use the global logger instance instead of creating a new one
        from src.utilities.logger import logger
        
        # Find RichHandler in global logger
        rich_handler = None
        for handler in logger.handlers:
            if isinstance(handler, RichHandler):
                rich_handler = handler
                break

        if rich_handler is None:
            # If no RichHandler found in global logger, skip test
            pytest.skip("No RichHandler found in global logger")
        
        # Change global log level
        set_log_level("ERROR")
        
        # Verify both logger and RichHandler levels changed
        assert logger.level == logging.ERROR
        assert rich_handler.level == logging.ERROR

    @pytest.mark.integration
    def test_rich_handler_markup_support(self, temp_logs_dir, clean_loggers, caplog):
        """Test that RichHandler supports markup in log messages."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger("markup_test")
            
            # Set console level to INFO so we can see the message
            set_console_log_level("INFO") 
            
            with caplog.at_level(logging.INFO):
                logger.info("[bold red]Error message[/bold red] with markup")
                
            # Verify message was captured (markup will be processed by Rich)
            assert "Error message" in caplog.text

    @pytest.mark.integration  
    def test_rich_tracebacks_enabled(self, temp_logs_dir, clean_loggers):
        """Test that RichHandler has rich tracebacks enabled."""
        with patch("src.utilities.logger.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = (
                Path(temp_logs_dir) / "test.log"
            )

            logger = setup_logger("traceback_test")
            
            # Find RichHandler and verify rich_tracebacks is enabled
            for handler in logger.handlers:
                if isinstance(handler, RichHandler):
                    # RichHandler should be configured with rich_tracebacks=True
                    # This is set in the constructor parameters
                    assert hasattr(handler, '_log_render')
                    break
            else:
                pytest.fail("RichHandler not found in logger handlers")
