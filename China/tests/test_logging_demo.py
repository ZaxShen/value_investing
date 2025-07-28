"""
Demo test to show test logging functionality.

This test demonstrates how the test logging system works:
- All test logs are written to tests/logs/test_run_TIMESTAMP.log
- Error-level logs are also written to tests/logs/test_errors_TIMESTAMP.log
- Log files persist after test completion for debugging
"""

import pytest
import logging


class TestLoggingDemo:
    """Demo tests to show logging functionality."""

    @pytest.mark.unit
    def test_basic_logging_demo(self, test_logs_dir):
        """Demonstrate basic logging to the test logs directory."""
        # Get a logger for this test
        logger = logging.getLogger("demo_test")
        
        # Log messages at different levels
        logger.debug("This is a debug message from test")
        logger.info("This is an info message from test")
        logger.warning("This is a warning message from test")
        logger.error("This is an error message from test")
        
        # Verify the test_logs_dir fixture works
        assert test_logs_dir.exists()
        assert test_logs_dir.name == "logs"
        
        # Show that we can create test-specific log files too
        test_specific_log = test_logs_dir / "demo_test_specific.log"
        
        # Create file handler for test-specific logging
        file_handler = logging.FileHandler(test_specific_log)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create test-specific logger
        specific_logger = logging.getLogger("test_specific")
        specific_logger.addHandler(file_handler)
        specific_logger.setLevel(logging.INFO)
        
        # Log to test-specific file
        specific_logger.info("This message goes to the test-specific log file")
        specific_logger.warning("This warning also goes to test-specific log")
        
        # Cleanup
        specific_logger.removeHandler(file_handler)
        file_handler.close()
        
        # Verify file was created
        assert test_specific_log.exists()

    @pytest.mark.unit
    def test_error_logging_demo(self):
        """Demonstrate error logging that goes to the error log file."""
        logger = logging.getLogger("error_demo")
        
        # These will appear in both the main log and error log
        logger.error("This error will appear in both log files")
        logger.critical("This critical error will also appear in both files")
        
        # These will only appear in the main log
        logger.info("This info message only goes to main log")
        logger.warning("This warning only goes to main log")

    @pytest.mark.unit
    def test_logging_with_context(self):
        """Show logging with additional context information."""
        logger = logging.getLogger("context_demo")
        
        # Log with extra context
        test_data = {"stock_code": "000001", "stock_name": "平安银行"}
        
        logger.info(f"Processing stock: {test_data['stock_code']} - {test_data['stock_name']}")
        logger.debug(f"Full test data: {test_data}")
        
        # Simulate some processing
        try:
            # Simulate potential error
            if test_data["stock_code"] == "000001":
                raise ValueError("Simulated processing error for demo")
        except ValueError as e:
            logger.error(f"Error processing {test_data['stock_code']}: {e}")
            # Don't re-raise for demo purposes
        
        logger.info("Test completed successfully")