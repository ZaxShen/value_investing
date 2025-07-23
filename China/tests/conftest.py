"""
Pytest configuration and shared fixtures for the stock analysis project.

This module provides comprehensive pytest configuration and reusable fixtures
for testing the Chinese stock analysis pipeline. It includes fixtures for:

- Temporary directory management for test isolation
- Logger cleanup to prevent test interference
- Mock directory creation for file system operations
- Log output capture for assertion testing
- Sample test data for consistent test scenarios

The fixtures are designed to support both unit and integration testing
of the async stock analysis pipeline with proper cleanup and isolation.
"""

import pytest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import patch
import sys
import os

# Add src to path for imports
# This ensures that test modules can import from the src package
# even when pytest is run from different working directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def temp_logs_dir():
    """
    Create a temporary logs directory for testing.
    
    This fixture provides isolated temporary directories for log file testing
    to prevent conflicts between test runs and ensure clean test environments.
    The directory is automatically cleaned up after each test.
    
    Yields:
        str: Path to temporary directory for log files
    
    Usage:
        def test_logging(temp_logs_dir):
            log_file = Path(temp_logs_dir) / "test.log"
            # Test logging operations...
    """
    # Create temporary directory with descriptive prefix for easy identification
    temp_dir = tempfile.mkdtemp(prefix="test_logs_")
    yield temp_dir
    # Cleanup: Remove temporary directory and all contents after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_loggers():
    """
    Clean up loggers after each test to avoid interference.
    
    This fixture ensures test isolation by preventing logger state from
    leaking between tests. It captures the initial logger state before
    test execution and restores it afterward, removing any loggers
    created during the test.
    
    This is critical for testing logging functionality because:
    - Loggers are global singletons that persist between tests
    - Handler accumulation can cause duplicate log messages
    - Logger level changes can affect subsequent tests
    
    Usage:
        def test_logger_functionality(clean_loggers):
            logger = get_logger("test")
            # Test operations...
            # Logger is automatically cleaned up
    """
    # Store original logger state before test execution
    # This creates a snapshot of all existing loggers
    original_loggers = logging.Logger.manager.loggerDict.copy()

    # Yield control to test execution
    yield

    # Post-test cleanup: Remove any loggers created during the test
    # This prevents logger state pollution between tests
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name not in original_loggers:
            logger = logging.getLogger(name)
            # Clear all handlers to prevent memory leaks and duplicate messages
            logger.handlers.clear()
            # Remove logger from global registry
            del logging.Logger.manager.loggerDict[name]


@pytest.fixture
def mock_logs_directory():
    """Mock the logs directory creation."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        yield mock_mkdir


@pytest.fixture
def capture_log_output():
    """
    Capture log output for testing.
    
    This fixture provides a mechanism to capture log messages during test
    execution for assertion and analysis. It creates an in-memory string
    buffer with a configured log handler that can be used to verify
    that logging operations produce expected output.
    
    Returns:
        Tuple[StringIO, Handler]: Log capture buffer and configured handler
    
    Usage:
        def test_logging_output(capture_log_output):
            log_capture, handler = capture_log_output
            logger = logging.getLogger("test")
            logger.addHandler(handler)
            logger.info("Test message")
            assert "Test message" in log_capture.getvalue()
    """
    import io

    # Create in-memory string buffer to capture log output
    log_capture = io.StringIO()
    
    # Create stream handler that writes to our capture buffer
    handler = logging.StreamHandler(log_capture)
    
    # Set to DEBUG level to capture all log messages
    handler.setLevel(logging.DEBUG)

    # Create standardized formatter for consistent output format
    # Format: LEVEL:logger_name:message
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)

    # Yield both capture buffer and handler for test use
    yield log_capture, handler

    # Cleanup: Close handler to release resources
    handler.close()


@pytest.fixture
def sample_test_data():
    """
    Provide sample data for testing.
    
    This fixture provides consistent, realistic test data that mirrors
    the actual Chinese stock market data structure used in the application.
    It includes representative stock codes, Chinese company names, and
    industry classifications to ensure tests use realistic data patterns.
    
    The data includes:
    - stock_codes: Valid Chinese stock codes (Shanghai/Shenzhen)
    - stock_names: Actual Chinese company names in simplified Chinese
    - industries: Real industry classifications used in Chinese markets
    - test_args/kwargs: Generic test parameters for function testing
    
    Returns:
        dict: Dictionary containing various test data categories
    
    Usage:
        def test_stock_analysis(sample_test_data):
            code = sample_test_data["stock_codes"][0]  # "000001"
            name = sample_test_data["stock_names"][0]  # "平安银行"
    """
    return {
        # Realistic Chinese stock codes: 
        # 000001 (Shenzhen), 000002 (Shenzhen), 600519 (Shanghai)
        "stock_codes": ["000001", "000002", "600519"],
        
        # Corresponding company names in simplified Chinese
        "stock_names": ["平安银行", "万科A", "贵州茅台"],
        
        # Industry classifications matching the companies above
        "industries": ["银行", "房地产开发", "白酒"],
        
        # Generic test parameters for function argument testing
        "test_args": [1, 2, 3],
        "test_kwargs": {"key1": "value1", "key2": "value2"},
    }
