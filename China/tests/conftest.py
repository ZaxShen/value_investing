"""
Pytest configuration and shared fixtures for the stock analysis project.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def temp_logs_dir():
    """Create a temporary logs directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_logs_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_loggers():
    """Clean up loggers after each test to avoid interference."""
    # Store original loggers
    original_loggers = logging.Logger.manager.loggerDict.copy()
    
    yield
    
    # Clean up any loggers created during the test
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name not in original_loggers:
            logger = logging.getLogger(name)
            logger.handlers.clear()
            del logging.Logger.manager.loggerDict[name]


@pytest.fixture
def mock_logs_directory():
    """Mock the logs directory creation."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        yield mock_mkdir


@pytest.fixture  
def capture_log_output():
    """Capture log output for testing."""
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    
    yield log_capture, handler
    
    handler.close()


@pytest.fixture
def sample_test_data():
    """Provide sample data for testing."""
    return {
        'stock_codes': ['000001', '000002', '600519'],
        'stock_names': ['平安银行', '万科A', '贵州茅台'],
        'industries': ['银行', '房地产开发', '白酒'],
        'test_args': [1, 2, 3],
        'test_kwargs': {'key1': 'value1', 'key2': 'value2'}
    }