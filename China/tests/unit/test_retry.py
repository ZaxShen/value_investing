"""
Unit tests for the retry mechanism with timeout handling.

This module tests the enhanced retry functionality that provides:
- Exponential backoff retry logic
- Timeout handling with ThreadPoolExecutor
- Comprehensive error logging
- Support for both sync and async contexts

The retry mechanism is critical for handling unreliable external APIs
like akshare that may timeout or fail intermittently.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
from concurrent.futures import TimeoutError as FutureTimeoutError

from src.utilities.retry import (
    retry_call,
    retry_with_backoff,
    RetryConfig,
    API_RETRY_CONFIG,
    NETWORK_RETRY_CONFIG,
)


class TestRetryCall:
    """Test the core retry_call function with timeout handling."""

    @pytest.mark.unit
    def test_retry_call_success_first_attempt(self):
        """Test successful function call on first attempt."""
        mock_func = Mock(return_value="success")
        
        result = retry_call(mock_func, "arg1", keyword="arg2", max_retries=3)
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", keyword="arg2")

    @pytest.mark.unit
    def test_retry_call_success_after_retries(self):
        """Test successful function call after initial failures."""
        mock_func = Mock()
        mock_func.side_effect = [ConnectionError("fail1"), ConnectionError("fail2"), "success"]
        
        result = retry_call(
            mock_func,
            max_retries=3,
            initial_delay=0.01,  # Fast test
            exceptions=(ConnectionError,)
        )
        
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.unit
    def test_retry_call_all_attempts_fail(self):
        """Test behavior when all retry attempts fail."""
        mock_func = Mock(side_effect=ConnectionError("persistent error"))
        
        with pytest.raises(ConnectionError, match="persistent error"):
            retry_call(
                mock_func,
                max_retries=2,
                initial_delay=0.01,
                exceptions=(ConnectionError,)
            )
        
        assert mock_func.call_count == 2

    @pytest.mark.unit
    def test_retry_call_timeout_handling(self):
        """Test timeout functionality with ThreadPoolExecutor."""
        def slow_function():
            time.sleep(0.1)  # 100ms function
            return "slow_result"
        
        # Test with sufficient timeout
        result = retry_call(slow_function, timeout=0.2, max_retries=1)
        assert result == "slow_result"
        
        # Test with insufficient timeout
        with pytest.raises(TimeoutError, match="timed out after 0.05 seconds"):
            retry_call(slow_function, timeout=0.05, max_retries=1)

    @pytest.mark.unit
    def test_retry_call_timeout_with_retries(self):
        """Test timeout handling across multiple retry attempts."""
        call_count = 0
        def intermittent_slow_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                time.sleep(0.1)  # First 2 calls timeout
            return f"success_on_attempt_{call_count}"
        
        result = retry_call(
            intermittent_slow_function,
            timeout=0.05,  # Will timeout first 2 attempts
            max_retries=3,
            initial_delay=0.01
        )
        
        assert result == "success_on_attempt_3"
        assert call_count == 3

    @pytest.mark.unit
    def test_retry_call_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        mock_func = Mock(side_effect=[Exception("fail"), "success"])
        
        start_time = time.time()
        result = retry_call(
            mock_func,
            max_retries=2,
            initial_delay=0.02,  # 20ms
            backoff_multiplier=2.0,
            exceptions=(Exception,)
        )
        elapsed = time.time() - start_time
        
        assert result == "success"
        # Should have at least the initial delay (20ms)
        assert elapsed >= 0.02
        assert mock_func.call_count == 2

    @pytest.mark.unit
    def test_retry_call_preserves_exception_types(self):
        """Test that specific exception types are properly handled."""
        # ConnectionError should trigger retry
        mock_func = Mock(side_effect=ConnectionError("network error"))
        with pytest.raises(ConnectionError):
            retry_call(mock_func, max_retries=1, exceptions=(ConnectionError,))
        
        # ValueError should NOT trigger retry (not in exceptions tuple)
        mock_func = Mock(side_effect=ValueError("value error"))
        with pytest.raises(ValueError):
            retry_call(mock_func, max_retries=3, exceptions=(ConnectionError,))
        
        assert mock_func.call_count == 1  # Should not retry ValueError


class TestRetryWithBackoffDecorator:
    """Test the retry_with_backoff decorator."""

    @pytest.mark.unit
    def test_decorator_basic_functionality(self):
        """Test decorator works with basic function."""
        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def decorated_function(x):
            if x < 2:
                raise ConnectionError("not ready")
            return f"result_{x}"
        
        # Should fail with x=1
        with pytest.raises(ConnectionError):
            decorated_function(1)
        
        # Should succeed with x=2
        result = decorated_function(2)
        assert result == "result_2"

    @pytest.mark.unit
    def test_decorator_preserves_metadata(self):
        """Test decorator preserves function metadata."""
        @retry_with_backoff()
        def documented_function(param: int) -> str:
            """This function has documentation."""
            return str(param)
        
        assert documented_function.__name__ == "documented_function"
        assert "documentation" in documented_function.__doc__


class TestRetryConfig:
    """Test the RetryConfig class."""

    @pytest.mark.unit
    def test_retry_config_initialization(self):
        """Test RetryConfig initializes with correct defaults."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.backoff_multiplier == 2.0
        assert config.timeout == 30.0
        assert config.exceptions == Exception

    @pytest.mark.unit
    def test_retry_config_custom_values(self):
        """Test RetryConfig with custom parameters."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            backoff_multiplier=1.5,
            timeout=60.0,
            exceptions=(ConnectionError, TimeoutError)
        )
        
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.backoff_multiplier == 1.5
        assert config.timeout == 60.0
        assert config.exceptions == (ConnectionError, TimeoutError)

    @pytest.mark.unit
    def test_retry_config_retry_method(self):
        """Test RetryConfig.retry() method."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        mock_func = Mock(return_value="config_success")
        
        result = config.retry(mock_func, "arg1", keyword="arg2")
        
        assert result == "config_success"
        mock_func.assert_called_once_with("arg1", keyword="arg2")

    @pytest.mark.unit
    def test_retry_config_with_timeout(self):
        """Test RetryConfig respects timeout parameter."""
        config = RetryConfig(timeout=0.05, max_retries=1)
        
        def slow_function():
            time.sleep(0.1)
            return "slow"
        
        with pytest.raises(TimeoutError):
            config.retry(slow_function)


class TestPredefinedConfigs:
    """Test the predefined retry configurations."""

    @pytest.mark.unit
    def test_api_retry_config(self):
        """Test API_RETRY_CONFIG has appropriate values for API calls."""
        assert API_RETRY_CONFIG.max_retries == 3
        assert API_RETRY_CONFIG.timeout == 45.0  # Generous for API calls
        assert ConnectionError in API_RETRY_CONFIG.exceptions
        assert TimeoutError in API_RETRY_CONFIG.exceptions

    @pytest.mark.unit
    def test_network_retry_config(self):
        """Test NETWORK_RETRY_CONFIG has appropriate values for network ops."""
        assert NETWORK_RETRY_CONFIG.max_retries == 5  # More retries for network
        assert NETWORK_RETRY_CONFIG.timeout == 30.0
        assert NETWORK_RETRY_CONFIG.backoff_multiplier == 1.5  # Gentler backoff

    @pytest.mark.unit
    def test_configs_are_different_instances(self):
        """Test that predefined configs are separate instances."""
        # Modifying one shouldn't affect the other
        original_api_retries = API_RETRY_CONFIG.max_retries
        API_RETRY_CONFIG.max_retries = 999
        
        assert NETWORK_RETRY_CONFIG.max_retries != 999
        
        # Reset for other tests
        API_RETRY_CONFIG.max_retries = original_api_retries


class TestRetryIntegration:
    """Integration tests for retry mechanism with logging."""

    @pytest.mark.unit
    def test_retry_logging_on_success(self, caplog):
        """Test retry mechanism logs timing for successful calls."""
        def slow_success():
            time.sleep(0.02)  # 20ms
            return "logged_success"
        
        with caplog.at_level("INFO"):
            result = retry_call(slow_success, timeout=1.0)
            
        assert result == "logged_success"
        
        # Should log timing for slow calls (>10ms)
        log_messages = [record.message for record in caplog.records]
        timing_logs = [msg for msg in log_messages if "completed in" in msg and "seconds" in msg]
        assert len(timing_logs) >= 1

    @pytest.mark.unit
    def test_retry_logging_on_timeout(self, caplog):
        """Test retry mechanism logs timeout warnings."""
        def timeout_function():
            time.sleep(0.1)
            return "never_reached"
        
        with caplog.at_level("WARNING"):
            with pytest.raises(TimeoutError):
                retry_call(timeout_function, timeout=0.05, max_retries=2, initial_delay=0.01)
        
        log_messages = [record.message for record in caplog.records]
        timeout_logs = [msg for msg in log_messages if "timed out" in msg.lower()]
        assert len(timeout_logs) >= 2  # Should log each timeout attempt

    @pytest.mark.unit  
    def test_retry_logging_on_failure(self, caplog):
        """Test retry mechanism logs failure warnings and final error."""
        def failing_function():
            raise ConnectionError("simulated network failure")
        
        with caplog.at_level("WARNING"):
            with pytest.raises(ConnectionError):
                retry_call(
                    failing_function,
                    max_retries=2,
                    initial_delay=0.01,
                    exceptions=(ConnectionError,)
                )
        
        log_messages = [record.message for record in caplog.records]
        
        # Should have warning logs for retry attempts
        retry_warnings = [msg for msg in log_messages if "failed (attempt" in msg and "Retrying" in msg]
        assert len(retry_warnings) >= 1
        
        # Should have final error log
        error_logs = [msg for msg in log_messages if "failed after" in msg and "attempts" in msg]
        assert len(error_logs) >= 1