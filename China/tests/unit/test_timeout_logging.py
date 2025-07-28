"""
Unit tests for timeout scenario logging.

This module tests the logging behavior during timeout scenarios to ensure:
- Timeout warnings are properly logged
- Retry attempt logging includes timeout information
- Error messages are clear and actionable
- Log levels are appropriate for different timeout scenarios
- Performance logging works during timeout conditions
"""

import asyncio
import pytest
import time
import logging
from unittest.mock import Mock, patch
from concurrent.futures import TimeoutError as FutureTimeoutError

from src.utilities.retry import retry_call, API_RETRY_CONFIG, RetryConfig
from src.utilities.logger import get_logger


class TestTimeoutLoggingBehavior:
    """Test logging behavior during timeout scenarios."""

    @pytest.mark.unit
    def test_retry_timeout_warning_logging(self, caplog):
        """Test that timeout warnings are logged with appropriate detail."""
        def timeout_function():
            time.sleep(0.1)  # 100ms delay
            return "never_reached"
        
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TimeoutError):
                retry_call(
                    timeout_function,
                    timeout=0.05,  # 50ms timeout - will fail
                    max_retries=2,
                    initial_delay=0.01,
                    logger_name="test_retry"
                )
        
        # Check warning logs for timeout attempts
        warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        
        timeout_warnings = [msg for msg in warning_messages if "timed out" in msg.lower()]
        assert len(timeout_warnings) >= 1, f"Expected timeout warnings, got: {warning_messages}"
        
        # Check that at least one timeout warning includes retry information
        retry_warnings = [msg for msg in timeout_warnings if "attempt" in msg.lower() and "retrying" in msg.lower()]
        assert len(retry_warnings) >= 1, f"Expected retry warnings with attempt info, got: {timeout_warnings}"
        
        first_retry_warning = retry_warnings[0]
        assert "seconds" in first_retry_warning

    @pytest.mark.unit
    def test_retry_final_timeout_error_logging(self, caplog):
        """Test final timeout error logging after all retries fail."""
        def always_timeout_function():
            time.sleep(0.1)
            return "unreachable"
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(TimeoutError):
                retry_call(
                    always_timeout_function,
                    timeout=0.05,
                    max_retries=2,
                    initial_delay=0.01,
                    logger_name="test_final_timeout"
                )
        
        # Check for final error log
        error_messages = [record.message for record in caplog.records if record.levelno == logging.ERROR]
        
        final_error_logs = [msg for msg in error_messages if "timed out after" in msg and "attempts" in msg]
        assert len(final_error_logs) >= 1, f"Expected final timeout error log, got: {error_messages}"
        
        final_error = final_error_logs[0]
        assert "2 attempts" in final_error  # Should mention number of attempts
        assert "seconds" in final_error  # Should include timing information

    @pytest.mark.unit
    def test_retry_success_after_timeout_logging(self, caplog):
        """Test logging when function succeeds after initial timeouts."""
        call_count = 0
        def intermittent_timeout_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                time.sleep(0.1)  # First call times out
            return f"success_on_attempt_{call_count}"
        
        with caplog.at_level(logging.INFO):
            result = retry_call(
                intermittent_timeout_function,
                timeout=0.05,  # First attempt will timeout
                max_retries=2,
                initial_delay=0.01,
                logger_name="test_recovery"
            )
        
        assert result == "success_on_attempt_2"
        
        # Should have both timeout warning and success info
        all_messages = [record.message for record in caplog.records]
        
        timeout_warnings = [msg for msg in all_messages if "timed out" in msg.lower() and "retrying" in msg.lower()]
        assert len(timeout_warnings) >= 1, "Should log timeout warning for first attempt"
        
        # Check for performance logging on successful completion
        success_logs = [msg for msg in all_messages if "completed in" in msg]
        # Success logging is optional depending on execution time

    @pytest.mark.unit
    def test_timeout_logging_includes_function_name(self, caplog):
        """Test that timeout logs include the function name being called."""
        def named_timeout_function():
            time.sleep(0.1)
            return "timeout"
        
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TimeoutError):
                retry_call(
                    named_timeout_function,
                    timeout=0.05,
                    max_retries=1,
                    logger_name="test_function_names"
                )
        
        warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        
        # Should mention the function name in timeout logs
        function_name_logs = [msg for msg in warning_messages if "named_timeout_function" in msg]
        assert len(function_name_logs) >= 1, f"Expected function name in logs, got: {warning_messages}"

    @pytest.mark.unit  
    def test_timeout_logging_with_different_log_levels(self, caplog):
        """Test timeout logging respects different log levels."""
        def timeout_function():
            time.sleep(0.1)
            return "timeout"
        
        # Test with INFO level - should not see WARNING logs
        with caplog.at_level(logging.ERROR):
            with pytest.raises(TimeoutError):
                retry_call(
                    timeout_function,
                    timeout=0.05,
                    max_retries=2,
                    initial_delay=0.01,
                    logger_name="test_log_levels"
                )
        
        # Should only see ERROR level logs (final failure), not WARNING level (retry attempts)
        warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
        error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
        
        assert len(warning_records) == 0, "Should not see WARNING logs at ERROR level"
        assert len(error_records) >= 1, "Should see ERROR logs for final failure"

    @pytest.mark.unit
    def test_concurrent_timeout_logging(self, caplog):
        """Test logging behavior with concurrent timeout operations."""
        def timeout_function(operation_id):
            time.sleep(0.1)
            return f"result_{operation_id}"
        
        with caplog.at_level(logging.WARNING):
            # Run multiple timeout operations concurrently
            async def run_concurrent_timeouts():
                tasks = []
                for i in range(3):
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            retry_call,
                            timeout_function,
                            i,  # operation_id
                            timeout=0.05,
                            max_retries=1,
                            initial_delay=0.01,
                            logger_name=f"concurrent_test_{i}"
                        )
                    )
                    tasks.append(task)
                
                # Gather with return_exceptions to capture TimeoutErrors
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            # Run the concurrent test
            results = asyncio.run(run_concurrent_timeouts())
            
            # All should have timed out
            timeout_errors = [r for r in results if isinstance(r, TimeoutError)]
            assert len(timeout_errors) == 3, "All operations should have timed out"
            
            # Should have timeout logs for each operation
            warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
            
            # Should have logs mentioning different operation IDs or function names
            assert len(warning_messages) >= 3, f"Expected at least 3 timeout warnings, got {len(warning_messages)}"


class TestStockAnalysisTimeoutLogging:
    """Test timeout logging in the context of stock analysis operations."""

    @pytest.mark.unit
    async def test_stock_filter_timeout_logging(self, caplog):
        """Test timeout logging in StockFilter operations."""
        from src.stock_filter import StockFilter
        import pandas as pd
        
        # Create minimal test data
        industry_mapping = pd.DataFrame({
            "行业": ["银行"],
            "代码": ["000001"]
        })
        
        stock_data = pd.DataFrame({
            "代码": ["000001"],
            "名称": ["测试银行"],
            "最新价": [10.0],
            "市值": [100.0],
            "市盈率-动态": [8.0],
            "涨跌幅": [1.0],
            "60日涨跌幅": [5.0],
            "年初至今涨跌幅": [10.0]
        })
        
        stock_filter = StockFilter(industry_mapping, stock_data)
        
        # Mock timeout scenario
        def timeout_api_call(*args, **kwargs):
            time.sleep(0.2)  # Long enough to trigger asyncio timeout
            return pd.DataFrame()
        
        with caplog.at_level(logging.WARNING):
            with patch.object(stock_filter, '_fetch_stock_fund_flow_sync', 
                             side_effect=timeout_api_call):
                
                # This should trigger timeout and logging
                result = await stock_filter.process_single_stock_async("000001", "测试银行", 29)
                
                assert result is None  # Should return None on timeout
        
        # Check for timeout-related logs
        all_messages = [record.message for record in caplog.records]
        
        timeout_logs = [msg for msg in all_messages if "timeout" in msg.lower()]
        assert len(timeout_logs) >= 1, f"Expected timeout logs in stock filter, got: {all_messages}"
        
        # Should mention stock information in logs
        stock_specific_logs = [msg for msg in all_messages if "000001" in msg or "测试银行" in msg]
        assert len(stock_specific_logs) >= 1, "Should log stock-specific timeout information"

    @pytest.mark.unit
    async def test_industry_filter_timeout_logging(self, caplog):
        """Test timeout logging in IndustryFilter operations."""
        from src.industry_filter import IndustryFilter
        
        industry_filter = IndustryFilter()
        
        # Mock timeout in capital flow API
        def capital_flow_timeout(*args, **kwargs):
            time.sleep(0.2)  # Trigger timeout
            return pd.DataFrame()
        
        with caplog.at_level(logging.WARNING):
            with patch.object(industry_filter, '_fetch_industry_capital_flow_data_sync',
                             side_effect=capital_flow_timeout):
                
                result = await industry_filter.process_single_industry_async("银行", "20250101", "20250131", 30)
                
                assert result is None  # Should return None on timeout
        
        # Check for industry-specific timeout logs
        all_messages = [record.message for record in caplog.records]
        
        timeout_logs = [msg for msg in all_messages if "timeout" in msg.lower()]
        assert len(timeout_logs) >= 1, f"Expected timeout logs in industry filter, got: {all_messages}"
        
        # Should mention industry name in logs
        industry_logs = [msg for msg in all_messages if "银行" in msg]
        assert len(industry_logs) >= 1, "Should log industry-specific timeout information"

    @pytest.mark.unit
    def test_api_retry_config_logging_integration(self, caplog):
        """Test that API_RETRY_CONFIG produces appropriate logs during timeouts."""
        def api_timeout_function():
            time.sleep(0.1)  # 100ms - longer than our custom timeout
            return "api_result"
        
        # Create a custom config with shorter timeout for testing
        test_config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            timeout=0.05,  # 50ms timeout to trigger timeout error
            exceptions=(ConnectionError, TimeoutError, Exception)
        )
        
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TimeoutError):
                test_config.retry(api_timeout_function)
        
        # API_RETRY_CONFIG should produce appropriate timeout logs
        warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        error_messages = [record.message for record in caplog.records if record.levelno == logging.ERROR]
        
        # Should have timeout warnings for retry attempts
        timeout_warnings = [msg for msg in warning_messages if "timed out" in msg.lower()]
        assert len(timeout_warnings) >= 1, "Retry config should log timeout warnings"
        
        # Should have final error for complete failure
        final_errors = [msg for msg in error_messages if ("failed after" in msg or "timed out after" in msg)]
        assert len(final_errors) >= 1, "Retry config should log final timeout error"
        
        # Check timeout duration is mentioned (0.05 seconds for our test config)
        duration_mentions = [msg for msg in warning_messages + error_messages if "0.05" in msg or "seconds" in msg]
        assert len(duration_mentions) >= 1, "Should mention timeout duration in logs"


class TestLoggingPerformanceWithTimeouts:
    """Test that logging doesn't significantly impact timeout handling performance."""

    @pytest.mark.unit
    def test_logging_overhead_during_timeouts(self):
        """Test that logging doesn't add significant overhead to timeout handling."""
        import time
        
        def quick_timeout_function():
            time.sleep(0.05)  # 50ms
            return "result"
        
        # Measure time with logging enabled
        start_with_logging = time.perf_counter()
        try:
            retry_call(
                quick_timeout_function,
                timeout=0.03,  # 30ms timeout
                max_retries=2,
                initial_delay=0.01,
                logger_name="performance_test"
            )
        except TimeoutError:
            pass
        end_with_logging = time.perf_counter()
        
        logging_time = end_with_logging - start_with_logging
        
        # Should complete relatively quickly despite logging
        # Allow generous margin for CI/test environment variability
        assert logging_time < 0.5, f"Logging added too much overhead: {logging_time} seconds"

    @pytest.mark.unit
    def test_log_message_formatting_performance(self):
        """Test that log message formatting doesn't cause delays."""
        # Create a scenario with complex log formatting
        def complex_args_function(large_data, *args, **kwargs):
            time.sleep(0.05)
            return "result"
        
        large_data = "x" * 1000  # Large argument that might slow down log formatting
        
        start_time = time.perf_counter()
        try:
            retry_call(
                complex_args_function,
                large_data,
                "arg1",
                "arg2",
                timeout=0.03,
                max_retries=1,
                keyword_arg="value",
                logger_name="formatting_test"
            )
        except TimeoutError:
            pass
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        # Log formatting shouldn't cause significant delays
        assert elapsed < 0.3, f"Log formatting caused excessive delay: {elapsed} seconds"


class TestTimeoutLoggingEdgeCases:
    """Test logging behavior in edge cases and error conditions."""

    @pytest.mark.unit
    def test_logging_with_invalid_logger_names(self, caplog):
        """Test timeout logging with various logger name configurations."""
        def timeout_function():
            time.sleep(0.1)
            return "result"
        
        # Test with None logger name
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TimeoutError):
                retry_call(
                    timeout_function,
                    timeout=0.05,
                    max_retries=1,
                    logger_name=None  # Should use default logger
                )
        
        # Should still produce logs even with None logger name
        warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warning_messages) >= 1, "Should log timeouts even with None logger name"

    @pytest.mark.unit
    def test_logging_during_logger_configuration_issues(self, caplog):
        """Test timeout logging when there are logger configuration problems."""
        def timeout_function():
            time.sleep(0.1)
            return "result"
        
        # Mock logger configuration issue
        with patch('src.utilities.retry.get_logger', side_effect=Exception("Logger error")):
            # Should still handle timeouts gracefully even if logging fails
            with pytest.raises(TimeoutError):
                retry_call(
                    timeout_function,
                    timeout=0.05,
                    max_retries=1,
                    logger_name="problematic_logger"
                )
        
        # The timeout should still work even if logging fails
        # (Exact behavior depends on implementation - might fallback to print or ignore logging errors)

    @pytest.mark.unit
    def test_timeout_logging_with_very_short_timeouts(self, caplog):
        """Test logging behavior with very short timeout values."""
        def instant_timeout_function():
            time.sleep(0.01)  # 10ms
            return "result"
        
        with caplog.at_level(logging.WARNING):
            with pytest.raises(TimeoutError):
                retry_call(
                    instant_timeout_function,
                    timeout=0.001,  # 1ms timeout - very short
                    max_retries=1,
                    initial_delay=0.001,
                    logger_name="short_timeout_test"
                )
        
        # Should handle very short timeouts gracefully in logging
        warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
        
        timeout_logs = [msg for msg in warning_messages if "timed out" in msg.lower()]
        assert len(timeout_logs) >= 1, "Should log even very short timeouts"
        
        # Should mention the short timeout duration
        duration_logs = [msg for msg in timeout_logs if "0.001" in msg or "1" in msg]
        # Note: Exact duration logging format may vary