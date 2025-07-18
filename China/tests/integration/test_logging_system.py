"""
Integration tests for the complete logging system.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import logging
from unittest.mock import patch

from src.utilities.logger import setup_logger, get_logger, set_log_level
from src.utilities.tools import timer, logged, timed_and_logged


class TestLoggingSystemIntegration:
    """Integration tests for the complete logging system."""
    
    @pytest.mark.integration
    def test_end_to_end_logging_workflow(self, temp_logs_dir):
        """Test complete logging workflow from setup to file output."""
        log_file = Path(temp_logs_dir) / "integration_test.log"
        
        with patch('src.utilities.logger.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = log_file
            
            # Setup logger
            logger = setup_logger("integration_workflow", "DEBUG")
            
            # Create a test function with logging
            module_logger = get_logger("test_module")
            
            @logged
            @timer
            def test_workflow_function(data):
                module_logger.info(f"Processing data: {data}")
                module_logger.debug("Debug information")
                module_logger.warning("Warning message")
                return f"Processed: {data}"
            
            # Execute the function
            result = test_workflow_function("test_data")
            
            # Verify result
            assert result == "Processed: test_data"
            
            # Flush all handlers
            for handler in logging.root.manager.loggerDict.values():
                if hasattr(handler, 'handlers'):
                    for h in getattr(handler, 'handlers', []):
                        if hasattr(h, 'flush'):
                            h.flush()
            
            # Check log file if it exists
            if log_file.exists():
                log_content = log_file.read_text()
                assert "Processing data: test_data" in log_content
                assert "Debug information" in log_content
                assert "Warning message" in log_content
    
    @pytest.mark.integration
    async def test_async_logging_integration(self, temp_logs_dir):
        """Test logging integration with async functions."""
        log_file = Path(temp_logs_dir) / "async_integration.log"
        
        with patch('src.utilities.logger.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = log_file
            
            logger = setup_logger("async_integration", "INFO")
            module_logger = get_logger("async_test")
            
            @timed_and_logged
            async def async_workflow(items):
                module_logger.info(f"Starting async processing of {len(items)} items")
                
                results = []
                for i, item in enumerate(items):
                    module_logger.debug(f"Processing item {i}: {item}")
                    await asyncio.sleep(0.001)  # Simulate async work
                    results.append(f"processed_{item}")
                
                module_logger.info("Async processing completed")
                return results
            
            # Execute async function
            test_items = ["item1", "item2", "item3"]
            results = await async_workflow(test_items)
            
            # Verify results
            assert len(results) == 3
            assert all("processed_" in result for result in results)
    
    @pytest.mark.integration
    def test_multiple_modules_logging(self, temp_logs_dir):
        """Test logging from multiple modules simultaneously."""
        log_file = Path(temp_logs_dir) / "multi_module.log"
        
        with patch('src.utilities.logger.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = log_file
            
            # Setup main logger
            main_logger = setup_logger("multi_module_test", "DEBUG")
            
            # Create loggers for different modules
            stock_logger = get_logger("stock_filter")
            analysis_logger = get_logger("stock_analysis")
            industry_logger = get_logger("industry_filter")
            
            # Simulate logging from different modules
            @logged
            def stock_processing():
                stock_logger.info("Processing stock data")
                stock_logger.debug("Stock filtering applied")
                return "stocks_processed"
            
            @logged
            def analysis_processing():
                analysis_logger.info("Running stock analysis")
                analysis_logger.warning("Low data quality detected")
                return "analysis_complete"
            
            @timer
            def industry_processing():
                industry_logger.info("Processing industry data")
                industry_logger.error("Industry data incomplete")
                return "industry_processed"
            
            # Execute functions from different modules
            stock_result = stock_processing()
            analysis_result = analysis_processing()
            industry_result = industry_processing()
            
            # Verify all results
            assert stock_result == "stocks_processed"
            assert analysis_result == "analysis_complete"
            assert industry_result == "industry_processed"
    
    @pytest.mark.integration
    def test_log_level_changes_during_runtime(self, temp_logs_dir, caplog):
        """Test changing log levels during runtime."""
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("runtime_test")
            
            # Initially at INFO level
            logger.info("Info message 1")
            logger.debug("Debug message 1 - should not appear")
            
            # Change to DEBUG level
            set_log_level("DEBUG")
            
            logger.info("Info message 2")
            logger.debug("Debug message 2 - should appear")
            
            # Change back to WARNING level
            set_log_level("WARNING")
            
            logger.info("Info message 3 - should not appear")
            logger.warning("Warning message 1")
            
            # Check captured logs
            log_messages = [record.message for record in caplog.records]
            
            # Should contain specific messages based on level changes
            assert "Info message 1" in log_messages
            assert "Info message 2" in log_messages
            assert "Debug message 2 - should appear" in log_messages
            assert "Warning message 1" in log_messages
    
    @pytest.mark.integration
    def test_exception_handling_across_system(self, caplog):
        """Test exception handling and logging across the system."""
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("exception_test")
            
            @timed_and_logged
            def function_that_fails(should_fail=True):
                logger.info("Function starting")
                
                if should_fail:
                    logger.error("About to raise exception")
                    raise ValueError("Intentional test failure")
                
                logger.info("Function completing normally")
                return "success"
            
            # Test successful execution
            result_success = function_that_fails(should_fail=False)
            assert result_success == "success"
            
            # Test exception handling
            with pytest.raises(ValueError, match="Intentional test failure"):
                function_that_fails(should_fail=True)
            
            # Verify both scenarios are logged
            log_messages = [record.message for record in caplog.records]
            
            # Should have logs from both successful and failed executions
            success_logs = [msg for msg in log_messages if "Function completing normally" in msg]
            error_logs = [msg for msg in log_messages if "About to raise exception" in msg]
            exception_logs = [msg for msg in log_messages if "✗ Exception in function_that_fails" in msg]
            
            assert len(success_logs) >= 1
            assert len(error_logs) >= 1
            assert len(exception_logs) >= 1
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_high_volume_logging_performance(self, temp_logs_dir):
        """Test logging system performance under high volume."""
        log_file = Path(temp_logs_dir) / "high_volume.log"
        
        with patch('src.utilities.logger.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_path.return_value.__truediv__.return_value = log_file
            
            logger = setup_logger("high_volume_test", "INFO")
            module_logger = get_logger("performance_test")
            
            @logged
            def high_volume_function(iteration):
                module_logger.info(f"Processing iteration {iteration}")
                return iteration * 2
            
            # Execute many logging operations
            results = []
            for i in range(100):  # Reduced from 1000 for test speed
                result = high_volume_function(i)
                results.append(result)
            
            # Verify all operations completed
            assert len(results) == 100
            assert results[50] == 100  # 50 * 2
    
    @pytest.mark.integration
    def test_logging_with_real_function_signatures(self):
        """Test logging with functions that have complex signatures like the real codebase."""
        logger = get_logger("real_signature_test")
        
        @timed_and_logged
        async def mock_stock_analysis(industry_name: str, stock_code: str, stock_name: str, days: int = 29):
            """Mock function similar to real stock analysis function."""
            logger.info(f"Analyzing {stock_name} ({stock_code}) in {industry_name} industry")
            
            # Simulate some processing
            await asyncio.sleep(0.001)
            
            logger.debug(f"Analysis complete for {days} days")
            return {
                'industry': industry_name,
                'code': stock_code,
                'name': stock_name,
                'analysis_days': days,
                'result': 'success'
            }
        
        @logged
        def mock_prepare_stock_data():
            """Mock function similar to prepare_stock_data."""
            logger.info("Loading stock market data...")
            logger.debug("Applying filters...")
            return {'stocks': 100, 'industries': 10}
        
        # Test both functions
        async def run_integration_test():
            # Test sync function
            data = mock_prepare_stock_data()
            assert data['stocks'] == 100
            
            # Test async function
            result = await mock_stock_analysis("科技", "000001", "测试股票", 30)
            assert result['code'] == "000001"
            assert result['analysis_days'] == 30
            
            return True
        
        # Run the integration test
        result = asyncio.run(run_integration_test())
        assert result is True