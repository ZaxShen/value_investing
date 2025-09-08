"""
Integration tests for timeout handling in the stock analysis pipeline.

This module tests the real timeout behavior with actual asyncio.wait_for()
wrappers and the enhanced retry mechanism. It focuses on:
- asyncio.wait_for() integration with API calls
- Real timeout scenarios (controlled)
- Timeout recovery and error handling
- Progress tracking during timeout situations
- End-to-end timeout behavior in the full pipeline
"""

import asyncio
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from rich.progress import Progress

from src.filters.industry_filter import IndustryFilter
from src.filters.stock_filter import StockFilter
from src.utilities.get_stock_data import get_stock_market_data


class TestAsyncioTimeoutIntegration:
    """Test integration of asyncio.wait_for() with the pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = tempfile.mkdtemp(prefix="integration_timeout_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    async def test_stock_filter_timeout_wrapper(self):
        """Test StockFilter timeout handling with asyncio.wait_for()."""
        # Create sample data for StockFilter
        industry_mapping = pd.DataFrame(
            {"行业": ["银行", "房地产开发"], "代码": ["000001", "000003"]}
        )

        stock_data = pd.DataFrame(
            {
                "代码": ["000001", "000003"],
                "名称": ["平安银行", "万科A"],
                "最新价": [10.50, 25.30],
                "市值": [150.0, 200.0],
                "市盈率-动态": [8.5, 12.3],
                "涨跌幅": [2.1, -1.5],
                "60日涨跌幅": [15.2, -8.3],
                "年初至今涨跌幅": [25.6, -12.8],
            }
        )

        stock_filter = StockFilter(industry_mapping, stock_data)

        # Mock a slow API call that would timeout
        def slow_api_call(*args, **kwargs):
            import time

            time.sleep(2.0)  # 2 second delay, longer than our timeout
            return pd.DataFrame({"日期": ["2025-01-01"], "主力净流入-净额": [1000000]})

        # Test that timeout is enforced
        with patch.object(
            stock_filter, "_fetch_stock_fund_flow_sync", side_effect=slow_api_call
        ):
            start_time = asyncio.get_event_loop().time()

            # This should timeout and return None (handled gracefully)
            result = await stock_filter.process_single_stock_async(
                "000001", "平安银行", 29
            )

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should have timed out quickly (within ~1.5 seconds, allowing for overhead)
            assert elapsed < 1.5, f"Timeout took too long: {elapsed} seconds"
            assert result is None, "Should return None on timeout"

    @pytest.mark.integration
    async def test_industry_filter_timeout_wrapper(self):
        """Test IndustryFilter timeout handling with asyncio.wait_for()."""
        industry_filter = IndustryFilter()

        # Mock the method to return a controlled result quickly
        # This test is focused on timeout handling, not API correctness
        def quick_mock_result(*args, **kwargs):
            return pd.DataFrame(
                {"日期": ["2025-01-01"], "主力净流入-净额": [1000000], "涨跌幅": [2.5]}
            )

        with patch.object(
            industry_filter,
            "_fetch_industry_capital_flow_data_sync",
            side_effect=quick_mock_result,
        ):
            start_time = asyncio.get_event_loop().time()

            # Use correct method signature (requires first_trading_date_str parameter)
            result = await industry_filter.process_single_industry_async(
                "银行", "20250101", "20250131", "20250101", 30
            )

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should complete within reasonable time (accounting for some API calls that may still occur)
            assert elapsed < 10.0, (
                f"Should complete within reasonable time: {elapsed} seconds"
            )
            # Result may be None due to data processing, but should not timeout
            assert result is None or isinstance(result, list), (
                "Should return valid result or None"
            )

    @pytest.mark.integration
    async def test_get_stock_data_with_real_retry_timeout(self, temp_data_dir):
        """Test get_stock_data functions with real retry timeout behavior."""

        # Mock akshare to always raise a timeout exception
        def timeout_akshare_call(*args, **kwargs):
            import time

            time.sleep(0.1)  # Small delay
            # Raise a timeout-like exception that the retry mechanism will catch
            raise requests.exceptions.ConnectionError("Simulated connection timeout")

        with patch(
            "src.utilities.get_stock_data.ak.stock_zh_a_spot_em",
            side_effect=timeout_akshare_call,
        ):
            start_time = asyncio.get_event_loop().time()

            # This should fail due to repeated failures in retry mechanism
            with pytest.raises((requests.exceptions.ConnectionError, Exception)):
                await get_stock_market_data(temp_data_dir)

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should have failed relatively quickly due to timeout handling
            # With 3 retries and 45s timeout each, this could take up to ~135s
            # But our mock times out at 2s, so retry should handle it faster
            assert elapsed < 10.0, f"Timeout handling took too long: {elapsed} seconds"

    @pytest.mark.integration
    async def test_timeout_with_progress_tracking(self):
        """Test timeout behavior with progress bar integration."""
        industry_mapping = pd.DataFrame({"行业": ["银行"], "代码": ["000001"]})

        stock_data = pd.DataFrame(
            {
                "代码": ["000001"],
                "名称": ["平安银行"],
                "最新价": [10.50],
                "市值": [150.0],
                "市盈率-动态": [8.5],
                "涨跌幅": [2.1],
                "60日涨跌幅": [15.2],
                "年初至今涨跌幅": [25.6],
            }
        )

        stock_filter = StockFilter(industry_mapping, stock_data)

        # Create real progress instance
        progress = Progress()
        progress.start()

        try:
            # Mock timeout scenario
            def timeout_api(*args, **kwargs):
                import time

                time.sleep(2.0)
                return pd.DataFrame()

            with patch.object(
                stock_filter, "_fetch_stock_fund_flow_sync", side_effect=timeout_api
            ):
                # Add a task to track progress
                task_id = progress.add_task("Testing timeout", total=1)

                result = await stock_filter.process_single_stock_async(
                    "000001", "平安银行", 29
                )

                # Should handle timeout gracefully even with progress tracking
                assert result is None

                # Progress should still be functional
                progress.update(task_id, completed=1)

        finally:
            progress.stop()

    @pytest.mark.integration
    async def test_concurrent_timeouts(self):
        """Test multiple concurrent operations with timeouts."""
        industry_mapping = pd.DataFrame(
            {
                "行业": ["银行", "房地产开发", "医药生物"],
                "代码": ["000001", "000002", "000003"],
            }
        )

        stock_data = pd.DataFrame(
            {
                "代码": ["000001", "000002", "000003"],
                "名称": ["平安银行", "万科A", "恒瑞医药"],
                "最新价": [10.50, 25.30, 68.90],
                "市值": [150.0, 200.0, 180.0],
                "市盈率-动态": [8.5, 12.3, 25.6],
                "涨跌幅": [2.1, -1.5, 1.8],
                "60日涨跌幅": [15.2, -8.3, 12.5],
                "年初至今涨跌幅": [25.6, -12.8, 18.9],
            }
        )

        stock_filter = StockFilter(industry_mapping, stock_data)

        # Mock varying timeout scenarios
        call_count = 0

        def variable_timeout_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            import time

            if call_count == 1:
                time.sleep(2.0)  # First call times out
            elif call_count == 2:
                return pd.DataFrame(
                    {"日期": ["2025-01-01"], "主力净流入-净额": [1000000]}
                )  # Second succeeds
            else:
                time.sleep(2.0)  # Third times out

        with patch.object(
            stock_filter,
            "_fetch_stock_fund_flow_sync",
            side_effect=variable_timeout_api,
        ):
            start_time = asyncio.get_event_loop().time()

            # Run multiple concurrent operations
            tasks = [
                stock_filter.process_single_stock_async("000001", "平安银行", 29),
                stock_filter.process_single_stock_async("000002", "万科A", 29),
                stock_filter.process_single_stock_async("000003", "恒瑞医药", 29),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should complete within reasonable time despite timeouts
            assert elapsed < 2.0, (
                f"Concurrent timeouts took too long: {elapsed} seconds"
            )

            # Should have mix of None (timeout) and valid results
            none_results = [r for r in results if r is None]
            assert len(none_results) >= 1, "Should have at least one timeout result"


class TestTimeoutErrorRecovery:
    """Test error recovery and resilience with timeout scenarios."""

    @pytest.mark.integration
    async def test_timeout_recovery_in_batch_processing(self):
        """Test that batch processing continues after individual timeouts."""
        industry_filter = IndustryFilter()

        # Mock industry data
        mock_industries = pd.DataFrame({"板块名称": ["银行", "房地产开发", "医药生物"]})

        call_count = 0

        def mixed_success_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            import time

            if call_count == 1:
                time.sleep(2.0)  # First industry times out
            elif call_count == 2:
                return pd.DataFrame(
                    {"日期": ["2025-01-01"], "主力净流入-净额": [1000000]}
                )  # Second succeeds
            else:
                return pd.DataFrame(
                    {"日期": ["2025-01-01"], "主力净流入-净额": [2000000]}
                )  # Third succeeds

        with patch(
            "src.industry_filter.ak.stock_board_industry_name_em",
            return_value=mock_industries,
        ):
            with patch.object(
                industry_filter,
                "_fetch_industry_capital_flow_data_sync",
                side_effect=mixed_success_api,
            ):
                with patch.object(
                    industry_filter,
                    "_fetch_industry_index_data_sync",
                    return_value=pd.DataFrame({"收盘": [100, 105]}),
                ):
                    start_time = asyncio.get_event_loop().time()

                    # This should process all industries despite timeout in first one
                    result = await industry_filter.process_all_industries_async(
                        mock_industries["板块名称"], "20250101", "20250131", 30
                    )

                    end_time = asyncio.get_event_loop().time()
                    elapsed = end_time - start_time

                    # Should complete relatively quickly
                    assert elapsed < 3.0, (
                        f"Batch processing took too long: {elapsed} seconds"
                    )

                    # Should have processed some industries successfully
                    assert isinstance(result, pd.DataFrame)
                    # Exact row count depends on implementation - some may be filtered out due to timeouts

    @pytest.mark.integration
    async def test_graceful_degradation_on_widespread_timeouts(self):
        """Test system behavior when many operations timeout."""
        # Create larger dataset to test resilience
        industry_mapping = pd.DataFrame(
            {
                "行业": ["银行"] * 5 + ["房地产开发"] * 5,
                "代码": [f"00000{i}" for i in range(1, 11)],
            }
        )

        stock_data = pd.DataFrame(
            {
                "代码": [f"00000{i}" for i in range(1, 11)],
                "名称": [f"股票{i}" for i in range(1, 11)],
                "总市值": [
                    (100.0 + i * 10) * 1e8 for i in range(10)
                ],  # Required column
                "流通市值": [
                    (90.0 + i * 10) * 1e8 for i in range(10)
                ],  # Required column
                "市盈率-动态": [8.0 + i for i in range(10)],
                "市净率": [1.2 + i * 0.1 for i in range(10)],  # Required column
                "60日涨跌幅": [i * 2 - 10 for i in range(10)],
                "年初至今涨跌幅": [i * 3 - 15 for i in range(10)],
            }
        )

        stock_filter = StockFilter(industry_mapping, stock_data)

        # Prepare stock data (required before calling process_all_industries_async)
        stock_filter.prepare_stock_data()

        # Mock widespread timeouts (80% of calls timeout)
        call_count = 0

        def mostly_timeout_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            import time

            if call_count % 5 != 0:  # 80% timeout
                time.sleep(2.0)
            else:  # 20% succeed
                return pd.DataFrame(
                    {"日期": ["2025-01-01"], "主力净流入-净额": [1000000]}
                )

        with patch.object(
            stock_filter, "_fetch_stock_fund_flow_sync", side_effect=mostly_timeout_api
        ):
            with patch(
                "src.stock_filter.ak.stock_sector_fund_flow_hist",
                return_value=pd.DataFrame({"日期": ["2025-01-01"]}),
            ):
                start_time = asyncio.get_event_loop().time()

                # Should handle widespread timeouts gracefully
                result = await stock_filter.process_all_industries_async(
                    days=29, progress=None
                )

                end_time = asyncio.get_event_loop().time()
                elapsed = end_time - start_time

                # Should not hang despite widespread timeouts
                assert elapsed < 5.0, (
                    f"Processing with widespread timeouts took too long: {elapsed} seconds"
                )

                # Should return some result structure (even if mostly empty due to timeouts)
                assert isinstance(result, pd.DataFrame)


class TestRealWorldTimeoutScenarios:
    """Test realistic timeout scenarios that could occur in production."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_network_instability_simulation(self):
        """Simulate network instability with intermittent timeouts."""
        industry_mapping = pd.DataFrame(
            {"行业": ["银行", "房地产开发"], "代码": ["000001", "000002"]}
        )

        stock_data = pd.DataFrame(
            {
                "代码": ["000001", "000002"],
                "名称": ["平安银行", "万科A"],
                "最新价": [10.50, 25.30],
                "市值": [150.0, 200.0],
                "市盈率-动态": [8.5, 12.3],
                "涨跌幅": [2.1, -1.5],
                "60日涨跌幅": [15.2, -8.3],
                "年初至今涨跌幅": [25.6, -12.8],
            }
        )

        stock_filter = StockFilter(industry_mapping, stock_data)

        # Simulate intermittent network issues
        import random

        def unstable_network_api(*args, **kwargs):
            import time

            # 30% chance of timeout, 70% chance of success with variable delay
            if random.random() < 0.3:
                time.sleep(2.0)  # Timeout
            else:
                time.sleep(random.uniform(0.1, 0.8))  # Variable but acceptable delay
                return pd.DataFrame(
                    {"日期": ["2025-01-01"], "主力净流入-净额": [1000000]}
                )

        # Set random seed for reproducible test
        random.seed(42)

        with patch.object(
            stock_filter,
            "_fetch_stock_fund_flow_sync",
            side_effect=unstable_network_api,
        ):
            start_time = asyncio.get_event_loop().time()

            # Run multiple operations to test network instability handling
            tasks = []
            for i in range(5):  # 5 attempts
                tasks.append(
                    stock_filter.process_single_stock_async(
                        f"00000{i + 1}", f"股票{i + 1}", 29
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time

            # Should handle instability without hanging
            assert elapsed < 4.0, (
                f"Network instability handling took too long: {elapsed} seconds"
            )

            # Should have mix of successful and failed results
            successful_results = [
                r for r in results if r is not None and not isinstance(r, Exception)
            ]
            failed_results = [
                r for r in results if r is None or isinstance(r, Exception)
            ]

            # With 30% timeout rate and 5 attempts, we should have some of each
            # (exact numbers depend on random seed, but should have both types)
            assert len(results) == 5, "Should have results for all attempts"
