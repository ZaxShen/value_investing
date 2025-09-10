"""
Integration tests for akshare API connectivity.

These tests make REAL API calls to verify that:
- akshare APIs are not down
- APIs return expected data structure
- Our functions work with real data

Run these separately since they're slow and can fail due to network issues.
"""

import akshare as ak
import pandas as pd
import pytest

from src.utilities.market_data_fetcher import (
    get_industry_stock_mapping_data,
    get_stock_market_data,
)


@pytest.mark.integration
@pytest.mark.slow
def test_akshare_stock_market_api_is_working():
    """Test that akshare stock market API is responding."""
    try:
        # Call real akshare API
        result = ak.stock_zh_a_spot_em()

        # Basic checks for real API response
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) > 1000  # Should have many stocks

        # Check expected columns exist
        expected_columns = ["代码", "名称", "最新价", "涨跌幅"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        print(f"✅ Stock market API working - got {len(result)} stocks")

    except Exception as e:
        pytest.fail(f"akshare stock market API is down or changed: {e}")


@pytest.mark.integration
@pytest.mark.slow
def test_akshare_industry_api_is_working():
    """Test that akshare industry APIs are responding."""
    try:
        # Test industry names API
        industries = ak.stock_board_industry_name_em()
        assert isinstance(industries, pd.DataFrame)
        assert not industries.empty
        assert "板块名称" in industries.columns

        # Test at least one industry stocks API
        first_industry = industries["板块名称"].iloc[0]
        industry_stocks = ak.stock_board_industry_cons_em(symbol=first_industry)

        assert isinstance(industry_stocks, pd.DataFrame)
        assert not industry_stocks.empty
        assert "代码" in industry_stocks.columns
        assert "名称" in industry_stocks.columns

        print(f"✅ Industry API working - got {len(industries)} industries")
        print(f"✅ Industry '{first_industry}' has {len(industry_stocks)} stocks")

    except Exception as e:
        pytest.fail(f"akshare industry API is down or changed: {e}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_our_functions_work_with_real_api():
    """Test our functions work with real akshare data (no mocking)."""
    import shutil
    import tempfile

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="integration_test_")

    try:
        # Test our stock market function with real API
        stock_result = await get_stock_market_data(temp_dir)
        assert isinstance(stock_result, pd.DataFrame)
        assert not stock_result.empty
        assert len(stock_result) > 1000

        print(
            f"✅ get_stock_market_data() works with real API - got {len(stock_result)} stocks"
        )

        # Test our industry mapping function with real API (but limit it)
        # Note: This will be slow since it calls many APIs
        industry_result = await get_industry_stock_mapping_data(temp_dir)
        assert isinstance(industry_result, pd.DataFrame)
        assert not industry_result.empty
        assert "行业" in industry_result.columns
        assert "代码" in industry_result.columns

        print(
            f"✅ get_industry_stock_mapping_data() works with real API - got {len(industry_result)} mappings"
        )

    except Exception as e:
        pytest.fail(f"Our functions failed with real API: {e}")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_api_data_quality():
    """Test that real API data has expected quality/format."""
    try:
        # Get sample of real data
        stock_data = ak.stock_zh_a_spot_em()

        # Check data quality
        sample = stock_data.head(10)

        # Stock codes should be 6 digits
        for code in sample["代码"]:
            assert isinstance(code, str), f"Stock code should be string: {code}"
            assert len(code) == 6, f"Stock code should be 6 digits: {code}"
            assert code.isdigit(), f"Stock code should be numeric: {code}"

        # Stock names should not be empty
        for name in sample["名称"]:
            assert isinstance(name, str), f"Stock name should be string: {name}"
            assert len(name.strip()) > 0, f"Stock name should not be empty: {name}"

        # Prices should be positive numbers
        for price in sample["最新价"]:
            if pd.notna(price):  # Skip NaN values
                assert isinstance(price, (int, float)), (
                    f"Price should be number: {price}"
                )
                assert price > 0, f"Price should be positive: {price}"

        print("✅ API data quality checks passed")

    except Exception as e:
        pytest.fail(f"API data quality issues: {e}")
