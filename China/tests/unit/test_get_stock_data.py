"""
Simple unit tests for the get_stock_data module.

Basic tests to verify that the stock data functions work correctly:
- Functions complete without errors
- Return non-empty DataFrames
- Cache files are created properly
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd

from src.utilities.get_stock_data import (
    get_stock_market_data,
    get_industry_stock_mapping_data,
)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_stock_data_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.mark.asyncio
async def test_get_stock_market_data_basic(temp_data_dir, mocker):
    """Test that get_stock_market_data returns non-empty DataFrame without errors."""
    # Mock the akshare API to return sample data
    mock_df = pd.DataFrame({
        "代码": ["000001", "000002"],
        "名称": ["平安银行", "万科A"],
        "最新价": [10.50, 25.30]
    })
    
    mocker.patch("src.utilities.get_stock_data.ak.stock_zh_a_spot_em", return_value=mock_df)
    result = await get_stock_market_data(temp_data_dir)
    
    # Basic checks
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) > 0
    assert "代码" in result.columns
    assert "名称" in result.columns


@pytest.mark.asyncio 
async def test_get_industry_mapping_data_basic(temp_data_dir, mocker):
    """Test that get_industry_stock_mapping_data returns non-empty DataFrame without errors."""
    # Mock industry names
    mock_industry_names = pd.DataFrame({"板块名称": ["银行", "房地产开发"]})
    
    # Mock industry stocks
    mock_industry_stocks = pd.DataFrame({
        "代码": ["000001", "000002"],
        "名称": ["平安银行", "万科A"]
    })
    
    mocker.patch("src.utilities.get_stock_data.ak.stock_board_industry_name_em", return_value=mock_industry_names)
    mocker.patch("src.utilities.get_stock_data.ak.stock_board_industry_cons_em", return_value=mock_industry_stocks)
    
    result = await get_industry_stock_mapping_data(temp_data_dir)
    
    # Basic checks
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) > 0
    assert "行业" in result.columns
    assert "代码" in result.columns


@pytest.mark.asyncio
async def test_cache_files_created(temp_data_dir, mocker):
    """Test that cache files are created properly."""
    mock_df = pd.DataFrame({"代码": ["000001"], "名称": ["测试"]})
    
    mocker.patch("src.utilities.get_stock_data.ak.stock_zh_a_spot_em", return_value=mock_df)
    await get_stock_market_data(temp_data_dir)
    
    # Check that cache file was created
    cache_files = [f for f in os.listdir(temp_data_dir) if f.startswith("stock_zh_a_spot_em_df-")]
    assert len(cache_files) == 1


@pytest.mark.asyncio
async def test_functions_handle_errors_gracefully(temp_data_dir, mocker):
    """Test that functions handle API errors gracefully."""
    # Test that exceptions from akshare are properly propagated
    mocker.patch("src.utilities.get_stock_data.ak.stock_zh_a_spot_em", side_effect=Exception("API Error"))
    
    with pytest.raises(Exception, match="API Error"):
        await get_stock_market_data(temp_data_dir)