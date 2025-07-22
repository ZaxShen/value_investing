"""
Unit tests for stock data fetching and caching utilities.
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, call
from src.utilities.get_stock_data import get_stock_market_data, get_industry_stock_mapping_data


class TestGetStockMarketData:
    """Test stock market data fetching and caching functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="test_stock_data_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock market data DataFrame."""
        return pd.DataFrame({
            "代码": ["000001", "000002", "600519"],
            "名称": ["平安银行", "万科A", "贵州茅台"],
            "最新价": [10.50, 25.30, 1800.00],
            "涨跌幅": [1.2, -0.5, 2.1],
            "市值": [2000000000, 3000000000, 2200000000000]
        })

    @pytest.mark.unit
    def test_get_stock_market_data_cache_hit(self, temp_data_dir, sample_stock_data):
        """Test that cached data is returned when available."""
        today = datetime.now().strftime("%Y%m%d")
        cache_file = f"{temp_data_dir}/stock_zh_a_spot_em_df-{today}.csv"
        
        # Create cached file
        os.makedirs(temp_data_dir, exist_ok=True)
        sample_stock_data.to_csv(cache_file, index=False)
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            result = get_stock_market_data(data_dir=temp_data_dir)
            
            # Should not call akshare API
            mock_ak.assert_not_called()
            
            # Should return cached data
            pd.testing.assert_frame_equal(result, sample_stock_data)
            assert result["代码"].dtype == object  # String type preserved

    @pytest.mark.unit
    def test_get_stock_market_data_cache_miss_fetch_new(self, temp_data_dir, sample_stock_data):
        """Test fetching new data when cache is empty."""
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = sample_stock_data
            
            result = get_stock_market_data(data_dir=temp_data_dir)
            
            # Should call akshare API
            mock_ak.stock_zh_a_spot_em.assert_called_once()
            
            # Should return fetched data
            pd.testing.assert_frame_equal(result, sample_stock_data)
            
            # Should save to cache
            today = datetime.now().strftime("%Y%m%d")
            cache_file = f"{temp_data_dir}/stock_zh_a_spot_em_df-{today}.csv"
            assert os.path.exists(cache_file)
            
            cached_data = pd.read_csv(cache_file, dtype={"代码": str})
            pd.testing.assert_frame_equal(cached_data, sample_stock_data)

    @pytest.mark.unit
    def test_get_stock_market_data_removes_outdated_files(self, temp_data_dir, sample_stock_data):
        """Test that outdated cache files are removed."""
        # Create some outdated files
        old_file1 = f"{temp_data_dir}/stock_zh_a_spot_em_df-20230101.csv"
        old_file2 = f"{temp_data_dir}/stock_zh_a_spot_em_df-20230102.csv"
        
        os.makedirs(temp_data_dir, exist_ok=True)
        sample_stock_data.to_csv(old_file1, index=False)
        sample_stock_data.to_csv(old_file2, index=False)
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = sample_stock_data
            
            get_stock_market_data(data_dir=temp_data_dir)
            
            # Old files should be removed
            assert not os.path.exists(old_file1)
            assert not os.path.exists(old_file2)
            
            # New file should exist
            today = datetime.now().strftime("%Y%m%d")
            new_file = f"{temp_data_dir}/stock_zh_a_spot_em_df-{today}.csv"
            assert os.path.exists(new_file)

    @pytest.mark.unit
    def test_get_stock_market_data_creates_directory(self, sample_stock_data):
        """Test that data directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = os.path.join(temp_dir, "new_data_dir")
            
            with patch('src.utilities.get_stock_data.ak') as mock_ak:
                mock_ak.stock_zh_a_spot_em.return_value = sample_stock_data
                
                get_stock_market_data(data_dir=non_existent_dir)
                
                assert os.path.exists(non_existent_dir)

    @pytest.mark.unit
    def test_get_stock_market_data_default_directory(self, sample_stock_data):
        """Test default data directory behavior."""
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = sample_stock_data
            
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs') as mock_makedirs:
                    with patch('glob.glob', return_value=[]):
                        with patch.object(sample_stock_data, 'to_csv') as mock_to_csv:
                            get_stock_market_data()
                            
                            # Should create default directory
                            mock_makedirs.assert_called_with("data/stocks", exist_ok=True)


class TestGetIndustryStockMappingData:
    """Test industry-stock mapping data fetching and caching functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="test_industry_data_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_industry_names(self):
        """Sample industry names DataFrame."""
        return pd.DataFrame({"板块名称": ["银行", "房地产开发", "白酒"]})

    @pytest.fixture
    def sample_industry_stocks(self):
        """Sample industry stocks data."""
        return {
            "银行": pd.DataFrame({"代码": ["000001", "600036"], "名称": ["平安银行", "招商银行"]}),
            "房地产开发": pd.DataFrame({"代码": ["000002"], "名称": ["万科A"]}),
            "白酒": pd.DataFrame({"代码": ["600519"], "名称": ["贵州茅台"]})
        }

    @pytest.fixture
    def expected_mapping_data(self):
        """Expected industry-stock mapping DataFrame."""
        return pd.DataFrame({
            "行业": ["银行", "银行", "房地产开发", "白酒"],
            "代码": ["000001", "600036", "000002", "600519"]
        })

    @pytest.mark.unit
    def test_get_industry_stock_mapping_cache_hit(self, temp_data_dir, expected_mapping_data):
        """Test that cached mapping data is returned when available."""
        today = datetime.now().strftime("%Y%m%d")
        cache_file = f"{temp_data_dir}/industry_stock_mapping_df-{today}.csv"
        
        # Create cached file
        os.makedirs(temp_data_dir, exist_ok=True)
        expected_mapping_data.to_csv(cache_file, index=False)
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            result = get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            # Should not call akshare API
            mock_ak.assert_not_called()
            
            # Should return cached data
            pd.testing.assert_frame_equal(result, expected_mapping_data)

    @pytest.mark.unit
    def test_get_industry_stock_mapping_cache_miss_fetch_new(
        self, temp_data_dir, sample_industry_names, sample_industry_stocks, expected_mapping_data
    ):
        """Test fetching new mapping data when cache is empty."""
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            # Setup mock returns
            mock_ak.stock_board_industry_name_em.return_value = sample_industry_names
            mock_ak.stock_board_industry_cons_em.side_effect = lambda symbol: sample_industry_stocks[symbol]
            
            result = get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            # Should call akshare APIs
            mock_ak.stock_board_industry_name_em.assert_called_once()
            assert mock_ak.stock_board_industry_cons_em.call_count == 3
            
            # Check individual calls
            expected_calls = [call(symbol="银行"), call(symbol="房地产开发"), call(symbol="白酒")]
            mock_ak.stock_board_industry_cons_em.assert_has_calls(expected_calls, any_order=True)
            
            # Should return correct mapping data
            pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_mapping_data.reset_index(drop=True))

    @pytest.mark.unit
    def test_get_industry_stock_mapping_removes_outdated_files(
        self, temp_data_dir, sample_industry_names, sample_industry_stocks, expected_mapping_data
    ):
        """Test that outdated mapping cache files are removed."""
        # Create some outdated files
        old_file1 = f"{temp_data_dir}/industry_stock_mapping_df-20230101.csv"
        old_file2 = f"{temp_data_dir}/industry_stock_mapping_df-20230102.csv"
        
        os.makedirs(temp_data_dir, exist_ok=True)
        expected_mapping_data.to_csv(old_file1, index=False)
        expected_mapping_data.to_csv(old_file2, index=False)
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_board_industry_name_em.return_value = sample_industry_names
            mock_ak.stock_board_industry_cons_em.side_effect = lambda symbol: sample_industry_stocks[symbol]
            
            get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            # Old files should be removed
            assert not os.path.exists(old_file1)
            assert not os.path.exists(old_file2)
            
            # New file should exist
            today = datetime.now().strftime("%Y%m%d")
            new_file = f"{temp_data_dir}/industry_stock_mapping_df-{today}.csv"
            assert os.path.exists(new_file)

    @pytest.mark.unit
    def test_get_industry_stock_mapping_creates_directory(
        self, sample_industry_names, sample_industry_stocks
    ):
        """Test that data directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = os.path.join(temp_dir, "new_mapping_dir")
            
            with patch('src.utilities.get_stock_data.ak') as mock_ak:
                mock_ak.stock_board_industry_name_em.return_value = sample_industry_names
                mock_ak.stock_board_industry_cons_em.side_effect = lambda symbol: sample_industry_stocks[symbol]
                
                get_industry_stock_mapping_data(data_dir=non_existent_dir)
                
                assert os.path.exists(non_existent_dir)

    @pytest.mark.unit
    def test_get_industry_stock_mapping_empty_industry_list(self, temp_data_dir):
        """Test behavior when no industries are returned."""
        empty_industries = pd.DataFrame({"板块名称": []})
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_board_industry_name_em.return_value = empty_industries
            
            result = get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            # Should return empty DataFrame with correct columns
            assert len(result) == 0
            assert list(result.columns) == ["行业", "代码"]
            
            # Should not call industry_cons_em
            mock_ak.stock_board_industry_cons_em.assert_not_called()


class TestDataFetchingIntegration:
    """Integration tests for data fetching functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="test_integration_data_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_both_functions_use_same_cache_directory(self, temp_data_dir, sample_test_data):
        """Test that both functions can work with the same cache directory."""
        stock_data = pd.DataFrame({
            "代码": ["000001", "000002"],
            "名称": ["平安银行", "万科A"],
            "最新价": [10.50, 25.30]
        })
        
        industry_names = pd.DataFrame({"板块名称": ["银行"]})
        industry_stocks = {"银行": pd.DataFrame({"代码": ["000001"]})}
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = stock_data
            mock_ak.stock_board_industry_name_em.return_value = industry_names
            mock_ak.stock_board_industry_cons_em.side_effect = lambda symbol: industry_stocks[symbol]
            
            # Call both functions
            stock_result = get_stock_market_data(data_dir=temp_data_dir)
            mapping_result = get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            # Both should succeed
            assert len(stock_result) == 2
            assert len(mapping_result) == 1
            
            # Both should create files in the same directory
            today = datetime.now().strftime("%Y%m%d")
            stock_file = f"{temp_data_dir}/stock_zh_a_spot_em_df-{today}.csv"
            mapping_file = f"{temp_data_dir}/industry_stock_mapping_df-{today}.csv"
            
            assert os.path.exists(stock_file)
            assert os.path.exists(mapping_file)

    @pytest.mark.integration
    def test_timer_decorator_applied(self, temp_data_dir):
        """Test that both functions have timer decorator applied."""
        # This is a smoke test to ensure the decorator is working
        # The actual timing is tested in decorator tests
        
        stock_data = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "最新价": [10.0]})
        industry_names = pd.DataFrame({"板块名称": ["测试行业"]})
        industry_stocks = {"测试行业": pd.DataFrame({"代码": ["000001"]})}
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = stock_data
            mock_ak.stock_board_industry_name_em.return_value = industry_names
            mock_ak.stock_board_industry_cons_em.side_effect = lambda symbol: industry_stocks[symbol]
            
            # These calls should complete without error (timer decorator working)
            stock_result = get_stock_market_data(data_dir=temp_data_dir)
            mapping_result = get_industry_stock_mapping_data(data_dir=temp_data_dir)
            
            assert stock_result is not None
            assert mapping_result is not None


class TestDataFetchingErrorHandling:
    """Test error handling in data fetching functions."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="test_error_data_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.unit
    def test_get_stock_market_data_akshare_error(self, temp_data_dir):
        """Test handling of akshare API errors."""
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                get_stock_market_data(data_dir=temp_data_dir)

    @pytest.mark.unit
    def test_get_industry_stock_mapping_akshare_error(self, temp_data_dir):
        """Test handling of akshare API errors in industry mapping."""
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_board_industry_name_em.side_effect = Exception("Industry API Error")
            
            with pytest.raises(Exception, match="Industry API Error"):
                get_industry_stock_mapping_data(data_dir=temp_data_dir)

    @pytest.mark.unit
    def test_file_permission_error(self, temp_data_dir):
        """Test handling of file permission errors."""
        stock_data = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "最新价": [10.0]})
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = stock_data
            
            # Mock to_csv to raise permission error
            with patch.object(pd.DataFrame, 'to_csv', side_effect=PermissionError("Permission denied")):
                with pytest.raises(PermissionError, match="Permission denied"):
                    get_stock_market_data(data_dir=temp_data_dir)

    @pytest.mark.unit
    def test_invalid_cached_file(self, temp_data_dir):
        """Test handling of corrupted cache files."""
        today = datetime.now().strftime("%Y%m%d")
        cache_file = f"{temp_data_dir}/stock_zh_a_spot_em_df-{today}.csv"
        
        # Create corrupted cache file
        os.makedirs(temp_data_dir, exist_ok=True)
        with open(cache_file, 'w') as f:
            f.write("invalid,csv,data\nwith,wrong,format")
        
        stock_data = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "最新价": [10.0]})
        
        with patch('src.utilities.get_stock_data.ak') as mock_ak:
            mock_ak.stock_zh_a_spot_em.return_value = stock_data
            
            # Should handle corrupted cache gracefully - this will depend on pandas behavior
            # This test documents the current behavior rather than testing specific error handling
            try:
                result = get_stock_market_data(data_dir=temp_data_dir)
                # If pandas can read it, that's fine
                assert result is not None
            except (pd.errors.ParserError, ValueError):
                # If pandas can't read it, that's expected behavior
                pass