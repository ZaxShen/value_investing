"""
Unit tests for the class-based architecture.

This module tests the StockFilter, IndustryFilter, and HoldingStockAnalyzer classes
with comprehensive mocking to avoid external API dependencies. It focuses on:
- Class initialization and dependency injection
- Method interfaces and parameter handling
- Progress tracking integration
- Error handling and timeout scenarios
- Data processing logic
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from rich.progress import Progress

from src.stock_filter import StockFilter
from src.industry_filter import IndustryFilter
from src.holding_stock_analyzer import HoldingStockAnalyzer


class TestStockFilter:
    """Test the StockFilter class functionality."""

    @pytest.fixture
    def sample_industry_mapping(self):
        """Sample industry-stock mapping data."""
        return pd.DataFrame({
            "行业": ["银行", "银行", "房地产开发", "房地产开发"],
            "代码": ["000001", "000002", "000003", "000004"]
        })

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock market data."""
        return pd.DataFrame({
            "代码": ["000001", "000002", "000003", "000004"],
            "名称": ["平安银行", "万科A", "招商银行", "保利地产"],
            "最新价": [10.50, 25.30, 45.20, 15.60],
            "总市值": [150.0e8, 200.0e8, 180.0e8, 120.0e8],  # In yuan (1.5 billion yuan)
            "流通市值": [140.0e8, 190.0e8, 170.0e8, 110.0e8],  # In yuan
            "市盈率-动态": [8.5, 12.3, 15.2, 9.8],
            "市净率": [1.2, 1.8, 2.1, 1.5],
            "涨跌幅": [2.1, -1.5, 0.8, 3.2],
            "60日涨跌幅": [15.2, -8.3, 12.1, 22.4],
            "年初至今涨跌幅": [25.6, -12.8, 18.9, 35.7]
        })

    @pytest.mark.unit
    def test_stock_filter_initialization(self, sample_industry_mapping, sample_stock_data):
        """Test StockFilter initializes correctly with required data."""
        filter_instance = StockFilter(sample_industry_mapping, sample_stock_data)
        
        assert filter_instance.industry_stock_mapping_df is not None
        assert filter_instance.stock_zh_a_spot_em_df is not None
        assert len(filter_instance.industry_stock_mapping_df) == 4
        assert len(filter_instance.stock_zh_a_spot_em_df) == 4

    @pytest.mark.unit
    async def test_stock_filter_run_analysis_interface(self, sample_industry_mapping, sample_stock_data):
        """Test StockFilter.run_analysis method interface."""
        filter_instance = StockFilter(sample_industry_mapping, sample_stock_data)
        
        # Mock the internal methods to avoid API calls
        with patch.object(filter_instance, 'process_all_industries_async', return_value=pd.DataFrame()):
            result = await filter_instance.run_analysis(days=29, progress=None)
            
            # Should complete without error
            assert result is None  # run_analysis doesn't return data, it saves files

    @pytest.mark.unit
    async def test_stock_filter_with_progress_tracking(self, sample_industry_mapping, sample_stock_data):
        """Test StockFilter integrates with progress tracking."""
        filter_instance = StockFilter(sample_industry_mapping, sample_stock_data)
        
        # Create mock progress instance
        mock_progress = Mock(spec=Progress)
        mock_progress.add_task.return_value = 1
        
        with patch.object(filter_instance, 'process_all_industries_async', return_value=pd.DataFrame()):
            await filter_instance.run_analysis(
                days=29, 
                progress=mock_progress,
                parent_task_id=1,
                batch_task_id=2
            )
            
            # Verify progress methods were called
            # Note: Exact calls depend on implementation details
            assert mock_progress.add_task.called or mock_progress.update.called

    @pytest.mark.unit
    async def test_stock_filter_timeout_handling(self, sample_industry_mapping, sample_stock_data):
        """Test StockFilter handles timeouts properly."""
        filter_instance = StockFilter(sample_industry_mapping, sample_stock_data)
        
        # Mock a timeout scenario in process_single_stock_async
        with patch.object(filter_instance, '_fetch_stock_fund_flow_sync', 
                         side_effect=asyncio.TimeoutError("Mocked timeout")):
            
            # Should handle timeout gracefully and return None
            result = await filter_instance.process_single_stock_async("000001", "测试股票", 29)
            assert result is None

    @pytest.mark.unit
    def test_stock_filter_sync_methods(self, sample_industry_mapping, sample_stock_data):
        """Test StockFilter synchronous helper methods."""
        filter_instance = StockFilter(sample_industry_mapping, sample_stock_data)
        
        # Test data filtering logic (if accessible)
        stock_market_df_filtered, industry_arr = filter_instance.prepare_stock_data()
        
        assert isinstance(stock_market_df_filtered, pd.DataFrame)
        assert isinstance(industry_arr, (pd.Series, np.ndarray))
        assert len(stock_market_df_filtered) <= len(sample_stock_data)  # Should filter some stocks


class TestIndustryFilter:
    """Test the IndustryFilter class functionality."""

    @pytest.mark.unit
    def test_industry_filter_initialization(self):
        """Test IndustryFilter initializes correctly."""
        filter_instance = IndustryFilter()
        
        # IndustryFilter doesn't require data in constructor
        assert filter_instance is not None

    @pytest.mark.unit
    async def test_industry_filter_run_analysis_interface(self):
        """Test IndustryFilter.run_analysis method interface."""
        filter_instance = IndustryFilter()
        
        # Mock the internal methods to avoid API calls
        with patch.object(filter_instance, 'process_all_industries_async', return_value=pd.DataFrame()):
            result = await filter_instance.run_analysis(days=30, progress=None)
            
            # Should complete without error
            assert result is None

    @pytest.mark.unit
    async def test_industry_filter_with_progress_tracking(self):
        """Test IndustryFilter integrates with progress tracking."""
        filter_instance = IndustryFilter()
        
        mock_progress = Mock(spec=Progress)
        mock_progress.add_task.return_value = 1
        
        with patch.object(filter_instance, 'process_all_industries_async', return_value=pd.DataFrame()):
            await filter_instance.run_analysis(
                days=30,
                progress=mock_progress,
                parent_task_id=1,
                batch_task_id=2
            )
            
            # Progress integration should work
            assert mock_progress.add_task.called or mock_progress.update.called

    @pytest.mark.unit
    async def test_industry_filter_timeout_handling(self):
        """Test IndustryFilter handles API timeouts properly."""
        filter_instance = IndustryFilter()
        
        # Mock timeout in process_single_industry_async
        with patch.object(filter_instance, '_fetch_industry_capital_flow_data_sync',
                         side_effect=asyncio.TimeoutError("Mocked timeout")):
            
            result = await filter_instance.process_single_industry_async("银行", "20250101", "20250131", 30)
            assert result is None  # Should handle timeout gracefully

    @pytest.mark.unit
    def test_industry_filter_column_generation(self):
        """Test IndustryFilter generates correct analysis columns."""
        filter_instance = IndustryFilter()
        
        columns_30_days = filter_instance._get_analysis_columns(30)
        columns_60_days = filter_instance._get_analysis_columns(60)
        
        assert isinstance(columns_30_days, list)
        assert isinstance(columns_60_days, list)
        assert len(columns_30_days) > 0
        assert len(columns_60_days) > 0
        
        # Different days should generate different columns
        assert columns_30_days != columns_60_days


class TestHoldingStockAnalyzer:
    """Test the HoldingStockAnalyzer class functionality."""

    @pytest.fixture
    def sample_holding_data(self):
        """Sample holding stocks data."""
        return {
            "银行股票": {
                "000001": "平安银行",
                "000002": "招商银行"
            },
            "地产股票": {
                "000003": "万科A",
                "000004": "保利地产"
            }
        }

    @pytest.fixture
    def sample_industry_mapping(self):
        """Sample industry mapping for analyzer."""
        return pd.DataFrame({
            "行业": ["银行", "银行", "房地产开发", "房地产开发"],
            "代码": ["000001", "000002", "000003", "000004"]
        })

    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock data for analyzer."""
        return pd.DataFrame({
            "代码": ["000001", "000002", "000003", "000004"],
            "名称": ["平安银行", "招商银行", "万科A", "保利地产"],
            "最新价": [10.50, 45.20, 25.30, 15.60],
            "涨跌幅": [2.1, 0.8, -1.5, 3.2]
        })

    @pytest.mark.unit
    def test_holding_stock_analyzer_initialization(self, sample_industry_mapping, sample_stock_data):
        """Test HoldingStockAnalyzer initializes correctly."""
        analyzer = HoldingStockAnalyzer(sample_industry_mapping, sample_stock_data)
        
        assert analyzer.industry_stock_mapping_df is not None
        assert analyzer.stock_zh_a_spot_em_df is not None
        assert len(analyzer.industry_stock_mapping_df) == 4
        assert len(analyzer.stock_zh_a_spot_em_df) == 4

    @pytest.mark.unit
    async def test_holding_stock_analyzer_run_analysis_with_data(
        self, sample_industry_mapping, sample_stock_data, sample_holding_data
    ):
        """Test HoldingStockAnalyzer.run_analysis with provided data."""
        analyzer = HoldingStockAnalyzer(sample_industry_mapping, sample_stock_data)
        
        # Mock the internal analysis method
        with patch.object(analyzer, 'analyze_single_stock', return_value={}):
            result = await analyzer.run_analysis(
                holding_stocks_data=sample_holding_data,
                days=30,
                progress=None
            )
            
            # Should complete without error
            assert result is None

    @pytest.mark.unit
    async def test_holding_stock_analyzer_run_analysis_from_files(
        self, sample_industry_mapping, sample_stock_data
    ):
        """Test HoldingStockAnalyzer.run_analysis_from_files method."""
        analyzer = HoldingStockAnalyzer(sample_industry_mapping, sample_stock_data)
        
        # Mock file loading and analysis
        with patch('os.path.exists', return_value=True):
            with patch('glob.glob', return_value=['mock_path/test.json']):
                with patch('builtins.open', create=True):
                    with patch('json.load', return_value={"000001": "平安银行"}):
                        with patch.object(analyzer, 'analyze_single_stock', return_value={}):
                            result = await analyzer.run_analysis_from_files(
                                dir_path="mock_path",
                                days=30,
                                progress=None
                            )
                        
                        assert result is None

    @pytest.mark.unit
    async def test_holding_stock_analyzer_with_progress_tracking(
        self, sample_industry_mapping, sample_stock_data, sample_holding_data
    ):
        """Test HoldingStockAnalyzer integrates with progress tracking."""
        analyzer = HoldingStockAnalyzer(sample_industry_mapping, sample_stock_data)
        
        mock_progress = Mock(spec=Progress)
        mock_progress.add_task.return_value = 1
        
        with patch.object(analyzer, 'analyze_single_stock', return_value={}):
            await analyzer.run_analysis(
                holding_stocks_data=sample_holding_data,
                progress=mock_progress,
                parent_task_id=1,
                batch_task_id=2
            )
            
            # Progress integration should work
            assert mock_progress.add_task.called or mock_progress.update.called

    @pytest.mark.unit
    async def test_holding_stock_analyzer_single_stock_analysis(
        self, sample_industry_mapping, sample_stock_data
    ):
        """Test HoldingStockAnalyzer.analyze_single_stock method."""
        analyzer = HoldingStockAnalyzer(sample_industry_mapping, sample_stock_data)
        
        # Mock API calls to avoid external dependencies
        with patch('src.holding_stock_analyzer.ak.stock_individual_fund_flow') as mock_fund_flow:
            mock_fund_flow.return_value = pd.DataFrame({
                "日期": ["2025-01-01", "2025-01-02"],
                "主力净流入-净额": [1000000, 2000000]
            })
            
            result = await analyzer.analyze_single_stock("银行", "000001", "平安银行", 30)
            
            assert isinstance(result, dict)
            # Should contain analysis results (exact structure depends on implementation)


class TestClassIntegration:
    """Integration tests for class-based architecture."""

    @pytest.mark.integration
    async def test_classes_work_together(self):
        """Test that all classes can be instantiated and used together."""
        # Create sample data
        industry_mapping = pd.DataFrame({
            "行业": ["银行", "银行"],
            "代码": ["000001", "000002"]
        })
        
        stock_data = pd.DataFrame({
            "代码": ["000001", "000002"],
            "名称": ["平安银行", "招商银行"],
            "最新价": [10.50, 45.20]
        })
        
        holding_data = {"银行股票": {"000001": "平安银行"}}
        
        # Mock all external API calls
        with patch('src.stock_filter.ak') as mock_ak:
            with patch('src.industry_filter.ak') as mock_ak2:
                with patch('src.holding_stock_analyzer.ak') as mock_ak3:
                    # Setup mock returns
                    mock_ak.stock_sector_fund_flow_hist.return_value = pd.DataFrame({"日期": ["2025-01-01"]})
                    mock_ak.stock_individual_fund_flow.return_value = pd.DataFrame({"日期": ["2025-01-01"]})
                    mock_ak2.stock_board_industry_name_em.return_value = pd.DataFrame({"板块名称": ["银行"]})
                    mock_ak3.stock_individual_fund_flow.return_value = pd.DataFrame({"日期": ["2025-01-01"]})
                    
                    # Test all classes can be instantiated
                    stock_filter = StockFilter(industry_mapping, stock_data)
                    industry_filter = IndustryFilter()
                    holding_analyzer = HoldingStockAnalyzer(industry_mapping, stock_data)
                    
                    assert stock_filter is not None
                    assert industry_filter is not None  
                    assert holding_analyzer is not None
                    
                    # Test they have expected methods
                    assert hasattr(stock_filter, 'run_analysis')
                    assert hasattr(industry_filter, 'run_analysis')
                    assert hasattr(holding_analyzer, 'run_analysis')
                    assert hasattr(holding_analyzer, 'run_analysis_from_files')

    @pytest.mark.unit
    def test_class_error_handling(self):
        """Test classes handle initialization errors properly."""
        # Test with invalid data types
        with pytest.raises((TypeError, AttributeError)):
            StockFilter("not_a_dataframe", "also_not_a_dataframe")
        
        # Test with None data
        with pytest.raises((TypeError, AttributeError)):
            StockFilter(None, None)
        
        # IndustryFilter should handle initialization gracefully
        industry_filter = IndustryFilter()
        assert industry_filter is not None