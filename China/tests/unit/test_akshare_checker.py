"""
Unit tests for the akshare connectivity checker module.
"""

import pytest
import time
import logging
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import requests.exceptions

from src.utilities.akshare_checker import (
    AkshareConnectivityChecker,
    ConnectivityStatus,
    check_akshare_connectivity,
    get_akshare_health_status,
    log_connectivity_status,
)


class TestAkshareConnectivityChecker:
    """Test AkshareConnectivityChecker class functionality."""

    @pytest.fixture
    def checker(self):
        """Create a connectivity checker instance for testing."""
        return AkshareConnectivityChecker(timeout=5, retry_attempts=2)

    @pytest.mark.unit
    def test_checker_initialization(self, checker):
        """Test that connectivity checker initializes with correct parameters."""
        assert checker.timeout == 5
        assert checker.retry_attempts == 2
        assert checker.logger is not None

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_check_basic_connectivity_success(self, mock_akshare, checker):
        """Test successful basic connectivity check."""
        # Mock successful akshare response
        mock_df = pd.DataFrame({'代码': ['000001'], '名称': ['平安银行']})
        mock_akshare.return_value = mock_df
        
        is_connected, error = checker.check_basic_connectivity()
        
        assert is_connected is True
        assert error is None
        mock_akshare.assert_called_once()

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_check_basic_connectivity_empty_data(self, mock_akshare, checker):
        """Test connectivity check with empty data response."""
        # Mock empty akshare response
        mock_akshare.return_value = pd.DataFrame()
        
        is_connected, error = checker.check_basic_connectivity()
        
        assert is_connected is False
        assert "Failed to establish akshare connectivity" in error

    @pytest.mark.unit 
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_check_basic_connectivity_request_exception(self, mock_akshare, checker):
        """Test connectivity check with network error."""
        # Mock network exception
        mock_akshare.side_effect = requests.exceptions.RequestException("Network error")
        
        is_connected, error = checker.check_basic_connectivity()
        
        assert is_connected is False
        assert "Failed to establish akshare connectivity" in error

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_check_basic_connectivity_general_exception(self, mock_akshare, checker):
        """Test connectivity check with general exception."""
        # Mock general exception
        mock_akshare.side_effect = Exception("API error")
        
        is_connected, error = checker.check_basic_connectivity()
        
        assert is_connected is False
        assert "Failed to establish akshare connectivity" in error

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    @patch('src.utilities.akshare_checker.time.sleep')
    def test_check_basic_connectivity_retry_logic(self, mock_sleep, mock_akshare, checker):
        """Test retry logic in connectivity check."""
        # First call fails, second succeeds
        mock_df = pd.DataFrame({'代码': ['000001'], '名称': ['平安银行']})
        mock_akshare.side_effect = [Exception("First failure"), mock_df]
        
        is_connected, error = checker.check_basic_connectivity()
        
        assert is_connected is True
        assert error is None
        assert mock_akshare.call_count == 2
        mock_sleep.assert_called_once_with(2)  # Exponential backoff

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, '_test_stock_spot')
    @patch.object(AkshareConnectivityChecker, '_test_industry_mapping')
    def test_check_api_endpoints_all_success(self, mock_industry, mock_stock, checker):
        """Test API endpoints check with all endpoints successful."""
        mock_stock.return_value = True
        mock_industry.return_value = True
        
        results = checker.check_api_endpoints()
        
        assert results["stock_spot"] is True
        assert results["industry_mapping"] is True

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, '_test_stock_spot')
    @patch.object(AkshareConnectivityChecker, '_test_industry_mapping')
    def test_check_api_endpoints_mixed_results(self, mock_industry, mock_stock, checker):
        """Test API endpoints check with mixed results."""
        mock_stock.return_value = True
        mock_industry.return_value = False
        
        results = checker.check_api_endpoints()
        
        assert results["stock_spot"] is True
        assert results["industry_mapping"] is False

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, '_test_stock_spot')
    @patch.object(AkshareConnectivityChecker, '_test_industry_mapping')
    def test_check_api_endpoints_exception_handling(self, mock_industry, mock_stock, checker):
        """Test API endpoints check with exception handling."""
        mock_stock.return_value = True
        mock_industry.side_effect = Exception("Endpoint error")
        
        results = checker.check_api_endpoints()
        
        assert results["stock_spot"] is True
        assert results["industry_mapping"] is False

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_test_stock_spot_success(self, mock_akshare, checker):
        """Test stock spot endpoint test."""
        # Mock successful response with sufficient data
        mock_df = pd.DataFrame({
            '代码': [f'00000{i}' for i in range(150)],
            '名称': [f'股票{i}' for i in range(150)]
        })
        mock_akshare.return_value = mock_df
        
        result = checker._test_stock_spot()
        
        assert result is True

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_test_stock_spot_insufficient_data(self, mock_akshare, checker):
        """Test stock spot endpoint test with insufficient data."""
        # Mock response with too few records
        mock_df = pd.DataFrame({
            '代码': ['000001'],
            '名称': ['平安银行']
        })
        mock_akshare.return_value = mock_df
        
        result = checker._test_stock_spot()
        
        assert result is False

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_board_industry_cons_em')
    @patch('src.utilities.akshare_checker.ak.stock_board_industry_name_em')
    def test_test_industry_mapping_success(self, mock_industry_names, mock_industry_cons, checker):
        """Test industry mapping endpoint test."""
        # Mock industry names response
        mock_industry_df = pd.DataFrame({
            '板块名称': [f'行业{i}' for i in range(15)]
        })
        mock_industry_names.return_value = mock_industry_df
        
        # Mock industry constituents response
        mock_cons_df = pd.DataFrame({
            '代码': ['000001', '000002', '000003'],
            '名称': ['平安银行', '万科A', '国农科技']
        })
        mock_industry_cons.return_value = mock_cons_df
        
        result = checker._test_industry_mapping()
        
        assert result is True
        # Verify both APIs were called
        mock_industry_names.assert_called_once()
        mock_industry_cons.assert_called_once_with(symbol='行业0')

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_measure_performance_success(self, mock_akshare, checker):
        """Test performance measurement."""
        # Mock successful response
        mock_df = pd.DataFrame({
            '代码': [f'00000{i}' for i in range(100)],
            '名称': [f'股票{i}' for i in range(100)]
        })
        mock_akshare.return_value = mock_df
        
        metrics = checker.measure_performance()
        
        assert "stock_data_response_time" in metrics
        assert "stock_data_count" in metrics
        assert metrics["stock_data_response_time"] >= 0  # Allow for very small response times
        assert metrics["stock_data_count"] == 100

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    def test_measure_performance_failure(self, mock_akshare, checker):
        """Test performance measurement with failure."""
        # Mock exception
        mock_akshare.side_effect = Exception("Performance test error")
        
        metrics = checker.measure_performance()
        
        assert metrics["stock_data_response_time"] == -1
        assert metrics["stock_data_count"] == 0

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    @patch.object(AkshareConnectivityChecker, 'check_api_endpoints')
    @patch.object(AkshareConnectivityChecker, 'measure_performance')
    def test_get_health_status_healthy(self, mock_perf, mock_endpoints, mock_basic, checker):
        """Test health status determination - healthy case."""
        mock_basic.return_value = (True, None)
        mock_endpoints.return_value = {"stock_spot": True, "industry_mapping": True}
        mock_perf.return_value = {"stock_data_response_time": 5.0, "stock_data_count": 100}
        
        status, details = checker.get_health_status()
        
        assert status == ConnectivityStatus.HEALTHY
        assert details["connectivity"] is True
        assert len(details["endpoints"]) == 2
        assert len(details["issues"]) == 0

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    def test_get_health_status_unavailable(self, mock_basic, checker):
        """Test health status determination - unavailable case."""
        mock_basic.return_value = (False, "Connection failed")
        
        status, details = checker.get_health_status()
        
        assert status == ConnectivityStatus.UNAVAILABLE
        assert details["connectivity"] is False
        assert len(details["issues"]) > 0

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    @patch.object(AkshareConnectivityChecker, 'check_api_endpoints')
    @patch.object(AkshareConnectivityChecker, 'measure_performance')
    def test_get_health_status_degraded_endpoints(self, mock_perf, mock_endpoints, mock_basic, checker):
        """Test health status determination - degraded due to endpoint failures."""
        mock_basic.return_value = (True, None)
        mock_endpoints.return_value = {"stock_spot": True, "industry_mapping": False}
        mock_perf.return_value = {"stock_data_response_time": 5.0, "stock_data_count": 100}
        
        status, details = checker.get_health_status()
        
        assert status == ConnectivityStatus.DEGRADED
        assert "industry_mapping" in str(details["issues"])

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    @patch.object(AkshareConnectivityChecker, 'check_api_endpoints')  
    @patch.object(AkshareConnectivityChecker, 'measure_performance')
    def test_get_health_status_degraded_performance(self, mock_perf, mock_endpoints, mock_basic, checker):
        """Test health status determination - degraded due to performance issues."""
        mock_basic.return_value = (True, None)
        mock_endpoints.return_value = {"stock_spot": True, "industry_mapping": True}
        mock_perf.return_value = {"stock_data_response_time": 15.0, "stock_data_count": 100}
        
        status, details = checker.get_health_status()
        
        assert status == ConnectivityStatus.DEGRADED
        assert "Slow response times" in str(details["issues"])


class TestModuleFunctions:
    """Test module-level functions."""

    @pytest.mark.unit
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    def test_check_akshare_connectivity_success(self, mock_basic):
        """Test quick connectivity check function - success."""
        mock_basic.return_value = (True, None)
        
        result = check_akshare_connectivity(timeout=10, retry_attempts=1)
        
        assert result is True

    @pytest.mark.unit  
    @patch.object(AkshareConnectivityChecker, 'check_basic_connectivity')
    def test_check_akshare_connectivity_failure(self, mock_basic):
        """Test quick connectivity check function - failure."""
        mock_basic.return_value = (False, "Connection failed")
        
        result = check_akshare_connectivity(timeout=10, retry_attempts=1)
        
        assert result is False

    @pytest.mark.unit
    @patch('src.utilities.akshare_checker._connectivity_checker')
    def test_get_akshare_health_status(self, mock_checker):
        """Test get health status function."""
        mock_status = ConnectivityStatus.HEALTHY
        mock_details = {"connectivity": True, "endpoints": {}, "performance": {}}
        mock_checker.get_health_status.return_value = (mock_status, mock_details)
        
        status, details = get_akshare_health_status()
        
        assert status == ConnectivityStatus.HEALTHY
        assert details["connectivity"] is True

    @pytest.mark.unit
    def test_log_connectivity_status_healthy(self, caplog):
        """Test logging connectivity status - healthy case."""
        caplog.set_level(logging.INFO, logger="stock_analysis.akshare_checker")
        status = ConnectivityStatus.HEALTHY
        details = {
            "connectivity": True,
            "endpoints": {"stock_spot": True, "industry_mapping": True},
            "performance": {"stock_data_response_time": 5.0, "stock_data_count": 100},
            "issues": []
        }
        
        log_connectivity_status(status, details)
        
        # Check that key messages are logged
        log_messages = caplog.text
        assert "🔗 AKSHARE CONNECTIVITY STATUS: HEALTHY" in log_messages
        assert "✅ Basic connectivity: OK" in log_messages
        assert "📡 Endpoint Status:" in log_messages
        assert "📊 Performance Metrics:" in log_messages

    @pytest.mark.unit
    def test_log_connectivity_status_unavailable(self, caplog):
        """Test logging connectivity status - unavailable case."""
        caplog.set_level(logging.INFO, logger="stock_analysis.akshare_checker")
        status = ConnectivityStatus.UNAVAILABLE
        details = {
            "connectivity": False,
            "endpoints": {},
            "performance": {},
            "issues": ["Network unreachable"]
        }
        
        log_connectivity_status(status, details)
        
        log_messages = caplog.text
        assert "🔗 AKSHARE CONNECTIVITY STATUS: UNAVAILABLE" in log_messages
        assert "❌ Basic connectivity: FAILED" in log_messages
        assert "⚠️  Issues Detected:" in log_messages
        assert "Network unreachable" in log_messages

    @pytest.mark.unit
    def test_log_connectivity_status_degraded(self, caplog):
        """Test logging connectivity status - degraded case."""
        caplog.set_level(logging.INFO, logger="stock_analysis.akshare_checker")
        status = ConnectivityStatus.DEGRADED
        details = {
            "connectivity": True,
            "endpoints": {"stock_spot": True, "industry_mapping": False},
            "performance": {"stock_data_response_time": 15.0, "stock_data_count": 50},
            "issues": ["Some endpoints unavailable", "Slow response times"]
        }
        
        log_connectivity_status(status, details)
        
        log_messages = caplog.text
        assert "🔗 AKSHARE CONNECTIVITY STATUS: DEGRADED" in log_messages
        assert "✅ Basic connectivity: OK" in log_messages
        assert "stock_spot: OK" in log_messages
        assert "industry_mapping: FAILED" in log_messages
        assert "Stock data response time: 15.0s" in log_messages


class TestConnectivityStatusEnum:
    """Test ConnectivityStatus enum."""

    @pytest.mark.unit
    def test_connectivity_status_values(self):
        """Test that ConnectivityStatus enum has expected values."""
        assert ConnectivityStatus.HEALTHY.value == "healthy"
        assert ConnectivityStatus.DEGRADED.value == "degraded"
        assert ConnectivityStatus.UNAVAILABLE.value == "unavailable"
        assert ConnectivityStatus.UNKNOWN.value == "unknown"


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    @pytest.mark.integration
    def test_checker_with_default_parameters(self):
        """Test checker with default parameters."""
        checker = AkshareConnectivityChecker()
        
        assert checker.timeout == 30
        assert checker.retry_attempts == 3

    @pytest.mark.integration
    @patch.object(AkshareConnectivityChecker, '_test_data_functions', return_value=True)
    @patch('src.utilities.akshare_checker.ak.stock_board_industry_cons_em')
    @patch('src.utilities.akshare_checker.ak.stock_zh_a_spot_em')
    @patch('src.utilities.akshare_checker.ak.stock_board_industry_name_em')
    @patch('src.utilities.akshare_checker.time.sleep')
    def test_full_health_check_simulation(self, mock_sleep, mock_industry, mock_stock, mock_industry_cons, mock_data_functions):
        """Test complete health check simulation with mixed results."""
        # Simulate network issues on first attempt, then success
        mock_stock_df = pd.DataFrame({
            '代码': [f'00000{i}' for i in range(200)],
            '名称': [f'股票{i}' for i in range(200)]
        })
        mock_industry_df = pd.DataFrame({
            '板块名称': [f'行业{i}' for i in range(20)]  # Fixed column name
        })
        mock_industry_cons_df = pd.DataFrame({
            '代码': ['000001', '000002', '000003'],
            '名称': ['平安银行', '万科A', '国农科技']
        })
        
        # First call fails, second succeeds
        mock_stock.side_effect = [
            requests.exceptions.Timeout("First timeout"),
            mock_stock_df,
            mock_stock_df,  # For performance test
            mock_stock_df   # For stock spot test
        ]
        mock_industry.return_value = mock_industry_df
        mock_industry_cons.return_value = mock_industry_cons_df
        
        checker = AkshareConnectivityChecker(timeout=5, retry_attempts=2)
        status, details = checker.get_health_status()
        
        # Should be healthy after retry
        assert status == ConnectivityStatus.HEALTHY
        assert details["connectivity"] is True
        assert details["endpoints"]["stock_spot"] is True
        assert details["endpoints"]["industry_mapping"] is True
        assert details["performance"]["stock_data_count"] == 200

    @pytest.mark.integration
    def test_global_connectivity_checker_instance(self):
        """Test that global connectivity checker instance works."""
        # This tests the module-level global instance
        status, details = get_akshare_health_status()
        
        # Should return a valid status (might be any status depending on network)
        assert isinstance(status, ConnectivityStatus)
        assert isinstance(details, dict)
        assert "timestamp" in details
        assert "connectivity" in details