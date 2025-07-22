"""
Akshare connectivity and health checker module.

This module provides functionality to verify akshare API connectivity
and health status before running the main application logic.
"""

import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import requests

import akshare as ak
from src.utilities.logger import get_logger
from src.utilities.tools import timer

logger = get_logger("akshare_checker")


class ConnectivityStatus(Enum):
    """Enumeration for akshare connectivity status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AkshareConnectivityChecker:
    """
    Comprehensive akshare API connectivity checker.

    This class provides methods to test akshare connectivity, measure response times,
    and determine the overall health status of akshare services.
    """

    def __init__(self, timeout: int = 30, retry_attempts: int = 3):
        """
        Initialize the connectivity checker.

        Args:
            timeout: Request timeout in seconds (default: 30)
            retry_attempts: Number of retry attempts (default: 3)
        """
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.logger = logger

    @timer
    def check_basic_connectivity(self) -> Tuple[bool, Optional[str]]:
        """
        Test basic akshare connectivity by making a simple API call.

        Returns:
            Tuple of (is_connected, error_message)
        """
        self.logger.info("🔍 Testing basic akshare connectivity...")

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.debug(
                    f"Connectivity check attempt {attempt}/{self.retry_attempts}"
                )

                # Use a lightweight API call for connectivity testing
                start_time = time.time()

                # Try to get a small sample of stock data
                test_data = ak.stock_zh_a_spot_em()

                if test_data is not None and not test_data.empty:
                    response_time = round(time.time() - start_time, 2)
                    self.logger.info(
                        f"✅ Akshare connectivity successful (response time: {response_time}s)"
                    )
                    return True, None
                else:
                    self.logger.warning(
                        f"⚠️  Akshare returned empty data on attempt {attempt}"
                    )

            except requests.exceptions.RequestException as e:
                error_msg = f"Network error on attempt {attempt}: {str(e)}"
                self.logger.warning(error_msg)

            except Exception as e:
                error_msg = f"Akshare API error on attempt {attempt}: {str(e)}"
                self.logger.warning(error_msg)

            # Wait before retry (except for last attempt)
            if attempt < self.retry_attempts:
                wait_time = 2**attempt  # Exponential backoff
                self.logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        final_error = f"Failed to establish akshare connectivity after {self.retry_attempts} attempts"
        self.logger.error(f"❌ {final_error}")
        return False, final_error

    @timer
    def check_api_endpoints(self) -> Dict[str, bool]:
        """
        Test multiple akshare API endpoints to assess overall health.

        Returns:
            Dictionary mapping endpoint names to their status
        """
        self.logger.info("🔍 Testing multiple akshare API endpoints...")

        endpoints = {
            "stock_spot": self._test_stock_spot,
            "industry_mapping": self._test_industry_mapping,
            "data_functions": self._test_data_functions,
        }

        results = {}
        for endpoint_name, test_func in endpoints.items():
            try:
                self.logger.debug(f"Testing {endpoint_name} endpoint...")
                results[endpoint_name] = test_func()
                status_icon = "✅" if results[endpoint_name] else "❌"
                self.logger.info(
                    f"{status_icon} {endpoint_name}: {'OK' if results[endpoint_name] else 'FAILED'}"
                )

            except Exception as e:
                self.logger.error(f"❌ {endpoint_name} endpoint test failed: {str(e)}")
                results[endpoint_name] = False

        return results

    def _test_stock_spot(self) -> bool:
        """Test stock spot data endpoint."""
        try:
            data = ak.stock_zh_a_spot_em()
            return data is not None and not data.empty and len(data) > 100
        except Exception as e:
            self.logger.debug(f"Stock spot test failed: {e}")
            return False

    def _test_industry_mapping(self) -> bool:
        """Test industry mapping data endpoint."""
        try:
            # Test industry classification data (same as get_industry_stock_mapping_data)
            industry_names = ak.stock_board_industry_name_em()
            if industry_names is None or industry_names.empty or len(industry_names) < 10:
                return False
            
            # Test the actual API used in get_industry_stock_mapping_data
            # Try to get industry constituents for the first industry
            first_industry = industry_names["板块名称"].iloc[0]
            industry_stocks = ak.stock_board_industry_cons_em(symbol=first_industry)
            
            return (industry_stocks is not None and 
                    not industry_stocks.empty and 
                    "代码" in industry_stocks.columns and
                    len(industry_stocks) > 0)
                    
        except Exception as e:
            self.logger.debug(f"Industry mapping test failed: {e}")
            return False

    def _test_data_functions(self) -> bool:
        """Test the exact same APIs used by get_stock_data functions."""
        try:
            # Import the actual data functions to test them directly
            from src.utilities.get_stock_data import get_stock_market_data, get_industry_stock_mapping_data
            
            # Test by calling the actual functions with a test directory
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Test stock market data function
                try:
                    stock_data = get_stock_market_data(data_dir=temp_dir)
                    if stock_data is None or stock_data.empty:
                        self.logger.debug("get_stock_market_data returned empty data")
                        return False
                except Exception as e:
                    self.logger.debug(f"get_stock_market_data failed: {e}")
                    return False
                
                # Test industry mapping data function (more intensive, so optional)
                try:
                    # This is a very expensive call, so we'll just test if it starts correctly
                    # by checking if the industry names API works
                    industry_names = ak.stock_board_industry_name_em()
                    if industry_names is None or industry_names.empty:
                        return False
                    
                    # Test one industry constituent call
                    first_industry = industry_names["板块名称"].iloc[0] 
                    industry_stocks = ak.stock_board_industry_cons_em(symbol=first_industry)
                    if industry_stocks is None or industry_stocks.empty:
                        return False
                        
                except Exception as e:
                    self.logger.debug(f"Industry mapping API test failed: {e}")
                    return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Data functions test failed: {e}")
            return False

    @timer
    def measure_performance(self) -> Dict[str, float]:
        """
        Measure akshare API performance metrics.

        Returns:
            Dictionary with performance metrics (response times, etc.)
        """
        self.logger.info("📊 Measuring akshare API performance...")

        metrics = {}

        # Test response time for stock data
        try:
            start_time = time.time()
            data = ak.stock_zh_a_spot_em()
            if data is not None and not data.empty:
                metrics["stock_data_response_time"] = round(time.time() - start_time, 2)
                metrics["stock_data_count"] = len(data)
                self.logger.info(
                    f"📈 Stock data: {metrics['stock_data_count']} records in {metrics['stock_data_response_time']}s"
                )

        except Exception as e:
            self.logger.warning(f"Performance test failed: {e}")
            metrics["stock_data_response_time"] = -1
            metrics["stock_data_count"] = 0

        return metrics

    @timer
    def get_health_status(self) -> Tuple[ConnectivityStatus, Dict[str, Any]]:
        """
        Get comprehensive akshare health status.

        Returns:
            Tuple of (status, details_dict)
        """
        self.logger.info("🏥 Performing comprehensive akshare health check...")

        details = {
            "timestamp": time.time(),
            "connectivity": False,
            "endpoints": {},
            "performance": {},
            "issues": [],
        }

        # Test basic connectivity
        is_connected, error = self.check_basic_connectivity()
        details["connectivity"] = is_connected

        if not is_connected:
            details["issues"].append(f"Basic connectivity failed: {error}")
            return ConnectivityStatus.UNAVAILABLE, details

        # Test individual endpoints
        endpoint_results = self.check_api_endpoints()
        details["endpoints"] = endpoint_results

        # Measure performance
        performance_metrics = self.measure_performance()
        details["performance"] = performance_metrics

        # Determine overall status
        failed_endpoints = [
            name for name, status in endpoint_results.items() if not status
        ]

        if len(failed_endpoints) == 0:
            if performance_metrics.get("stock_data_response_time", 0) < 10:
                status = ConnectivityStatus.HEALTHY
                self.logger.info("✅ Akshare status: HEALTHY - All systems operational")
            else:
                status = ConnectivityStatus.DEGRADED
                details["issues"].append("Slow response times detected")
                self.logger.warning(
                    "⚠️  Akshare status: DEGRADED - Performance issues detected"
                )

        elif len(failed_endpoints) < len(endpoint_results):
            status = ConnectivityStatus.DEGRADED
            details["issues"].append(f"Some endpoints unavailable: {failed_endpoints}")
            self.logger.warning(
                f"⚠️  Akshare status: DEGRADED - Issues with: {failed_endpoints}"
            )

        else:
            status = ConnectivityStatus.UNAVAILABLE
            details["issues"].append("All endpoints unavailable")
            self.logger.error("❌ Akshare status: UNAVAILABLE - All endpoints failed")

        return status, details


# Global instance for easy access
_connectivity_checker = AkshareConnectivityChecker()


def check_cached_data_availability() -> Dict[str, bool]:
    """
    Check if cached data is available locally (no network needed).
    
    Returns:
        Dictionary indicating which data is available locally
    """
    from datetime import datetime
    import os
    
    today = datetime.now().strftime("%Y%m%d")
    data_dir = "data/stocks"
    
    cached_files = {
        "stock_market_data": os.path.exists(f"{data_dir}/stock_zh_a_spot_em_df-{today}.csv"),
        "industry_mapping_data": os.path.exists(f"{data_dir}/industry_stock_mapping_df-{today}.csv")
    }
    
    logger.info(f"📁 Cached data availability: {cached_files}")
    return cached_files


@timer
def check_akshare_connectivity(timeout: int = 30, retry_attempts: int = 3) -> bool:
    """
    Quick akshare connectivity check function.

    Args:
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts

    Returns:
        True if akshare is accessible, False otherwise
    """
    checker = AkshareConnectivityChecker(timeout=timeout, retry_attempts=retry_attempts)
    is_connected, _ = checker.check_basic_connectivity()
    return is_connected


@timer
def get_akshare_health_status() -> Tuple[ConnectivityStatus, Dict[str, Any]]:
    """
    Get comprehensive akshare health status.

    Returns:
        Tuple of (status, details_dict)
    """
    return _connectivity_checker.get_health_status()


def log_connectivity_status(
    status: ConnectivityStatus, details: Dict[str, Any]
) -> None:
    """
    Log akshare connectivity status in a formatted way.

    Args:
        status: Connectivity status
        details: Health check details
    """
    logger.info("=" * 60)
    logger.info(f"🔗 AKSHARE CONNECTIVITY STATUS: {status.value.upper()}")
    logger.info("=" * 60)

    if details.get("connectivity"):
        logger.info("✅ Basic connectivity: OK")
    else:
        logger.error("❌ Basic connectivity: FAILED")

    # Log endpoint status
    if details.get("endpoints"):
        logger.info("\n📡 Endpoint Status:")
        for endpoint, status_ok in details["endpoints"].items():
            icon = "✅" if status_ok else "❌"
            logger.info(f"  {icon} {endpoint}: {'OK' if status_ok else 'FAILED'}")

    # Log performance metrics
    if details.get("performance"):
        logger.info("\n📊 Performance Metrics:")
        perf = details["performance"]
        if "stock_data_response_time" in perf and perf["stock_data_response_time"] > 0:
            logger.info(
                f"  📈 Stock data response time: {perf['stock_data_response_time']}s"
            )
            logger.info(
                f"  📊 Stock data records: {perf.get('stock_data_count', 'N/A')}"
            )

    # Log issues
    if details.get("issues"):
        logger.warning("\n⚠️  Issues Detected:")
        for issue in details["issues"]:
            logger.warning(f"  • {issue}")

    logger.info("=" * 60)

