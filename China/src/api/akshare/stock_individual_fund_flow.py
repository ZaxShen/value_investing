"""
Centralized akshare API module for stock_individual_fund_flow functionality.

This module provides a centralized interface for fetching individual stock fund flow data
from akshare, with built-in retry mechanisms, market detection, and consistent configuration.
Used by both stock_filter.py and watchlist_analyzer.py to eliminate code duplication.
"""

import asyncio
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

# Configure environment before akshare import
configure_environment()
import akshare as ak

# Initialize logger for this module
logger = get_logger("api.akshare.stock_individual_fund_flow")


class StockIndividualFundFlowConfig(BaseModel):
    """
    Configuration model for ak.stock_individual_fund_flow API parameters.

    This model validates and provides default values for the API parameters.
    """

    stock: str = ""
    market: str = ""
    date: str = ""
    period_count: list[int] = [1, 5, 29]


def get_market_by_stock_code(stock_code: str) -> str:
    """
    Determine the market based on stock code prefix.

    Args:
        stock_code: Stock code (e.g., "000001", "600001", "301001")

    Returns:
        Market identifier: "sh" for Shanghai, "sz" for Shenzhen, "bj" for Beijing
    """
    if stock_code.startswith("6"):
        return "sh"  # Shanghai Stock Exchange
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        return "sz"  # Shenzhen Stock Exchange
    else:
        return "bj"  # Beijing Stock Exchange


def fetch_stock_fund_flow_sync(stock_code: str, market: str) -> pd.DataFrame:
    """
    Fetch stock individual fund flow data with retry mechanism.

    Args:
        stock_code: Stock code (e.g., "000001")
        market: Market identifier (e.g., "sz" for Shenzhen, "sh" for Shanghai)

    Returns:
        DataFrame containing historical fund flow data for the specified stock

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching fund flow data for %s on %s market", stock_code, market)
    
    return API_RETRY_CONFIG.retry(
        ak.stock_individual_fund_flow, stock=stock_code, market=market
    )


async def fetch_stock_fund_flow_async(stock_code: str, market: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch stock individual fund flow data asynchronously with automatic market detection.

    Args:
        stock_code: Stock code (e.g., "000001")
        market: Market identifier. If None, will be auto-detected from stock code

    Returns:
        DataFrame containing historical fund flow data for the specified stock

    Raises:
        Exception: If API call fails after all retries
    """
    if market is None:
        market = get_market_by_stock_code(stock_code)
    
    return await asyncio.to_thread(fetch_stock_fund_flow_sync, stock_code, market)


def process_fund_flow_for_periods(
    df: pd.DataFrame, 
    period_counts: list[int],
    target_date_idx: Optional[int] = None
) -> tuple[list[float], list[float]]:
    """
    Process fund flow data to calculate fund flows and price changes for multiple periods.

    Args:
        df: DataFrame containing fund flow data with columns ['主力净流入-净额', '收盘价', '日期']
        period_counts: List of period counts to calculate (e.g., [1, 5, 29])
        target_date_idx: Index of target date in DataFrame. If None, uses last row

    Returns:
        Tuple of (fund_flows, price_changes) lists for each period
    """
    if target_date_idx is None:
        target_date_idx = len(df) - 1

    fund_flows = []
    price_changes = []

    for period in period_counts:
        # Calculate fund flow for this period (sum of last N days)
        start_idx = max(0, target_date_idx - period + 1)
        end_idx = target_date_idx + 1
        period_data = df.iloc[start_idx:end_idx]
        
        if len(period_data) >= period:
            fund_flow = round(period_data["主力净流入-净额"].sum() / 1e8, 1)
        else:
            fund_flow = 0.0
        fund_flows.append(fund_flow)

        # Calculate price change for this period (N days ago vs target date)
        required_data_points = period + 1
        if len(df) >= required_data_points and target_date_idx >= period:
            first_price_idx = target_date_idx - period
            first_price = df.iloc[first_price_idx]["收盘价"]
            last_price = df.iloc[target_date_idx]["收盘价"]
            
            if first_price == 0:
                price_change = 0.0
            else:
                price_change = round(
                    (last_price - first_price) / first_price * 100, 1
                )
        else:
            price_change = 0.0
        price_changes.append(price_change)

    return fund_flows, price_changes


class StockIndividualFundFlowAPI:
    """
    Centralized API handler for stock individual fund flow operations.
    
    This class provides a high-level interface for fund flow operations
    with consistent configuration and error handling.
    """

    def __init__(self, config: Optional[StockIndividualFundFlowConfig] = None):
        """
        Initialize the API handler with configuration.

        Args:
            config: Configuration object. If None, uses default values
        """
        self.config = config or StockIndividualFundFlowConfig()

    def fetch_sync(self, stock_code: str, market: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch fund flow data synchronously with automatic market detection.

        Args:
            stock_code: Stock code
            market: Market identifier. If None, auto-detected

        Returns:
            DataFrame with fund flow data
        """
        if market is None:
            market = get_market_by_stock_code(stock_code)
        
        return fetch_stock_fund_flow_sync(stock_code, market)

    async def fetch_async(self, stock_code: str, market: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch fund flow data asynchronously with automatic market detection.

        Args:
            stock_code: Stock code
            market: Market identifier. If None, auto-detected

        Returns:
            DataFrame with fund flow data
        """
        return await fetch_stock_fund_flow_async(stock_code, market)

    def process_periods(
        self, 
        df: pd.DataFrame, 
        period_counts: Optional[list[int]] = None,
        target_date_idx: Optional[int] = None
    ) -> tuple[list[float], list[float]]:
        """
        Process fund flow data for multiple periods using config or custom periods.

        Args:
            df: DataFrame with fund flow data
            period_counts: Custom period counts. If None, uses config values
            target_date_idx: Target date index. If None, uses last row

        Returns:
            Tuple of (fund_flows, price_changes)
        """
        periods = period_counts or self.config.period_count
        return process_fund_flow_for_periods(df, periods, target_date_idx)