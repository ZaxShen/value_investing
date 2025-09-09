"""
Centralized akshare API module for stock board industry functionality.

This module provides a centralized interface for fetching industry board data
from akshare, including industry names, historical data, and related operations.
Used to eliminate code duplication across different analysis modules.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel

from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

# Configure environment before akshare import
configure_environment()
import akshare as ak

# Initialize logger for this module
logger = get_logger("api.akshare.stock_board_industry")


class StockBoardIndustryHistConfig(BaseModel):
    """
    Configuration model for ak.stock_board_industry_hist_em API parameters.

    This model validates and provides default values for the API parameters.
    """

    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    period: str = "日k"
    period_count: list[int] = [1, 5, 29]
    adjust: str = ""


def date_converter(date_str: str, period: str, period_count: int) -> str:
    """
    Convert date by adding/subtracting periods.

    Args:
        date_str: Date string in YYYYMMDD format
        period: Period type ("日k", "周k", "月k")
        period_count: Number of periods to add (positive) or subtract (negative)

    Returns:
        New date string in YYYYMMDD format

    Raises:
        ValueError: If period is not supported
    """
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    
    if period == "日k":
        new_date = date_obj + timedelta(days=period_count)
    elif period == "周k":
        new_date = date_obj + timedelta(weeks=period_count)
    elif period == "月k":
        new_date = date_obj + relativedelta(months=period_count)
    else:
        error_msg = f"Invalid period '{period}'. Must be one of: '日k', '周k', '月k'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return new_date.strftime("%Y%m%d")


def fetch_industry_names_sync() -> pd.DataFrame:
    """
    Fetch industry names data with retry mechanism.

    Returns:
        DataFrame containing industry names and related information

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching industry names data")
    
    return API_RETRY_CONFIG.retry(ak.stock_board_industry_name_em)


def fetch_industry_hist_sync(
    symbol: str,
    start_date: str,
    end_date: str, 
    period: str = "日k",
    adjust: str = ""
) -> pd.DataFrame:
    """
    Fetch industry historical data with retry mechanism.

    Args:
        symbol: Industry symbol/name
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        period: Period type ("日k", "周k", "月k")
        adjust: Adjustment type ("", "qfq", "hfq")

    Returns:
        DataFrame containing historical industry data

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching industry hist data for %s (%s to %s)", symbol, start_date, end_date)
    
    return API_RETRY_CONFIG.retry(
        ak.stock_board_industry_hist_em,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        adjust=adjust
    )


async def fetch_industry_names_async() -> pd.DataFrame:
    """
    Fetch industry names data asynchronously.

    Returns:
        DataFrame containing industry names and related information
    """
    return await asyncio.to_thread(fetch_industry_names_sync)


async def fetch_industry_hist_async(
    symbol: str,
    start_date: str,
    end_date: str,
    period: str = "日k", 
    adjust: str = ""
) -> pd.DataFrame:
    """
    Fetch industry historical data asynchronously.

    Args:
        symbol: Industry symbol/name
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        period: Period type ("日k", "周k", "月k") 
        adjust: Adjustment type ("", "qfq", "hfq")

    Returns:
        DataFrame containing historical industry data
    """
    return await asyncio.to_thread(
        fetch_industry_hist_sync, symbol, start_date, end_date, period, adjust
    )


def resolve_date_range(
    config: StockBoardIndustryHistConfig,
    sample_symbol: Optional[str] = None
) -> tuple[str, str]:
    """
    Resolve start_date and end_date based on configuration rules.

    Args:
        config: Configuration object with date and period settings
        sample_symbol: Optional sample symbol to use for fetching latest date

    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format

    Raises:
        ValueError: If period is invalid or date resolution fails
    """
    # Use the maximum period_count for date calculations when needed
    max_period_count = max(config.period_count) if config.period_count else 29
    
    if config.start_date and config.end_date:
        # Both dates provided - use as is
        return config.start_date, config.end_date
    elif config.start_date and not config.end_date:
        # Only start_date provided - calculate end_date using max period
        end_date = date_converter(config.start_date, config.period, max_period_count)
        return config.start_date, end_date
    elif not config.start_date and config.end_date:
        # Only end_date provided - calculate start_date using max period
        start_date = date_converter(config.end_date, config.period, -max_period_count)
        return start_date, config.end_date
    else:
        # Both dates empty - get latest date from API call
        if not sample_symbol:
            # Use a default industry for date resolution
            industry_names = fetch_industry_names_sync()
            sample_symbol = industry_names["板块名称"].iloc[0]
        
        today = datetime.today()
        temp_start = (today - timedelta(days=365)).strftime("%Y%m%d")
        temp_end = today.strftime("%Y%m%d")

        temp_data = fetch_industry_hist_sync(
            symbol=sample_symbol,
            start_date=temp_start,
            end_date=temp_end,
            period=config.period,
            adjust=config.adjust
        )

        # Get the latest available date and set as end_date
        latest_date = temp_data["日期"].iloc[-1].replace("-", "")
        start_date = date_converter(latest_date, config.period, -max_period_count)
        
        return start_date, latest_date


def calculate_price_changes_for_periods(
    df: pd.DataFrame,
    period_counts: list[int],
    close_column: str = "收盘",
    target_idx: Optional[int] = None
) -> list[float]:
    """
    Calculate price changes for multiple periods from historical data.

    Args:
        df: DataFrame with historical price data
        period_counts: List of period counts to calculate
        close_column: Name of the closing price column
        target_idx: Index of target date. If None, uses last row

    Returns:
        List of price change percentages for each period
    """
    if target_idx is None:
        target_idx = len(df) - 1

    price_changes = []
    df_len = len(df)
    
    for period in period_counts:
        # For N-day change, we need to go back N+1 positions to compare with N days ago
        period_lookback = min(period + 1, df_len)
        if period_lookback > 1:  # Need at least 2 data points
            current_price = df[close_column].iloc[target_idx]
            past_price = df[close_column].iloc[target_idx - period_lookback + 1]
            
            if past_price != 0:
                price_change = round((current_price - past_price) / past_price * 100, 2)
            else:
                price_change = 0.0
        else:
            price_change = 0.0
        price_changes.append(price_change)

    return price_changes


class StockBoardIndustryAPI:
    """
    Centralized API handler for stock board industry operations.
    
    This class provides a high-level interface for industry-related operations
    with consistent configuration and error handling.
    """

    def __init__(self, config: Optional[StockBoardIndustryHistConfig] = None):
        """
        Initialize the API handler with configuration.

        Args:
            config: Configuration object. If None, uses default values
        """
        self.config = config or StockBoardIndustryHistConfig()

    def fetch_names_sync(self) -> pd.DataFrame:
        """
        Fetch industry names synchronously.

        Returns:
            DataFrame with industry names and information
        """
        return fetch_industry_names_sync()

    async def fetch_names_async(self) -> pd.DataFrame:
        """
        Fetch industry names asynchronously.

        Returns:
            DataFrame with industry names and information
        """
        return await fetch_industry_names_async()

    def fetch_hist_sync(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        adjust: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch industry historical data synchronously with config defaults.

        Args:
            symbol: Industry symbol/name
            start_date: Start date. If None, uses config value
            end_date: End date. If None, uses config value
            period: Period type. If None, uses config value
            adjust: Adjustment type. If None, uses config value

        Returns:
            DataFrame with historical data
        """
        return fetch_industry_hist_sync(
            symbol=symbol,
            start_date=start_date or self.config.start_date,
            end_date=end_date or self.config.end_date,
            period=period or self.config.period,
            adjust=adjust or self.config.adjust
        )

    async def fetch_hist_async(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        adjust: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch industry historical data asynchronously with config defaults.

        Args:
            symbol: Industry symbol/name
            start_date: Start date. If None, uses config value
            end_date: End date. If None, uses config value
            period: Period type. If None, uses config value
            adjust: Adjustment type. If None, uses config value

        Returns:
            DataFrame with historical data
        """
        return await fetch_industry_hist_async(
            symbol=symbol,
            start_date=start_date or self.config.start_date,
            end_date=end_date or self.config.end_date,
            period=period or self.config.period,
            adjust=adjust or self.config.adjust
        )

    def resolve_dates(self, sample_symbol: Optional[str] = None) -> tuple[str, str]:
        """
        Resolve start and end dates based on config rules.

        Args:
            sample_symbol: Optional sample symbol for date resolution

        Returns:
            Tuple of (start_date, end_date)
        """
        return resolve_date_range(self.config, sample_symbol)

    def calculate_price_changes(
        self,
        df: pd.DataFrame,
        period_counts: Optional[list[int]] = None,
        close_column: str = "收盘",
        target_idx: Optional[int] = None
    ) -> list[float]:
        """
        Calculate price changes for periods using config or custom periods.

        Args:
            df: DataFrame with historical data
            period_counts: Custom period counts. If None, uses config values
            close_column: Name of closing price column
            target_idx: Target date index. If None, uses last row

        Returns:
            List of price change percentages
        """
        periods = period_counts or self.config.period_count
        return calculate_price_changes_for_periods(df, periods, close_column, target_idx)