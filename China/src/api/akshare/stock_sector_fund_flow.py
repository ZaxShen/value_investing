"""
Centralized akshare API module for stock sector fund flow functionality.

This module provides a centralized interface for fetching sector fund flow data
from akshare, including capital flow analysis and related operations.
Used to eliminate code duplication across different analysis modules.
"""

import asyncio
from datetime import datetime
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
logger = get_logger("api.akshare.stock_sector_fund_flow")


class StockSectorFundFlowConfig(BaseModel):
    """
    Configuration model for ak.stock_sector_fund_flow_hist API parameters.

    This model validates and provides default values for the API parameters.
    """

    symbol: str = ""
    period_count: list[int] = [1, 5, 29]


def fetch_sector_fund_flow_sync(symbol: str) -> pd.DataFrame:
    """
    Fetch sector fund flow historical data with retry mechanism.

    Args:
        symbol: Sector symbol identifier (e.g., industry name)

    Returns:
        DataFrame containing historical sector fund flow data (~120 days)

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching sector fund flow data for %s", symbol)
    
    return API_RETRY_CONFIG.retry(ak.stock_sector_fund_flow_hist, symbol=symbol)


async def fetch_sector_fund_flow_async(symbol: str) -> pd.DataFrame:
    """
    Fetch sector fund flow historical data asynchronously.

    Args:
        symbol: Sector symbol identifier (e.g., industry name)

    Returns:
        DataFrame containing historical sector fund flow data
    """
    return await asyncio.to_thread(fetch_sector_fund_flow_sync, symbol)


def validate_fund_flow_data_availability(period_unit: str) -> bool:
    """
    Validate if fund flow data is available for the given period unit.

    Fund flow data is only available for daily periods in recent 120 days.

    Args:
        period_unit: Period unit ("日", "周", "月")

    Returns:
        bool: True if fund flow data is available, False otherwise
    """
    return period_unit == "日"


def process_fund_flow_for_periods(
    df: pd.DataFrame,
    period_counts: list[int],
    end_date: Optional[datetime] = None,
    fund_flow_column: str = "主力净流入-净额"
) -> list[float]:
    """
    Process fund flow data to calculate fund flows for multiple periods.

    Args:
        df: DataFrame containing fund flow data with date and fund flow columns
        period_counts: List of period counts to calculate (e.g., [1, 5, 29])
        end_date: End date for filtering. If None, uses all available data
        fund_flow_column: Name of the fund flow column

    Returns:
        List of fund flow values (in 100 million RMB) for each period
    """
    # Ensure date column is datetime
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], format="%Y-%m-%d")

    # Filter data up to end_date if specified
    if end_date is not None:
        filtered_df = df[df["日期"] <= end_date]
        if len(filtered_df) == 0:
            logger.warning(
                "End date %s filters out all data. Using all available data (range: %s to %s)",
                end_date.strftime("%Y%m%d"),
                df["日期"].min(),
                df["日期"].max()
            )
            filtered_df = df
    else:
        filtered_df = df

    fund_flows = []
    
    for period in period_counts:
        if len(filtered_df) >= period:
            # Take the last N periods
            period_data = filtered_df.iloc[-period:]
            fund_flow = round(period_data[fund_flow_column].sum() / 1e8, 1)
        else:
            # Use all available data if not enough for the period
            fund_flow = round(filtered_df[fund_flow_column].sum() / 1e8, 1)
        fund_flows.append(fund_flow)

    return fund_flows


def create_empty_fund_flow_data() -> pd.DataFrame:
    """
    Create empty fund flow DataFrame with standard structure.

    This is used when fund flow data is not available (e.g., for non-daily periods).

    Returns:
        DataFrame with empty/null fund flow data structure
    """
    return pd.DataFrame({
        "日期": ["0000-01-01"],
        "主力净流入-净额": [None],
        "主力净流入-净占比": [None],
        "超大单净流入-净额": [None],
        "超大单净流入-净占比": [None],
        "大单净流入-净额": [None],
        "大单净流入-净占比": [None],
        "中单净流入-净额": [None],
        "中单净流入-净占比": [None],
        "小单净流入-净额": [None],
        "小单净流入-净占比": [None],
    })


class StockSectorFundFlowAPI:
    """
    Centralized API handler for stock sector fund flow operations.
    
    This class provides a high-level interface for sector fund flow operations
    with consistent configuration and error handling.
    """

    def __init__(self, config: Optional[StockSectorFundFlowConfig] = None):
        """
        Initialize the API handler with configuration.

        Args:
            config: Configuration object. If None, uses default values
        """
        self.config = config or StockSectorFundFlowConfig()

    def fetch_sync(self, symbol: str) -> pd.DataFrame:
        """
        Fetch sector fund flow data synchronously.

        Args:
            symbol: Sector symbol identifier

        Returns:
            DataFrame with fund flow data
        """
        return fetch_sector_fund_flow_sync(symbol)

    async def fetch_async(self, symbol: str) -> pd.DataFrame:
        """
        Fetch sector fund flow data asynchronously.

        Args:
            symbol: Sector symbol identifier

        Returns:
            DataFrame with fund flow data
        """
        return await fetch_sector_fund_flow_async(symbol)

    def validate_availability(self, period_unit: str) -> bool:
        """
        Validate if fund flow data is available for the period unit.

        Args:
            period_unit: Period unit to validate

        Returns:
            bool: True if available, False otherwise
        """
        return validate_fund_flow_data_availability(period_unit)

    def fetch_with_validation(
        self, 
        symbol: str, 
        period_unit: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch fund flow data with validation and filtering.

        Args:
            symbol: Sector symbol identifier
            period_unit: Period unit for validation
            end_date: End date in YYYYMMDD format for filtering

        Returns:
            DataFrame with fund flow data or empty structure if not available
        """
        if not self.validate_availability(period_unit):
            return create_empty_fund_flow_data()

        # Fetch all available flow data
        flow_data = self.fetch_sync(symbol)

        # Convert date column to datetime
        flow_data["日期"] = pd.to_datetime(flow_data["日期"], format="%Y-%m-%d")

        # Filter data up to the configured end_date if specified
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y%m%d")
            filtered_flow_data = flow_data[flow_data["日期"] <= end_date_obj]

            # If filtering results in no data, use all available data instead
            if len(filtered_flow_data) == 0:
                logger.warning(
                    "Config end_date %s filters out all data for %s. "
                    "Using all available data (range: %s to %s)",
                    end_date,
                    symbol,
                    flow_data["日期"].min(),
                    flow_data["日期"].max()
                )
                filtered_flow_data = flow_data
        else:
            filtered_flow_data = flow_data

        return filtered_flow_data

    async def fetch_with_validation_async(
        self,
        symbol: str,
        period_unit: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch fund flow data asynchronously with validation and filtering.

        Args:
            symbol: Sector symbol identifier
            period_unit: Period unit for validation
            end_date: End date in YYYYMMDD format for filtering

        Returns:
            DataFrame with fund flow data or empty structure if not available
        """
        if not self.validate_availability(period_unit):
            return create_empty_fund_flow_data()

        # Fetch all available flow data
        flow_data = await self.fetch_async(symbol)

        # Convert date column to datetime
        flow_data["日期"] = pd.to_datetime(flow_data["日期"], format="%Y-%m-%d")

        # Filter data up to the configured end_date if specified
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y%m%d")
            filtered_flow_data = flow_data[flow_data["日期"] <= end_date_obj]

            # If filtering results in no data, use all available data instead
            if len(filtered_flow_data) == 0:
                logger.warning(
                    "Config end_date %s filters out all data for %s. "
                    "Using all available data (range: %s to %s)",
                    end_date,
                    symbol,
                    flow_data["日期"].min(),
                    flow_data["日期"].max()
                )
                filtered_flow_data = flow_data
        else:
            filtered_flow_data = flow_data

        return filtered_flow_data

    def process_periods(
        self,
        df: pd.DataFrame,
        period_counts: Optional[list[int]] = None,
        end_date: Optional[datetime] = None,
        fund_flow_column: str = "主力净流入-净额"
    ) -> list[float]:
        """
        Process fund flow data for multiple periods using config or custom periods.

        Args:
            df: DataFrame with fund flow data
            period_counts: Custom period counts. If None, uses config values
            end_date: End date for filtering
            fund_flow_column: Name of fund flow column

        Returns:
            List of fund flow values for each period
        """
        periods = period_counts or self.config.period_count
        return process_fund_flow_for_periods(df, periods, end_date, fund_flow_column)