"""
Stock analysis and holding report generation for Chinese equity markets.

This module provides comprehensive analysis of individual stocks and generates
detailed holding reports. It analyzes stock performance, fund flows, and
calculates key financial metrics for investment decision making.
"""

import asyncio
import functools
import glob
import json
import os
from typing import Optional, List, Any, Callable

import akshare as ak
import pandas as pd
from tqdm import tqdm
from src.utilities.get_stock_data import (
    get_stock_market_data,
    get_industry_stock_mapping_data,
)
from src.utilities.tools import timer
from src.utilities.logger import get_logger
from src.utilities.akshare_checker import (
    check_akshare_connectivity,
    get_akshare_health_status,
    log_connectivity_status,
    ConnectivityStatus,
    check_cached_data_availability,
)

# Initialize logger for this module
logger = get_logger("stock_analysis")


class StockDataManager:
    """Singleton class to manage stock market data with lazy loading and connectivity checking."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._stock_market_data: Optional[pd.DataFrame] = None
            self._industry_mapping_data: Optional[pd.DataFrame] = None
            self._connectivity_checked: bool = False
            self._connectivity_status: Optional[ConnectivityStatus] = None
            self._initialized = True

    def _ensure_connectivity(self) -> None:
        """Ensure akshare connectivity is verified before data operations."""
        if not self._connectivity_checked:
            logger.info("🔍 Verifying akshare connectivity before data operations...")

            # First, check if we have cached data available
            cached_data = check_cached_data_availability()
            all_cached = all(cached_data.values())

            if all_cached:
                logger.info(
                    "📁 All required data is cached locally - skipping network connectivity check"
                )
                self._connectivity_checked = True
                self._connectivity_status = ConnectivityStatus.HEALTHY
                return

            try:
                logger.info(
                    "🌐 Cached data not available - performing network connectivity check..."
                )

                # Perform comprehensive health check
                status, details = get_akshare_health_status()
                self._connectivity_status = status
                self._connectivity_checked = True

                # Log the connectivity status
                log_connectivity_status(status, details)

                # Handle different status levels
                if status == ConnectivityStatus.UNAVAILABLE:
                    error_msg = "Akshare services are completely unavailable. Cannot proceed with data operations."
                    logger.error(f"❌ {error_msg}")
                    raise ConnectionError(error_msg)

                elif status == ConnectivityStatus.DEGRADED:
                    logger.warning(
                        "⚠️  Akshare services are degraded but operational. Proceeding with caution..."
                    )

                else:  # HEALTHY
                    logger.info(
                        "✅ Akshare services are healthy. Proceeding with data operations..."
                    )

            except Exception as e:
                # Mark as checked to avoid repeated failures during testing
                self._connectivity_checked = True
                self._connectivity_status = ConnectivityStatus.UNKNOWN
                error_msg = f"Connectivity check failed: {str(e)}"
                logger.error(error_msg)
                # Don't raise the error during testing - let the data fetching handle it
                if not getattr(__builtins__, "__TESTING__", False):
                    raise ConnectionError(error_msg)

    @property
    def stock_market_data(self) -> pd.DataFrame:
        """Get stock market data, loading it if necessary after connectivity check."""
        self._ensure_connectivity()

        if self._stock_market_data is None:
            logger.debug("Loading stock market data")
            self._stock_market_data = get_stock_market_data()

        # Ensure we have valid data before returning
        assert (
            self._stock_market_data is not None
        ), "Stock market data should not be None after loading"
        return self._stock_market_data

    @property
    def industry_mapping_data(self) -> pd.DataFrame:
        """Get industry mapping data, loading it if necessary after connectivity check."""
        self._ensure_connectivity()

        if self._industry_mapping_data is None:
            logger.debug("Loading industry mapping data")
            self._industry_mapping_data = get_industry_stock_mapping_data()

        # Ensure we have valid data before returning
        assert (
            self._industry_mapping_data is not None
        ), "Industry mapping data should not be None after loading"
        return self._industry_mapping_data

    @property
    def connectivity_status(self) -> Optional[ConnectivityStatus]:
        """Get the current akshare connectivity status."""
        return self._connectivity_status

    def reset_connectivity_check(self) -> None:
        """Reset connectivity check to force re-verification on next data access."""
        logger.info("🔄 Resetting akshare connectivity check...")
        self._connectivity_checked = False
        self._connectivity_status = None


# Global instance of the data manager
_data_manager = StockDataManager()


def validate_stock_name(stock_code: str, stock_name: str, df: pd.DataFrame) -> None:
    """
    Validate that the stock name matches the stock code in the dataset.

    Args:
        stock_code: Stock code to validate (e.g., "000001")
        stock_name: Expected stock name
        df: DataFrame containing stock data with "代码" and "名称" columns

    Raises:
        ValueError: If stock name doesn't match or stock code not found
    """
    try:
        actual_name = df[df["代码"] == stock_code]["名称"].values[0]
        if actual_name != stock_name:
            raise ValueError(
                f"Stock name mismatch for {stock_code}: {stock_name} != {actual_name}"
            )
    except (IndexError, KeyError):
        raise ValueError(f"Stock code {stock_code} not found")


def run_in_executor(func: Callable) -> Callable:
    """
    Decorator to run blocking functions in thread pool executor.

    This decorator converts synchronous blocking functions into asynchronous
    functions by running them in a thread pool executor.

    Args:
        func: The synchronous function to be executed in a thread pool

    Returns:
        Async wrapper function that executes the original function in a thread pool
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


@run_in_executor
def fetch_stock_individual_fund_flow(stock_code: str, market: str) -> pd.DataFrame:
    """
    Fetch stock individual fund flow data - wrapped for async execution.

    Args:
        stock_code: Stock code (e.g., "000001")
        market: Market identifier (e.g., "sz", "sh", "bj")

    Returns:
        DataFrame containing historical fund flow data for the specified stock
    """
    return ak.stock_individual_fund_flow(stock=stock_code, market=market)


@run_in_executor
def fetch_stock_sector_fund_flow_hist(symbol: str) -> pd.DataFrame:
    """
    Fetch stock sector fund flow historical data - wrapped for async execution.

    Args:
        symbol: Sector symbol identifier

    Returns:
        DataFrame containing historical sector fund flow data
    """
    return ak.stock_sector_fund_flow_hist(symbol=symbol)


async def stock_analysis(
    industry_name: str, stock_code: str, stock_name: str, days: int = 29
) -> Optional[List[Any]]:
    """
    Perform comprehensive analysis of a single stock including fund flow and performance metrics.

    This function analyzes a stock's financial performance, fund flow patterns,
    and calculates key metrics for investment decision making.

    Args:
        industry_name: Industry classification of the stock
        stock_code: Stock code (e.g., "000001")
        stock_name: Stock name for validation and display
        days: Number of days to analyze (default: 29)

    Returns:
        List containing analysis results with financial metrics, or None if
        analysis fails or stock doesn't meet criteria
    """
    logger.debug(
        "Processing %s (%s) in %s industry", stock_name, stock_code, industry_name
    )
    # Determine the market based on the stock code
    if stock_code.startswith("6"):
        market = "sh"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        market = "sz"
    else:
        market = "bj"

    # Get stock market data through data manager
    stock_data = _data_manager.stock_market_data
    stock_row = stock_data[stock_data["代码"] == stock_code]

    # Extract the stock's market data
    stock_total_market_value = round(
        stock_row["总市值"].values[0] / 1e8, 0
    )  # Convert to 100M
    stock_circulating_market_value = round(
        stock_row["流通市值"].values[0] / 1e8, 0  # Convert to 100M
    )
    stock_pe_dynamic = stock_row["市盈率-动态"].values[0]
    stock_pb = stock_row["市净率"].values[0]
    stock_60d_change = stock_row["60日涨跌幅"].values[0]
    stock_ytd_change = stock_row["年初至今涨跌幅"].values[0]

    # Extract the historical data of the stock (async)
    stock_individual_fund_flow_df = await fetch_stock_individual_fund_flow(
        stock_code, market
    )
    if len(stock_individual_fund_flow_df) < days:
        logger.warning(
            "Skipping %s (%s) due to insufficient data for the last %d days",
            stock_name,
            stock_code,
            days,
        )
        return None
    stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[-days:]
    # Get the main net inflow data
    stock_main_net_flow = stock_individual_fund_flow_df["主力净流入-净额"].sum()
    stock_main_net_flow = round(stock_main_net_flow / 1e8, 2)  # Convert to billions
    # Calculate change percentage
    stock_1st_price = stock_individual_fund_flow_df.iloc[-days]["收盘价"]
    stock_last_price = stock_individual_fund_flow_df.iloc[-1]["收盘价"]
    stock_price_change_percentage = (
        (stock_last_price - stock_1st_price) / stock_1st_price * 100
    )
    stock_price_change_percentage = round(stock_price_change_percentage, 2)

    return [
        industry_name,
        stock_code,
        stock_name,
        stock_total_market_value,
        stock_circulating_market_value,
        stock_pe_dynamic,
        stock_pb,
        stock_last_price,
        stock_main_net_flow,
        stock_price_change_percentage,
        stock_60d_change,
        stock_ytd_change,
    ]


@timer
async def main() -> None:
    """
    Main function to execute stock analysis and generate holding reports.

    This function reads stock holding data from JSON files, performs comprehensive
    analysis on each stock, and generates detailed reports with financial metrics
    and performance indicators.
    """
    dir_path = "data/holding_stocks"
    days = 29

    # Initialize a pandas Dataframe to hold industry names,
    # industry main net flow, and industry index change percentage
    df = pd.DataFrame(
        columns=[
            "账户",
            "行业",
            "代码",
            "名称",
            "总市值(亿)",
            "流通市值(亿)",
            "市盈率-动态",
            "市净率",
            "收盘价",
            f"{days}日主力净流入-总净额(亿)",
            f"{days}日涨跌幅(%)",
            "60日涨跌幅(%)",
            "年初至今涨跌幅(%)",
        ]
    )

    # Count total stocks for progress tracking
    total_stocks = 0
    account_files = list(glob.glob(os.path.join(dir_path, "*.json")))

    logger.info("📊 Scanning holding stock files...")
    for file in account_files:
        with open(file, "r", encoding="utf-8") as f:
            holding_stocks = json.load(f)
            total_stocks += len(holding_stocks)

    logger.info(
        "📈 Found %d accounts with %d total holdings to analyze",
        len(account_files),
        total_stocks,
    )

    # Main pipeline stages
    pipeline_stages = [
        "📊 Loading market data",
        "🔍 Processing holdings",
        "📅 Getting report date",
        "💾 Saving final report",
    ]

    with tqdm(
        total=len(pipeline_stages),
        desc="Stock Analysis Pipeline",
        unit="stage",
        leave=True,
    ) as main_pbar:
        # Stage 1: Load market data (this triggers data manager connectivity check)
        main_pbar.set_description("📊 Loading market data and validating connectivity")
        logger.info("Loading market data through data manager...")
        stock_data = _data_manager.stock_market_data
        industry_data = _data_manager.industry_mapping_data
        main_pbar.update(1)
        logger.info("✅ Market data loaded successfully")

        # Stage 2: Process all holdings
        main_pbar.set_description("🔍 Analyzing individual stock holdings")
        logger.info(
            "Starting individual stock analysis for %d holdings...", total_stocks
        )

        processed_count = 0
        with tqdm(
            total=total_stocks,
            desc="Processing holdings",
            unit="stock",
            leave=False,
            position=1,
        ) as stock_pbar:
            for file in account_files:
                with open(file, "r", encoding="utf-8") as f:
                    account_name = os.path.splitext(os.path.basename(file))[0]
                    holding_stocks = json.load(f)

                    stock_pbar.set_description(
                        f"Processing {account_name} ({len(holding_stocks)} stocks)"
                    )
                    logger.info(
                        "Processing account: %s with %d holdings",
                        account_name,
                        len(holding_stocks),
                    )

                    for stock_code, stock_name in holding_stocks.items():
                        try:
                            validate_stock_name(stock_code, stock_name, stock_data)
                            industry_name = industry_data[
                                industry_data["代码"] == stock_code
                            ]["行业"].values[0]

                            result = await stock_analysis(
                                industry_name=industry_name,
                                stock_code=stock_code,
                                stock_name=stock_name,
                                days=days,
                            )

                            if result is not None:
                                df.loc[len(df)] = [f"{account_name}"] + result
                                processed_count += 1

                            stock_pbar.update(1)

                        except Exception as e:
                            logger.error(
                                "Error processing %s (%s): %s",
                                stock_name,
                                stock_code,
                                str(e),
                            )
                            stock_pbar.update(1)

            stock_pbar.set_description(
                f"✅ Processed {processed_count}/{total_stocks} holdings successfully"
            )

        main_pbar.update(1)
        logger.info(
            "✅ Stock analysis completed: %d/%d stocks processed successfully",
            processed_count,
            total_stocks,
        )

        # Stage 3: Get report date
        main_pbar.set_description("📅 Getting report date from market data")
        logger.info("Fetching latest market date for report naming...")
        stock_sector_data = await fetch_stock_sector_fund_flow_hist("证券")
        last_date = stock_sector_data.iloc[-1]["日期"]
        last_date_str = last_date.strftime("%Y%m%d")
        main_pbar.update(1)
        logger.info("✅ Report date determined: %s", last_date_str)

        # Stage 4: Save final report
        main_pbar.set_description("💾 Saving holding analysis report")
        df.to_csv(f"{dir_path}/reports/持股报告-{last_date_str}.csv", index=True)
        main_pbar.update(1)

        main_pbar.set_description("🎉 Stock analysis pipeline completed successfully!")
        logger.info(
            "✅ Report saved: %s/reports/持股报告-%s.csv", dir_path, last_date_str
        )
        logger.info("🎉 Stock analysis pipeline completed successfully!")

    print(
        f"📋 Stock analysis completed! Processed {processed_count} holdings from {len(account_files)} accounts."
    )


if __name__ == "__main__":
    asyncio.run(main())
