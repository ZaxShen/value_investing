"""
Stock analysis and holding report generation for Chinese equity markets.

This module provides a WatchlistsAnalyzer class that encapsulates comprehensive
analysis of individual stocks and generates detailed holding reports. It analyzes
stock performance, fund flows, and calculates key financial metrics for
investment decision making.
"""

import asyncio
import glob
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

configure_environment()
import akshare as ak

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Initialize logger for this module
logger = get_logger("watchlists_analyzer")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class StockIndividualFundFlowConfig(BaseModel):
    """
    Configuration model for ak.stock_individual_fund_flow API parameters.

    This model validates and provides default values for the API parameters.
    """

    date: str = ""  # Date of fund flow data (format: YYYYMMDD)
    period_count: list = []  # Number of days for fund flow analysis
    config_name: str = ""  # If this config for PROD or other purpose


def load_stock_individual_fund_flow_config(
    config_name: Optional[str] = None,
) -> StockIndividualFundFlowConfig:
    """
    Load configuration for stock_individual_fund_flow API from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        StockIndividualFundFlowConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_dir = Path(
        "data/input/watchlists_analyzer/akshare/stock_individual_fund_flow/"
    )
    if config_name is None:
        config_name = "config.yml"
    config_path = Path(config_dir, config_name)

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load and validate config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return StockIndividualFundFlowConfig(**config_data)


class WatchlistsAnalyzer:
    """
    A class to encapsulate holding stock analysis functionality.

    This class manages the analysis of individual stocks for holding reports,
    providing comprehensive financial metrics, fund flow analysis, and
    performance tracking for investment portfolios.
    """

    # Class constants for analysis parameters
    # TODO: Make Class constants to config.yml
    REPORT_DIR = "data/watchlists/reports"
    watchlists_DIR = "data/watchlists"
    DAYS_LOOKBACK_PERIOD = 100  # Days to look back for sufficient trading data

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
        config_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the WatchlistsAnalyzer with market data.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
            config_name: YAML config file name for API parameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including backtesting parameters
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df
        self.config = load_stock_individual_fund_flow_config(config_name)

        # Store additional arguments for flexibility
        self.args = args
        self.kwargs = kwargs

        # Resolve dates in the class config
        self._resolve_config()

    def _date_converter(self, date_str: str, period_count: int) -> str:
        """
        Convert date by adding/subtracting days.

        Args:
            date_str: Date string in YYYYMMDD format
            period_count: Number of days to add (positive) or subtract (negative)

        Returns:
            New date string in YYYYMMDD format
        """
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        new_date = date_obj + timedelta(days=period_count)
        return new_date.strftime("%Y%m%d")

    def _resolve_config(self) -> None:
        """
        Resolve start_date, end_date, or other data in config based on the configuration rules.
        Modifies the config object in-place.
        """
        config = self.config

        # Override with kwargs if provided
        if "start_date" in self.kwargs:
            config.start_date = self.kwargs["start_date"]
        if "end_date" in self.kwargs:
            config.end_date = self.kwargs["end_date"]
        if "period_count" in self.kwargs:
            config.period_count = self.kwargs["period_count"]

        # Set start_date & end_date
        if config.start_date and config.end_date:
            # Both dates provided - use as is
            pass
        elif config.start_date and not config.end_date:
            # Only start_date provided - calculate end_date
            config.end_date = self._date_converter(
                config.start_date, config.period_count
            )
        elif not config.start_date and config.end_date:
            # Only end_date provided - calculate start_date
            config.start_date = self._date_converter(
                config.end_date, -config.period_count
            )
        else:
            # Both dates empty - get latest date from API call first
            # Use default stock to get latest available date
            temp_data = API_RETRY_CONFIG.retry(
                ak.stock_individual_fund_flow, stock=config.stock, market=config.market
            )

            # Get the latest available date and set as end_date
            latest_date = temp_data["日期"].iloc[-1].replace("-", "")
            config.end_date = latest_date
            config.start_date = self._date_converter(
                config.end_date, -config.period_count
            )

    def _get_analysis_columns(self, days: int) -> List[str]:
        """
        Generate analysis column names with dynamic days parameter.

        Args:
            days: Number of days for fund flow analysis

        Returns:
            List of column names for analysis results
        """
        return [
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

    def _get_market_by_stock_code(self, stock_code: str) -> str:
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

    def validate_stock_name(self, stock_code: str, stock_name: str) -> None:
        """
        Validate that the stock name matches the stock code in the dataset.

        Args:
            stock_code: Stock code to validate (e.g., "000001")
            stock_name: Expected stock name

        Raises:
            ValueError: If stock name doesn't match or stock code not found
        """
        try:
            actual_name = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ]["名称"].values[0]
            if actual_name != stock_name:
                raise ValueError(
                    f"Stock name mismatch for {stock_code}: {stock_name} != {actual_name}"
                )
        except (IndexError, KeyError):
            raise ValueError(f"Stock code {stock_code} not found")

    def load_watchlists_from_files(
        self, dir_path: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Load holding stocks data from JSON files in the specified directory.

        Args:
            dir_path: Directory path containing JSON files (default: class constant)

        Returns:
            Dictionary with account names as keys and {stock_code: stock_name}
            dictionaries as values

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If JSON files are malformed
        """
        if dir_path is None:
            dir_path = self.watchlists_DIR

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Holding stocks directory not found: {dir_path}")

        watchlists_data = {}
        json_files = glob.glob(os.path.join(dir_path, "*.json"))

        if not json_files:
            logger.warning("No JSON files found in directory: %s", dir_path)
            return watchlists_data

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    account_name = os.path.splitext(os.path.basename(file_path))[0]
                    watchlists = json.load(f)

                    # Validate JSON structure
                    if not isinstance(watchlists, dict):
                        raise ValueError(
                            f"JSON file {file_path} should contain a dictionary"
                        )

                    for stock_code, stock_name in watchlists.items():
                        if not isinstance(stock_code, str) or not isinstance(
                            stock_name, str
                        ):
                            raise ValueError(
                                f"Invalid stock data in {file_path}: {stock_code} -> {stock_name}"
                            )

                    watchlists_data[account_name] = watchlists
                    logger.info(
                        "Loaded %d stocks for account '%s'",
                        len(watchlists),
                        account_name,
                    )

            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Error loading JSON file %s: %s", file_path, str(e))
                raise ValueError(f"Malformed JSON file {file_path}: {str(e)}")
            except Exception as e:
                logger.error("Error reading file %s: %s", file_path, str(e))
                raise

        logger.info("Loaded holding stocks for %d accounts", len(watchlists_data))
        return watchlists_data

    def _fetch_stock_fund_flow_sync(self, stock_code: str, market: str) -> pd.DataFrame:
        """
        Fetch stock individual fund flow data with retry mechanism.

        Args:
            stock_code: Stock code (e.g., "000001")
            market: Market identifier (e.g., "sz", "sh", "bj")

        Returns:
            DataFrame containing historical fund flow data for the specified stock
        """
        return API_RETRY_CONFIG.retry(
            ak.stock_individual_fund_flow, stock=stock_code, market=market
        )

    def _fetch_sector_fund_flow_sync(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock sector fund flow historical data with retry mechanism.

        Args:
            symbol: Sector symbol identifier

        Returns:
            DataFrame containing historical sector fund flow data
        """
        return API_RETRY_CONFIG.retry(ak.stock_sector_fund_flow_hist, symbol=symbol)

    async def analyze_single_stock(
        self,
        industry_name: str,
        stock_code: str,
        stock_name: str,
        days: Optional[int] = None,
    ) -> Optional[List[Any]]:
        """
        Perform comprehensive analysis of a single stock including fund flow and performance metrics.

        This method analyzes a stock's financial performance, fund flow patterns,
        and calculates key metrics for investment decision making.

        Args:
            industry_name: Industry classification of the stock
            stock_code: Stock code (e.g., "000001")
            stock_name: Stock name for validation and display
            days: Number of days to analyze (default: class constant)

        Returns:
            List containing analysis results with financial metrics, or None if
            analysis fails or stock doesn't meet criteria
        """
        if days is None:
            days = self.config.period_count

        logger.debug(
            "Processing %s (%s) in %s industry", stock_name, stock_code, industry_name
        )

        # Determine the market based on the stock code
        market = self._get_market_by_stock_code(stock_code)

        try:
            # Extract the stock's market data
            stock_data = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ].iloc[0]

            stock_total_market_value = round(stock_data["总市值"] / 1e8, 0)
            stock_circulating_market_value = round(stock_data["流通市值"] / 1e8, 0)
            stock_pe_dynamic = stock_data["市盈率-动态"]
            stock_pb = stock_data["市净率"]
            stock_60d_change = stock_data["60日涨跌幅"]
            stock_ytd_change = stock_data["年初至今涨跌幅"]

            # Extract the historical data of the stock (async with retry)
            stock_individual_fund_flow_df = await asyncio.to_thread(
                self._fetch_stock_fund_flow_sync, stock_code, market
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
            stock_main_net_flow = round(
                stock_individual_fund_flow_df["主力净流入-净额"].sum() / 1e8, 2
            )

            # Calculate change percentage with division by zero protection
            stock_1st_price = stock_individual_fund_flow_df.iloc[0]["收盘价"]
            stock_last_price = stock_individual_fund_flow_df.iloc[-1]["收盘价"]

            if stock_1st_price == 0:
                logger.warning(
                    "First price is zero for %s (%s), skipping price change calculation",
                    stock_name,
                    stock_code,
                )
                return None

            stock_price_change_percentage = round(
                (stock_last_price - stock_1st_price) / stock_1st_price * 100, 2
            )

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

        except Exception as e:
            logger.error("Error processing %s (%s): %s", stock_name, stock_code, str(e))
            return None

    def _save_report(self, df: pd.DataFrame, last_date_str: str) -> None:
        """
        Save holding analysis report to CSV file.

        Args:
            df: DataFrame containing analysis results
            last_date_str: Date string for report naming
        """
        config = self.config
        # Check if config file for PROD
        if config.config_name.upper() == "PROD":
            config_name = ""
        elif config.config_name == "":
            # Only PROD config allows to use empty config_name
            config_name = "-UNKNOWN"
        else:
            config_name = f"-{config.config_name}"

        try:
            report_path = f"{self.REPORT_DIR}/持股报告-{last_date_str}{config_name}.csv"
            df.to_csv(report_path, index=True)
            logger.info("Report saved to %s", report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save holding report: %s", str(e))
            raise

    async def run_analysis(
        self,
        watchlists_data: Dict[str, Dict[str, str]],
        days: Optional[int] = None,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional["TaskID"] = None,
        _batch_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Run the complete holding stock analysis pipeline.

        This method orchestrates the entire holding stock analysis process including
        stock validation, analysis, and report generation.

        Args:
            watchlists_data: Dictionary with account names as keys and
                                {stock_code: stock_name} dictionaries as values
            days: Number of days to analyze (default: class constant)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        if days is None:
            days = self.config.period_count

        # Initialize DataFrame with analysis columns
        columns = self._get_analysis_columns(days)
        df = pd.DataFrame(columns=columns)

        # Process each account's holdings
        for account_name, watchlists in watchlists_data.items():
            for stock_code, stock_name in watchlists.items():
                try:
                    # Validate stock name
                    self.validate_stock_name(stock_code, stock_name)

                    # Get industry name
                    industry_name = self.industry_stock_mapping_df[
                        self.industry_stock_mapping_df["代码"] == stock_code
                    ]["行业"].values[0]

                    # Analyze the stock
                    result = await self.analyze_single_stock(
                        industry_name=industry_name,
                        stock_code=stock_code,
                        stock_name=stock_name,
                        days=days,
                    )

                    if result is not None:
                        df.loc[len(df)] = [account_name] + result

                except Exception as e:
                    logger.error(
                        "Error processing %s (%s) for account %s: %s",
                        stock_name,
                        stock_code,
                        account_name,
                        str(e),
                    )
                    continue

        # Get report date with retry mechanism
        stock_sector_data = await asyncio.to_thread(
            self._fetch_sector_fund_flow_sync, "证券"
        )
        last_date = stock_sector_data.iloc[-1]["日期"]
        last_date_str = last_date.strftime("%Y%m%d")

        # Save the report
        self._save_report(df, last_date_str)

    async def run_analysis_from_files(
        self,
        dir_path: Optional[str] = None,
        days: Optional[int] = None,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional["TaskID"] = None,
        batch_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Load holding stocks from JSON files and run the complete analysis pipeline.

        This is a convenience method that combines loading JSON files and running analysis.

        Args:
            dir_path: Directory path containing JSON files (default: class constant)
            days: Number of days to analyze (default: class constant)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Load holding stocks data from JSON files
        watchlists_data = self.load_watchlists_from_files(dir_path)

        if not watchlists_data:
            logger.warning("No holding stocks data loaded, skipping analysis")
            return

        # Run the analysis with loaded data
        await self.run_analysis(
            watchlists_data=watchlists_data,
            days=days,
            _progress=progress,
            _parent_task_id=parent_task_id,
            _batch_task_id=batch_task_id,
        )


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    progress: Optional["Progress"] = None,
    parent_task_id: Optional["TaskID"] = None,
    batch_task_id: Optional["TaskID"] = None,
    config_name: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
    """
    Main function to execute holding stock analysis and generate reports.

    This function creates a WatchlistsAnalyzer instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        industry_stock_mapping_df: DataFrame containing industry-stock mapping
        stock_zh_a_spot_em_df: DataFrame containing stock market data
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        config_name: YAML config file name. If None, uses default config
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments including backtesting parameters
    """
    watchlists_analyzer = WatchlistsAnalyzer(
        industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name, *args, **kwargs
    )
    watchlists_data = watchlists_analyzer.load_watchlists_from_files()
    await watchlists_analyzer.run_analysis(
        watchlists_data=watchlists_data,
        _progress=progress,
        _parent_task_id=parent_task_id,
        _batch_task_id=batch_task_id,
    )
