"""
Stock analysis from watchlist report generation for Chinese equity markets.

This module provides a WatchlistAnalyzer class that encapsulates comprehensive
analysis of individual stocks and generates detailed watchlist reports. It analyzes
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
from src.api.akshare import StockIndividualFundFlowAPI, StockIndividualFundFlowConfig, get_market_by_stock_code

configure_environment()
import akshare as ak

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Initialize logger for this module
logger = get_logger("watchlist_analyzer")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class FileConfig(BaseModel):
    """
    Configuration model for file-related settings.

    This model handles file configuration metadata including config name,
    version, and description.
    """

    config_name: str = "PROD"  # If this config for PROD or other purpose
    description: str = ""  # Configuration description
    version: str = ""  # Configuration version


# StockIndividualFundFlowConfig now imported from centralized API module


class WatchlistAnalyzerConfig(BaseModel):
    """
    Configuration model for WatchlistAnalyzer class parameters.

    This model validates and provides default values for the WatchlistAnalyzer class constants.
    """

    days_lookback_period: int = 100  # Days to look back for sufficient trading data
    watchlist_dir: str = ""  # Source data directory
    report_dir: str = ""  # Report directory


class Config(BaseModel):
    """
    Configuration model for nested YAML structure supporting both akshare and WatchlistAnalyzer configs.

    This model handles the nested structure from data/config/watchlist_analyzer.
    """

    akshare: Dict[str, Dict[str, Any]] = {}
    watchlist_analyzer: Dict[str, Any] = {}
    file_config: Dict[str, Any] = {}


def load_config(
    config_name: Optional[str] = None,
) -> tuple[StockIndividualFundFlowConfig, WatchlistAnalyzerConfig, FileConfig]:
    """
    Load nested configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        tuple: (akshare_config, watchlist_analyzer_config, file_config)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_dir = Path("data/config/watchlist_analyzer/")
    if config_name is None:
        config_name = "config.yml"
    config_path = Path(config_dir, config_name)

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Check if it's nested format (has 'akshare' key) or flat format
    if "akshare" in config_data:
        # Nested format - extract each section
        config = Config(**config_data)

        # Extract akshare config
        akshare_data = config.akshare.get("stock_individual_fund_flow", {})
        akshare_config = StockIndividualFundFlowConfig(**akshare_data)

        # Extract watchlist analyzer config
        watchlist_config = WatchlistAnalyzerConfig(**config.watchlist_analyzer)

        # Extract file config
        file_config = FileConfig(**config.file_config)

        return akshare_config, watchlist_config, file_config
    else:
        # Flat format - use existing logic for backward compatibility
        # Extract config_name if present, otherwise use PROD
        config_name_val = config_data.pop("config_name", "PROD")
        akshare_config = StockIndividualFundFlowConfig(**config_data)
        # Use default watchlist analyzer config
        watchlist_config = WatchlistAnalyzerConfig()
        file_config = FileConfig(config_name=config_name_val)

        return akshare_config, watchlist_config, file_config


def load_stock_individual_fund_flow_config(
    config_name: Optional[str] = None,
) -> StockIndividualFundFlowConfig:
    """
    Load configuration for stock_individual_fund_flow API from YAML file.

    Maintains backward compatibility by returning only the akshare config.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        StockIndividualFundFlowConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    akshare_config, _, _ = load_config(config_name)
    return akshare_config


class WatchlistAnalyzer:
    """
    A class to encapsulate watchlist stock analysis functionality.

    This class manages the analysis of individual stocks reports,
    providing comprehensive financial metrics, fund flow analysis, and
    performance tracking for investment portfolios.
    """

    # Class constants for analysis parameters
    # REPORT_DIR = "data/watchlist/reports"
    # WATCHLIST_DIR = "data/watchlist"
    # DAYS_LOOKBACK_PERIOD = 100  # Days to look back for sufficient trading data

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
        config_name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the WatchlistAnalyzer with market data.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
            config_name: YAML config file name for API parameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including backtesting parameters
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df

        # Load both akshare and watchlist analyzer configs
        self.config, self.analyzer_config, self.file_config = load_config(config_name)

        # Apply class constants from config
        self.DAYS_LOOKBACK_PERIOD = self.analyzer_config.days_lookback_period
        self.WATCHLIST_DIR = self.analyzer_config.watchlist_dir
        self.REPORT_DIR = self.analyzer_config.report_dir

        # Store additional arguments for flexibility
        self.args = args
        self.kwargs = kwargs
        
        # Initialize centralized API handler
        self.fund_flow_api = StockIndividualFundFlowAPI(self.config)

        # Initialize date-related attributes
        self.last_date = None
        self.last_date_str = None

        # Resolve dates in the class config
        self._resolve_config()

        # Initialize the actual last date for consistent date handling across methods
        self._initialize_last_date()

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
        Resolve config based on kwargs if provided.
        For watchlist analyzer, we fetch all available data and slice dynamically.
        """
        config = self.config

        # Override with kwargs if provided
        if "period_count" in self.kwargs:
            config.period_count = self.kwargs["period_count"]
        if "date" in self.kwargs:
            config.date = self.kwargs["date"]

        # Set market automatically if empty
        if not config.market and config.stock:
            config.market = self._get_market_by_stock_code(config.stock)

    def _initialize_last_date(self) -> None:
        """
        Initialize the actual last date for consistent date handling across all methods.

        This method determines the actual last date based on config.date and available data,
        ensuring all methods use the same reference date instead of iloc[-1].
        """
        try:
            # Fetch sector data synchronously during initialization
            stock_sector_data = self._fetch_sector_fund_flow_sync("证券")

            # Use specified date from config if provided, otherwise use latest date
            if self.config.date:
                # Convert config date to datetime for comparison
                target_date = datetime.strptime(self.config.date, "%Y%m%d")

                # Find the closest available date (same or previous)
                available_dates = pd.to_datetime(stock_sector_data["日期"])
                valid_dates = available_dates[available_dates <= target_date]

                if len(valid_dates) > 0:
                    # Use the latest date that's not later than the target date
                    self.last_date = valid_dates.max()
                    logger.info(
                        "Using specified date %s (actual: %s)",
                        self.config.date,
                        self.last_date.strftime("%Y%m%d"),
                    )
                else:
                    # Fallback to earliest available date if target is before all data
                    self.last_date = available_dates.min()
                    logger.warning(
                        "Target date %s is before all available data, using earliest date: %s",
                        self.config.date,
                        self.last_date.strftime("%Y%m%d"),
                    )
            else:
                # Use the latest available date (default behavior)
                self.last_date = stock_sector_data.iloc[-1]["日期"]

            self.last_date_str = self.last_date.strftime("%Y%m%d")
            logger.debug("Initialized last_date: %s", self.last_date_str)

        except Exception as e:
            logger.error("Failed to initialize last_date: %s", str(e))
            # Set a fallback date if initialization fails
            self.last_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.last_date_str = self.last_date.strftime("%Y%m%d")

    def _get_data_for_date(self, df: pd.DataFrame, target_date: datetime) -> pd.Series:
        """
        Get the row from a DataFrame that corresponds to the target date or the closest previous date.

        Args:
            df: DataFrame with a '日期' column
            target_date: Target date to find

        Returns:
            Series corresponding to the target date or closest previous date
        """
        # Convert dates to datetime if they aren't already
        df_dates = pd.to_datetime(df["日期"])

        # Find dates that are <= target_date
        valid_dates = df_dates[df_dates <= target_date]

        if len(valid_dates) > 0:
            # Get the latest valid date
            actual_date = valid_dates.max()
            # Return the row for that date
            return df[df_dates == actual_date].iloc[
                -1
            ]  # Use iloc[-1] in case of duplicates
        else:
            # Fallback to earliest available date
            earliest_date = df_dates.min()
            return df[df_dates == earliest_date].iloc[0]

    def _get_analysis_columns(self) -> List[str]:
        """
        Generate analysis column names with dynamic periods from config.

        Returns:
            List of column names for analysis results including all periods
        """
        base_columns = [
            "账户",
            "行业",
            "代码",
            "名称",
            "总市值(亿)",
            "流通市值(亿)",
            "市盈率-动态",
            "市净率",
            "收盘价",
        ]

        # Add dynamic fund flow columns for each period
        fund_flow_columns = []
        price_change_columns = []
        for period in self.config.period_count:
            fund_flow_columns.append(f"{period}日主力净流入-总净额(亿)")
            price_change_columns.append(f"{period}日涨跌幅(%)")

        # Add fixed period columns
        fixed_columns = [
            "60日涨跌幅(%)",
            "年初至今涨跌幅(%)",
        ]

        return base_columns + fund_flow_columns + price_change_columns + fixed_columns

    def _get_market_by_stock_code(self, stock_code: str) -> str:
        """
        Determine the market based on stock code prefix.
        Delegates to centralized API module.
        """
        return get_market_by_stock_code(stock_code)

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

    def load_watchlist_from_files(
        self, dir_path: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Load watchlist stocks data from JSON files in the specified directory.

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
            dir_path = self.WATCHLIST_DIR

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Watchlist stocks directory not found: {dir_path}")

        watchlist_data = {}
        json_files = glob.glob(os.path.join(dir_path, "*.json"))

        if not json_files:
            logger.warning("No JSON files found in directory: %s", dir_path)
            return watchlist_data

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    account_name = os.path.splitext(os.path.basename(file_path))[0]
                    watchlist = json.load(f)

                    # Validate JSON structure
                    if not isinstance(watchlist, dict):
                        raise ValueError(
                            f"JSON file {file_path} should contain a dictionary"
                        )

                    for stock_code, stock_name in watchlist.items():
                        if not isinstance(stock_code, str) or not isinstance(
                            stock_name, str
                        ):
                            raise ValueError(
                                f"Invalid stock data in {file_path}: {stock_code} -> {stock_name}"
                            )

                    watchlist_data[account_name] = watchlist
                    logger.info(
                        "Loaded %d stocks for account '%s'",
                        len(watchlist),
                        account_name,
                    )

            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Error loading JSON file %s: %s", file_path, str(e))
                raise ValueError(f"Malformed JSON file {file_path}: {str(e)}")
            except Exception as e:
                logger.error("Error reading file %s: %s", file_path, str(e))
                raise

        logger.info("Loaded watchlist stocks for %d accounts", len(watchlist_data))
        return watchlist_data

    def _fetch_stock_fund_flow_sync(self, stock_code: str, market: str) -> pd.DataFrame:
        """
        Fetch stock individual fund flow data with retry mechanism.
        Delegates to centralized API module.
        """
        return self.fund_flow_api.fetch_sync(stock_code, market)

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
    ) -> Optional[List[Any]]:
        """
        Perform comprehensive analysis of a single stock including fund flow and performance metrics.

        This method analyzes a stock's financial performance, fund flow patterns,
        and calculates key metrics for all periods defined in config.

        Args:
            industry_name: Industry classification of the stock
            stock_code: Stock code (e.g., "000001")
            stock_name: Stock name for validation and display

        Returns:
            List containing analysis results with financial metrics for all periods,
            or None if analysis fails or stock doesn't meet criteria
        """
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

            # Check if we have enough data for the maximum period required
            max_period = max(self.config.period_count)
            if len(stock_individual_fund_flow_df) < max_period:
                logger.warning(
                    "Skipping %s (%s) due to insufficient data for the last %d days",
                    stock_name,
                    stock_code,
                    max_period,
                )
                return None

            # Get the actual data row for the target date instead of always using iloc[-1]
            # Ensure last_date is a datetime object
            if self.last_date is None:
                # Fallback to latest date if initialization failed
                actual_target_date = pd.to_datetime(
                    stock_individual_fund_flow_df.iloc[-1]["日期"]
                )
            else:
                actual_target_date = pd.to_datetime(self.last_date)

            actual_data_row = self._get_data_for_date(
                stock_individual_fund_flow_df, actual_target_date
            )

            # Base stock data
            base_data = [
                industry_name,
                stock_code,
                stock_name,
                stock_total_market_value,
                stock_circulating_market_value,
                stock_pe_dynamic,
                stock_pb,
                actual_data_row["收盘价"],  # Closing price for the actual target date
            ]

            # Calculate fund flows and price changes using centralized processing
            # Get the index of the actual target date in the DataFrame
            df_dates = pd.to_datetime(stock_individual_fund_flow_df["日期"])
            target_idx = None
            for idx, date in enumerate(df_dates):
                if date <= actual_target_date:
                    target_idx = idx

            if target_idx is None:
                # If no valid date found, use the first available
                target_idx = 0
            
            # Use centralized processing with target date index
            fund_flows, price_changes = self.fund_flow_api.process_periods(
                stock_individual_fund_flow_df, self.config.period_count, target_idx
            )
            
            # Round to 2 decimal places to match original behavior
            fund_flows = [round(ff, 2) for ff in fund_flows]
            price_changes = [round(pc, 2) for pc in price_changes]

            # Fixed period data (60-day and YTD are from market data)
            fixed_data = [stock_60d_change, stock_ytd_change]

            return base_data + fund_flows + price_changes + fixed_data

        except Exception as e:
            logger.error("Error processing %s (%s): %s", stock_name, stock_code, str(e))
            return None

    def _save_report(self, df: pd.DataFrame, last_date_str: str) -> None:
        """
        Save watchlist analysis report to CSV file.

        Args:
            df: DataFrame containing analysis results
            last_date_str: Date string for report naming
        """
        file_config = self.file_config
        # Check if config file for PROD
        if file_config.config_name.upper() == "PROD":
            config_name = ""
        elif file_config.config_name == "":
            # Only PROD config allows to use empty config_name
            config_name = "-UNKNOWN"
        else:
            config_name = f"-{file_config.config_name}"

        try:
            report_path = (
                f"{self.REPORT_DIR}/自选股报告-{last_date_str}{config_name}.csv"
            )
            df.to_csv(report_path, index=True)
            logger.info("Report saved to %s", report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save watchlist report: %s", str(e))
            raise

    async def run_analysis(
        self,
        watchlist_data: Dict[str, Dict[str, str]],
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional["TaskID"] = None,
        _batch_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Run the complete watchlist analysis pipeline.

        This method orchestrates the entire watchlist analysis process including
        stock validation, analysis, and report generation with dynamic periods.

        Args:
            watchlist_data: Dictionary with account names as keys and
                                {stock_code: stock_name} dictionaries as values
            _progress: Optional Rich Progress instance for hierarchical progress tracking
            _parent_task_id: Optional parent task ID for hierarchical progress structure
            _batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Initialize DataFrame with dynamic analysis columns
        columns = self._get_analysis_columns()
        df = pd.DataFrame(columns=columns)

        # Process each watchlist
        for account_name, watchlist in watchlist_data.items():
            for stock_code, stock_name in watchlist.items():
                try:
                    # Validate stock name
                    self.validate_stock_name(stock_code, stock_name)

                    # Get industry name
                    industry_name = self.industry_stock_mapping_df[
                        self.industry_stock_mapping_df["代码"] == stock_code
                    ]["行业"].values[0]

                    # Analyze the stock (now handles multiple periods automatically)
                    result = await self.analyze_single_stock(
                        industry_name=industry_name,
                        stock_code=stock_code,
                        stock_name=stock_name,
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

        # Use the pre-determined last_date from initialization
        # This ensures consistency across all analysis methods
        if self.last_date_str is None:
            logger.warning("last_date_str not initialized, re-initializing...")
            self._initialize_last_date()

        # Ensure last_date_str is not None before saving
        if self.last_date_str is None:
            # Ultimate fallback if initialization still fails
            self.last_date_str = datetime.now().strftime("%Y%m%d")
            logger.error(
                "Failed to determine last_date, using current date as fallback"
            )

        # Save the report
        self._save_report(df, self.last_date_str)

    async def run_analysis_from_files(
        self,
        dir_path: Optional[str] = None,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional["TaskID"] = None,
        _batch_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Load watchlist from JSON files and run the complete analysis pipeline.

        This is a convenience method that combines loading JSON files and running analysis.

        Args:
            dir_path: Directory path containing JSON files (default: class constant)
            _progress: Optional Rich Progress instance for hierarchical progress tracking
            _parent_task_id: Optional parent task ID for hierarchical progress structure
            _batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Load watchlist data from JSON files
        watchlist_data = self.load_watchlist_from_files(dir_path)

        if not watchlist_data:
            logger.warning("No watchlist data loaded, skipping analysis")
            return

        # Run the analysis with loaded data (periods are now from config)
        await self.run_analysis(
            watchlist_data=watchlist_data,
            _progress=_progress,
            _parent_task_id=_parent_task_id,
            _batch_task_id=_batch_task_id,
        )


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    config_name: Optional[str] = None,
    _progress: Optional["Progress"] = None,
    _parent_task_id: Optional["TaskID"] = None,
    _batch_task_id: Optional["TaskID"] = None,
    *args,
    **kwargs,
) -> None:
    """
    Main function to execute watchlist stock analysis and generate reports.

    This function creates a WatchlistAnalyzer instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        industry_stock_mapping_df: DataFrame containing industry-stock mapping
        stock_zh_a_spot_em_df: DataFrame containing stock market data
        config_name: YAML config file name. If None, uses default config
        _progress: Optional Rich Progress instance for hierarchical progress tracking
        _parent_task_id: Optional parent task ID for hierarchical progress structure
        _batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments including backtesting parameters
    """
    watchlist_analyzer = WatchlistAnalyzer(
        industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name, *args, **kwargs
    )
    watchlist_data = watchlist_analyzer.load_watchlist_from_files()
    await watchlist_analyzer.run_analysis(
        watchlist_data=watchlist_data,
        _progress=_progress,
        _parent_task_id=_parent_task_id,
        _batch_task_id=_batch_task_id,
    )
