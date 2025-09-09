"""
Industry analysis and filtering for Chinese equity markets.

This module provides an IndustryFilter class that encapsulates comprehensive
analysis of industry sectors including fund flow analysis, index performance
tracking, and industry-level filtering. It generates detailed reports on
industry performance and capital flows.
"""

import asyncio
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.api.akshare import (
    StockBoardIndustryAPI,
    StockBoardIndustryHistConfig,
    StockSectorFundFlowAPI,
    date_converter,
    resolve_date_range,
)

configure_environment()

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("industry_filter")

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


# StockBoardIndustryHistConfig now imported from centralized API module


class IndustryFilterConfig(BaseModel):
    """
    Configuration model for IndustryFilter class parameters.

    This model validates and provides default values for the IndustryFilter class constants.
    """

    target_period_count: int = 29  # Specific period from period_count list to use for filtering
    min_main_net_inflow_yi: int = 20  # Minimum main net inflow in 100 million RMB
    max_price_change_percent: int = 8  # Maximum price change percentage
    batch_size: int = 3  # Batch size for concurrent processing
    days_lookback_period: int = 100  # Days to look back for sufficient trading data
    trading_days_60: int = 60  # 60 trading days for analysis
    report_dir: str = "data/stocks/reports"  # Report directory


class Config(BaseModel):
    """
    Configuration model for nested YAML structure supporting both akshare and IndustryFilter configs.

    This model handles the nested structure from data/config/industry_filter.
    """

    akshare: Dict[str, Dict[str, Any]] = {}
    industry_filter: Dict[str, Any] = {}
    file_config: Dict[str, Any] = {}


def load_config(
    config_name: Optional[str] = None,
) -> tuple[StockBoardIndustryHistConfig, IndustryFilterConfig, FileConfig]:
    """
    Load nested configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        tuple: (akshare_config, industry_filter_config, file_config)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if config_name is None:
        config_name = "config.yml"

    # Handle both relative and absolute paths
    if config_name.startswith("data/config/"):
        config_path = Path(config_name)
    else:
        config_dir = Path("data/config/industry_filter/")
        config_path = config_dir / config_name

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Check if it's nested format (has 'akshare' key) or flat format
    if "akshare" in config_data:
        # Nested format - extract each section
        configs = Config(**config_data)

        # Extract akshare config
        akshare_data = configs.akshare.get("stock_board_industry_hist_em", {})
        akshare_config = StockBoardIndustryHistConfig(**akshare_data)

        # Extract industry filter config
        industry_filter_config = IndustryFilterConfig(**configs.industry_filter)

        # Extract file config
        file_config = FileConfig(**configs.file_config)

        return akshare_config, industry_filter_config, file_config
    else:
        # Flat format - use existing logic for backward compatibility
        # Extract config_name if present, otherwise use PROD
        config_name_val = config_data.pop("config_name", "PROD")
        akshare_config = StockBoardIndustryHistConfig(**config_data)
        # Use default industry filter config
        industry_filter_config = IndustryFilterConfig()
        file_config = FileConfig(config_name=config_name_val)

        return akshare_config, industry_filter_config, file_config


class IndustryFilter:
    """
    A class to encapsulate industry filtering and analysis functionality.

    This class manages industry data analysis including fund flow tracking,
    index performance analysis, and comprehensive industry filtering based
    on various financial metrics.
    """

    def __init__(self, config_name: Optional[str] = None):
        """Initialize the IndustryFilter.

        Args:
            config_name: YAML config file name for API parameters
        """
        # Load both akshare and industry filter configs
        self.akshare_config, self.filter_config, self.file_config = load_config(
            config_name
        )

        # Apply class constants from config
        self.TARGET_PERIOD_COUNT = self.filter_config.target_period_count
        self.MIN_MAIN_NET_INFLOW_YI = self.filter_config.min_main_net_inflow_yi
        self.MAX_PRICE_CHANGE_PERCENT = self.filter_config.max_price_change_percent
        self.BATCH_SIZE = self.filter_config.batch_size
        self.DAYS_LOOKBACK_PERIOD = self.filter_config.days_lookback_period
        self.TRADING_DAYS_60 = self.filter_config.trading_days_60
        self.REPORT_DIR = self.filter_config.report_dir

        # Validate target_period_count is in period_count list
        if self.TARGET_PERIOD_COUNT not in self.akshare_config.period_count:
            raise ValueError(
                f"target_period_count ({self.TARGET_PERIOD_COUNT}) must be one of the "
                f"period_count values: {self.akshare_config.period_count}"
            )

        # Initialize date-related attributes
        self.end_date = None
        self.end_date_str = None

        # Initialize dynamic period counts (will be set based on available data)
        self.fund_period_counts = {}  # Will store actual counts for each period
        
        # Initialize centralized API handlers
        self.industry_api = StockBoardIndustryAPI(self.akshare_config)
        self.sector_flow_api = StockSectorFundFlowAPI()

        # Resolve dates in the class config
        self._resolve_akshare_config()

        # Initialize the actual end date for consistent date handling across methods
        self._initialize_end_date()

        # Initialize industry data and dates to ensure first_trading_date_str is available
        self._get_dates

    def _date_converter(self, date_str: str, period: str, period_count: int) -> str:
        """
        Convert date by adding/subtracting periods.
        Delegates to centralized API module.
        """
        return date_converter(date_str, period, period_count)

    def _resolve_akshare_config(self) -> None:
        """
        Resolve start_date, end_date, or other data in config based on the configuration rules.
        Modifies the config object in-place.
        """
        config = self.akshare_config

        # Define period unit: 日, 周, 月
        try:
            self.period_unit = config.period[0]
        except IndexError:
            error_msg = (
                f"Invalid period '{config.period}'. Must be one of: '日k', '周k', '月k'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set start_date & end_date
        # Use the maximum period_count for date calculations when needed
        max_period_count = max(config.period_count) if config.period_count else 29

        if config.start_date and config.end_date:
            # Both dates provided - use as is
            pass
        elif config.start_date and not config.end_date:
            # Only start_date provided - calculate end_date using max period
            config.end_date = self._date_converter(
                config.start_date, config.period, max_period_count
            )
        elif not config.start_date and config.end_date:
            # Only end_date provided - calculate start_date using max period
            config.start_date = self._date_converter(
                config.end_date, config.period, -max_period_count
            )
        else:
            # Both dates empty - use centralized date resolution
            start_date, end_date = resolve_date_range(config)
            config.start_date = start_date
            config.end_date = end_date

    def _initialize_end_date(self) -> None:
        """
        Initialize the actual end date for consistent date handling across all methods.

        This method determines the actual end date based on config.end_date and available data,
        ensuring all methods use the same reference date instead of iloc[-1].
        """
        try:
            # Use the end_date from config that was resolved in _resolve_akshare_config
            if self.akshare_config.end_date:
                self.end_date = datetime.strptime(
                    self.akshare_config.end_date, "%Y%m%d"
                )
                self.end_date_str = self.akshare_config.end_date
                logger.debug("Initialized end_date: %s", self.end_date_str)
            else:
                # Fallback to current date if no end_date is set
                self.end_date = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                self.end_date_str = self.end_date.strftime("%Y%m%d")
                logger.warning(
                    "No end_date configured, using current date: %s", self.end_date_str
                )

        except Exception as e:
            logger.error("Failed to initialize end_date: %s", str(e))
            # Set a fallback date if initialization fails
            self.end_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.end_date_str = self.end_date.strftime("%Y%m%d")

    def _get_analysis_columns(self) -> List[str]:
        """
        Generate analysis column names with dynamic periods from config.

        Returns:
            List of column names for analysis results including all periods
        """
        base_columns = [
            "行业",
        ]

        # Add dynamic fund flow columns for each period
        fund_flow_columns = []
        price_change_columns = []

        for period in self.akshare_config.period_count:
            fund_flow_columns.append(f"{period}{self.period_unit}主力净流入-总净额(亿)")
            price_change_columns.append(f"{period}{self.period_unit}涨跌幅(%)")

        # Add fixed period columns
        fixed_columns = [
            "60日涨跌幅(%)",
            "年初至今涨跌幅(%)",
        ]

        return base_columns + fund_flow_columns + price_change_columns + fixed_columns

    @cached_property
    def _get_dates(self) -> pd.Series:
        """
        Get industry names and date ranges for analysis with retry mechanism.

        This method retrieves the list of industry names and calculates
        appropriate date ranges for industry analysis, ensuring sufficient
        trading days for meaningful analysis.

        Returns:
            Series of industry names
        """
        # Get the list of industry names using centralized API
        industry_data = self.industry_api.fetch_names_sync()
        industry_arr = industry_data["板块名称"]

        # Get date related variables
        today = datetime.today().date()
        this_year = today.year
        # A consecutive date that greater than 60 trading days
        date_100_days_ago = today - timedelta(days=self.DAYS_LOOKBACK_PERIOD)
        # Define first_date, the range to fetch industry data
        if datetime(this_year, 1, 1).date() < date_100_days_ago:
            first_date = datetime(this_year, 1, 1)
        else:
            first_date = date_100_days_ago
        first_date_str = first_date.strftime("%Y%m%d")
        last_date_str = today.strftime("%Y%m%d")

        # Define last_date, the range to fetch industry data using centralized API
        hist_data = self.industry_api.fetch_hist_sync(
            symbol=industry_arr[0],  # use any industry to get the latest date
            start_date=first_date_str,
            end_date=last_date_str,
        )
        dates = hist_data["日期"].values

        self.last_date_str = dates[-1].replace("-", "")

        # Get the 1st trading date
        first_trading_date = datetime(datetime.today().year, 1, 1).date()
        while first_trading_date.strftime("%Y-%m-%d") not in dates:
            first_trading_date += timedelta(days=1)
        self.first_trading_date_str = first_trading_date.strftime("%Y-%m-%d")

        return industry_arr

    def _validate_industry_capital_flow_data_sync(self) -> bool:
        """
        Validate industry capital flow data.
        Delegates to centralized API module.

        Returns:
            bool if capital flow data available
        """
        return self.sector_flow_api.validate_availability(self.period_unit)

    def _fetch_industry_capital_flow_data_sync(
        self,
        industry_name: str,
    ) -> pd.DataFrame:
        """
        Fetch industry capital flow data with retry mechanism.

        Args:
            industry_name: Name of the industry to analyze

        Returns:
            DataFrame containing recent capital flow data for the industry
        """
        # Use centralized API with validation and filtering
        return self.sector_flow_api.fetch_with_validation(
            symbol=industry_name,
            period_unit=self.period_unit,
            end_date=self.akshare_config.end_date
        )

    def _fetch_industry_index_data_sync(
        self,
        industry_name: str,
    ) -> pd.DataFrame:
        """
        Fetch industry index historical data with retry mechanism.

        Args:
            industry_name: Name of the industry to analyze

        Returns:
            DataFrame containing historical index data for the industry
        """
        return self.industry_api.fetch_hist_sync(symbol=industry_name)

    async def process_single_industry_async(
        self,
        industry_name: str,
    ) -> Optional[List[Any]]:
        """
        Process a single industry asynchronously to calculate performance metrics.

        This method analyzes an industry's capital flow, index performance,
        and calculates key metrics for different time periods.

        Args:
            industry_name: Name of the industry to analyze

        Returns:
            List containing industry analysis results, or None if analysis fails
        """

        async with REQUEST_SEMAPHORE:
            try:
                # Fetch industry capital flow data with timeout
                try:
                    stock_sector_fund_flow_hist_df = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._fetch_industry_capital_flow_data_sync,
                            industry_name,
                        ),
                        timeout=45.0,  # 45 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout fetching capital flow data for industry %s, skipping",
                        industry_name,
                    )
                    return None
                # Calculate fund flows using centralized processing
                fund_flows = self.sector_flow_api.process_periods(
                    stock_sector_fund_flow_hist_df, 
                    self.akshare_config.period_count
                )

                # Fetch industry index data with timeout
                try:
                    stock_board_industry_hist_em_df = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._fetch_industry_index_data_sync,
                            industry_name,
                        ),
                        timeout=45.0,  # 45 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout fetching index data for industry %s, skipping",
                        industry_name,
                    )
                    return None
                # Get the index of the last trading date
                industry_last_index = stock_board_industry_hist_em_df["收盘"].iloc[-1]

                # Calculate price changes using centralized processing
                price_changes = self.industry_api.calculate_price_changes(
                    stock_board_industry_hist_em_df,
                    self.akshare_config.period_count,
                    close_column="收盘"
                )

                # Calculate fixed period changes (60-day and YTD)
                days_60_lookback = min(self.TRADING_DAYS_60 + 1, df_len)
                if days_60_lookback > 1:  # Need at least 2 data points
                    industry_60_index = stock_board_industry_hist_em_df["收盘"].iloc[
                        -days_60_lookback
                    ]
                    industry_index_change_perc_60 = round(
                        (industry_last_index - industry_60_index)
                        / industry_60_index
                        * 100,
                        2,
                    )
                else:
                    industry_index_change_perc_60 = 0.0
                # Get the index of the 1st trading date (with bounds checking)
                first_trading_date_filter = stock_board_industry_hist_em_df[
                    stock_board_industry_hist_em_df["日期"]
                    == self.first_trading_date_str
                ]
                if len(first_trading_date_filter) > 0:
                    industry_1st_trading_date_index = first_trading_date_filter[
                        "收盘"
                    ].iloc[0]
                else:
                    # Fallback to first available date if exact date not found
                    industry_1st_trading_date_index = stock_board_industry_hist_em_df[
                        "收盘"
                    ].iloc[0]
                # Calculate YTD price change
                industry_index_change_perc_ytd = round(
                    (industry_last_index - industry_1st_trading_date_index)
                    / industry_1st_trading_date_index
                    * 100,
                    2,
                )
                # Build return data with dynamic structure
                # Base data
                result = [industry_name]

                # Add fund flows for each period
                result.extend(fund_flows)

                # Add price changes for each period
                result.extend(price_changes)

                # Add fixed period data
                result.extend(
                    [industry_index_change_perc_60, industry_index_change_perc_ytd]
                )

                # Log the results (simplified for dynamic periods)
                logger.debug(
                    "%s: fund_flows=%s, price_changes=%s, 60d=%.2f%%, ytd=%.2f%%",
                    industry_name,
                    fund_flows,
                    price_changes,
                    industry_index_change_perc_60,
                    industry_index_change_perc_ytd,
                )

                return result

            except Exception as e:
                logger.error("Error processing %s: %s", industry_name, str(e))
                return None

    async def process_all_industries_async(
        self,
        industry_arr: pd.Series,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional[int] = None,
        _batch_task_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all industries concurrently with batch processing.

        This method orchestrates the analysis of all industries using
        batch processing to avoid overwhelming the API while maintaining
        good performance.

        Args:
            industry_arr: Series containing industry names
            _progress: Optional Rich Progress instance for hierarchical progress tracking
            _parent_task_id: Optional parent task ID for hierarchical progress structure
            _batch_task_id: Optional pre-created batch task ID for proper hierarchy display

        Returns:
            DataFrame containing analysis results for all industries
        """
        # Define columns for consistency
        columns = self._get_analysis_columns()

        all_industries_df = pd.DataFrame(columns=columns)

        # Process industries with some concurrency but not too much to avoid overwhelming the API
        batch_size = self.BATCH_SIZE
        total_batches = (len(industry_arr) + batch_size - 1) // batch_size

        # Use pre-created batch task if provided, otherwise create new one
        if _progress is not None and _batch_task_id is not None:
            # Make the pre-created batch task visible and configure it
            _progress.update(
                _batch_task_id,
                total=total_batches,
                visible=True,
                description="    🏢 Industry Filter: Processing batches",
            )
        elif _progress is not None:
            # Fallback: create new batch task (will appear at bottom)
            _batch_task_id = _progress.add_task(
                "🏢 Processing industry analysis batches",
                total=total_batches,
                visible=True,
            )

        for i in range(0, len(industry_arr), batch_size):
            batch = industry_arr[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "Processing industry batch %d/%d",
                batch_num,
                total_batches,
            )

            # Update batch progress if available
            if _progress is not None and _batch_task_id is not None:
                _progress.update(
                    _batch_task_id,
                    completed=batch_num - 1,
                    description=f"    🏢 Industry Filter: Processing batch {batch_num}/{total_batches} ({len(batch)} industries)",
                )

            # Create tasks for the current batch
            tasks = [
                self.process_single_industry_async(industry_name)
                for industry_name in batch
            ]

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    all_industries_df.loc[len(all_industries_df)] = result

            # Update batch progress after completion
            if _progress is not None and _batch_task_id is not None:
                _progress.advance(_batch_task_id)

        # Remove batch progress bar when finished (subtask cleanup)
        if _progress is not None and _batch_task_id is not None:
            _progress.update(
                _batch_task_id,
                description="    ✅ Industry Filter: All batches completed",
            )
            await asyncio.sleep(0.5)  # Brief display of completion
            _progress.remove_task(_batch_task_id)

        return all_industries_df

    def _save_reports(self, all_industries_df: pd.DataFrame) -> None:
        """
        Save analysis reports to CSV files.

        Args:
            all_industries_df: DataFrame containing all analysis results
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

        # Use the target period from config for filtering
        target_period_count = self.TARGET_PERIOD_COUNT
        
        # Define column names for filtering and sorting
        main_net_inflow_col = f"{target_period_count}{self.period_unit}主力净流入-总净额(亿)"
        price_change_col = f"{target_period_count}{self.period_unit}涨跌幅(%)"
        
        logger.info(
            "Applying filters using %d-day period: %s > %.1f and %s < %.1f",
            target_period_count,
            main_net_inflow_col, 
            self.MIN_MAIN_NET_INFLOW_YI,
            price_change_col,
            self.MAX_PRICE_CHANGE_PERCENT
        )
        
        # Check if the required columns exist
        missing_cols = []
        if main_net_inflow_col not in all_industries_df.columns:
            missing_cols.append(main_net_inflow_col)
        if price_change_col not in all_industries_df.columns:
            missing_cols.append(price_change_col)
            
        if missing_cols:
            logger.error("Missing required columns for filtering: %s", missing_cols)
            logger.info("Available columns: %s", list(all_industries_df.columns))
            raise ValueError(f"Missing columns for {target_period_count}-day filtering: {missing_cols}")

        # Sort all_industries_df (columns already have correct names from _get_analysis_columns)
        all_industries_df = all_industries_df.sort_values(
            by=[main_net_inflow_col, price_change_col],
            ascending=[False, True],
        )
        all_industries_df.reset_index(inplace=True, drop=True)

        # Apply additional filters to create filtered_df
        filtered_df = all_industries_df[
            (all_industries_df[main_net_inflow_col] > self.MIN_MAIN_NET_INFLOW_YI)
            & (all_industries_df[price_change_col] < self.MAX_PRICE_CHANGE_PERCENT)
        ].copy()
        
        logger.info(
            "Filtered from %d to %d industries using %d-day criteria",
            len(all_industries_df),
            len(filtered_df),
            target_period_count
        )

        # Sort filtered DataFrame
        filtered_df = filtered_df.sort_values(
            by=[main_net_inflow_col, price_change_col],
            ascending=[False, True],
        )
        filtered_df.reset_index(inplace=True, drop=True)

        # Output the raw report with error handling
        try:
            raw_report_path = f"{self.REPORT_DIR}/行业筛选报告-raw-{self.end_date_str}{config_name}.csv"
            all_industries_df.to_csv(raw_report_path, index=True)
            logger.info("Report saved to %s", raw_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save raw industry report: %s", str(e))
            raise

        # Output the filtered report with error handling
        try:
            filtered_report_path = (
                f"{self.REPORT_DIR}/行业筛选报告-{self.end_date_str}{config_name}.csv"
            )
            filtered_df.to_csv(filtered_report_path, index=True)
            logger.info("Filtered report saved to %s", filtered_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save filtered industry report: %s", str(e))
            raise

    async def run_analysis(
        self,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional[int] = None,
        _batch_task_id: Optional[int] = None,
    ) -> None:
        """
        Run the complete industry filtering pipeline.

        This method orchestrates the entire industry filtering process including
        data preparation, industry analysis, result filtering, and report generation.

        Args:
            _progress: Optional Rich Progress instance for hierarchical progress tracking
            _parent_task_id: Optional parent task ID for hierarchical progress structure
            _batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Get dates and industry data
        industry_arr = self._get_dates

        # Process all industries with progress tracking
        all_industries_df = await self.process_all_industries_async(
            industry_arr,
            _progress=_progress,
            _parent_task_id=_parent_task_id,
            _batch_task_id=_batch_task_id,
        )

        # Save reports (raw and filtered)
        self._save_reports(all_industries_df)


async def main(
    config_name: Optional[str] = None,
    _progress: Optional["Progress"] = None,
    _parent_task_id: Optional[int] = None,
    _batch_task_id: Optional[int] = None,
) -> None:
    """
    Main function to execute the complete industry filtering pipeline.

    This function creates an IndustryFilter instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        config_name: YAML config file name. If None, uses default config
        _progress: Optional Rich Progress instance for hierarchical progress tracking
        _parent_task_id: Optional parent task ID for hierarchical progress structure
        _batch_task_id: Optional pre-created batch task ID for proper hierarchy display
    """
    industry_filter = IndustryFilter(config_name)
    await industry_filter.run_analysis(
        _progress=_progress,
        _parent_task_id=_parent_task_id,
        _batch_task_id=_batch_task_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
