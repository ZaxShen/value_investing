"""
Stock filtering and analysis for Chinese equity markets.

This module provides a StockFilter class that encapsulates asynchronous functions
to filter and analyze Chinese stocks based on various financial metrics including
market cap, P/E ratio, and capital flow. It processes stocks by industry with
concurrency controls to respect API rate limits.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment

configure_environment()

import akshare as ak
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Initialize logger for this module
logger = get_logger("stock_filter")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class FileConfig(BaseModel):
    """
    Configuration model for file-related settings.

    This model handles file configuration metadata including config name,
    version, and description.
    """

    config_name: str = "PROD"
    description: str = ""
    version: str = ""


class StockIndividualFundFlowConfig(BaseModel):
    """
    Configuration model for ak.stock_individual_fund_flow API parameters.

    This model validates and provides default values for the API parameters.
    """

    stock: str = ""
    market: str = ""
    date: str = ""
    period_count: List[int] = [1, 5, 29]


class StockFilterConfig(BaseModel):
    """
    Configuration model for StockFilter class parameters.

    This model validates and provides default values for the StockFilter class constants.
    """

    max_market_cap_yi: int = 200
    min_pe_ratio: int = 0
    max_pe_ratio: int = 50
    min_main_net_inflow_yi: int = 1
    max_price_change_percent: int = 10
    target_period_count: int = 5
    batch_size: int = 3
    report_dir: str = "data/stocks/reports"


class Config(BaseModel):
    """
    Configuration model for nested YAML structure supporting both akshare and StockFilter configs.

    This model handles the nested structure from data/config/stock_filter.
    """

    akshare: dict = {}
    stock_filter: StockFilterConfig = StockFilterConfig()
    file_config: FileConfig = FileConfig()


def load_config(
    config_name: Optional[str] = None,
) -> tuple[StockIndividualFundFlowConfig, StockFilterConfig, FileConfig]:
    """
    Load nested configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        tuple: (akshare_config, stock_filter_config, file_config)

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
        config_dir = Path("data/config/stock_filter/")
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
        if "stock_individual_fund_flow" in configs.akshare:
            akshare_data = configs.akshare["stock_individual_fund_flow"]
            akshare_config = StockIndividualFundFlowConfig(**akshare_data)
        else:
            akshare_config = StockIndividualFundFlowConfig()

        # Extract stock filter config
        filter_config = configs.stock_filter

        # Extract file config
        file_config = configs.file_config

        return akshare_config, filter_config, file_config
    else:
        raise ValueError("Legacy flat config format not supported. Use nested format.")


class StockFilter:
    """
    A class to encapsulate stock filtering and analysis functionality.

    This class manages the industry-stock mapping and stock market data,
    providing asynchronous methods to filter and analyze Chinese stocks
    based on various financial metrics.
    """

    # Column definitions
    STOCK_DATA_COLS = [
        "代码",
        "名称",
        "总市值",
        "流通市值",
        "市盈率-动态",
        "市净率",
        "60日涨跌幅",
        "年初至今涨跌幅",
    ]

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
        config_name: Optional[str] = None,
    ):
        """
        Initialize the StockFilter with market data and configuration.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
            config_name: YAML config file name for API parameters
        """
        # Load configuration
        self.akshare_config, self.filter_config, self.file_config = load_config(
            config_name
        )

        # Apply class constants from config
        self.MAX_MARKET_CAP_YI = self.filter_config.max_market_cap_yi
        self.MIN_PE_RATIO = self.filter_config.min_pe_ratio
        self.MAX_PE_RATIO = self.filter_config.max_pe_ratio
        self.MIN_MAIN_NET_INFLOW_YI = self.filter_config.min_main_net_inflow_yi
        self.MAX_PRICE_CHANGE_PERCENT = self.filter_config.max_price_change_percent
        self.TARGET_PERIOD_COUNT = self.filter_config.target_period_count
        self.BATCH_SIZE = self.filter_config.batch_size
        self.REPORT_DIR = self.filter_config.report_dir

        # Validate target_period_count is in period_count list
        if self.TARGET_PERIOD_COUNT not in self.akshare_config.period_count:
            raise ValueError(
                f"target_period_count ({self.TARGET_PERIOD_COUNT}) must be one of the "
                f"period_count values: {self.akshare_config.period_count}"
            )

        # Store market data
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df
        self.stock_market_df_filtered: Optional[pd.DataFrame] = None
        self.industry_arr: Optional[np.ndarray] = None

        # Initialize date-related attributes for consistent date handling
        self._initialize_analysis_date()

    def _initialize_analysis_date(self) -> None:
        """
        Initialize the actual analysis date for consistent date handling across all methods.

        This method determines the actual analysis date based on config.date and available data,
        ensuring all methods use the same reference date for fund flow calculations.
        """
        from datetime import datetime

        try:
            if self.akshare_config.date:
                # Use the configured date
                self.analysis_date = datetime.strptime(
                    self.akshare_config.date, "%Y%m%d"
                )
                self.analysis_date_str = self.akshare_config.date
                logger.debug(
                    "Using configured analysis date: %s", self.analysis_date_str
                )
            else:
                # Use current date as fallback
                self.analysis_date = datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                self.analysis_date_str = self.analysis_date.strftime("%Y%m%d")
                logger.info(
                    "No analysis date configured, using current date: %s",
                    self.analysis_date_str,
                )

        except Exception as e:
            logger.error("Failed to initialize analysis date: %s", str(e))
            # Set a fallback date if initialization fails
            from datetime import datetime

            self.analysis_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.analysis_date_str = self.analysis_date.strftime("%Y%m%d")

    def _get_analysis_columns(self) -> List[str]:
        """
        Generate analysis column names with dynamic periods from config.

        Returns:
            List of column names for analysis results including all periods
        """
        base_columns = [
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
        for period in self.akshare_config.period_count:
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

    def _save_reports(self, all_industries_df: pd.DataFrame) -> None:
        """
        Save analysis reports to CSV files.

        Args:
            all_industries_df: DataFrame containing all analysis results
        """
        # Use the consistent analysis date initialized during construction
        last_date_str = self.analysis_date_str

        # Output the all_industries_df to a CSV file with error handling
        try:
            raw_report_path = f"{self.REPORT_DIR}/股票筛选报告-raw-{last_date_str}.csv"
            all_industries_df.to_csv(raw_report_path, index=True)
            logger.info("Report saved to %s", raw_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save raw report: %s", str(e))
            raise

        # Apply additional filters using the target period from config
        target_period_count = self.TARGET_PERIOD_COUNT
        main_net_inflow_col = f"{target_period_count}日主力净流入-总净额(亿)"
        price_change_col = f"{target_period_count}日涨跌幅(%)"
        
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
        
        df = all_industries_df[
            (all_industries_df[main_net_inflow_col] > self.MIN_MAIN_NET_INFLOW_YI)
            & (all_industries_df[price_change_col] < self.MAX_PRICE_CHANGE_PERCENT)
        ]
        
        logger.info(
            "Filtered from %d to %d stocks using %d-day criteria",
            len(all_industries_df),
            len(df),
            target_period_count
        )

        # Sort the DataFrame by price change percentage
        df = df.sort_values(by=[price_change_col])
        df.reset_index(inplace=True, drop=True)

        # Output the filtered DataFrame to a CSV file with error handling
        try:
            filtered_report_path = f"{self.REPORT_DIR}/股票筛选报告-{last_date_str}.csv"
            df.to_csv(filtered_report_path, index=True)
            logger.info("Filtered report saved to %s", filtered_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save filtered report: %s", str(e))
            raise

    def prepare_stock_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare and filter stock market data based on market cap and P/E ratio criteria.

        This method loads stock market data and industry mapping, applies filtering
        criteria (market cap < 20 billion RMB, P/E ratio between 0-50), and returns
        the filtered dataset along with unique industry names.

        Returns:
            Tuple containing:
                - stock_market_df_filtered: DataFrame with filtered stock data including
                  columns for industry, stock code, name, market cap, P/E ratio, etc.
                - industry_arr: Array of unique industry names for further processing
        """

        # Use class constant for columns

        # Filter stock market data using class constants
        stock_market_df_filtered = self.stock_zh_a_spot_em_df[
            (self.stock_zh_a_spot_em_df["总市值"] < self.MAX_MARKET_CAP_YI * 1e8)
            & (self.stock_zh_a_spot_em_df["市盈率-动态"] > self.MIN_PE_RATIO)
            & (self.stock_zh_a_spot_em_df["市盈率-动态"] < self.MAX_PE_RATIO)
        ]
        # Extract required data
        stock_market_df_filtered = stock_market_df_filtered[self.STOCK_DATA_COLS]

        # Inner join industry_stock_mapping_df with stock_market_df_filtered
        stock_market_df_filtered = pd.merge(
            self.industry_stock_mapping_df,
            stock_market_df_filtered,
            on="代码",
            how="inner",
        )

        # Organize the columns
        stock_market_df_filtered.columns = ["行业"] + self.STOCK_DATA_COLS

        # Get unique industry names
        industry_arr = stock_market_df_filtered["行业"].unique()

        # Store for later use
        self.stock_market_df_filtered = stock_market_df_filtered
        self.industry_arr = industry_arr

        logger.info(
            "Loaded %d stocks across %d industries",
            len(stock_market_df_filtered),
            len(industry_arr),
        )
        return stock_market_df_filtered, industry_arr

    def _fetch_stock_fund_flow_sync(self, stock_code: str, market: str) -> pd.DataFrame:
        """
        Fetch stock individual fund flow data with retry mechanism.

        Args:
            stock_code: Stock code (e.g., "000001")
            market: Market identifier (e.g., "sz" for Shenzhen, "sh" for Shanghai)

        Returns:
            DataFrame containing historical fund flow data for the specified stock
        """
        return API_RETRY_CONFIG.retry(
            ak.stock_individual_fund_flow, stock=stock_code, market=market
        )

    async def process_single_stock_async(
        self,
        stock_code: str,
        stock_name: str,
        industry_name: str,
    ) -> Optional[List[Any]]:
        """
        Process a single stock asynchronously with multi-period fund flow analysis.

        This method fetches historical fund flow data for a stock, calculates
        key financial metrics for multiple time periods, and applies filtering criteria.
        It respects API rate limits using semaphores.

        Args:
            stock_code: Stock code (e.g., "000001")
            stock_name: Stock name for display purposes
            industry_name: Industry classification

        Returns:
            List containing stock analysis results with data for all configured periods,
            or None if stock doesn't meet criteria or has insufficient data
        """
        async with REQUEST_SEMAPHORE:
            logger.debug(
                "Processing %s (%s) in %s industry",
                stock_name,
                stock_code,
                industry_name,
            )

            # Determine the market based on the stock code
            market = self._get_market_by_stock_code(stock_code)

            try:
                # Ensure data is prepared
                if self.stock_market_df_filtered is None:
                    raise ValueError(
                        "Stock data not prepared. Call prepare_stock_data() first."
                    )

                # Extract the stock's market data
                stock_data = self.stock_market_df_filtered[
                    self.stock_market_df_filtered["代码"] == stock_code
                ].iloc[0]  # More efficient than multiple queries

                stock_total_market_value = round(stock_data["总市值"] / 1e8, 0)
                stock_circulating_market_value = round(stock_data["流通市值"] / 1e8, 0)
                stock_pe_dynamic = stock_data["市盈率-动态"]
                stock_pb = stock_data["市净率"]
                stock_60d_change = stock_data["60日涨跌幅"]
                stock_ytd_change = stock_data["年初至今涨跌幅"]

                # Extract the historical data of the stock (async) with timeout
                try:
                    stock_individual_fund_flow_df = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._fetch_stock_fund_flow_sync, stock_code, market
                        ),
                        timeout=60.0,  # 60 second timeout for fund flow data
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout fetching fund flow data for %s (%s) after 60 seconds, skipping",
                        stock_name,
                        stock_code,
                    )
                    return None

                # Get the maximum period to ensure we have enough data
                max_period = max(self.akshare_config.period_count)

                if len(stock_individual_fund_flow_df) < max_period:
                    logger.warning(
                        "Skipping %s (%s) due to insufficient data for max period %d days",
                        stock_name,
                        stock_code,
                        max_period,
                    )
                    return None

                # For multi-period analysis, we need at least max_period + 1 data points
                full_data_needed = max_period + 1
                if len(stock_individual_fund_flow_df) >= full_data_needed:
                    # Use full data for calculations
                    analysis_df = stock_individual_fund_flow_df
                else:
                    # Use what we have with warning
                    logger.warning(
                        "Limited data for %s (%s): have %d days, need %d for optimal analysis",
                        stock_name,
                        stock_code,
                        len(stock_individual_fund_flow_df),
                        full_data_needed,
                    )
                    analysis_df = stock_individual_fund_flow_df

                # Calculate fund flows and price changes for each period
                fund_flows = []
                price_changes = []

                for period in self.akshare_config.period_count:
                    # Calculate fund flow for this period (sum of last N days)
                    if len(analysis_df) >= period:
                        period_fund_df = analysis_df.iloc[-period:]
                        fund_flow = round(
                            period_fund_df["主力净流入-净额"].sum() / 1e8, 1
                        )
                    else:
                        fund_flow = 0.0
                    fund_flows.append(fund_flow)

                    # Calculate price change for this period (N days ago vs today)
                    required_data_points = period + 1
                    if len(analysis_df) >= required_data_points:
                        first_price = analysis_df.iloc[-required_data_points]["收盘价"]
                        last_price = analysis_df.iloc[-1]["收盘价"]
                        if first_price == 0:
                            price_change = 0.0
                        else:
                            price_change = round(
                                (last_price - first_price) / first_price * 100, 1
                            )
                    else:
                        price_change = 0.0
                    price_changes.append(price_change)

                # Build return data with dynamic structure
                base_data = [
                    industry_name,
                    stock_code,
                    stock_name,
                    stock_total_market_value,
                    stock_circulating_market_value,
                    stock_pe_dynamic,
                    stock_pb,
                    analysis_df.iloc[-1]["收盘价"],  # Latest closing price
                ]

                # Add fund flows and price changes for all periods
                result = (
                    base_data
                    + fund_flows
                    + price_changes
                    + [stock_60d_change, stock_ytd_change]
                )

                return result

            except Exception as e:
                logger.error(
                    "Error processing %s (%s): %s", stock_name, stock_code, str(e)
                )
                return None

    async def process_single_industry_async(self, industry_name: str) -> pd.DataFrame:
        """
        Analyze stocks in a given industry with multi-period fund flow and price metrics.

        This method processes all stocks within a specific industry, fetching
        their fund flow data and calculating key financial metrics for multiple
        time periods. It uses concurrent processing to improve performance while
        respecting API limits.

        Args:
            industry_name: The industry name to analyze (e.g., "银行")

        Returns:
            DataFrame containing analysis results for all stocks in the industry,
            with columns for market cap, P/E ratio, and fund flows/price changes
            for all configured periods
        """
        # Ensure data is prepared
        if self.stock_market_df_filtered is None:
            raise ValueError(
                "Stock data not prepared. Call prepare_stock_data() first."
            )

        # Extract all qualified stocks from stock_market_df_filtered
        stocks = self.stock_market_df_filtered[
            self.stock_market_df_filtered["行业"] == industry_name
        ][["代码", "名称"]]

        # Define columns for consistency
        columns = self._get_analysis_columns()

        df = pd.DataFrame(columns=columns)

        # Create tasks for concurrent processing
        tasks = []
        for row in stocks.itertuples():
            task = self.process_single_stock_async(
                str(row.代码), str(row.名称), industry_name
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and add to DataFrame
        for result in results:
            if result is not None and not isinstance(result, Exception):
                try:
                    # Ensure result is a list before adding to DataFrame
                    if isinstance(result, list):
                        df.loc[len(df)] = result
                    else:
                        logger.warning("Unexpected result type: %s", type(result))
                except Exception as e:
                    logger.warning("Failed to add result to DataFrame: %s", str(e))

        return df

    async def process_all_industries_async(
        self,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional["TaskID"] = None,  # noqa: ARG002
        batch_task_id: Optional["TaskID"] = None,
    ) -> pd.DataFrame:
        """
        Process all industries concurrently with batch processing and rate limiting.

        This method orchestrates the analysis of all industries by processing them
        in batches to avoid overwhelming the API. It implements proper error handling
        and result aggregation.

        Args:
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display

        Returns:
            DataFrame containing consolidated analysis results from all industries,
            with complete financial metrics and fund flow data
        """
        # Define columns for consistency
        columns = self._get_analysis_columns()

        # Store results in a list to avoid repeated concatenation
        result_dfs = []

        # Ensure data is prepared
        if self.industry_arr is None:
            raise ValueError(
                "Industry data not prepared. Call prepare_stock_data() first."
            )

        # Process industries with some concurrency but not too much to avoid overwhelming the API
        batch_size = self.BATCH_SIZE
        total_batches = (len(self.industry_arr) + batch_size - 1) // batch_size

        # Use pre-created batch task if provided, otherwise create new one
        if progress is not None and batch_task_id is not None:
            # Make the pre-created batch task visible and configure it
            progress.update(
                batch_task_id,
                total=total_batches,
                visible=True,
                description="    📊 Stock Filter: Processing batches",
            )
        elif progress is not None:
            # Fallback: create new batch task (will appear at bottom)
            batch_task_id = progress.add_task(
                "📊 Processing industry batches", total=total_batches, visible=True
            )

        for i in range(0, len(self.industry_arr), batch_size):
            batch = self.industry_arr[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "Processing industry batch %d/%d",
                batch_num,
                total_batches,
            )

            # Update batch progress if available
            if progress is not None and batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    completed=batch_num - 1,
                    description=f"    📊 Stock Filter: Processing batch {batch_num}/{total_batches} ({len(batch)} industries)",
                )

            # Create tasks for the current batch
            tasks = [
                self.process_single_industry_async(industry_name)
                for industry_name in batch
            ]

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid results
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    try:
                        # Check if result is a DataFrame and not empty
                        # Use explicit type check to satisfy type checker
                        if (
                            hasattr(result, "empty")
                            and hasattr(result, "iloc")
                            and not isinstance(result, BaseException)
                        ):
                            # Now type checker knows result is not an Exception
                            if not result.empty:  # type: ignore[attr-defined]
                                result_dfs.append(result)
                    except Exception as e:
                        logger.warning("Error checking result validity: %s", str(e))

            # Update batch progress after completion
            if progress is not None and batch_task_id is not None:
                progress.advance(batch_task_id)

        # Remove batch progress bar when finished (subtask cleanup)
        if progress is not None and batch_task_id is not None:
            progress.update(
                batch_task_id, description="    ✅ Stock Filter: All batches completed"
            )
            await asyncio.sleep(0.5)  # Brief display of completion
            progress.remove_task(batch_task_id)

        # Concatenate all results at once, or return empty DataFrame if no results
        if result_dfs:
            all_industries_df = pd.concat(result_dfs, ignore_index=True)
        else:
            all_industries_df = pd.DataFrame(columns=columns)
            all_industries_df["代码"] = all_industries_df["代码"].astype(str)

        return all_industries_df

    async def run_analysis(
        self,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional["TaskID"] = None,  # noqa: ARG002
        _batch_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Run the complete stock filtering pipeline.

        This method orchestrates the entire stock filtering process including
        data preparation, industry analysis, result filtering, and report generation.

        Args:
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Prepare data
        self.prepare_stock_data()

        # Process all industries with progress tracking
        all_industries_df = await self.process_all_industries_async(
            _progress,
            _parent_task_id,
            _batch_task_id,
        )

        # Save reports (raw and filtered)
        self._save_reports(all_industries_df)


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    config_name: Optional[str] = None,
    _progress: Optional["Progress"] = None,
    _parent_task_id: Optional["TaskID"] = None,  # noqa: ARG002
    _batch_task_id: Optional["TaskID"] = None,
    *args,
    **kwargs,
) -> None:
    """
    Main async function to execute the complete stock filtering pipeline.

    This function creates a StockFilter instance and runs the complete analysis.
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
    stock_filter = StockFilter(
        industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name, *args, **kwargs
    )
    await stock_filter.run_analysis(
        _progress=_progress,
        _parent_task_id=_parent_task_id,
        _batch_task_id=_batch_task_id,
    )
