"""
Stock filtering and analysis for Chinese equity markets.

This module provides a StockFilter class that encapsulates asynchronous functions
to filter and analyze Chinese stocks based on various financial metrics including
market cap, P/E ratio, and capital flow. It processes stocks by industry with
concurrency controls to respect API rate limits.
"""

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
configure_environment()

import akshare as ak
import numpy as np
import pandas as pd

from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("stock_filter")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class StockFilter:
    """
    A class to encapsulate stock filtering and analysis functionality.

    This class manages the industry-stock mapping and stock market data,
    providing asynchronous methods to filter and analyze Chinese stocks
    based on various financial metrics.
    """

    # Class constants for filtering criteria
    MAX_MARKET_CAP_YI = 200  # Maximum market cap in 100 million RMB
    MIN_PE_RATIO = 0
    MAX_PE_RATIO = 50
    MIN_MAIN_NET_INFLOW_YI = 1  # Minimum main net inflow in 100 million RMB
    MAX_PRICE_CHANGE_PERCENT = 10
    BATCH_SIZE = 3

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

    REPORT_DIR = "data/stocks/reports"

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
    ):
        """
        Initialize the StockFilter with market data.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df
        self.stock_market_df_filtered = None
        self.industry_arr = None

    def _get_analysis_columns(self, days: int) -> List[str]:
        """
        Generate analysis column names with dynamic days parameter.

        Args:
            days: Number of days for fund flow analysis

        Returns:
            List of column names for analysis results
        """
        return [
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

    def _save_reports(self, all_industries_df: pd.DataFrame, days: int) -> None:
        """
        Save analysis reports to CSV files.

        Args:
            all_industries_df: DataFrame containing all analysis results
            days: Number of days used for analysis (for filtering and naming)
        """
        # Define the report date with retry mechanism
        sector_fund_flow = API_RETRY_CONFIG.retry(
            ak.stock_sector_fund_flow_hist, symbol="证券"
        )
        last_date = sector_fund_flow.iloc[-1]["日期"]
        last_date_str = last_date.strftime("%Y%m%d")

        # Output the all_industries_df to a CSV file with error handling
        try:
            raw_report_path = f"{self.REPORT_DIR}/股票筛选报告-raw-{last_date_str}.csv"
            all_industries_df.to_csv(raw_report_path, index=True)
            logger.info("Report saved to %s", raw_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save raw report: %s", str(e))
            raise

        # Apply additional filters to all_industries_df
        main_net_inflow_col = f"{days}日主力净流入-总净额(亿)"
        price_change_col = f"{days}日涨跌幅(%)"
        df = all_industries_df[
            (all_industries_df[main_net_inflow_col] > self.MIN_MAIN_NET_INFLOW_YI)
            & (all_industries_df[price_change_col] < self.MAX_PRICE_CHANGE_PERCENT)
        ]

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
        days: int = 29,
    ) -> Optional[List[Any]]:
        """
        Process a single stock asynchronously with fund flow analysis.

        This method fetches historical fund flow data for a stock, calculates
        key financial metrics, and applies filtering criteria. It respects API
        rate limits using semaphores.

        Args:
            stock_code: Stock code (e.g., "000001")
            stock_name: Stock name for display purposes
            industry_name: Industry classification
            days: Number of days to analyze for fund flow (default: 29)

        Returns:
            List containing stock analysis results, or None if stock doesn't
            meet criteria or has insufficient data
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
                        asyncio.to_thread(self._fetch_stock_fund_flow_sync, stock_code, market),
                        timeout=60.0  # 60 second timeout for fund flow data
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout fetching fund flow data for %s (%s) after 60 seconds, skipping",
                        stock_name, stock_code
                    )
                    return None

                if len(stock_individual_fund_flow_df) < days:
                    logger.warning(
                        "Skipping %s (%s) due to insufficient data for the last %d days",
                        stock_name,
                        stock_code,
                        days,
                    )
                    return None

                stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[
                    -days:
                ]

                # Get the main net inflow data
                stock_main_net_flow = round(
                    stock_individual_fund_flow_df["主力净流入-净额"].sum() / 1e8, 1
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
                    (stock_last_price - stock_1st_price) / stock_1st_price * 100, 1
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
                logger.error(
                    "Error processing %s (%s): %s", stock_name, stock_code, str(e)
                )
                return None

    async def process_single_industry_async(
        self, industry_name: str, days: int = 29
    ) -> pd.DataFrame:
        """
        Analyze stocks in a given industry by extracting fund flow and price metrics.

        This method processes all stocks within a specific industry, fetching
        their fund flow data and calculating key financial metrics. It uses
        concurrent processing to improve performance while respecting API limits.

        Args:
            industry_name: The industry name to analyze (e.g., "银行")
            days: Number of days to analyze for fund flow calculation (default: 29)

        Returns:
            DataFrame containing analysis results for all stocks in the industry,
            with columns for market cap, P/E ratio, fund flow, and price changes
        """
        # Extract all qualified stocks from stock_market_df_filtered
        stocks = self.stock_market_df_filtered[
            self.stock_market_df_filtered["行业"] == industry_name
        ][["代码", "名称"]]

        # Define columns for consistency
        columns = self._get_analysis_columns(days)

        df = pd.DataFrame(columns=columns)

        # Create tasks for concurrent processing
        tasks = []
        for row in stocks.itertuples():
            task = self.process_single_stock_async(
                row.代码, row.名称, industry_name, days
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and add to DataFrame
        for result in results:
            if result is not None and not isinstance(result, Exception):
                df.loc[len(df)] = result

        return df

    async def process_all_industries_async(
        self,
        days: int = 29,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all industries concurrently with batch processing and rate limiting.

        This method orchestrates the analysis of all industries by processing them
        in batches to avoid overwhelming the API. It implements proper error handling
        and result aggregation.

        Args:
            days: Number of days to analyze for fund flow (default: 29)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display

        Returns:
            DataFrame containing consolidated analysis results from all industries,
            with complete financial metrics and fund flow data
        """
        # Define columns for consistency
        columns = self._get_analysis_columns(days)

        # Store results in a list to avoid repeated concatenation
        result_dfs = []

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
                self.process_single_industry_async(industry_name, days)
                for industry_name in batch
            ]

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid results
            for result in batch_results:
                if (
                    result is not None
                    and not isinstance(result, Exception)
                    and not result.empty
                ):
                    result_dfs.append(result)

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
        days: int = 29,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> None:
        """
        Run the complete stock filtering pipeline.

        This method orchestrates the entire stock filtering process including
        data preparation, industry analysis, result filtering, and report generation.

        Args:
            days: Number of days to analyze for fund flow (default: 29)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Prepare data
        self.prepare_stock_data()

        # Process all industries with progress tracking
        all_industries_df = await self.process_all_industries_async(
            days,
            progress,
            parent_task_id,
            batch_task_id,
        )

        # Save reports (raw and filtered)
        self._save_reports(all_industries_df, days)


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    progress: Optional["Progress"] = None,
    parent_task_id: Optional[int] = None,
    batch_task_id: Optional[int] = None,
) -> None:
    """
    Main async function to execute the complete stock filtering pipeline.

    This function creates a StockFilter instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        industry_stock_mapping_df: DataFrame containing industry-stock mapping
        stock_zh_a_spot_em_df: DataFrame containing stock market data
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display
    """
    stock_filter = StockFilter(industry_stock_mapping_df, stock_zh_a_spot_em_df)
    await stock_filter.run_analysis(29, progress, parent_task_id, batch_task_id)
