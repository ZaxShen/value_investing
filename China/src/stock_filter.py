"""
Stock filtering and analysis for Chinese equity markets.

This module provides a StockAnalyzer class that encapsulates asynchronous functions
to filter and analyze Chinese stocks based on various financial metrics including
market cap, P/E ratio, and capital flow. It processes stocks by industry with
concurrency controls to respect API rate limits.
"""

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd

from src.utilities.logger import get_logger

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("stock_filter")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class StockAnalyzer:
    """
    A class to encapsulate stock filtering and analysis functionality.

    This class manages the industry-stock mapping and stock market data,
    providing asynchronous methods to filter and analyze Chinese stocks
    based on various financial metrics.
    """

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
    ):
        """
        Initialize the StockAnalyzer with market data.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df
        self.stock_market_df_filtered = None
        self.industry_arr = None

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

        # Define required COLS
        COLS = [
            "代码",
            "名称",
            "总市值",
            "流通市值",
            "市盈率-动态",
            "市净率",
            "60日涨跌幅",
            "年初至今涨跌幅",
        ]

        # Filter stock market data
        # 总市值 < 200 亿, 0 < 动态市盈率 < 50
        stock_market_df_filtered = self.stock_zh_a_spot_em_df[
            (self.stock_zh_a_spot_em_df["总市值"] < 200 * 1e8)
            & (self.stock_zh_a_spot_em_df["市盈率-动态"] > 0)
            & (self.stock_zh_a_spot_em_df["市盈率-动态"] < 50)
        ]
        # Extract required data
        stock_market_df_filtered = stock_market_df_filtered[COLS]

        # Inner join industry_stock_mapping_df with stock_market_df_filtered
        stock_market_df_filtered = pd.merge(
            self.industry_stock_mapping_df,
            stock_market_df_filtered,
            on="代码",
            how="inner",
        )

        # Organize the columns
        stock_market_df_filtered.columns = ["行业"] + COLS

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
        Fetch stock individual fund flow data - synchronous version.

        Args:
            stock_code: Stock code (e.g., "000001")
            market: Market identifier (e.g., "sz" for Shenzhen, "sh" for Shanghai)

        Returns:
            DataFrame containing historical fund flow data for the specified stock
        """
        return ak.stock_individual_fund_flow(stock=stock_code, market=market)

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
            if stock_code.startswith("6"):
                market = "sh"
            elif stock_code.startswith("0") or stock_code.startswith("3"):
                market = "sz"
            else:
                market = "bj"

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

                # Extract the historical data of the stock (async)
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

                stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[
                    -days:
                ]

                # Get the main net inflow data
                stock_main_net_flow = round(
                    stock_individual_fund_flow_df["主力净流入-净额"].sum() / 1e8, 1
                )

                # Calculate change percentage
                stock_1st_price = stock_individual_fund_flow_df.iloc[0]["收盘价"]
                stock_last_price = stock_individual_fund_flow_df.iloc[-1]["收盘价"]
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
        columns = [
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
        columns = [
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

        # Store results in a list to avoid repeated concatenation
        result_dfs = []

        # Process industries with some concurrency but not too much to avoid overwhelming the API
        batch_size = 3
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

        # Define the report date
        last_date = ak.stock_sector_fund_flow_hist(symbol="证券").iloc[-1]["日期"]
        last_date_str = last_date.strftime("%Y%m%d")

        # Define the directory for reports
        REPORT_DIR = "data/stocks/reports"

        # Output the all_industries_df to a CSV file
        all_industries_df.to_csv(
            f"{REPORT_DIR}/股票筛选报告-raw-{last_date_str}.csv", index=True
        )
        logger.info(
            "Report saved to %s/股票筛选报告-raw-%s.csv", REPORT_DIR, last_date_str
        )

        # Apply additional filters to all_industries_df
        df = all_industries_df[
            (all_industries_df[f"{days}日主力净流入-总净额(亿)"] > 1)
            & (all_industries_df[f"{days}日涨跌幅(%)"] < 10)
        ]

        # Sort the DataFrame by pe and {days} change percentage
        df = df.sort_values(by=[f"{days}日涨跌幅(%)"])
        df.reset_index(inplace=True, drop=True)

        # Output the filtered DataFrame to a CSV file
        df.to_csv(f"{REPORT_DIR}/股票筛选报告-{last_date_str}.csv", index=True)
        logger.info(
            "Filtered report saved to %s/股票筛选报告-%s.csv", REPORT_DIR, last_date_str
        )


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    progress: Optional["Progress"] = None,
    parent_task_id: Optional[int] = None,
    batch_task_id: Optional[int] = None,
) -> None:
    """
    Main async function to execute the complete stock filtering pipeline.

    This function creates a StockAnalyzer instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        industry_stock_mapping_df: DataFrame containing industry-stock mapping
        stock_zh_a_spot_em_df: DataFrame containing stock market data
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display
    """
    analyzer = StockAnalyzer(industry_stock_mapping_df, stock_zh_a_spot_em_df)
    await analyzer.run_analysis(29, progress, parent_task_id, batch_task_id)
