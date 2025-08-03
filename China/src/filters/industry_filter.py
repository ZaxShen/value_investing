"""
Industry analysis and filtering for Chinese equity markets.

This module provides an IndustryFilter class that encapsulates comprehensive
analysis of industry sectors including fund flow analysis, index performance
tracking, and industry-level filtering. It generates detailed reports on
industry performance and capital flows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
configure_environment()

import akshare as ak
import pandas as pd

from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("industry_filter")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class IndustryFilter:
    """
    A class to encapsulate industry filtering and analysis functionality.

    This class manages industry data analysis including fund flow tracking,
    index performance analysis, and comprehensive industry filtering based
    on various financial metrics.
    """

    # Class constants for filtering criteria
    MIN_MAIN_NET_INFLOW_YI = 20  # Minimum main net inflow in 100 million RMB
    MAX_PRICE_CHANGE_PERCENT = 8  # Maximum price change percentage
    BATCH_SIZE = 3
    DAYS_ANALYSIS_PERIOD = 29  # Default analysis period in days
    DAYS_LOOKBACK_PERIOD = 100  # Days to look back for sufficient trading data
    TRADING_DAYS_60 = 60  # 60 trading days for analysis

    # Report directory
    REPORT_DIR = "data/stocks/reports"

    def __init__(self):
        """Initialize the IndustryFilter."""
        pass

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
            f"{days}日主力净流入-总净额(亿)",
            f"{days}日涨跌幅(%)",
            "60日涨跌幅(%)",
            "年初至今涨跌幅(%)",
        ]

    def get_dates(self) -> Tuple[pd.Series, str, str, str]:
        """
        Get industry names and date ranges for analysis with retry mechanism.

        This method retrieves the list of industry names and calculates
        appropriate date ranges for industry analysis, ensuring sufficient
        trading days for meaningful analysis.

        Returns:
            Tuple containing:
                - industry_arr: Series of industry names
                - first_date_str: Start date in %Y%m%d format
                - last_date_str: End date in %Y%m%d format
                - first_trading_date_str: First trading date in %Y-%m-%d format
        """
        # Get the list of industry names with retry
        industry_data = API_RETRY_CONFIG.retry(ak.stock_board_industry_name_em)
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

        # Define last_date, the range to fetch industry data with retry
        hist_data = API_RETRY_CONFIG.retry(
            ak.stock_board_industry_hist_em,
            symbol=industry_arr[0],
            start_date=first_date_str,
            end_date=last_date_str,
            period="日k",
            adjust="",
        )
        dates = hist_data["日期"].values

        last_date_str = dates[-1].replace("-", "")

        # Get the 1st trading date
        first_trading_date = datetime(datetime.today().year, 1, 1).date()
        while first_trading_date.strftime("%Y-%m-%d") not in dates:
            first_trading_date += timedelta(days=1)
        first_trading_date_str = first_trading_date.strftime("%Y-%m-%d")

        return industry_arr, first_date_str, last_date_str, first_trading_date_str

    def _fetch_industry_capital_flow_data_sync(
        self, industry_name: str, days: int
    ) -> pd.DataFrame:
        """
        Fetch industry capital flow data with retry mechanism.

        Args:
            industry_name: Name of the industry to analyze
            days: Number of recent days to fetch data for

        Returns:
            DataFrame containing recent capital flow data for the industry
        """
        flow_data = API_RETRY_CONFIG.retry(
            ak.stock_sector_fund_flow_hist, symbol=industry_name
        )
        return flow_data.iloc[-days:]

    def _fetch_industry_index_data_sync(
        self,
        industry_name: str,
        first_date_str: str,
        last_date_str: str,
    ) -> pd.DataFrame:
        """
        Fetch industry index historical data with retry mechanism.

        Args:
            industry_name: Name of the industry to analyze
            first_date_str: Start date in %Y%m%d format
            last_date_str: End date in %Y%m%d format

        Returns:
            DataFrame containing historical index data for the industry
        """
        return API_RETRY_CONFIG.retry(
            ak.stock_board_industry_hist_em,
            symbol=industry_name,
            start_date=first_date_str,
            end_date=last_date_str,
            period="日k",
            adjust="",
        )

    async def process_single_industry_async(
        self,
        industry_name: str,
        first_date_str: str,
        last_date_str: str,
        first_trading_date_str: str,
        days: int = 29,
    ) -> Optional[List[Any]]:
        """
        Process a single industry asynchronously to calculate performance metrics.

        This method analyzes an industry's capital flow, index performance,
        and calculates key metrics for different time periods.

        Args:
            industry_name: Name of the industry to analyze
            first_date_str: Start date in %Y%m%d format
            last_date_str: End date in %Y%m%d format
            first_trading_date_str: First trading date in %Y-%m-%d format
            days: Number of days to analyze (default: 29)

        Returns:
            List containing industry analysis results, or None if analysis fails
        """

        async with REQUEST_SEMAPHORE:
            try:
                # Fetch industry capital flow data with timeout
                try:
                    stock_sector_fund_flow_hist_df = await asyncio.wait_for(
                        asyncio.to_thread(self._fetch_industry_capital_flow_data_sync, industry_name, days),
                        timeout=45.0  # 45 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching capital flow data for industry %s, skipping", industry_name)
                    return None
                # Calculate main net flow
                industry_main_net_flow = stock_sector_fund_flow_hist_df[
                    "主力净流入-净额"
                ].sum()
                industry_main_net_flow = round(
                    industry_main_net_flow / 1e8, 1
                )  # Convert to 100M

                # Fetch industry index data with timeout
                try:
                    stock_board_industry_hist_em = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._fetch_industry_index_data_sync,
                            industry_name,
                            first_date_str,
                            last_date_str,
                        ),
                        timeout=45.0  # 45 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching index data for industry %s, skipping", industry_name)
                    return None
                # Get the index of the last trading date
                industry_last_index = stock_board_industry_hist_em["收盘"].iloc[-1]
                # Get the index of the desired trading date
                industry_days_index = stock_board_industry_hist_em["收盘"].iloc[-days]
                # Get the index of 60 trading days ago
                industry_60_index = stock_board_industry_hist_em["收盘"].iloc[
                    -self.TRADING_DAYS_60
                ]
                # Get the index of the 1st trading date
                industry_1st_trading_date_index = stock_board_industry_hist_em[
                    stock_board_industry_hist_em["日期"] == first_trading_date_str
                ]["收盘"].iloc[0]
                # Calculate index change percentage
                industry_index_change_perc_days = (
                    (industry_last_index - industry_days_index)
                    / industry_days_index
                    * 100
                )
                industry_index_change_perc_days = round(
                    industry_index_change_perc_days, 2
                )

                industry_index_change_perc_60 = (
                    (industry_last_index - industry_60_index) / industry_60_index * 100
                )
                industry_index_change_perc_60 = round(industry_index_change_perc_60, 2)

                industry_index_change_perc_ytd = (
                    (industry_last_index - industry_1st_trading_date_index)
                    / industry_1st_trading_date_index
                    * 100
                )
                industry_index_change_perc_ytd = round(
                    industry_index_change_perc_ytd, 2
                )
                # Log the results
                logger.debug(
                    "%s: %s, %s%%, %s%%, %s%%",
                    industry_name,
                    industry_main_net_flow,
                    industry_index_change_perc_days,
                    industry_index_change_perc_60,
                    industry_index_change_perc_ytd,
                )
                return [
                    industry_name,
                    industry_main_net_flow,
                    industry_index_change_perc_days,
                    industry_index_change_perc_60,
                    industry_index_change_perc_ytd,
                ]

            except Exception as e:
                logger.error("Error processing %s: %s", industry_name, str(e))
                return None

    async def process_all_industries_async(
        self,
        industry_arr: pd.Series,
        first_date_str: str,
        last_date_str: str,
        first_trading_date_str: str,
        days: int = 29,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all industries concurrently with batch processing.

        This method orchestrates the analysis of all industries using
        batch processing to avoid overwhelming the API while maintaining
        good performance.

        Args:
            industry_arr: Series containing industry names
            first_date_str: Start date in %Y%m%d format
            last_date_str: End date in %Y%m%d format
            first_trading_date_str: First trading date in %Y-%m-%d format
            days: Number of days to analyze (default: 29)

        Returns:
            DataFrame containing analysis results for all industries
        """
        # Define columns for consistency
        columns = self._get_analysis_columns(days)

        all_industries_df = pd.DataFrame(columns=columns)

        # Process industries with some concurrency but not too much to avoid overwhelming the API
        batch_size = self.BATCH_SIZE
        total_batches = (len(industry_arr) + batch_size - 1) // batch_size

        # Use pre-created batch task if provided, otherwise create new one
        if progress is not None and batch_task_id is not None:
            # Make the pre-created batch task visible and configure it
            progress.update(
                batch_task_id,
                total=total_batches,
                visible=True,
                description="    🏢 Industry Filter: Processing batches",
            )
        elif progress is not None:
            # Fallback: create new batch task (will appear at bottom)
            batch_task_id = progress.add_task(
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
            if progress is not None and batch_task_id is not None:
                progress.update(
                    batch_task_id,
                    completed=batch_num - 1,
                    description=f"    🏢 Industry Filter: Processing batch {batch_num}/{total_batches} ({len(batch)} industries)",
                )

            # Create tasks for the current batch
            tasks = [
                self.process_single_industry_async(
                    industry_name,
                    first_date_str,
                    last_date_str,
                    first_trading_date_str,
                    days,
                )
                for industry_name in batch
            ]

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            for result in batch_results:
                if result is not None and not isinstance(result, Exception):
                    all_industries_df.loc[len(all_industries_df)] = result

            # Update batch progress after completion
            if progress is not None and batch_task_id is not None:
                progress.advance(batch_task_id)

        # Remove batch progress bar when finished (subtask cleanup)
        if progress is not None and batch_task_id is not None:
            progress.update(
                batch_task_id,
                description="    ✅ Industry Filter: All batches completed",
            )
            await asyncio.sleep(0.5)  # Brief display of completion
            progress.remove_task(batch_task_id)

        return all_industries_df

    def _save_reports(
        self, all_industries_df: pd.DataFrame, days: int, last_date_str: str
    ) -> None:
        """
        Save analysis reports to CSV files.

        Args:
            all_industries_df: DataFrame containing all analysis results
            days: Number of days used for analysis (for filtering and naming)
            last_date_str: Last date string for report naming
        """
        # Sort all_industries_df
        all_industries_df = all_industries_df.sort_values(
            by=[f"{days}日主力净流入-总净额(亿)", f"{days}日涨跌幅(%)"],
            ascending=[False, True],
        )
        all_industries_df.reset_index(inplace=True, drop=True)

        # Output the raw report with error handling
        try:
            raw_report_path = f"{self.REPORT_DIR}/行业筛选报告-raw-{last_date_str}.csv"
            all_industries_df.to_csv(raw_report_path, index=True)
            logger.info("Report saved to %s", raw_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save raw industry report: %s", str(e))
            raise

        # Apply additional filters to all_industries_df
        main_net_inflow_col = f"{days}日主力净流入-总净额(亿)"
        price_change_col = f"{days}日涨跌幅(%)"
        filtered_df = all_industries_df[
            (all_industries_df[main_net_inflow_col] > self.MIN_MAIN_NET_INFLOW_YI)
            & (all_industries_df[price_change_col] < self.MAX_PRICE_CHANGE_PERCENT)
        ]

        # Sort filtered DataFrame
        filtered_df = filtered_df.sort_values(
            by=[main_net_inflow_col, price_change_col],
            ascending=[False, True],
        )
        filtered_df.reset_index(inplace=True, drop=True)

        # Output the filtered report with error handling
        try:
            filtered_report_path = f"{self.REPORT_DIR}/行业筛选报告-{last_date_str}.csv"
            filtered_df.to_csv(filtered_report_path, index=True)
            logger.info("Filtered report saved to %s", filtered_report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save filtered industry report: %s", str(e))
            raise

    async def run_analysis(
        self,
        days: int = None,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> None:
        """
        Run the complete industry filtering pipeline.

        This method orchestrates the entire industry filtering process including
        data preparation, industry analysis, result filtering, and report generation.

        Args:
            days: Number of days to analyze for fund flow (default: class constant)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        if days is None:
            days = self.DAYS_ANALYSIS_PERIOD

        # Get dates and industry data
        industry_arr, first_date_str, last_date_str, first_trading_date_str = (
            self.get_dates()
        )

        # Process all industries with progress tracking
        all_industries_df = await self.process_all_industries_async(
            industry_arr,
            first_date_str,
            last_date_str,
            first_trading_date_str,
            days=days,
            progress=progress,
            parent_task_id=parent_task_id,
            batch_task_id=batch_task_id,
        )

        # Save reports (raw and filtered)
        self._save_reports(all_industries_df, days, last_date_str)


async def main(
    progress: Optional["Progress"] = None,
    parent_task_id: Optional[int] = None,
    batch_task_id: Optional[int] = None,
) -> None:
    """
    Main function to execute the complete industry filtering pipeline.

    This function creates an IndustryFilter instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display
    """
    industry_filter = IndustryFilter()
    await industry_filter.run_analysis(
        progress=progress, parent_task_id=parent_task_id, batch_task_id=batch_task_id
    )


if __name__ == "__main__":
    asyncio.run(main())
