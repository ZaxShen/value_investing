"""
Industry analysis and filtering for Chinese equity markets.

This module provides comprehensive analysis of industry sectors including
fund flow analysis, index performance tracking, and industry-level filtering.
It generates detailed reports on industry performance and capital flows.
"""

import asyncio
import functools
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Any, Callable
from src.utilities.tools import timer
from src.utilities.logger import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("industry_filter")


def get_dates() -> Tuple[pd.Series, str, str, str]:
    """
    Get industry names and date ranges for analysis.

    This function retrieves the list of industry names and calculates
    appropriate date ranges for industry analysis, ensuring sufficient
    trading days for meaningful analysis.

    Returns:
        Tuple containing:
            - industry_arr: Series of industry names
            - first_date_str: Start date in %Y%m%d format
            - last_date_str: End date in %Y%m%d format
            - first_trading_date_str: First trading date in %Y-%m-%d format
    """
    # Get the list of industry names
    industry_arr = ak.stock_board_industry_name_em()["板块名称"]

    # Get date related variabels
    today = datetime.today().date()
    this_year = today.year
    # A consecutive date that gearter than 60 trading days
    date_100_days_ago = today - timedelta(days=100)
    # Define first_date, the range to fetch industry data
    if datetime(this_year, 1, 1).date() < date_100_days_ago:
        first_date = datetime(this_year, 1, 1)
    else:
        first_date = date_100_days_ago
    first_date_str = first_date.strftime("%Y%m%d")
    last_date_str = today.strftime("%Y%m%d")

    # Define last_date, the range to fetch industry data
    dates = ak.stock_board_industry_hist_em(
        symbol=industry_arr[0],
        start_date=first_date_str,
        end_date=last_date_str,
        period="日k",
        adjust="",
    )["日期"].values

    last_date_str = dates[-1].replace("-", "")

    # Get the 1st trading date
    first_trading_date = datetime(datetime.today().year, 1, 1).date()
    while first_trading_date.strftime("%Y-%m-%d") not in dates:
        first_trading_date += timedelta(days=1)
    first_trading_date_str = first_trading_date.strftime("%Y-%m-%d")

    return industry_arr, first_date_str, last_date_str, first_trading_date_str


# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


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
def fetch_industry_capital_flow_data(industry_name: str, days: int) -> pd.DataFrame:
    """
    Fetch industry capital flow data - wrapped for async execution.

    Args:
        industry_name: Name of the industry to analyze
        days: Number of recent days to fetch data for

    Returns:
        DataFrame containing recent capital flow data for the industry
    """
    return ak.stock_sector_fund_flow_hist(symbol=industry_name).iloc[-days:]


@run_in_executor
def fetch_industry_index_data(
    industry_name: str,
    first_date_str: str,
    last_date_str: str,
) -> pd.DataFrame:
    """
    Fetch industry index historical data - wrapped for async execution.

    Args:
        industry_name: Name of the industry to analyze
        first_date_str: Start date in %Y%m%d format
        last_date_str: End date in %Y%m%d format

    Returns:
        DataFrame containing historical index data for the industry
    """
    stock_board_industry_hist_em = ak.stock_board_industry_hist_em(
        symbol=industry_name,
        start_date=first_date_str,
        end_date=last_date_str,
        period="日k",
        adjust="",
    )
    return stock_board_industry_hist_em


async def process_single_industry_async(
    industry_name: str,
    first_date_str: str,
    last_date_str: str,
    first_trading_date_str: str,
    days: int = 29,
) -> Optional[List[Any]]:
    """
    Process a single industry asynchronously to calculate performance metrics.

    This function analyzes an industry's capital flow, index performance,
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
            # Fetch industry capital flow data
            stock_sector_fund_flow_hist_df = await fetch_industry_capital_flow_data(
                industry_name, days
            )
            # Calculate main net flow
            industry_main_net_flow = stock_sector_fund_flow_hist_df["主力净流入-净额"].sum()
            industry_main_net_flow = round(industry_main_net_flow / 1e8, 1)  # Convert to 100M

            # Fetch industry index data
            stock_board_industry_hist_em = await fetch_industry_index_data(
                industry_name,
                first_date_str,
                last_date_str,
            )
            # Get the index of the last trading date
            industry_last_index = stock_board_industry_hist_em["收盘"].iloc[-1]
            # Get the index of the desired trading date
            industry_days_index = stock_board_industry_hist_em["收盘"].iloc[-days]
            # Get the index of 60 trading days ago
            industry_60_index = stock_board_industry_hist_em["收盘"].iloc[-60]
            # Get the index of the 1st trading date
            industry_1st_trading_date_index = stock_board_industry_hist_em[
                stock_board_industry_hist_em["日期"] == first_trading_date_str
            ]["收盘"].iloc[0]
            # Calcuate index change percentage
            industry_index_change_perc_days = (
                (industry_last_index - industry_days_index) / industry_days_index * 100
            )
            industry_index_change_perc_days = round(industry_index_change_perc_days, 2)

            industry_index_change_perc_60 = (
                (industry_last_index - industry_60_index) / industry_60_index * 100
            )
            industry_index_change_perc_60 = round(industry_index_change_perc_60, 2)

            industry_index_change_perc_ytd = (
                (industry_last_index - industry_1st_trading_date_index)
                / industry_1st_trading_date_index
                * 100
            )
            industry_index_change_perc_ytd = round(industry_index_change_perc_ytd, 2)
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


@timer
async def process_all_industries_async(
    industry_arr: pd.Series,
    first_date_str: str,
    last_date_str: str,
    first_trading_date_str: str,
    days: int = 29,
    progress: Optional["Progress"] = None,
    parent_task_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Process all industries concurrently with batch processing.

    This function orchestrates the analysis of all industries using
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
    columns = [
        "行业",
        f"{days}日主力净流入-总净额(亿)",
        f"{days}日涨跌幅(%)",
        "60日涨跌幅(%)",
        "年初至今涨跌幅(%)",
    ]

    all_industries_df = pd.DataFrame(columns=columns)

    # Process industries with some concurrency but not too much to avoid overwhelming the API
    batch_size = 3
    total_batches = (len(industry_arr) + batch_size - 1) // batch_size
    
    # Create batch processing progress bar if progress is provided
    batch_task_id = None
    if progress is not None:
        batch_task_id = progress.add_task(
            "🏢 Processing industry analysis batches", 
            total=total_batches,
            visible=True
        )

    for i in range(0, len(industry_arr), batch_size):
        batch = industry_arr[i : i + batch_size]
        batch_num = i//batch_size + 1
        
        logger.info(
            "Processing industry batch %d/%d",
            batch_num,
            total_batches,
        )
        
        # Update batch progress if available
        if progress is not None and batch_task_id is not None:
            progress.update(
                batch_task_id, 
                completed=batch_num-1,
                description=f"\tIndustry Filter: Processing batch {batch_num}/{total_batches} ({len(batch)} industries)"
            )

        # Create tasks for the current batch
        tasks = [
            process_single_industry_async(
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
            description="✅ All industry analysis batches completed"
        )
        await asyncio.sleep(0.5)  # Brief display of completion
        progress.remove_task(batch_task_id)

    return all_industries_df


async def main(progress: Optional["Progress"] = None, parent_task_id: Optional[int] = None) -> None:
    """
    Main function to execute the complete industry filtering pipeline.

    This function orchestrates the entire industry analysis process including
    data preparation, industry analysis, result filtering, and report generation.
    
    Args:
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
    """
    days = 29

    industry_arr, first_date_str, last_date_str, first_trading_date_str = get_dates()

    # Process all industries with progress tracking
    all_industries_df = await process_all_industries_async(
        industry_arr,
        first_date_str,
        last_date_str,
        first_trading_date_str,
        days=29,
        progress=progress,
        parent_task_id=parent_task_id
    )

    # Define the directory for reports
    REPORT_DIR = "data/stocks/reports"

    # Sort all_industries_df
    all_industries_df = all_industries_df.sort_values(
        by=[f"{days}日主力净流入-总净额(亿)", f"{days}日涨跌幅(%)"],
        ascending=[False, True],
    )
    all_industries_df.reset_index(inplace=True, drop=True)
    # Output the all_industries_df to a CSV file
    all_industries_df.to_csv(f"{REPORT_DIR}/行业筛选报告-raw-{last_date_str}.csv", index=True)
    logger.info("Report saved to %s/行业筛选报告-raw-%s.csv", REPORT_DIR, last_date_str)

    # Apply additional filters to all_industries_df
    df = all_industries_df[
        (all_industries_df[f"{days}日主力净流入-总净额(亿)"] > 20)
        & (all_industries_df[f"{days}日涨跌幅(%)"] < 8)
    ]

    # Sort df
    df = df.sort_values(
        by=[f"{days}日主力净流入-总净额(亿)", f"{days}日涨跌幅(%)"],
        ascending=[False, True],
    )
    df.reset_index(inplace=True, drop=True)

    # Output the filtered DataFrame to a CSV file
    df.to_csv(f"{REPORT_DIR}/行业筛选报告-{last_date_str}.csv", index=True)
    logger.info("Filtered report saved to %s/行业筛选报告-%s.csv", REPORT_DIR, last_date_str)


if __name__ == "__main__":
    asyncio.run(main())
