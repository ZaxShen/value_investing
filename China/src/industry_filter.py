import asyncio
import functools
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from src.utilities.tools import timer, verbose
from src.utilities.logger import get_logger

# Initialize logger for this module
logger = get_logger("industry_filter")


def get_dates():
    """
    returns:
        - industry_arr: arr
        - first_date_str: %Y%m%d
        - last_date_str: %Y%m%d
        - first_trading_date_str: %Y-%m-%d
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


def run_in_executor(func):
    """Decorator to run blocking functions in thread pool executor"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


@run_in_executor
def fetch_indsutry_capital_flow_data(industry_name, days):
    return ak.stock_sector_fund_flow_hist(symbol=industry_name).iloc[-days:]


@run_in_executor
def fetch_industry_index_data(
    industry_name,
    first_date_str,
    last_date_str,
):
    stock_board_industry_hist_em = ak.stock_board_industry_hist_em(
        symbol=industry_name,
        start_date=first_date_str,
        end_date=last_date_str,
        period="日k",
        adjust="",
    )
    return stock_board_industry_hist_em


async def process_single_industry_async(
    industry_name,
    first_date_str,
    last_date_str,
    first_trading_date_str,
    days=29,
):

    async with REQUEST_SEMAPHORE:
        try:
            # Fetch industry capital flow data
            stock_sector_fund_flow_hist_df = await fetch_indsutry_capital_flow_data(
                industry_name, days
            )
            # Calculate main net flow
            industry_main_net_flow = stock_sector_fund_flow_hist_df[
                "主力净流入-净额"
            ].sum()
            industry_main_net_flow = round(
                industry_main_net_flow / 1e8, 1
            )  # Convert to 100M

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
            logger.debug(f"{industry_name}: {industry_main_net_flow}, {industry_index_change_perc_days}%, {industry_index_change_perc_60}%, {industry_index_change_perc_ytd}%")
            return [
                industry_name,
                industry_main_net_flow,
                industry_index_change_perc_days,
                industry_index_change_perc_60,
                industry_index_change_perc_ytd,
            ]

        except Exception as e:
            logger.error(f"Error processing {industry_name}: {str(e)}")
            return None


@timer
async def process_all_industries_async(
    industry_arr,
    first_date_str,
    last_date_str,
    first_trading_date_str,
    days=29,
):
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

    for i in range(0, len(industry_arr), batch_size):
        batch = industry_arr[i : i + batch_size]
        logger.info(f"Processing industry batch {i//batch_size + 1}/{(len(industry_arr) + batch_size - 1)//batch_size}")

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

    return all_industries_df


async def main():
    """Main async function"""
    days = 29

    industry_arr, first_date_str, last_date_str, first_trading_date_str = get_dates()

    # Process all industries
    all_industries_df = await process_all_industries_async(
        industry_arr,
        first_date_str,
        last_date_str,
        first_trading_date_str,
        days=29,
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
    all_industries_df.to_csv(
        f"{REPORT_DIR}/行业筛选报告-raw-{last_date_str}.csv", index=True
    )
    logger.info(f"Report saved to {REPORT_DIR}/行业筛选报告-raw-{last_date_str}.csv")

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
    logger.info(f"Filtered report saved to {REPORT_DIR}/行业筛选报告-{last_date_str}.csv")


if __name__ == "__main__":
    asyncio.run(main())
