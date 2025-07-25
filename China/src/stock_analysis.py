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
from src.utilities.get_stock_data import (
    get_stock_market_data,
    get_industry_stock_mapping_data,
)
from src.utilities.tools import timer
from src.utilities.logger import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("stock_analysis")


stock_zh_a_spot_em_df = get_stock_market_data()
industry_stock_mapping_df = get_industry_stock_mapping_data()


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
            raise ValueError(f"Stock name mismatch for {stock_code}: {stock_name} != {actual_name}")
    except (IndexError, KeyError):
        raise ValueError(f"Stock code {stock_code} not found")




def fetch_stock_individual_fund_flow_sync(stock_code: str, market: str) -> pd.DataFrame:
    """
    Fetch stock individual fund flow data - synchronous version.

    Args:
        stock_code: Stock code (e.g., "000001")
        market: Market identifier (e.g., "sz", "sh", "bj")

    Returns:
        DataFrame containing historical fund flow data for the specified stock
    """
    return ak.stock_individual_fund_flow(stock=stock_code, market=market)


def fetch_stock_sector_fund_flow_hist_sync(symbol: str) -> pd.DataFrame:
    """
    Fetch stock sector fund flow historical data - synchronous version.

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
    logger.debug("Processing %s (%s) in %s industry", stock_name, stock_code, industry_name)
    # Determine the market based on the stock code
    if stock_code.startswith("6"):
        market = "sh"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        market = "sz"
    else:
        market = "bj"

    # Extract the stock's market data
    stock_total_market_value = (
        stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code]["总市值"].values[0] / 1e8
    )  # Convert to 100M
    stock_total_market_value = round(stock_total_market_value, 0)
    stock_circulating_market_value = (
        stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code]["流通市值"].values[0]
        / 1e8
    )  # Convert to 100M
    stock_circulating_market_value = round(stock_circulating_market_value, 0)
    stock_pe_dynamic = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
        "市盈率-动态"
    ].values[0]
    stock_pb = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code]["市净率"].values[
        0
    ]
    stock_60d_change = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
        "60日涨跌幅"
    ].values[0]
    stock_ytd_change = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
        "年初至今涨跌幅"
    ].values[0]

    # Extract the historical data of the stock (async)
    stock_individual_fund_flow_df = await asyncio.to_thread(fetch_stock_individual_fund_flow_sync, stock_code, market)
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
    stock_price_change_percentage = (stock_last_price - stock_1st_price) / stock_1st_price * 100
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


# @timer
async def main(progress: Optional["Progress"] = None, parent_task_id: Optional[int] = None, batch_task_id: Optional[int] = None) -> None:
    """
    Main function to execute stock analysis and generate holding reports.

    This function reads stock holding data from JSON files, performs comprehensive
    analysis on each stock, and generates detailed reports with financial metrics
    and performance indicators.
    
    Args:
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display (unused in this script)
    """
    dir_path = "data/holding_stocks"
    days = 29
    # Initialize a pandas Dataframe to hold industry names, industry main net flow, and industry index change percentage
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

    for file in glob.glob(os.path.join(dir_path, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            account_name = os.path.splitext(os.path.basename(file))[0]
            holding_stocks = json.load(f)
            for stock_code, stock_name in holding_stocks.items():
                validate_stock_name(stock_code, stock_name, stock_zh_a_spot_em_df)
                industry_name = industry_stock_mapping_df[
                    industry_stock_mapping_df["代码"] == stock_code
                ]["行业"].values[0]
                result = await stock_analysis(
                    industry_name=industry_name,
                    stock_code=stock_code,
                    stock_name=stock_name,
                    days=days,
                )
                if result is not None:
                    df.loc[len(df)] = [f"{account_name}"] + result

    # Define the report date (async)
    stock_sector_data = await asyncio.to_thread(fetch_stock_sector_fund_flow_hist_sync, "证券")
    last_date = stock_sector_data.iloc[-1]["日期"]
    last_date_str = last_date.strftime("%Y%m%d")
    # Output the df to a CSV file
    df.to_csv(f"{dir_path}/reports/持股报告-{last_date_str}.csv", index=True)
    logger.info("Report saved to %s/reports/持股报告-%s.csv", dir_path, last_date_str)


if __name__ == "__main__":
    asyncio.run(main())
