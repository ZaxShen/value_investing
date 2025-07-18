import os
import pandas as pd
import akshare as ak
import glob
import json
import asyncio
import functools
from datetime import datetime
from src.utilities.get_stock_data import (
    get_stock_market_data,
    get_industry_stock_mapping_data,
)
from src.utilities.tools import timer, verbose
from src.utilities.logger import get_logger

# Initialize logger for this module
logger = get_logger("stock_analysis")


stock_zh_a_spot_em_df = get_stock_market_data()
industry_stock_mapping_df = get_industry_stock_mapping_data()


def validate_stock_name(stock_code, stock_name, df):
    try:
        actual_name = df[df["代码"] == stock_code]["名称"].values[0]
        if actual_name != stock_name:
            raise ValueError(
                f"Stock name mismatch for {stock_code}: {stock_name} != {actual_name}"
            )
    except (IndexError, KeyError):
        raise ValueError(f"Stock code {stock_code} not found")


def run_in_executor(func):
    """Decorator to run blocking functions in thread pool executor"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


@run_in_executor
def fetch_stock_individual_fund_flow(stock_code, market):
    """Fetch stock individual fund flow data - wrapped for async execution"""
    return ak.stock_individual_fund_flow(stock=stock_code, market=market)


@run_in_executor
def fetch_stock_sector_fund_flow_hist(symbol):
    """Fetch stock sector fund flow history - wrapped for async execution"""
    return ak.stock_sector_fund_flow_hist(symbol=symbol)


async def stock_analysis(industry_name, stock_code, stock_name, days=29):
    logger.debug(f"Processing {stock_name} ({stock_code}) in {industry_name} industry")
    # Determine the market based on the stock code
    if stock_code.startswith("6"):
        market = "sh"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        market = "sz"
    else:
        market = "bj"

    # Extract the stock's market data
    stock_total_market_value = (
        stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
            "总市值"
        ].values[0]
        / 1e8
    )  # Convert to 100M
    stock_total_market_value = round(stock_total_market_value, 0)
    stock_circulating_market_value = (
        stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
            "流通市值"
        ].values[0]
        / 1e8
    )  # Convert to 100M
    stock_circulating_market_value = round(stock_circulating_market_value, 0)
    stock_pe_dynamic = stock_zh_a_spot_em_df[
        stock_zh_a_spot_em_df["代码"] == stock_code
    ]["市盈率-动态"].values[0]
    stock_pb = stock_zh_a_spot_em_df[stock_zh_a_spot_em_df["代码"] == stock_code][
        "市净率"
    ].values[0]
    stock_60d_change = stock_zh_a_spot_em_df[
        stock_zh_a_spot_em_df["代码"] == stock_code
    ]["60日涨跌幅"].values[0]
    stock_ytd_change = stock_zh_a_spot_em_df[
        stock_zh_a_spot_em_df["代码"] == stock_code
    ]["年初至今涨跌幅"].values[0]

    # Extract the historical data of the stock (async)
    stock_individual_fund_flow_df = await fetch_stock_individual_fund_flow(
        stock_code, market
    )
    if len(stock_individual_fund_flow_df) < days:
        logger.warning(f"Skipping {stock_name} ({stock_code}) due to insufficient data for the last {days} days")
        return None
    stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[-days:]
    # Get the main net inflow data
    stock_main_net_flow = stock_individual_fund_flow_df["主力净流入-净额"].sum()
    stock_main_net_flow = round(stock_main_net_flow / 1e8, 2)  # Convert to billions
    # Calculate change percentage
    stock_1st_price = stock_individual_fund_flow_df.iloc[-days]["收盘价"]
    stock_last_price = stock_individual_fund_flow_df.iloc[-1]["收盘价"]
    stock_price_change_percentage = (
        (stock_last_price - stock_1st_price) / stock_1st_price * 100
    )
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


@timer
async def main():
    DIR_PATH = "data/holding_stocks"
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

    for file in glob.glob(os.path.join(DIR_PATH, "*.json")):
        with open(file, "r") as f:
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
    stock_sector_data = await fetch_stock_sector_fund_flow_hist("证券")
    last_date = stock_sector_data.iloc[-1]["日期"]
    last_date_str = last_date.strftime("%Y%m%d")
    # Output the df to a CSV file
    df.to_csv(f"{DIR_PATH}/reports/持股报告-{last_date_str}.csv", index=True)
    logger.info(f"Report saved to {DIR_PATH}/reports/持股报告-{last_date_str}.csv")


if __name__ == "__main__":
    asyncio.run(main())
