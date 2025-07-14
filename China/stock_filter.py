import asyncio
import time
import functools
import akshare as ak
import pandas as pd
from utilities.get_stock_data import (
    get_stock_market_data,
    get_industry_stock_mapping_data,
)


def prepare_stock_data():
    """
    Prepare and filter stock market data.

    Returns:
        tuple: (stock_market_df_filtered, industry_arr)
            - stock_market_df_filtered: DataFrame with filtered stock data
            - industry_arr: Array of unique industry names
    """
    print("Loading stock market data...")
    stock_zh_a_spot_em_df = get_stock_market_data()
    industry_stock_mapping_df = get_industry_stock_mapping_data()

    # Filter stock market data
    # 总市值 < 200 亿, 0 < 动态市盈率 < 50
    stock_market_df_filtered = stock_zh_a_spot_em_df[
        (stock_zh_a_spot_em_df["总市值"] < 200 * 1e8)
        & (stock_zh_a_spot_em_df["市盈率-动态"] > 0)
        & (stock_zh_a_spot_em_df["市盈率-动态"] < 50)
    ]

    # Release memory early
    del stock_zh_a_spot_em_df

    # Inner join industry_stock_mapping_df with stock_market_df_filtered
    stock_market_df_filtered = pd.merge(
        industry_stock_mapping_df,
        stock_market_df_filtered[
            [
                "代码",
                "名称",
                "总市值",
                "流通市值",
                "市盈率-动态",
                "市净率",
                "60日涨跌幅",
                "年初至今涨跌幅",
            ]
        ],
        on="代码",
        how="inner",
    )

    # Organize the columns
    stock_market_df_filtered.columns = [
        "行业",
        "代码",
        "名称",
        "总市值",
        "流通市值",
        "市盈率-动态",
        "市净率",
        "60日涨跌幅",
        "年初至今涨跌幅",
    ]

    # Get unique industry names
    industry_arr = stock_market_df_filtered["行业"].unique()

    # Release memory
    del industry_stock_mapping_df

    print(
        f"Loaded {len(stock_market_df_filtered)} stocks across {len(industry_arr)} industries"
    )
    return stock_market_df_filtered, industry_arr


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
def fetch_stock_data(stock_code, market):
    """Fetch stock individual fund flow data - wrapped for async execution"""
    return ak.stock_individual_fund_flow(stock=stock_code, market=market)


async def process_single_stock_async(
    stock_code, stock_name, industry_name, stock_market_df_filtered, days=29
):
    """
    Process a single stock asynchronously

    Args:
        stock_code: Stock code
        stock_name: Stock name
        industry_name: Industry name
        stock_market_df_filtered: DataFrame with filtered stock data
        days: Number of days to analyze
    """
    async with REQUEST_SEMAPHORE:
        print(f"Processing {stock_name} ({stock_code}) in {industry_name} industry...")

        # Determine the market based on the stock code
        if stock_code.startswith("6"):
            market = "sh"
        elif stock_code.startswith("0") or stock_code.startswith("3"):
            market = "sz"
        else:
            market = "bj"

        try:
            # Extract the stock's market data
            stock_data = stock_market_df_filtered[
                stock_market_df_filtered["代码"] == stock_code
            ].iloc[
                0
            ]  # More efficient than multiple queries

            stock_total_market_value = round(stock_data["总市值"] / 1e8, 0)
            stock_circulating_market_value = round(stock_data["流通市值"] / 1e8, 0)
            stock_pe_dynamic = stock_data["市盈率-动态"]
            stock_pb = stock_data["市净率"]
            stock_60d_change = stock_data["60日涨跌幅"]
            stock_ytd_change = stock_data["年初至今涨跌幅"]

            # Extract the historical data of the stock (async)
            stock_individual_fund_flow_df = await fetch_stock_data(stock_code, market)

            if len(stock_individual_fund_flow_df) < days:
                print(
                    f"Skipping {stock_name} ({stock_code}) due to insufficient data for the last {days} days."
                )
                return None

            stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[-days:]

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
            print(f"Error processing {stock_name} ({stock_code}): {str(e)}")
            return None


async def process_single_industry_async(
    industry_name, stock_market_df_filtered, days=29
):
    """
    Analyzes stocks in a given industry by extracting their main net inflow
    and price change percentage over the last `days` days.

    Args:
        industry_name: The industry name to analyze
        stock_market_df_filtered: DataFrame with filtered stock data
        days: Number of days to consider for net flow calculation

    Returns:
        DataFrame with stock analysis results
    """
    # Extract all qualified stocks from stock_market_df_filtered
    stocks = stock_market_df_filtered[
        stock_market_df_filtered["行业"] == industry_name
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
        task = process_single_stock_async(
            row.代码, row.名称, industry_name, stock_market_df_filtered, days
        )
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and add to DataFrame
    for result in results:
        if result is not None and not isinstance(result, Exception):
            df.loc[len(df)] = result

    return df


async def process_all_industries_async(stock_market_df_filtered, industry_arr, days=29):
    """
    Process all industries concurrently

    Args:
        stock_market_df_filtered: DataFrame with filtered stock data
        industry_arr: Array of industry names to process
        days: Number of days to analyze

    Returns:
        DataFrame with results from all industries
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

    all_industries_df = pd.DataFrame(columns=columns)
    all_industries_df["代码"] = all_industries_df["代码"].astype(str)

    # Process industries with some concurrency but not too much to avoid overwhelming the API
    batch_size = 3

    for i in range(0, len(industry_arr), batch_size):
        batch = industry_arr[i : i + batch_size]
        print(
            f"Processing industry batch {i//batch_size + 1}/{(len(industry_arr) + batch_size - 1)//batch_size}"
        )

        # Create tasks for the current batch
        tasks = [
            process_single_industry_async(industry_name, stock_market_df_filtered, days)
            for industry_name in batch
        ]

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        for result in batch_results:
            if result is not None and not isinstance(result, Exception):
                all_industries_df = pd.concat(
                    [all_industries_df, result], ignore_index=True
                )

    return all_industries_df


async def main():
    """Main async function"""
    days = 29

    print("Starting async stock analysis...")
    start_time = time.time()

    # Prepare data
    stock_market_df_filtered, industry_arr = prepare_stock_data()

    # Process all industries
    all_industries_df = await process_all_industries_async(
        stock_market_df_filtered, industry_arr, days
    )

    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    print("=" * 100)

    # Define the report date
    last_date = ak.stock_sector_fund_flow_hist(symbol="证券").iloc[-1]["日期"]
    last_date_str = last_date.strftime("%Y%m%d")

    # Define the directory for reports
    REPORT_DIR = "data/stocks/reports"

    # Output the all_industries_df to a CSV file
    all_industries_df.to_csv(
        f"{REPORT_DIR}/股票筛选报告-raw-{last_date_str}.csv", index=True
    )
    print(f"Report saved to {REPORT_DIR}/股票筛选报告-raw-{last_date_str}.csv")

    # Apply additional filters to all_industries_df
    df = all_industries_df[
        (all_industries_df[f"{days}日主力净流入-总净额(亿)"] > 1)
        & (all_industries_df[f"{days}日涨跌幅(%)"] < 10)
    ]

    # Sort the DataFrame by pe and {days} change percentage
    df.sort_values(by=["市盈率-动态", f"{days}日涨跌幅(%)"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Output the filtered DataFrame to a CSV file
    df.to_csv(f"{REPORT_DIR}/股票筛选报告-{last_date_str}.csv", index=True)
    print(f"Filtered report saved to {REPORT_DIR}/股票筛选报告-{last_date_str}.csv")


if __name__ == "__main__":
    asyncio.run(main())
