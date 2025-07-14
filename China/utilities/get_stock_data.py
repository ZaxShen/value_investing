import os
import glob
import pandas as pd
import akshare as ak
from datetime import datetime


def get_stock_market_data(data_dir="data/stocks"):
    """
    Fetch stock market data with caching.
    Returns cached data if available for today, otherwise fetches new data.
    """
    today = datetime.now().strftime("%Y%m%d")
    file_path = f"{data_dir}/stock_zh_a_spot_em_df-{today}.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={"代码": str}, index_col=0)

    # Delete outdated files
    for f in glob.glob(f"{data_dir}/stock_zh_a_spot_em_df-*.csv"):
        os.remove(f)

    # Fetch and save new data
    stock_df = ak.stock_zh_a_spot_em()
    os.makedirs(data_dir, exist_ok=True)
    stock_df.to_csv(file_path, index=False)

    return stock_df


def get_industry_stock_mapping_data(data_dir="data/stocks"):
    """
    Fetch industry-stock mapping data with caching.
    Returns cached data if available for today, otherwise fetches new data.
    """
    today = datetime.now().strftime("%Y%m%d")
    file_path = f"{data_dir}/industry_stock_mapping_df-{today}.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={"代码": str}, index_col=0)

    # Delete outdated files
    for f in glob.glob(f"{data_dir}/industry_stock_mapping_df-*.csv"):
        os.remove(f)

    # Fetch new data
    industry_names = ak.stock_board_industry_name_em()["板块名称"]
    mapping_df = pd.DataFrame(columns=["行业", "代码"])

    for industry_name in industry_names:
        industry_stocks = ak.stock_board_industry_cons_em(symbol=industry_name)
        for stock_code in industry_stocks["代码"]:
            mapping_df.loc[len(mapping_df)] = [industry_name, stock_code]

    # Save data
    os.makedirs(data_dir, exist_ok=True)
    mapping_df.to_csv(file_path, index=False)

    return mapping_df
