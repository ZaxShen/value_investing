"""
Data fetching and caching utilities for Chinese stock market data.

This module provides functions to fetch and cache stock market data and
industry-stock mapping data using the akshare library. It implements
intelligent caching to avoid repeated API calls and improve performance.
"""

import os
import glob
import asyncio
import pandas as pd
import akshare as ak
from datetime import datetime
from typing import Optional
from .tools import timer


# @timer
async def get_stock_market_data(data_dir: str = "data/stocks") -> pd.DataFrame:
    """
    Fetch stock market data with caching.

    Args:
        data_dir: Directory to store cached data files

    Returns:
        DataFrame containing stock market data with columns including
        stock codes, names, prices, and financial metrics
    """
    today = datetime.now().strftime("%Y%m%d")
    file_path = f"{data_dir}/stock_zh_a_spot_em_df-{today}.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={"代码": str})

    # Delete outdated files
    for f in glob.glob(f"{data_dir}/stock_zh_a_spot_em_df-*.csv"):
        os.remove(f)

    # Fetch and save new data
    stock_df = await asyncio.to_thread(ak.stock_zh_a_spot_em)
    os.makedirs(data_dir, exist_ok=True)
    stock_df.to_csv(file_path, index=False)

    return stock_df


# @timer
async def get_industry_stock_mapping_data(data_dir: str = "data/stocks") -> pd.DataFrame:
    """
    Fetch industry-stock mapping data with caching.

    Args:
        data_dir: Directory to store cached data files

    Returns:
        DataFrame containing industry-stock mapping with columns for
        industry names and corresponding stock codes
    """
    today = datetime.now().strftime("%Y%m%d")
    file_path = f"{data_dir}/industry_stock_mapping_df-{today}.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path, dtype={"代码": str})

    # Delete outdated files
    for f in glob.glob(f"{data_dir}/industry_stock_mapping_df-*.csv"):
        os.remove(f)

    # Fetch new data
    industry_names = await asyncio.to_thread(ak.stock_board_industry_name_em)
    industry_names = industry_names["板块名称"]
    mapping_df = pd.DataFrame(columns=["行业", "代码"])

    for industry_name in industry_names:
        industry_stocks = await asyncio.to_thread(ak.stock_board_industry_cons_em, symbol=industry_name)
        for stock_code in industry_stocks["代码"]:
            mapping_df.loc[len(mapping_df)] = [industry_name, stock_code]

    # Save data
    os.makedirs(data_dir, exist_ok=True)
    mapping_df.to_csv(file_path, index=False)

    return mapping_df
