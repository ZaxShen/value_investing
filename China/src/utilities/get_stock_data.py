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
from typing import Optional, List
from .tools import timer
from rich.progress import Progress, TaskID
from rich.console import Console

# Semaphore to control concurrent API requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)
console = Console()


@timer
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


async def _fetch_industry_stocks(industry_name: str) -> List[tuple]:
    """
    Fetch stocks for a single industry with semaphore control.
    
    Args:
        industry_name: Name of the industry to fetch stocks for
        
    Returns:
        List of (industry_name, stock_code) tuples
    """
    async with REQUEST_SEMAPHORE:
        try:
            industry_stocks = await asyncio.to_thread(
                ak.stock_board_industry_cons_em, symbol=industry_name
            )
            return [(industry_name, code) for code in industry_stocks["代码"]]
        except Exception as e:
            print(f"Error fetching data for industry {industry_name}: {e}")
            return []


@timer
async def get_industry_stock_mapping_data(data_dir: str = "data/stocks") -> pd.DataFrame:
    """
    Fetch industry-stock mapping data with caching and optimized concurrent processing.

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

    # Fetch industry names
    industry_names = await asyncio.to_thread(ak.stock_board_industry_name_em)
    industry_names = industry_names["板块名称"]

    # Process industries concurrently with batching and progress tracking
    batch_size = 10
    all_mappings = []
    
    with Progress(console=console) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(industry_names)} industries...", 
            total=len(industry_names)
        )
        
        for i in range(0, len(industry_names), batch_size):
            batch = industry_names[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [_fetch_industry_stocks(industry_name) for industry_name in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            processed_count = 0
            for result in batch_results:
                if isinstance(result, list):
                    all_mappings.extend(result)
                    processed_count += 1
                elif isinstance(result, Exception):
                    processed_count += 1  # Count failed ones too
            
            progress.update(task, advance=len(batch))
            
            # Add small delay between batches to be respectful to API
            await asyncio.sleep(0.1)

    # Convert to DataFrame efficiently using list of tuples
    if all_mappings:
        mapping_df = pd.DataFrame(all_mappings, columns=["行业", "代码"])
    else:
        mapping_df = pd.DataFrame(columns=["行业", "代码"])

    # Save data
    os.makedirs(data_dir, exist_ok=True)
    mapping_df.to_csv(file_path, index=False)

    return mapping_df
