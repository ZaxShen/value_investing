"""
Data fetching and caching utilities for Chinese stock market data.

This module provides functions to fetch and cache stock market data and
industry-stock mapping data using the akshare library. It implements
intelligent caching to avoid repeated API calls and improve performance
with comprehensive retry mechanisms.
"""

import asyncio
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Handle imports for both module and standalone execution
try:
    # Import settings first to disable tqdm before akshare import
    from src.settings import configure_environment
    from src.utilities.logger import get_logger
    from src.utilities.retry import API_RETRY_CONFIG
    configure_environment()  # Ensure tqdm is disabled
except ModuleNotFoundError:
    # When running as standalone script, add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.settings import configure_environment
    from src.utilities.logger import get_logger
    from src.utilities.retry import API_RETRY_CONFIG
    configure_environment()  # Ensure tqdm is disabled

import akshare as ak
import pandas as pd
from rich.console import Console
from rich.progress import Progress

# Semaphore to control concurrent API requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)
console = Console()

# Initialize logger for this module
logger = get_logger("get_stock_data")


# @timer
async def get_stock_market_data(data_dir: str = "data/stocks") -> pd.DataFrame:
    """
    Fetch stock market data with caching and progress tracking.

    Args:
        data_dir: Directory to store cached data files

    Returns:
        DataFrame containing stock market data with columns including
        stock codes, names, prices, and financial metrics
    """
    today = datetime.now().strftime("%Y%m%d")
    file_path = f"{data_dir}/stock_zh_a_spot_em_df-{today}.csv"

    if os.path.exists(file_path):
        logger.info("Loading cached stock market data from %s", file_path)
        return pd.read_csv(file_path, dtype={"代码": str})

    # Delete outdated files
    logger.info("Removing outdated stock market data files")
    for f in glob.glob(f"{data_dir}/stock_zh_a_spot_em_df-*.csv"):
        os.remove(f)

    # Fetch and save new data with retry mechanism and progress tracking
    with Progress(console=console) as progress:
        task = progress.add_task(
            "[cyan]Fetching stock market data from akshare API...", 
            total=None  # Indeterminate progress for single API call
        )
        
        logger.info("Fetching new stock market data from akshare API")
        try:
            stock_df = await asyncio.to_thread(
                API_RETRY_CONFIG.retry, ak.stock_zh_a_spot_em
            )
            
            # Update progress to show completion
            progress.update(task, completed=1, total=1, description="[green]✓ Stock market data fetched successfully")
            await asyncio.sleep(0.5)  # Brief pause to show completion
            
            os.makedirs(data_dir, exist_ok=True)
            stock_df.to_csv(file_path, index=False)
            logger.info("Successfully saved stock market data to %s", file_path)
            return stock_df
        except Exception as e:
            progress.update(task, description="[red]✗ Failed to fetch stock market data")
            logger.error("Failed to fetch stock market data: %s", str(e))
            raise


async def _fetch_industry_stocks(industry_name: str) -> List[tuple]:
    """
    Fetch stocks for a single industry with semaphore control and retry mechanism.

    Args:
        industry_name: Name of the industry to fetch stocks for

    Returns:
        List of (industry_name, stock_code) tuples
    """
    async with REQUEST_SEMAPHORE:
        try:
            industry_stocks = await asyncio.to_thread(
                API_RETRY_CONFIG.retry,
                ak.stock_board_industry_cons_em,
                symbol=industry_name,
            )
            return [(industry_name, code) for code in industry_stocks["代码"]]
        except Exception as e:
            logger.error(
                "Error fetching data for industry %s: %s", industry_name, str(e)
            )
            return []


# @timer
async def get_industry_stock_mapping_data(
    data_dir: str = "data/stocks",
) -> pd.DataFrame:
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
        logger.info("Loading cached industry mapping data from %s", file_path)
        return pd.read_csv(file_path, dtype={"代码": str})

    # Delete outdated files
    logger.info("Removing outdated industry mapping data files")
    for f in glob.glob(f"{data_dir}/industry_stock_mapping_df-*.csv"):
        os.remove(f)

    # Fetch industry names with retry mechanism
    logger.info("Fetching industry names from akshare API")
    try:
        industry_data = await asyncio.to_thread(
            API_RETRY_CONFIG.retry, ak.stock_board_industry_name_em
        )
        industry_names = industry_data["板块名称"]
        logger.info("Found %d industries to process", len(industry_names))
    except Exception as e:
        logger.error("Failed to fetch industry names: %s", str(e))
        raise

    # Process industries concurrently with batching and progress tracking
    batch_size = 10
    all_mappings = []

    with Progress(console=console) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(industry_names)} industries...",
            total=len(industry_names),
        )

        for i in range(0, len(industry_names), batch_size):
            batch = industry_names[i : i + batch_size]

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
        logger.info(
            "Successfully processed %d industry-stock mappings", len(all_mappings)
        )
    else:
        mapping_df = pd.DataFrame(columns=["行业", "代码"])
        logger.warning("No industry-stock mappings were successfully processed")

    # Save data with error handling
    try:
        os.makedirs(data_dir, exist_ok=True)
        mapping_df.to_csv(file_path, index=False)
        logger.info("Successfully saved industry mapping data to %s", file_path)
    except Exception as e:
        logger.error("Failed to save industry mapping data: %s", str(e))
        raise

    return mapping_df


async def main():
    """
    Main function to test data fetching functionality.
    
    This function fetches both industry mapping and stock market data
    concurrently and displays basic statistics about the retrieved data.
    
    Returns:
        Tuple of (industry_stock_mapping_df, stock_zh_a_spot_em_df)
    """
    logger.info("Starting data fetching test...")
    
    try:
        # Fetch both datasets concurrently
        industry_stock_mapping_df, stock_zh_a_spot_em_df = await asyncio.gather(
            get_industry_stock_mapping_data(), get_stock_market_data()
        )
        
        # Display basic statistics
        logger.info("Data fetching completed successfully!")
        logger.info("Industry mapping data: %d rows, %d columns", 
                   len(industry_stock_mapping_df), len(industry_stock_mapping_df.columns))
        logger.info("Stock market data: %d rows, %d columns", 
                   len(stock_zh_a_spot_em_df), len(stock_zh_a_spot_em_df.columns))
        
        # Show sample data
        console.print("\n[bold green]Industry Mapping Data Sample:[/bold green]")
        console.print(industry_stock_mapping_df.head())
        
        console.print("\n[bold green]Stock Market Data Sample:[/bold green]")
        console.print(stock_zh_a_spot_em_df.head())
        
        return industry_stock_mapping_df, stock_zh_a_spot_em_df
        
    except Exception as e:
        logger.error("Data fetching failed: %s", str(e))
        raise


if __name__ == "__main__":
    """
    When running this script directly, execute the main function.
    
    Usage:
        uv run python src/utilities/get_stock_data.py
    
    Note: Use 'uv run' to ensure all dependencies are available.
    """
    asyncio.run(main())
