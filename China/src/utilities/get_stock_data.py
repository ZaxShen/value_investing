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
    from src.utilities.retry import API_RETRY_CONFIG, retry_call

    configure_environment()  # Ensure tqdm is disabled
except ModuleNotFoundError:
    # When running as standalone script, add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.settings import configure_environment
    from src.utilities.logger import get_logger
    from src.utilities.retry import API_RETRY_CONFIG, retry_call

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
async def get_stock_market_data(
    data_dir: str = "data/stocks", progress: Progress = None
) -> pd.DataFrame:
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
    use_own_progress = progress is None
    if use_own_progress:
        progress = Progress(console=console)
        progress.start()

    # Create separate progress tasks for each market
    sh_task = progress.add_task(
        "    [cyan]Fetching SH stock market data...", total=None
    )
    sz_task = progress.add_task(
        "    [cyan]Fetching SZ stock market data...", total=None
    )
    bj_task = progress.add_task(
        "    [cyan]Fetching BJ stock market data...", total=None
    )

    logger.info("Fetching stock market data from three markets concurrently")

    async def fetch_market_data(market_func, task_id, market_name):
        """Fetch data for a specific market and update its progress."""
        try:
            result = await asyncio.to_thread(retry_call, market_func, timeout=None)
            progress.update(
                task_id,
                completed=1,
                total=1,
                description=f"    [green]✓ {market_name} stock market data fetched successfully",
            )
            return result
        except Exception:
            progress.update(
                task_id,
                description=f"[red]✗ Failed to fetch {market_name} stock market data",
            )
            raise

    try:
        # Fetch data from three markets concurrently with individual progress tracking
        sh_df, sz_df, bj_df = await asyncio.gather(
            fetch_market_data(ak.stock_sh_a_spot_em, sh_task, "SH"),
            fetch_market_data(ak.stock_sz_a_spot_em, sz_task, "SZ"),
            fetch_market_data(ak.stock_bj_a_spot_em, bj_task, "BJ"),
        )

        await asyncio.sleep(0.5)  # Brief pause to show all completions

        # Concatenate the three DataFrames
        stock_df = pd.concat([sh_df, sz_df, bj_df], ignore_index=True)
        logger.info(
            "Combined data from three markets: SH(%d), SZ(%d), BJ(%d) = Total(%d)",
            len(sh_df),
            len(sz_df),
            len(bj_df),
            len(stock_df),
        )

        os.makedirs(data_dir, exist_ok=True)
        stock_df.to_csv(file_path, index=False)
        logger.info("Successfully saved stock market data to %s", file_path)
        return stock_df
    except Exception as e:
        progress.update(task, description="[red]✗ Failed to fetch stock market data")
        logger.error("Failed to fetch stock market data: %s", str(e))
        raise
    finally:
        if use_own_progress:
            progress.stop()


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
    data_dir: str = "data/stocks", progress: Progress = None
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

    use_own_progress = progress is None
    if use_own_progress:
        progress = Progress(console=console)
        progress.start()

    task = progress.add_task(
        f"[cyan]Processing {len(industry_names)} industries...",
        total=len(industry_names),
    )

    try:
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

        # Update progress to show completion
        progress.update(
            task, description="    [green]✓ Industry data processed successfully"
        )
    finally:
        if use_own_progress:
            progress.stop()

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
    using a shared progress display to avoid Rich conflicts.

    Returns:
        Tuple of (industry_stock_mapping_df, stock_zh_a_spot_em_df)
    """
    logger.info("Starting data fetching test...")

    try:
        # Use single Progress context for both operations
        with Progress(console=console) as progress:
            industry_stock_mapping_df, stock_zh_a_spot_em_df = await asyncio.gather(
                get_industry_stock_mapping_data(progress=progress),
                get_stock_market_data(progress=progress),
            )

        # Display basic statistics
        logger.info("Data fetching completed successfully!")
        logger.info(
            "Industry mapping data: %d rows, %d columns",
            len(industry_stock_mapping_df),
            len(industry_stock_mapping_df.columns),
        )
        logger.info(
            "Stock market data: %d rows, %d columns",
            len(stock_zh_a_spot_em_df),
            len(stock_zh_a_spot_em_df.columns),
        )

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
