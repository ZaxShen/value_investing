"""
Centralized akshare API module for stock market data functionality.

This module provides a centralized interface for fetching market data from akshare,
including spot data for different exchanges (Shanghai, Shenzhen, Beijing) and 
unified market data operations. Used to eliminate code duplication across utilities.
"""

import asyncio
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG, retry_call

# Configure environment before akshare import
configure_environment()
import akshare as ak

# Initialize logger for this module
logger = get_logger("api.akshare.stock_market_data")


class StockMarketDataConfig(BaseModel):
    """
    Configuration model for stock market data API parameters.

    This model validates and provides default values for market data operations.
    """

    markets: List[str] = ["SH", "SZ", "BJ"]  # Markets to fetch data from
    timeout: Optional[float] = None  # Timeout for API calls


def fetch_shanghai_spot_sync() -> pd.DataFrame:
    """
    Fetch Shanghai A-share spot market data with retry mechanism.

    Returns:
        DataFrame containing Shanghai A-share market data

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching Shanghai A-share spot data")
    
    return API_RETRY_CONFIG.retry(ak.stock_sh_a_spot_em)


def fetch_shenzhen_spot_sync() -> pd.DataFrame:
    """
    Fetch Shenzhen A-share spot market data with retry mechanism.

    Returns:
        DataFrame containing Shenzhen A-share market data

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching Shenzhen A-share spot data")
    
    return API_RETRY_CONFIG.retry(ak.stock_sz_a_spot_em)


def fetch_beijing_spot_sync() -> pd.DataFrame:
    """
    Fetch Beijing A-share spot market data with retry mechanism.

    Returns:
        DataFrame containing Beijing A-share market data

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching Beijing A-share spot data")
    
    return API_RETRY_CONFIG.retry(ak.stock_bj_a_spot_em)


def fetch_all_a_shares_spot_sync(timeout: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch all A-share spot market data with retry mechanism.

    This is a unified call to get all A-share market data across all exchanges.

    Args:
        timeout: Optional timeout for the API call

    Returns:
        DataFrame containing all A-share market data

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching all A-share spot data")
    
    return retry_call(ak.stock_zh_a_spot_em, timeout=timeout)


async def fetch_shanghai_spot_async() -> pd.DataFrame:
    """
    Fetch Shanghai A-share spot market data asynchronously.

    Returns:
        DataFrame containing Shanghai A-share market data
    """
    return await asyncio.to_thread(fetch_shanghai_spot_sync)


async def fetch_shenzhen_spot_async() -> pd.DataFrame:
    """
    Fetch Shenzhen A-share spot market data asynchronously.

    Returns:
        DataFrame containing Shenzhen A-share market data
    """
    return await asyncio.to_thread(fetch_shenzhen_spot_sync)


async def fetch_beijing_spot_async() -> pd.DataFrame:
    """
    Fetch Beijing A-share spot market data asynchronously.

    Returns:
        DataFrame containing Beijing A-share market data
    """
    return await asyncio.to_thread(fetch_beijing_spot_sync)


async def fetch_all_a_shares_spot_async(timeout: Optional[float] = None) -> pd.DataFrame:
    """
    Fetch all A-share spot market data asynchronously.

    Args:
        timeout: Optional timeout for the API call

    Returns:
        DataFrame containing all A-share market data
    """
    return await asyncio.to_thread(fetch_all_a_shares_spot_sync, timeout)


async def fetch_multiple_markets_async(
    markets: List[str],
    progress_callbacks: Optional[dict] = None
) -> dict[str, pd.DataFrame]:
    """
    Fetch market data for multiple exchanges concurrently.

    Args:
        markets: List of market codes ("SH", "SZ", "BJ")
        progress_callbacks: Optional dict mapping market codes to progress callback functions

    Returns:
        Dictionary mapping market codes to their respective DataFrames

    Raises:
        ValueError: If unknown market code is provided
    """
    market_functions = {
        "SH": fetch_shanghai_spot_async,
        "SZ": fetch_shenzhen_spot_async, 
        "BJ": fetch_beijing_spot_async,
    }
    
    # Validate market codes
    invalid_markets = set(markets) - set(market_functions.keys())
    if invalid_markets:
        raise ValueError(f"Unknown market codes: {invalid_markets}. Valid codes: {list(market_functions.keys())}")
    
    # Create tasks for each market
    tasks = []
    for market in markets:
        task = market_functions[market]()
        tasks.append((market, task))
    
    # Execute all tasks concurrently
    results = {}
    completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    for i, ((market, _), result) in enumerate(zip(tasks, completed_tasks)):
        if isinstance(result, Exception):
            logger.error("Failed to fetch data for market %s: %s", market, result)
            results[market] = pd.DataFrame()  # Return empty DataFrame on error
        else:
            results[market] = result
            logger.debug("Successfully fetched data for market %s: %d records", market, len(result))
            
        # Call progress callback if provided
        if progress_callbacks and market in progress_callbacks:
            progress_callbacks[market]()
    
    return results


def combine_market_data(market_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine market data from multiple exchanges into a single DataFrame.

    Args:
        market_dataframes: Dictionary mapping market codes to DataFrames

    Returns:
        Combined DataFrame with all market data

    Raises:
        ValueError: If no valid data is provided
    """
    valid_dfs = [(market, df) for market, df in market_dataframes.items() if not df.empty]
    
    if not valid_dfs:
        logger.warning("No valid market data to combine")
        return pd.DataFrame()
    
    combined_dfs = []
    for market, df in valid_dfs:
        # Add market column for identification
        df_copy = df.copy()
        df_copy['市场'] = market
        combined_dfs.append(df_copy)
        logger.debug("Added %d records from %s market", len(df_copy), market)
    
    combined = pd.concat(combined_dfs, ignore_index=True)
    logger.info("Combined market data: %d total records from %d markets", len(combined), len(valid_dfs))
    
    return combined


class StockMarketDataAPI:
    """
    Centralized API handler for stock market data operations.
    
    This class provides a high-level interface for market data operations
    with consistent configuration and error handling.
    """

    def __init__(self, config: Optional[StockMarketDataConfig] = None):
        """
        Initialize the API handler with configuration.

        Args:
            config: Configuration object. If None, uses default values
        """
        self.config = config or StockMarketDataConfig()

    def fetch_all_sync(self, timeout: Optional[float] = None) -> pd.DataFrame:
        """
        Fetch all A-share market data synchronously.

        Args:
            timeout: Override timeout from config

        Returns:
            DataFrame with all A-share market data
        """
        actual_timeout = timeout if timeout is not None else self.config.timeout
        return fetch_all_a_shares_spot_sync(actual_timeout)

    async def fetch_all_async(self, timeout: Optional[float] = None) -> pd.DataFrame:
        """
        Fetch all A-share market data asynchronously.

        Args:
            timeout: Override timeout from config

        Returns:
            DataFrame with all A-share market data
        """
        actual_timeout = timeout if timeout is not None else self.config.timeout
        return await fetch_all_a_shares_spot_async(actual_timeout)

    def fetch_market_sync(self, market: str) -> pd.DataFrame:
        """
        Fetch data for a specific market synchronously.

        Args:
            market: Market code ("SH", "SZ", "BJ")

        Returns:
            DataFrame with market data

        Raises:
            ValueError: If unknown market code is provided
        """
        market_functions = {
            "SH": fetch_shanghai_spot_sync,
            "SZ": fetch_shenzhen_spot_sync,
            "BJ": fetch_beijing_spot_sync,
        }
        
        if market not in market_functions:
            raise ValueError(f"Unknown market code: {market}. Valid codes: {list(market_functions.keys())}")
        
        return market_functions[market]()

    async def fetch_market_async(self, market: str) -> pd.DataFrame:
        """
        Fetch data for a specific market asynchronously.

        Args:
            market: Market code ("SH", "SZ", "BJ")

        Returns:
            DataFrame with market data

        Raises:
            ValueError: If unknown market code is provided
        """
        market_functions = {
            "SH": fetch_shanghai_spot_async,
            "SZ": fetch_shenzhen_spot_async,
            "BJ": fetch_beijing_spot_async,
        }
        
        if market not in market_functions:
            raise ValueError(f"Unknown market code: {market}. Valid codes: {list(market_functions.keys())}")
        
        return await market_functions[market]()

    async def fetch_multiple_markets(
        self,
        markets: Optional[List[str]] = None,
        progress_callbacks: Optional[dict] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple markets concurrently using config or custom markets.

        Args:
            markets: List of market codes. If None, uses config values
            progress_callbacks: Optional progress callbacks

        Returns:
            Dictionary mapping market codes to DataFrames
        """
        target_markets = markets or self.config.markets
        return await fetch_multiple_markets_async(target_markets, progress_callbacks)

    def combine_data(self, market_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine market data from multiple exchanges.

        Args:
            market_dataframes: Dictionary mapping market codes to DataFrames

        Returns:
            Combined DataFrame with all market data
        """
        return combine_market_data(market_dataframes)