"""
Centralized akshare API module for stock board constituents functionality.

This module provides a centralized interface for fetching industry board constituents
from akshare, including industry-stock mapping and constituent analysis operations.
Used to eliminate code duplication across utilities and analysis modules.
"""

import asyncio
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel

from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

# Configure environment before akshare import
configure_environment()
import akshare as ak

# Initialize logger for this module
logger = get_logger("api.akshare.stock_board_constituents")


class StockBoardConstituentsConfig(BaseModel):
    """
    Configuration model for stock board constituents API parameters.

    This model validates and provides default values for constituents operations.
    """

    batch_size: int = 5  # Number of concurrent requests for batch processing
    include_delisted: bool = False  # Whether to include delisted stocks


def fetch_industry_constituents_sync(industry_name: str) -> pd.DataFrame:
    """
    Fetch constituents for a specific industry with retry mechanism.

    Args:
        industry_name: Name of the industry (e.g., "银行", "房地产")

    Returns:
        DataFrame containing stock codes and names for the industry

    Raises:
        Exception: If API call fails after all retries
    """
    logger.debug("Fetching constituents for industry: %s", industry_name)
    
    return API_RETRY_CONFIG.retry(ak.stock_board_industry_cons_em)(symbol=industry_name)


async def fetch_industry_constituents_async(industry_name: str) -> pd.DataFrame:
    """
    Fetch constituents for a specific industry asynchronously.

    Args:
        industry_name: Name of the industry

    Returns:
        DataFrame containing stock codes and names for the industry
    """
    return await asyncio.to_thread(fetch_industry_constituents_sync, industry_name)


def process_constituents_data(
    df: pd.DataFrame, 
    include_delisted: bool = False
) -> pd.DataFrame:
    """
    Process and clean constituents data.

    Args:
        df: Raw constituents DataFrame from akshare
        include_delisted: Whether to include delisted stocks

    Returns:
        Cleaned DataFrame with standardized columns
    """
    if df.empty:
        return df
    
    # Standardize column names if needed
    df = df.copy()
    
    # Filter out delisted stocks if requested
    if not include_delisted and "状态" in df.columns:
        df = df[df["状态"] != "退市"]
        logger.debug("Filtered out delisted stocks, remaining: %d", len(df))
    
    # Ensure stock codes are strings with proper formatting
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.zfill(6)
    
    return df


async def fetch_multiple_industries_constituents_async(
    industry_names: List[str],
    batch_size: int = 5,
    include_delisted: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Fetch constituents for multiple industries concurrently with batch processing.

    Args:
        industry_names: List of industry names to fetch
        batch_size: Number of concurrent requests per batch
        include_delisted: Whether to include delisted stocks

    Returns:
        Dictionary mapping industry names to their constituent DataFrames
    """
    results = {}
    semaphore = asyncio.Semaphore(batch_size)
    
    async def fetch_single_industry(industry: str) -> tuple[str, pd.DataFrame]:
        async with semaphore:
            try:
                df = await fetch_industry_constituents_async(industry)
                processed_df = process_constituents_data(df, include_delisted)
                logger.debug("Fetched %d constituents for industry: %s", len(processed_df), industry)
                return industry, processed_df
            except Exception as e:
                logger.error("Failed to fetch constituents for industry %s: %s", industry, e)
                return industry, pd.DataFrame()
    
    # Create tasks for all industries
    tasks = [fetch_single_industry(industry) for industry in industry_names]
    
    # Execute all tasks concurrently
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in completed_results:
        if isinstance(result, Exception):
            logger.error("Task failed with exception: %s", result)
            continue
        
        industry, df = result
        results[industry] = df
    
    logger.info("Fetched constituents for %d industries", len(results))
    return results


def create_industry_stock_mapping(
    constituents_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create industry-stock mapping from constituents data.

    Args:
        constituents_data: Dictionary mapping industry names to constituent DataFrames

    Returns:
        DataFrame with columns ["行业", "代码", "名称"] for industry-stock mapping
    """
    mapping_rows = []
    
    for industry_name, df in constituents_data.items():
        if df.empty:
            continue
        
        # Extract stock codes and names
        for _, row in df.iterrows():
            if "代码" in row and "名称" in row:
                mapping_rows.append({
                    "行业": industry_name,
                    "代码": str(row["代码"]).zfill(6),
                    "名称": str(row["名称"])
                })
    
    if not mapping_rows:
        logger.warning("No valid mapping data found")
        return pd.DataFrame(columns=["行业", "代码", "名称"])
    
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Remove duplicates (a stock might be in multiple industries)
    original_count = len(mapping_df)
    mapping_df = mapping_df.drop_duplicates(subset=["代码"], keep="first")
    
    if len(mapping_df) < original_count:
        logger.info("Removed %d duplicate stock entries", original_count - len(mapping_df))
    
    logger.info("Created industry-stock mapping with %d unique stocks across %d industries", 
                len(mapping_df), len(constituents_data))
    
    return mapping_df


def get_stocks_by_industries(
    mapping_df: pd.DataFrame,
    target_industries: List[str]
) -> Dict[str, List[str]]:
    """
    Get stock codes grouped by industries from mapping DataFrame.

    Args:
        mapping_df: Industry-stock mapping DataFrame
        target_industries: List of industries to filter

    Returns:
        Dictionary mapping industry names to lists of stock codes
    """
    result = {}
    
    for industry in target_industries:
        industry_stocks = mapping_df[mapping_df["行业"] == industry]["代码"].tolist()
        result[industry] = industry_stocks
        logger.debug("Industry %s has %d stocks", industry, len(industry_stocks))
    
    return result


def get_all_stocks_from_industries(
    mapping_df: pd.DataFrame,
    target_industries: List[str]
) -> Set[str]:
    """
    Get all unique stock codes from specified industries.

    Args:
        mapping_df: Industry-stock mapping DataFrame
        target_industries: List of industries to include

    Returns:
        Set of unique stock codes
    """
    filtered_df = mapping_df[mapping_df["行业"].isin(target_industries)]
    stocks = set(filtered_df["代码"].tolist())
    
    logger.info("Found %d unique stocks across %d industries", len(stocks), len(target_industries))
    return stocks


class StockBoardConstituentsAPI:
    """
    Centralized API handler for stock board constituents operations.
    
    This class provides a high-level interface for constituents operations
    with consistent configuration and error handling.
    """

    def __init__(self, config: Optional[StockBoardConstituentsConfig] = None):
        """
        Initialize the API handler with configuration.

        Args:
            config: Configuration object. If None, uses default values
        """
        self.config = config or StockBoardConstituentsConfig()

    def fetch_industry_sync(self, industry_name: str) -> pd.DataFrame:
        """
        Fetch constituents for a specific industry synchronously.

        Args:
            industry_name: Name of the industry

        Returns:
            DataFrame with constituents data
        """
        df = fetch_industry_constituents_sync(industry_name)
        return process_constituents_data(df, self.config.include_delisted)

    async def fetch_industry_async(self, industry_name: str) -> pd.DataFrame:
        """
        Fetch constituents for a specific industry asynchronously.

        Args:
            industry_name: Name of the industry

        Returns:
            DataFrame with constituents data
        """
        df = await fetch_industry_constituents_async(industry_name)
        return process_constituents_data(df, self.config.include_delisted)

    async def fetch_multiple_industries(
        self,
        industry_names: List[str],
        batch_size: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch constituents for multiple industries using config or custom batch size.

        Args:
            industry_names: List of industry names
            batch_size: Override batch size from config

        Returns:
            Dictionary mapping industry names to constituent DataFrames
        """
        actual_batch_size = batch_size or self.config.batch_size
        return await fetch_multiple_industries_constituents_async(
            industry_names, actual_batch_size, self.config.include_delisted
        )

    def create_mapping(
        self,
        constituents_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create industry-stock mapping from constituents data.

        Args:
            constituents_data: Dictionary mapping industry names to DataFrames

        Returns:
            Industry-stock mapping DataFrame
        """
        return create_industry_stock_mapping(constituents_data)

    def get_stocks_by_industries(
        self,
        mapping_df: pd.DataFrame,
        target_industries: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get stock codes grouped by industries.

        Args:
            mapping_df: Industry-stock mapping DataFrame
            target_industries: List of industries to filter

        Returns:
            Dictionary mapping industry names to stock code lists
        """
        return get_stocks_by_industries(mapping_df, target_industries)

    def get_all_stocks_from_industries(
        self,
        mapping_df: pd.DataFrame,
        target_industries: List[str]
    ) -> Set[str]:
        """
        Get all unique stock codes from specified industries.

        Args:
            mapping_df: Industry-stock mapping DataFrame
            target_industries: List of industries to include

        Returns:
            Set of unique stock codes
        """
        return get_all_stocks_from_industries(mapping_df, target_industries)