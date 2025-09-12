"""
FHPS (分红派息送股) data caching for Chinese equity markets.

This module provides caching functionality for FHPS data in two phases:
Phase 1: Cache historical FHPS data for specified years
Phase 2: Cache target year FHPS data with historical prices

This script handles the data fetching and caching operations separately from filtering.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from pydantic import BaseModel

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
from src.utilities.logger import get_logger
from src.utilities.trading_calendar import get_previous_trading_day

configure_environment()

# akshare imports after environment configuration
import akshare as ak

# Initialize logger for this module
logger = get_logger("fhps_caching")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class Phase1CachingConfig(BaseModel):
    """Configuration model for Phase 1 caching (historical FHPS data)."""

    historical_years: List[int] = [2020, 2021, 2022, 2023]
    cache_dir: str = "data/fhps"
    skip_existing: bool = True


class Phase2CachingConfig(BaseModel):
    """Configuration model for Phase 2 caching (target year with prices)."""

    target_years: List[int] = [2024]
    cache_dir: str = "data/fhps"
    skip_existing: bool = True


class LatestFileConfig(BaseModel):
    """Configuration model for latest file settings."""

    stock_fhps_em_latest_filename: str = "stock_fhps_em-latest.csv"


class FileConfig(BaseModel):
    """Configuration model for file metadata."""

    config_name: str = "PROD"
    description: str = ""
    version: str = "1.9.3"


class FhpsCachingConfig(BaseModel):
    """Main configuration model for FHPS caching operations."""

    phase1_caching: Phase1CachingConfig = Phase1CachingConfig()
    phase2_caching: Phase2CachingConfig = Phase2CachingConfig()
    latest_file: LatestFileConfig = LatestFileConfig()
    file_config: FileConfig = FileConfig()


def load_config(config_name: Optional[str] = None) -> FhpsCachingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        FhpsCachingConfig: Loaded configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_dir = Path("input/config/caching/fhps/")
    config_name = config_name or "config"
    config_path = config_dir / f"{config_name}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return FhpsCachingConfig(**config_data)


class FhpsCaching:
    """
    Cache FHPS data for Chinese equity markets.

    This class handles two phases of caching:
    - Phase 1: Historical FHPS data for multiple years
    - Phase 2: Target year FHPS data with historical price enrichment
    """

    def __init__(self, config_name: Optional[str] = None):
        """
        Initialize FhpsCaching with configuration.

        Args:
            config_name: Optional config name to use for loading configuration
        """
        # Load configuration
        self.config = load_config(config_name)

        self.logger = get_logger("fhps_caching")

    async def get_stock_price_async(
        self, stock_code: str, date: datetime
    ) -> Optional[float]:
        """
        Get stock price asynchronously for the previous trading day before the given date.
        Uses trading calendar to find valid trading days and implements robust fallback.

        Args:
            stock_code: Stock code (6-digit format)
            date: Ex-dividend date (will get price from previous trading day)

        Returns:
            Stock price from previous trading day or None if not found
        """
        max_attempts = 5  # Prevent infinite loops

        # Get the previous trading day before the ex-dividend date
        current_date = get_previous_trading_day(date)

        for _ in range(max_attempts):
            try:
                async with REQUEST_SEMAPHORE:
                    # Use asyncio.to_thread for non-blocking akshare calls
                    df_price = await asyncio.to_thread(
                        ak.stock_zh_a_hist,
                        stock_code,
                        period="daily",
                        start_date=current_date.strftime("%Y%m%d"),
                        end_date=current_date.strftime("%Y%m%d"),
                        adjust="",  # 不复权 - for accurate fill-right analysis
                    )
                    if not df_price.empty:
                        price = df_price["收盘"].iloc[0]  # Close price
                        self.logger.debug(
                            f"Found price for {stock_code} on {current_date.strftime('%Y-%m-%d')}: {price}"
                        )
                        return price

                    # No data for this date, try previous trading day
                    self.logger.debug(
                        f"No price data for {stock_code} on {current_date.strftime('%Y-%m-%d')}, trying previous trading day"
                    )
                    current_date = get_previous_trading_day(current_date)

            except Exception as e:
                self.logger.error(
                    f"Error fetching price for {stock_code} on {current_date}: {e}"
                )
                # Try previous trading day on API error too
                current_date = get_previous_trading_day(current_date)

        self.logger.warning(
            f"Could not find price for {stock_code} after {max_attempts} attempts"
        )
        return None

    async def phase1_caching(self) -> None:
        """
        Phase 1 caching: Fetch and cache historical FHPS data for specified years.

        For each year in historical_years, fetches stock_fhps_em data and caches
        it after filtering out rows with NaN in "送转股份-送转总比例".
        """
        self.logger.info("Starting Phase 1 caching (historical FHPS data)")

        # Create cache directory
        cache_dir = self.config.phase1_caching.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        for year in self.config.phase1_caching.historical_years:
            cache_filename = f"stock_fhps_em-{year}.csv"
            cache_path = os.path.join(cache_dir, cache_filename)

            # Skip if file exists and skip_existing is True
            if self.config.phase1_caching.skip_existing and os.path.exists(cache_path):
                self.logger.info(
                    f"✅ CACHED FILE EXISTS - Skipping {year}: {cache_path}"
                )
                print(f"✅ CACHED FILE EXISTS - Skipping {year}: {cache_path}")
                continue

            self.logger.info(f"🌐 Fetching FHPS data for year {year}")
            print(f"🌐 Fetching FHPS data for year {year}")

            try:
                # Fetch FHPS data for end of year
                stock_fhps_em_df = await asyncio.to_thread(
                    ak.stock_fhps_em, date=f"{year}1231"
                )

                if stock_fhps_em_df is None or stock_fhps_em_df.empty:
                    self.logger.warning(f"No FHPS data found for year {year}")
                    print(f"⚠️ No FHPS data found for year {year}")
                    continue

                # Filter out NaN values in "送转股份-送转总比例"
                df_filtered = stock_fhps_em_df.dropna(
                    subset=["送转股份-送转总比例"]
                ).copy()

                # Ensure stock codes are strings with proper 6-digit format
                if "代码" in df_filtered.columns:
                    df_filtered.loc[:, "代码"] = df_filtered["代码"].apply(
                        lambda x: str(x).zfill(6)
                    )

                # Reset index and save to cache
                df_filtered.reset_index(drop=True).to_csv(
                    cache_path, index=True, encoding="utf-8-sig"
                )

                self.logger.info(
                    f"✅ Cached {len(df_filtered)} FHPS records for {year}: {cache_path}"
                )
                print(
                    f"✅ Cached {len(df_filtered)} FHPS records for {year}: {cache_path}"
                )

            except Exception as e:
                error_msg = f"Failed to cache FHPS data for year {year}: {str(e)}"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

        self.logger.info("Phase 1 caching completed")
        print("✅ Phase 1 caching completed")

    async def phase2_caching(self) -> None:
        """
        Phase 2 caching: Fetch target year FHPS data and enrich with historical prices.

        For each year in target_years, fetches stock_fhps_em data, filters it,
        and enriches with 除权除息前日股价 (pre ex-dividend price).
        """
        self.logger.info("Starting Phase 2 caching (target year with prices)")
        print("🚀 Starting Phase 2 caching (target year with prices)")

        # Create cache directory
        cache_dir = self.config.phase2_caching.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        for year in self.config.phase2_caching.target_years:
            cache_filename = f"stock_fhps_em_with_prices-{year}.csv"
            cache_path = os.path.join(cache_dir, cache_filename)

            # Skip if file exists and skip_existing is True
            if self.config.phase2_caching.skip_existing and os.path.exists(cache_path):
                self.logger.info(
                    f"✅ CACHED FILE EXISTS - Skipping {year}: {cache_path}"
                )
                print(f"✅ CACHED FILE EXISTS - Skipping {year}: {cache_path}")
                continue

            self.logger.info(f"🌐 Fetching FHPS data with prices for year {year}")
            print(f"🌐 Fetching FHPS data with prices for year {year}")

            try:
                # Fetch FHPS data for end of year
                stock_fhps_em_df = await asyncio.to_thread(
                    ak.stock_fhps_em, date=f"{year}1231"
                )

                if stock_fhps_em_df is None or stock_fhps_em_df.empty:
                    self.logger.warning(f"No FHPS data found for year {year}")
                    print(f"⚠️ No FHPS data found for year {year}")
                    continue

                # Filter out NaN values in "送转股份-送转总比例"
                df_filtered = stock_fhps_em_df.dropna(
                    subset=["送转股份-送转总比例"]
                ).copy()

                # Ensure stock codes are strings with proper 6-digit format
                if "代码" in df_filtered.columns:
                    df_filtered.loc[:, "代码"] = df_filtered["代码"].apply(
                        lambda x: str(x).zfill(6)
                    )

                # Convert ex-dividend date to datetime if not already
                if "除权除息日" in df_filtered.columns:
                    df_filtered.loc[:, "除权除息日"] = pd.to_datetime(
                        df_filtered["除权除息日"], format="%Y-%m-%d"
                    )

                # Filter stocks with ex-dividend dates before today
                today = datetime.today()
                filter_past = df_filtered.loc[:, "除权除息日"] < today
                df_past_dates = df_filtered[filter_past]

                self.logger.info(
                    f"Found {len(df_past_dates)} stocks with past ex-dividend dates for {year}"
                )
                print(
                    f"📊 Found {len(df_past_dates)} stocks with past ex-dividend dates for {year}"
                )

                # Enrich with historical prices
                self.logger.info("💰 Fetching historical prices for all stocks...")
                print("💰 Fetching historical prices for all stocks...")

                ex_prices = []
                for _, row in df_past_dates.iterrows():
                    stock_code = str(row["代码"]).zfill(6)
                    ex_date = row["除权除息日"]
                    try:
                        ex_price = await self.get_stock_price_async(stock_code, ex_date)
                        ex_prices.append(ex_price)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to get price for {stock_code} on {ex_date}: {e}"
                        )
                        ex_prices.append(None)

                # Add the historical prices column
                df_enriched = df_past_dates.copy()
                df_enriched.loc[:, "除权除息前日股价"] = ex_prices

                # Reset index and save to cache
                df_enriched.reset_index(drop=True).to_csv(
                    cache_path, index=True, encoding="utf-8-sig"
                )

                self.logger.info(
                    f"✅ Cached {len(df_enriched)} enriched FHPS records for {year}: {cache_path}"
                )
                print(
                    f"✅ Cached {len(df_enriched)} enriched FHPS records for {year}: {cache_path}"
                )

                # Also create the "latest" file for the most recent target year
                latest_filename = self.config.latest_file.stock_fhps_em_latest_filename
                latest_path = os.path.join(cache_dir, latest_filename)
                df_enriched.reset_index(drop=True).to_csv(
                    latest_path, index=True, encoding="utf-8-sig"
                )

                self.logger.info(f"✅ Created latest file: {latest_path}")
                print(f"✅ Created latest file: {latest_path}")

            except Exception as e:
                error_msg = (
                    f"Failed to cache FHPS data with prices for year {year}: {str(e)}"
                )
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

        self.logger.info("Phase 2 caching completed")
        print("🚀 Phase 2 caching completed")

    async def run_caching(self) -> None:
        """
        Run both phases of FHPS caching operations.
        """
        self.logger.info("Starting FHPS caching operations")
        print("🚀 Starting FHPS caching operations")

        try:
            # Run Phase 1 caching
            await self.phase1_caching()

            # Run Phase 2 caching
            await self.phase2_caching()

            self.logger.info("✅ All FHPS caching operations completed successfully")
            print("🎉 All FHPS caching operations completed successfully")

        except Exception as e:
            error_msg = f"FHPS caching failed: {str(e)}"
            self.logger.error(error_msg)
            print(f"💥 {error_msg}")
            raise


async def main():
    """Main entry point for running FHPS caching operations."""
    try:
        caching = FhpsCaching()
        await caching.run_caching()
    except Exception as e:
        print(f"💥 FHPS caching failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
