"""
Trading calendar utility for Chinese stock markets.

This module provides functionality to determine trading days and non-trading days
for Chinese stock exchanges, with intelligent caching for performance.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set

import akshare as ak
import pandas as pd

from src.utilities.logger import get_logger

logger = get_logger("trading_calendar")


class TradingCalendar:
    """Chinese stock market trading calendar with caching."""
    
    def __init__(self, cache_dir: str = "data/reference"):
        """
        Initialize trading calendar with cache directory.
        
        Args:
            cache_dir: Directory to store cached trading calendar data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._trading_days_cache = {}
        self._non_trading_days_cache = {}
    
    def _get_cache_path(self, year: int) -> Path:
        """Get cache file path for a given year."""
        return self.cache_dir / f"trading_calendar_{year}.json"
    
    def _load_trading_days_from_cache(self, year: int) -> Set[str]:
        """Load trading days from cache file."""
        cache_path = self._get_cache_path(year)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    trading_days = set(data.get('trading_days', []))
                    logger.info(f"Loaded {len(trading_days)} trading days for {year} from cache")
                    return trading_days
            except Exception as e:
                logger.error(f"Error loading trading calendar cache for {year}: {e}")
        
        return set()
    
    def _save_trading_days_to_cache(self, year: int, trading_days: Set[str]) -> None:
        """Save trading days to cache file."""
        cache_path = self._get_cache_path(year)
        
        try:
            data = {
                'year': year,
                'trading_days': list(trading_days),
                'generated_at': datetime.now().isoformat(),
                'total_days': len(trading_days)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cached {len(trading_days)} trading days for {year}")
            
        except Exception as e:
            logger.error(f"Error saving trading calendar cache for {year}: {e}")
    
    def _fetch_trading_days_from_api(self, year: int) -> Set[str]:
        """Fetch trading days from akshare API."""
        try:
            logger.info(f"Fetching trading days for {year} from akshare API...")
            
            # Get trading days for the year using akshare
            # We'll use Shanghai exchange as reference (SH and SZ have same trading days)
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # Fetch actual trading data to determine trading days
            # Use a representative stock to get trading days
            df = ak.stock_zh_a_hist(
                symbol="000001",  # Ping An Bank (SZ) - always active
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            
            trading_days = set()
            if not df.empty:
                # Ensure the date column is datetime type
                if '日期' in df.columns:
                    # Convert to datetime if it's not already
                    df['日期'] = pd.to_datetime(df['日期'])
                    # Convert dates to string format YYYY-MM-DD
                    trading_days = set(df['日期'].dt.strftime('%Y-%m-%d').tolist())
                else:
                    # Fallback: use index if '日期' column not found
                    if hasattr(df.index, 'strftime'):
                        trading_days = set(df.index.strftime('%Y-%m-%d').tolist())
                    else:
                        df.index = pd.to_datetime(df.index)
                        trading_days = set(df.index.strftime('%Y-%m-%d').tolist())
            
            logger.info(f"Fetched {len(trading_days)} trading days for {year}")
            return trading_days
            
        except Exception as e:
            logger.error(f"Error fetching trading days for {year}: {e}")
            return set()
    
    def get_trading_days(self, year: int) -> Set[str]:
        """
        Get trading days for a given year.
        
        Args:
            year: Year to get trading days for
            
        Returns:
            Set of trading day strings in YYYY-MM-DD format
        """
        # Check memory cache first
        if year in self._trading_days_cache:
            return self._trading_days_cache[year]
        
        # Try to load from file cache
        trading_days = self._load_trading_days_from_cache(year)
        
        # If cache miss, fetch from API
        if not trading_days:
            trading_days = self._fetch_trading_days_from_api(year)
            if trading_days:
                self._save_trading_days_to_cache(year, trading_days)
        
        # Store in memory cache
        self._trading_days_cache[year] = trading_days
        return trading_days
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day.
        
        Args:
            date: Date to check
            
        Returns:
            True if the date is a trading day, False otherwise
        """
        date_str = date.strftime('%Y-%m-%d')
        year = date.year
        
        trading_days = self.get_trading_days(year)
        return date_str in trading_days
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before the given date.
        
        Args:
            date: Reference date
            
        Returns:
            Previous trading day
        """
        # Start from the day before
        current_date = date - timedelta(days=1)
        max_attempts = 10  # Prevent infinite loops
        
        for _ in range(max_attempts):
            if self.is_trading_day(current_date):
                return current_date
            
            # If we hit a year boundary, make sure we have that year's data
            if current_date.year != date.year:
                self.get_trading_days(current_date.year)
            
            current_date -= timedelta(days=1)
        
        # Fallback: if we can't find a trading day, just return the original date - 1
        logger.warning(f"Could not find previous trading day for {date}, using fallback")
        return date - timedelta(days=1)


# Global instance for easy access
_trading_calendar = None


def get_trading_calendar() -> TradingCalendar:
    """Get global trading calendar instance."""
    global _trading_calendar
    if _trading_calendar is None:
        _trading_calendar = TradingCalendar()
    return _trading_calendar


def get_previous_trading_day(date: datetime) -> datetime:
    """
    Convenience function to get previous trading day.
    
    Args:
        date: Reference date
        
    Returns:
        Previous trading day
    """
    calendar = get_trading_calendar()
    return calendar.get_previous_trading_day(date)


def is_trading_day(date: datetime) -> bool:
    """
    Convenience function to check if date is a trading day.
    
    Args:
        date: Date to check
        
    Returns:
        True if trading day, False otherwise
    """
    calendar = get_trading_calendar()
    return calendar.is_trading_day(date)