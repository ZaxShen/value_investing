"""
FHPS (分红派息送股) filtering and analysis for Chinese equity markets.

This module provides a FhpsFilter class that encapsulates asynchronous functions
to filter and analyze Chinese stocks with dividend/split plans. It processes
stocks with ex-dividend dates, calculates performance metrics, and adds fund flow
data for comprehensive analysis.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pandas as pd
import yaml
from pydantic import BaseModel

from src.api.akshare import (
    StockIndividualFundFlowAPI,
    StockIndividualFundFlowConfig,
    get_market_by_stock_code,
)

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
from src.utilities.logger import get_logger

configure_environment()

# akshare imports now handled through centralized API modules
import akshare as ak

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Initialize logger for this module
logger = get_logger("fhps_filter")

# Create a semaphore to limit concurrent requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class FileConfig(BaseModel):
    """
    Configuration model for file-related settings.

    This model handles file configuration metadata including config name,
    version, and description.
    """

    config_name: str = "PROD"
    description: str = ""
    version: str = ""


class FhpsFilterConfig(BaseModel):
    """
    Configuration model for FhpsFilter class parameters.

    This model validates and provides default values for the FhpsFilter class constants.
    """

    # FHPS data configuration
    fhps_date: str = ""  # Date for FHPS data query (YYYYMMDD format)
    min_transfer_ratio: float = 1.0  # Minimum transfer ratio to filter
    max_price_change_percent: float = (
        10.0  # Maximum price change percentage for filtering
    )

    # Processing configuration
    batch_size: int = 10  # Batch size for concurrent processing

    # Output configuration
    report_dir: str = "output/reports/filters/fhps_filter"
    output_filename_template: str = "除权除息股票-{date}.csv"


class Config(BaseModel):
    """
    Configuration model for nested YAML structure supporting both akshare and FhpsFilter configs.

    This model handles the nested structure from input/config/filters/fhps_filter.
    """

    akshare: Dict[str, Dict[str, Any]] = {}
    fhps_filter: FhpsFilterConfig = FhpsFilterConfig()
    file_config: FileConfig = FileConfig()


def load_config(
    config_name: Optional[str] = None,
) -> Tuple[StockIndividualFundFlowConfig, FhpsFilterConfig, FileConfig]:
    """
    Load nested configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        tuple: (akshare_config, fhps_filter_config, file_config)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_dir = Path("input/config/filters/fhps_filter/")
    config_name = config_name or "config"
    config_path = config_dir / f"{config_name}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Check if it's nested format (has 'akshare' key) or flat format
    if "akshare" in config_data:
        # Nested format - extract each section
        configs = Config(**config_data)
        akshare_config = StockIndividualFundFlowConfig(
            **configs.akshare.get("stock_individual_fund_flow", {})
        )
        fhps_filter_config = configs.fhps_filter
        file_config = configs.file_config
    else:
        # Flat format - assume all config is for akshare
        akshare_config = StockIndividualFundFlowConfig(**config_data)
        fhps_filter_config = FhpsFilterConfig()
        file_config = FileConfig()

    return akshare_config, fhps_filter_config, file_config


class FhpsFilter:
    """
    Filter and analyze stocks with dividend/split plans (FHPS).

    This class processes stocks with ex-dividend dates, calculates performance metrics,
    and enriches data with fund flow information for comprehensive analysis.
    """

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
        config_name: Optional[str] = None,
    ):
        """
        Initialize FhpsFilter with market data and configuration.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
            config_name: Optional config name to use for loading configuration
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df

        # Load configuration
        self.akshare_config, self.filter_config, self.file_config = load_config(
            config_name
        )

        # Apply class constants from config
        # Default to end of year if no date specified (more likely to have FHPS data)
        self.FHPS_DATE = self.filter_config.fhps_date or "20241231"
        self.MIN_TRANSFER_RATIO = self.filter_config.min_transfer_ratio
        self.MAX_PRICE_CHANGE_PERCENT = self.filter_config.max_price_change_percent
        self.BATCH_SIZE = self.filter_config.batch_size
        self.REPORT_DIR = self.filter_config.report_dir
        self.OUTPUT_FILENAME_TEMPLATE = self.filter_config.output_filename_template

        self.logger = get_logger("fhps_filter")

    async def _get_cached_fhps_data(self) -> Optional[pd.DataFrame]:
        """
        Get FHPS data with caching support.

        Returns:
            DataFrame with FHPS data or None if no data available
        """
        # Define cache paths
        cache_dir = "data/fhps"
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = f"stock_fhps_em-{self.FHPS_DATE}.csv"
        cache_path = os.path.join(cache_dir, cache_filename)

        # Check if cached file exists
        if os.path.exists(cache_path):
            self.logger.info(f"✅ CACHED DATA FOUND! Loading FHPS data from: {cache_path}")
            print(f"✅ CACHED DATA FOUND! Loading FHPS data from: {cache_path}")
            try:
                df = pd.read_csv(cache_path, encoding="utf-8-sig")
                # Ensure stock codes are strings with proper 6-digit format
                if "代码" in df.columns:
                    df["代码"] = df["代码"].apply(lambda x: str(x).zfill(6))
                self.logger.info(f"✅ Successfully loaded {len(df)} FHPS records from cache")
                print(f"✅ Successfully loaded {len(df)} FHPS records from cache")
                return df
            except Exception as e:
                self.logger.error(f"❌ Error loading cached FHPS data: {e}")
                print(f"❌ Error loading cached FHPS data: {e}")
                self.logger.info("🔄 Falling back to API fetch...")
                print("🔄 Falling back to API fetch...")
                # Fall through to fetch fresh data

        # Fetch fresh data from API
        self.logger.info(f"🌐 NO CACHE FOUND - Fetching fresh FHPS data from API for date: {self.FHPS_DATE}")
        print(f"🌐 NO CACHE FOUND - Fetching fresh FHPS data from API for date: {self.FHPS_DATE}")
        try:
            stock_fhps_em_df = await asyncio.to_thread(
                ak.stock_fhps_em, date=self.FHPS_DATE
            )

            if stock_fhps_em_df is not None and not stock_fhps_em_df.empty:
                # Ensure stock codes are strings with proper 6-digit format
                if "代码" in stock_fhps_em_df.columns:
                    stock_fhps_em_df["代码"] = stock_fhps_em_df["代码"].apply(lambda x: str(x).zfill(6))
                
                # Cache the data
                self.logger.info(f"💾 Caching {len(stock_fhps_em_df)} FHPS records to: {cache_path}")
                print(f"💾 Caching {len(stock_fhps_em_df)} FHPS records to: {cache_path}")
                stock_fhps_em_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
                self.logger.info("✅ FHPS data successfully cached for future use")
                print("✅ FHPS data successfully cached for future use")

            return stock_fhps_em_df

        except Exception as api_error:
            error_msg = (
                f"Failed to fetch FHPS data for date {self.FHPS_DATE}: {str(api_error)}"
            )
            self.logger.error(error_msg)
            return None

    async def get_stock_price_async(
        self, stock_code: str, date: datetime
    ) -> Optional[float]:
        """
        Get stock price for a specific date using asyncio.to_thread.

        Args:
            stock_code: Stock code (6-digit format)
            date: Date to get price for

        Returns:
            Close price for the specified date, or None if not found
        """
        try:
            async with REQUEST_SEMAPHORE:
                # Use asyncio.to_thread for non-blocking akshare calls
                df_price = await asyncio.to_thread(
                    ak.stock_zh_a_hist,
                    stock_code,
                    period="daily",
                    start_date=date.strftime("%Y%m%d"),
                    end_date=date.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                if not df_price.empty:
                    return df_price["收盘"].iloc[0]  # Close price
                return None
        except Exception as e:
            self.logger.error(f"Error fetching price for {stock_code} on {date}: {e}")
            return None

    async def get_fund_flow_data(self, stock_code: str) -> Dict[str, Optional[float]]:
        """
        Get fund flow data for a stock using the configured periods.

        Args:
            stock_code: Stock code (6-digit format)

        Returns:
            Dictionary containing fund flow data for different periods
        """
        fund_flow_api = StockIndividualFundFlowAPI(self.akshare_config)

        try:
            # Determine market from stock code
            market = get_market_by_stock_code(stock_code)

            # Get fund flow DataFrame for the stock
            fund_flow_df = await fund_flow_api.fetch_async(stock_code, market)

            # Process the data for the configured periods
            fund_flows, price_changes = fund_flow_api.process_periods(
                fund_flow_df, self.akshare_config.period_count
            )

            result = {}
            for i, period in enumerate(self.akshare_config.period_count):
                # Fund flow data (already in 亿)
                if i < len(fund_flows):
                    result[f"{period}日主力净流入-总净额(亿)"] = fund_flows[i]
                else:
                    result[f"{period}日主力净流入-总净额(亿)"] = None

                # Price change data
                if i < len(price_changes):
                    result[f"{period}日涨跌幅(%)"] = price_changes[i]
                else:
                    result[f"{period}日涨跌幅(%)"] = None

            return result

        except Exception as e:
            self.logger.error(f"Error fetching fund flow data for {stock_code}: {e}")
            # Return default structure with None values
            result = {}
            for period in self.akshare_config.period_count:
                result[f"{period}日主力净流入-总净额(亿)"] = None
                result[f"{period}日涨跌幅(%)"] = None
            return result

    def get_stock_market_data(self, stock_code: str) -> Dict[str, Any]:
        """
        Get market data for a stock from the spot market DataFrame.

        Args:
            stock_code: Stock code (6-digit format)

        Returns:
            Dictionary containing market data (market cap, P/E, P/B, etc.)
        """
        try:
            stock_row = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ]

            if stock_row.empty:
                self.logger.warning(f"No market data found for stock {stock_code}")
                return {
                    "总市值(亿)": None,
                    "流通市值(亿)": None,
                    "市盈率-动态": None,
                    "市净率": None,
                    "60日涨跌幅(%)": None,
                    "年初至今涨跌幅(%)": None,
                }

            row = stock_row.iloc[0]
            return {
                "总市值(亿)": row.get("总市值", None),
                "流通市值(亿)": row.get("流通市值", None),
                "市盈率-动态": row.get("市盈率-动态", None),
                "市净率": row.get("市净率", None),
                "60日涨跌幅(%)": row.get("60日涨跌幅", None),
                "年初至今涨跌幅(%)": row.get("年初至今涨跌幅", None),
            }

        except Exception as e:
            self.logger.error(f"Error getting market data for {stock_code}: {e}")
            return {
                "总市值(亿)": None,
                "流通市值(亿)": None,
                "市盈率-动态": None,
                "市净率": None,
                "60日涨跌幅(%)": None,
                "年初至今涨跌幅(%)": None,
            }

    def get_stock_industry(self, stock_code: str) -> str:
        """
        Get industry for a stock from the industry mapping DataFrame.

        Args:
            stock_code: Stock code (6-digit format)

        Returns:
            Industry name or "未知" if not found
        """
        try:
            industry_row = self.industry_stock_mapping_df[
                self.industry_stock_mapping_df["代码"] == stock_code
            ]

            if industry_row.empty:
                return "未知"

            return industry_row.iloc[0].get("行业", "未知")

        except Exception as e:
            self.logger.error(f"Error getting industry for {stock_code}: {e}")
            return "未知"

    def get_today_stock_price(self, stock_code: str) -> Optional[float]:
        """
        Get today's stock price from cached market data.

        Args:
            stock_code: Stock code (6-digit format)

        Returns:
            Current stock price or None if not found
        """
        try:
            stock_row = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ]

            if stock_row.empty:
                return None

            return stock_row.iloc[0].get("最新价", None)

        except Exception as e:
            self.logger.error(f"Error getting today's price for {stock_code}: {e}")
            return None

    async def run_analysis(
        self,
        _progress: Optional["Progress"] = None,
        _parent_task_id: Optional["TaskID"] = None,
    ) -> None:
        """
        Run the complete FHPS filter analysis.

        Args:
            _progress: Optional Progress instance for tracking
            _parent_task_id: Optional parent task ID
        """
        self.logger.info("Starting FHPS filter analysis")

        try:
            # Update progress
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=10,
                    description="📊 Fetching FHPS data...",
                )

            # Fetch FHPS data (with caching)
            stock_fhps_em_df = await self._get_cached_fhps_data()

            if stock_fhps_em_df is None or stock_fhps_em_df.empty:
                self.logger.warning(f"No FHPS data found for date {self.FHPS_DATE}")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="⚠️ No FHPS data available for the specified date",
                    )
                return

            self.logger.info(
                f"Initial FHPS data contains {len(stock_fhps_em_df)} records"
            )

            # Filter stocks with transfer ratios (remove NaN values first)
            df = stock_fhps_em_df.dropna(subset=["送转股份-送转总比例"])
            self.logger.info(f"After removing NaN transfer ratios: {len(df)} stocks")

            # Convert ex-dividend date to datetime
            df.loc[:, "除权除息日"] = pd.to_datetime(
                df["除权除息日"], format="%Y-%m-%d"
            )

            # Filter stocks with ex-dividend dates before today
            today = datetime.today()
            filter_past = df.loc[:, "除权除息日"] < today
            df_filtered = df[filter_past]

            self.logger.info(
                f"After ex-dividend date filter (< today): {len(df_filtered)} stocks"
            )

            # Apply minimum transfer ratio filter
            df_filtered = df_filtered[
                df_filtered["送转股份-送转总比例"] >= self.MIN_TRANSFER_RATIO
            ]
            self.logger.info(
                f"After transfer ratio filter (>= {self.MIN_TRANSFER_RATIO}): {len(df_filtered)} stocks"
            )

            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=30,
                    description=f"📈 Getting prices for {len(df_filtered)} filtered FHPS stocks...",
                )

            # First, get basic price data for all FHPS stocks to apply price change filter
            price_results = []

            # Process stocks in smaller batches for price data only
            stock_list = list(df_filtered.iterrows())
            price_batches = [
                stock_list[i : i + self.BATCH_SIZE]
                for i in range(0, len(stock_list), self.BATCH_SIZE)
            ]

            self.logger.info(
                f"Getting prices for {len(df_filtered)} FHPS stocks in {len(price_batches)} batches"
            )

            # Create a nested progress bar for detailed batch tracking if available
            batch_progress_task = None
            if _progress:
                batch_progress_task = _progress.add_task(
                    "    📊 Processing FHPS price batches",
                    total=len(price_batches),
                    visible=True,
                )

            for batch_idx, batch in enumerate(price_batches):
                if _progress and _parent_task_id:
                    progress_pct = 30 + (batch_idx / len(price_batches)) * 40
                    _progress.update(
                        _parent_task_id,
                        completed=progress_pct,
                        description=f"📈 Processing price batch {batch_idx + 1}/{len(price_batches)}...",
                    )

                # Update batch progress
                if _progress and batch_progress_task:
                    _progress.update(
                        batch_progress_task,
                        completed=batch_idx,
                        description=f"Batch {batch_idx + 1}/{len(price_batches)}: {len(batch)} stocks",
                    )

                # Get basic price data for this batch
                for stock_idx, (original_idx, row) in enumerate(batch):
                    stock_code = str(row["代码"]).zfill(6)  # Convert to string and pad to 6 digits
                    ex_date = row["除权除息日"]

                    # Update batch progress with current stock
                    if _progress and batch_progress_task:
                        _progress.update(
                            batch_progress_task,
                            description=f"Batch {batch_idx + 1}/{len(price_batches)}: {stock_code} ({stock_idx + 1}/{len(batch)})",
                        )

                    try:
                        # Get ex-dividend price from API (only what we need)
                        ex_price = await self.get_stock_price_async(stock_code, ex_date)

                        # Get today's price from cached market data (much faster)
                        today_price = self.get_today_stock_price(stock_code)

                        # Calculate price change
                        price_change_pct = None
                        if (
                            ex_price is not None
                            and today_price is not None
                            and ex_price != 0
                        ):
                            price_change_pct = round(
                                ((today_price - ex_price) / ex_price) * 100, 2
                            )

                        if price_change_pct is not None:
                            price_results.append(
                                {
                                    "original_idx": original_idx,
                                    "row": row,
                                    "ex_price": ex_price,
                                    "today_price": today_price,
                                    "price_change_pct": price_change_pct,
                                }
                            )

                    except Exception as e:
                        self.logger.error(f"Error getting prices for {stock_code}: {e}")
                        continue

                # Complete this batch
                if _progress and batch_progress_task:
                    _progress.update(
                        batch_progress_task,
                        completed=batch_idx + 1,
                        description=f"✅ Batch {batch_idx + 1} completed ({len([r for r in price_results if r['original_idx'] in [idx for idx, _ in batch]])} prices fetched)",
                    )

            # Remove batch progress bar when done
            if _progress and batch_progress_task:
                await asyncio.sleep(0.5)  # Brief pause to show completion
                _progress.remove_task(batch_progress_task)

            # Apply price change filter
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=70,
                    description="🔍 Applying price change filters...",
                )

            # Filter by price change percentage (< max_price_change_percent)
            filtered_stocks = [
                stock
                for stock in price_results
                if stock["price_change_pct"] < self.MAX_PRICE_CHANGE_PERCENT
            ]

            self.logger.info(
                f"After price change filter (<{self.MAX_PRICE_CHANGE_PERCENT}%): {len(filtered_stocks)} stocks"
            )

            if not filtered_stocks:
                self.logger.warning("No stocks passed the price change filter")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="⚠️ No stocks passed the price change filter",
                    )
                return

            # Now enrich the filtered stocks with full data (industry + fund flow)
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=75,
                    description=f"📊 Enriching {len(filtered_stocks)} filtered stocks with fund flow data...",
                )

            # Create a nested progress bar for enrichment tracking
            enrichment_progress_task = None
            if _progress:
                enrichment_progress_task = _progress.add_task(
                    "    💰 Enriching with fund flow data",
                    total=len(filtered_stocks),
                    visible=True,
                )

            all_results = []

            for i, stock_info in enumerate(filtered_stocks):
                row = stock_info["row"]
                stock_code = str(row["代码"]).zfill(6)  # Convert to string and pad to 6 digits
                stock_name = row["名称"]

                # Update main progress
                if _progress and _parent_task_id:
                    progress_pct = 75 + (i / len(filtered_stocks)) * 15
                    _progress.update(
                        _parent_task_id,
                        completed=progress_pct,
                        description=f"📊 Enriching stock {i + 1}/{len(filtered_stocks)}: {stock_code}",
                    )

                # Update enrichment progress
                if _progress and enrichment_progress_task:
                    _progress.update(
                        enrichment_progress_task,
                        completed=i,
                        description=f"Processing {stock_code} - {stock_name}",
                    )

                try:
                    # Get fund flow data
                    fund_flow_data = await self.get_fund_flow_data(stock_code)

                    # Get market data
                    market_data = self.get_stock_market_data(stock_code)

                    # Get industry
                    industry = self.get_stock_industry(stock_code)

                    # Build complete result following strict column sequence
                    result = {
                        # Column 1: Row number starting from 0 (blank column name)
                        "": i,
                        # Column 2: 行业
                        "行业": industry,
                        # Column 3: 代码
                        "代码": stock_code,
                        # Column 4: 名称
                        "名称": stock_name,
                        # Column 5: 总市值(亿)
                        "总市值(亿)": market_data.get("总市值(亿)"),
                        # Column 6: 流通市值(亿)
                        "流通市值(亿)": market_data.get("流通市值(亿)"),
                        # Column 7: 市盈率-动态
                        "市盈率-动态": market_data.get("市盈率-动态"),
                        # Column 8: 市净率
                        "市净率": market_data.get("市净率"),
                        # Column 9: 送转股份-送转总比例
                        "送转股份-送转总比例": row["送转股份-送转总比例"],
                        # Column 10: 除权除息日
                        "除权除息日": stock_info["row"]["除权除息日"].strftime(
                            "%Y-%m-%d"
                        )
                        if isinstance(stock_info["row"]["除权除息日"], datetime)
                        else str(stock_info["row"]["除权除息日"]),
                        # Column 11: 除权除息日股价
                        "除权除息日股价": stock_info["ex_price"],
                        # Column 12: {today}股价
                        f"{today.strftime('%Y%m%d')}股价": stock_info["today_price"],
                        # Column 13: 自除权出息日起涨跌幅(%)
                        "自除权出息日起涨跌幅(%)": stock_info["price_change_pct"],
                    }

                    # Add dynamic fund flow and price change columns based on configured periods
                    column_index = 14  # Start from column 14
                    for period in self.akshare_config.period_count:
                        # Fund flow columns
                        fund_flow_key = f"{period}日主力净流入-总净额(亿)"
                        result[fund_flow_key] = fund_flow_data.get(fund_flow_key)
                        column_index += 1

                    for period in self.akshare_config.period_count:
                        # Price change columns
                        price_change_key = f"{period}日涨跌幅(%)"
                        result[price_change_key] = fund_flow_data.get(price_change_key)
                        column_index += 1

                    # Add final market data columns
                    result.update(
                        {
                            # 60日涨跌幅(%)
                            "60日涨跌幅(%)": market_data.get("60日涨跌幅(%)"),
                            # 年初至今涨跌幅(%)
                            "年初至今涨跌幅(%)": market_data.get("年初至今涨跌幅(%)"),
                        }
                    )

                    all_results.append(result)

                    # Update enrichment progress with success
                    if _progress and enrichment_progress_task:
                        _progress.update(
                            enrichment_progress_task,
                            completed=i + 1,
                            description=f"✅ {stock_code} - {stock_name} completed",
                        )

                except Exception as e:
                    self.logger.error(f"Error enriching stock {stock_code}: {e}")
                    # Update enrichment progress with error
                    if _progress and enrichment_progress_task:
                        _progress.update(
                            enrichment_progress_task,
                            completed=i + 1,
                            description=f"❌ {stock_code} - {stock_name} failed",
                        )
                    continue

            # Remove enrichment progress bar when done
            if _progress and enrichment_progress_task:
                await asyncio.sleep(0.5)  # Brief pause to show completion
                _progress.remove_task(enrichment_progress_task)

            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id, completed=90, description="📝 Generating report..."
                )

            # Create DataFrame and save report
            if all_results:
                result_df = pd.DataFrame(all_results)

                # Sort by price change percentage
                result_df = result_df.sort_values(
                    by=["自除权出息日起涨跌幅(%)"], ascending=True
                )

                # Reset the first column to sequential 0-based indexing after sorting
                result_df.iloc[:, 0] = pd.Series(
                    range(len(result_df)), index=result_df.index
                )

                # Create output directory
                os.makedirs(self.REPORT_DIR, exist_ok=True)

                # Generate output filename
                today_str = datetime.now().strftime("%Y%m%d")
                output_filename = self.OUTPUT_FILENAME_TEMPLATE.format(date=today_str)
                output_path = os.path.join(self.REPORT_DIR, output_filename)

                # Save to CSV
                result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

                self.logger.info(
                    f"FHPS analysis completed. Report saved to: {output_path}"
                )
                self.logger.info(f"Total stocks processed: {len(result_df)}")
            else:
                self.logger.warning("No results to save - all stock processing failed")

            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=100,
                    description="✅ FHPS analysis completed",
                )

        except Exception as e:
            error_msg = f"FHPS filter analysis failed: {str(e)}"
            self.logger.error(error_msg)
            if _progress and _parent_task_id:
                _progress.update(_parent_task_id, description=f"❌ {error_msg}")
            raise
