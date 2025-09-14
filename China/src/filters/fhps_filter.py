"""
FHPS (分红派息送股) filtering and analysis for Chinese equity markets.

This module provides filtering-only functionality for FHPS data using pre-cached files.
It processes stocks with ex-dividend dates, applies multi-year historical filtering,
and enriches data with fund flow information for comprehensive analysis.

Note: This script assumes data has already been cached by fhps_caching.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

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

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Initialize logger for this module
logger = get_logger("fhps_filter")


class InputFilesConfig(BaseModel):
    """Configuration model for input files."""

    stock_fhps_em_latest_filename: str = "stock_fhps_em-latest.csv"
    data_dir: str = "data/fhps"


class HistoricalFilterConfig(BaseModel):
    """Configuration model for historical filtering."""

    previous_years: List[int] = [2020, 2021, 2022, 2023]
    require_historical_presence: bool = True


class FhpsFilterConfig(BaseModel):
    """Configuration model for FhpsFilter class parameters."""

    min_transfer_ratio: float = 1.0
    max_price_change_percent: float = 5.0
    max_circulating_market_cap_yi: float = 50.0
    min_pe_ratio: float = 0.0
    batch_size: int = 5
    report_dir: str = "output/reports/filters/fhps_filter"
    output_filename: str = "除权除息股票-{date}.csv"


class FileConfig(BaseModel):
    """Configuration model for file-related settings."""

    config_name: str = "PROD"
    description: str = ""
    version: str = "1.9.3"


class Config(BaseModel):
    """Configuration model for nested YAML structure supporting input files, historical filtering, akshare and FhpsFilter configs."""

    input_files: InputFilesConfig = InputFilesConfig()
    historical_filter: HistoricalFilterConfig = HistoricalFilterConfig()
    akshare: Dict[str, Dict[str, Any]] = {}
    fhps_filter: FhpsFilterConfig = FhpsFilterConfig()
    file_config: FileConfig = FileConfig()


def load_config(
    config_name: Optional[str] = None,
) -> Tuple[
    InputFilesConfig,
    HistoricalFilterConfig,
    StockIndividualFundFlowConfig,
    FhpsFilterConfig,
    FileConfig,
]:
    """
    Load nested configuration from YAML file.

    Args:
        config_name: YAML config file name. If None, uses default config

    Returns:
        tuple: (input_files_config, historical_filter_config, akshare_config, fhps_filter_config, file_config)

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

    # Extract each section
    configs = Config(**config_data)
    input_files_config = configs.input_files
    historical_filter_config = configs.historical_filter
    akshare_config = StockIndividualFundFlowConfig(
        **configs.akshare.get("stock_individual_fund_flow", {})
    )
    fhps_filter_config = configs.fhps_filter
    file_config = configs.file_config

    return (
        input_files_config,
        historical_filter_config,
        akshare_config,
        fhps_filter_config,
        file_config,
    )


class FhpsFilter:
    """
    Filter and analyze stocks with dividend/split plans (FHPS) using cached data.

    This class processes pre-cached FHPS data, applies multi-year historical filtering,
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
        (
            self.input_files_config,
            self.historical_filter_config,
            self.akshare_config,
            self.config,
            self.file_config,
        ) = load_config(config_name)

        # Apply class constants from config
        self.MIN_TRANSFER_RATIO = self.config.min_transfer_ratio
        self.MAX_PRICE_CHANGE_PERCENT = self.config.max_price_change_percent
        self.MAX_CIRCULATING_MARKET_CAP_YI = self.config.max_circulating_market_cap_yi
        self.MIN_PE_RATIO = self.config.min_pe_ratio
        self.BATCH_SIZE = self.config.batch_size
        self.REPORT_DIR = self.config.report_dir
        self.OUTPUT_FILENAME = self.config.output_filename

        self.logger = get_logger("fhps_filter")

    def _load_latest_fhps_data(self) -> Optional[pd.DataFrame]:
        """
        Load the latest FHPS data from cached file.

        Returns:
            DataFrame with latest FHPS data or None if not available
        """
        data_dir = self.input_files_config.data_dir
        latest_filename = self.input_files_config.stock_fhps_em_latest_filename
        latest_path = os.path.join(data_dir, latest_filename)

        if not os.path.exists(latest_path):
            error_msg = f"Latest FHPS file not found: {latest_path}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        try:
            df = pd.read_csv(latest_path, encoding="utf-8-sig")
            # Ensure stock codes are strings with proper 6-digit format
            if "代码" in df.columns:
                df["代码"] = df["代码"].apply(lambda x: str(x).zfill(6))

            # Convert date column back to datetime if it's a string
            if "除权除息日" in df.columns:
                df["除权除息日"] = pd.to_datetime(df["除权除息日"])

            self.logger.info(
                f"✅ Loaded {len(df)} records from latest FHPS file: {latest_path}"
            )
            print(f"✅ Loaded {len(df)} records from latest FHPS file: {latest_path}")
            return df

        except Exception as e:
            error_msg = f"Error loading latest FHPS file {latest_path}: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def _load_historical_stock_codes(self) -> Set[str]:
        """
        Load stock codes from historical years for multi-year filtering.

        Returns:
            Set of stock codes that appeared in previous years
        """
        all_historical_codes = set()
        data_dir = self.input_files_config.data_dir

        for year in self.historical_filter_config.previous_years:
            filename = f"stock_fhps_em-{year}.csv"
            file_path = os.path.join(data_dir, filename)

            if not os.path.exists(file_path):
                self.logger.warning(f"Historical file not found: {file_path}")
                print(f"⚠️ Historical file not found: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
                if "代码" in df.columns:
                    year_codes = set(
                        df["代码"].apply(lambda x: str(x).zfill(6)).tolist()
                    )
                    all_historical_codes.update(year_codes)
                    self.logger.info(
                        f"Loaded {len(year_codes)} stock codes from {year}"
                    )
                    print(f"📊 Loaded {len(year_codes)} stock codes from {year}")

            except Exception as e:
                self.logger.error(f"Error loading historical file {file_path}: {e}")
                print(f"❌ Error loading historical file {file_path}: {e}")
                continue

        self.logger.info(
            f"Total unique historical stock codes: {len(all_historical_codes)}"
        )
        print(f"📈 Total unique historical stock codes: {len(all_historical_codes)}")
        return all_historical_codes

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

            # Convert market cap values to 亿 and round to 2 decimal places
            total_market_cap = row.get("总市值", None)
            if total_market_cap is not None:
                total_market_cap = round(total_market_cap / 1e8, 2)

            circulating_market_cap = row.get("流通市值", None)
            if circulating_market_cap is not None:
                circulating_market_cap = round(circulating_market_cap / 1e8, 2)

            return {
                "总市值(亿)": total_market_cap,
                "流通市值(亿)": circulating_market_cap,
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
        Run the complete FHPS filter analysis using cached data.

        Args:
            _progress: Optional Progress instance for tracking
            _parent_task_id: Optional parent task ID
        """
        self.logger.info("Starting FHPS filter analysis (filtering-only mode)")

        try:
            # Update progress
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=10,
                    description="📂 Loading cached FHPS data...",
                )

            # Load latest FHPS data
            df_latest = self._load_latest_fhps_data()
            if df_latest is None or df_latest.empty:
                error_msg = "No latest FHPS data available"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="❌ No latest FHPS data available",
                    )
                raise RuntimeError(error_msg)

            # Load historical stock codes for multi-year filtering
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=20,
                    description="📊 Loading historical stock codes...",
                )

            historical_codes = self._load_historical_stock_codes()

            if (
                self.historical_filter_config.require_historical_presence
                and not historical_codes
            ):
                error_msg = "No historical stock codes found, but historical presence is required"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="❌ No historical data for filtering",
                    )
                raise RuntimeError(error_msg)

            # Apply historical filtering if required
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=30,
                    description="🔍 Applying historical presence filter...",
                )

            if self.historical_filter_config.require_historical_presence:
                # Filter stocks that appear in both latest and historical data
                df_latest["代码_str"] = df_latest["代码"].astype(str).str.zfill(6)
                mask = df_latest["代码_str"].isin(historical_codes)
                df_filtered = df_latest[mask].copy()
                df_filtered.drop(columns=["代码_str"], inplace=True)

                self.logger.info(
                    f"Historical filtering: {len(df_latest)} → {len(df_filtered)} stocks (must appear in previous years)"
                )
                print(
                    f"📊 Historical filtering: {len(df_latest)} → {len(df_filtered)} stocks (must appear in previous years)"
                )
            else:
                df_filtered = df_latest.copy()

            if df_filtered.empty:
                self.logger.warning("No stocks passed the historical filtering")
                print("⚠️ No stocks passed the historical filtering")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="⚠️ No stocks passed historical filtering",
                    )
                return

            # Apply minimum transfer ratio filter
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=40,
                    description="📈 Applying transfer ratio filter...",
                )

            df_filtered = df_filtered[
                df_filtered["送转股份-送转总比例"] >= self.MIN_TRANSFER_RATIO
            ]
            self.logger.info(
                f"After transfer ratio filter (>= {self.MIN_TRANSFER_RATIO}): {len(df_filtered)} stocks"
            )
            print(
                f"📈 After transfer ratio filter (>= {self.MIN_TRANSFER_RATIO}): {len(df_filtered)} stocks"
            )

            if df_filtered.empty:
                self.logger.warning("No stocks passed the transfer ratio filter")
                print("⚠️ No stocks passed the transfer ratio filter")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="⚠️ No stocks passed transfer ratio filter",
                    )
                return

            # Apply price change filtering and enrich with market data
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=50,
                    description=f"💰 Processing {len(df_filtered)} stocks...",
                )

            valid_stocks = []
            market_cap_filtered = 0
            pe_ratio_filtered = 0
            price_change_filtered = 0

            for _, row in df_filtered.iterrows():
                stock_code = str(row["代码"]).zfill(6)

                # Get market data first for early filtering
                market_data = self.get_stock_market_data(stock_code)

                # Apply market data filters to save API costs
                # Filter by circulating market cap
                circulating_market_cap = market_data.get("流通市值(亿)", None)
                if (
                    circulating_market_cap is not None
                    and circulating_market_cap >= self.MAX_CIRCULATING_MARKET_CAP_YI
                ):
                    market_cap_filtered += 1
                    self.logger.debug(
                        f"Filtered out {stock_code}: 流通市值 {circulating_market_cap}亿 >= {self.MAX_CIRCULATING_MARKET_CAP_YI}亿"
                    )
                    continue

                # Filter by P/E ratio
                pe_ratio = market_data.get("市盈率-动态", None)
                if pe_ratio is not None and pe_ratio <= self.MIN_PE_RATIO:
                    pe_ratio_filtered += 1
                    self.logger.debug(
                        f"Filtered out {stock_code}: 市盈率-动态 {pe_ratio} <= {self.MIN_PE_RATIO}"
                    )
                    continue

                # Calculate price change if we have the cached ex-dividend price
                ex_price = row.get("除权除息前日股价", None)
                today_price = self.get_today_stock_price(stock_code)
                price_change_pct = None

                if ex_price is not None and today_price is not None and ex_price != 0:
                    price_change_pct = round(
                        ((today_price - ex_price) / ex_price) * 100, 2
                    )

                    # Apply price change filter
                    if price_change_pct >= self.MAX_PRICE_CHANGE_PERCENT:
                        price_change_filtered += 1
                        self.logger.debug(
                            f"Filtered out {stock_code}: price change {price_change_pct}% >= {self.MAX_PRICE_CHANGE_PERCENT}%"
                        )
                        continue

                # Stock passed all filters, add to valid list
                valid_stocks.append(
                    {
                        "row": row,
                        "stock_code": stock_code,
                        "market_data": market_data,
                        "ex_price": ex_price,
                        "today_price": today_price,
                        "price_change_pct": price_change_pct,
                    }
                )

            # Log filtering statistics
            total_initial = len(df_filtered)
            total_valid = len(valid_stocks)
            total_filtered_out = (
                market_cap_filtered + pe_ratio_filtered + price_change_filtered
            )

            self.logger.info("Filtering statistics:")
            self.logger.info(
                f"  - Initial stocks after transfer ratio filter: {total_initial}"
            )
            self.logger.info(
                f"  - 流通市值 >= {self.MAX_CIRCULATING_MARKET_CAP_YI}亿: {market_cap_filtered} filtered"
            )
            self.logger.info(
                f"  - 市盈率-动态 <= {self.MIN_PE_RATIO}: {pe_ratio_filtered} filtered"
            )
            self.logger.info(
                f"  - Price change >= {self.MAX_PRICE_CHANGE_PERCENT}%: {price_change_filtered} filtered"
            )
            self.logger.info(f"  - Total filtered out: {total_filtered_out}")
            self.logger.info(f"  - Valid stocks for enrichment: {total_valid}")

            print("📊 Filtering statistics:")
            print(f"  - Initial stocks: {total_initial}")
            print(f"  - Market cap filtered: {market_cap_filtered}")
            print(f"  - P/E ratio filtered: {pe_ratio_filtered}")
            print(f"  - Price change filtered: {price_change_filtered}")
            print(f"  - Valid stocks: {total_valid}")

            if not valid_stocks:
                self.logger.warning("No stocks passed all filters")
                print("⚠️ No stocks passed all filters")
                if _progress and _parent_task_id:
                    _progress.update(
                        _parent_task_id,
                        completed=100,
                        description="⚠️ No stocks passed all filters",
                    )
                return

            # Enrich valid stocks with fund flow data
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=60,
                    description=f"💰 Enriching {len(valid_stocks)} stocks with fund flow...",
                )

            all_results = []
            for i, stock_info in enumerate(valid_stocks):
                row = stock_info["row"]
                stock_code = stock_info["stock_code"]
                market_data = stock_info["market_data"]

                if _progress and _parent_task_id:
                    progress_pct = 60 + (i / len(valid_stocks)) * 30
                    _progress.update(
                        _parent_task_id,
                        completed=progress_pct,
                        description=f"💰 Processing {i + 1}/{len(valid_stocks)}: {stock_code}",
                    )

                try:
                    # Get fund flow data
                    fund_flow_data = await self.get_fund_flow_data(stock_code)

                    # Get industry
                    industry = self.get_stock_industry(stock_code)

                    # Build complete result
                    result = {
                        "": i,  # Row number
                        "行业": industry,
                        "代码": stock_code,
                        "名称": row.get("名称", ""),
                        "总市值(亿)": market_data.get("总市值(亿)"),
                        "流通市值(亿)": market_data.get("流通市值(亿)"),
                        "市盈率-动态": market_data.get("市盈率-动态"),
                        "市净率": market_data.get("市净率"),
                        "送转股份-送转总比例": row.get("送转股份-送转总比例"),
                        "除权除息日": row.get("除权除息日").strftime("%Y-%m-%d")
                        if isinstance(row.get("除权除息日"), datetime)
                        else str(row.get("除权除息日", "")),
                        "除权除息前日股价": stock_info["ex_price"],
                        "当前股价": stock_info["today_price"],
                        "自除权除息前日起涨跌幅(%)": stock_info["price_change_pct"],
                    }

                    # Add dynamic fund flow and price change columns
                    for period in self.akshare_config.period_count:
                        fund_flow_key = f"{period}日主力净流入-总净额(亿)"
                        result[fund_flow_key] = fund_flow_data.get(fund_flow_key)

                    for period in self.akshare_config.period_count:
                        price_change_key = f"{period}日涨跌幅(%)"
                        result[price_change_key] = fund_flow_data.get(price_change_key)

                    # Add final market data columns
                    result.update(
                        {
                            "60日涨跌幅(%)": market_data.get("60日涨跌幅(%)"),
                            "年初至今涨跌幅(%)": market_data.get("年初至今涨跌幅(%)"),
                        }
                    )

                    all_results.append(result)

                except Exception as e:
                    self.logger.error(f"Error enriching stock {stock_code}: {e}")
                    continue

            # Generate report
            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=90,
                    description="📝 Generating report...",
                )

            if all_results:
                result_df = pd.DataFrame(all_results)

                # Sort by price change percentage
                result_df = result_df.sort_values(
                    by=["自除权除息前日起涨跌幅(%)"], ascending=True
                )

                # Reset the first column to sequential 0-based indexing after sorting
                result_df.iloc[:, 0] = pd.Series(
                    range(len(result_df)), index=result_df.index
                )

                # Create output directory
                os.makedirs(self.REPORT_DIR, exist_ok=True)

                # Generate output filename
                today_str = datetime.now().strftime("%Y%m%d")
                output_filename = self.OUTPUT_FILENAME.format(date=today_str)
                output_path = os.path.join(self.REPORT_DIR, output_filename)

                # Save to CSV
                result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

                self.logger.info(
                    f"FHPS filter analysis completed. Report saved to: {output_path}"
                )
                self.logger.info(f"Total stocks in report: {len(result_df)}")
                print(
                    f"✅ FHPS filter analysis completed. Report saved to: {output_path}"
                )
                print(f"📊 Total stocks in report: {len(result_df)}")
            else:
                self.logger.warning("No results to save - all stock processing failed")
                print("⚠️ No results to save - all stock processing failed")

            if _progress and _parent_task_id:
                _progress.update(
                    _parent_task_id,
                    completed=100,
                    description="✅ FHPS filter analysis completed",
                )

        except Exception as e:
            error_msg = f"FHPS filter analysis failed: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            if _progress and _parent_task_id:
                _progress.update(_parent_task_id, description=f"❌ {error_msg}")
            raise


async def main():
    """Main entry point for running FHPS filter analysis."""
    try:
        # Note: In actual usage, you would pass the required DataFrames
        # This is just a placeholder for testing the script structure
        print(
            "❌ Error: FhpsFilter requires industry_stock_mapping_df and stock_zh_a_spot_em_df"
        )
        print(
            "Please run this through the main application that provides these DataFrames"
        )
    except Exception as e:
        print(f"❌ FHPS filter analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
