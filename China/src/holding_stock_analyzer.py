"""
Stock analysis and holding report generation for Chinese equity markets.

This module provides a HoldingStockAnalyzer class that encapsulates comprehensive
analysis of individual stocks and generates detailed holding reports. It analyzes
stock performance, fund flows, and calculates key financial metrics for
investment decision making.
"""

import asyncio
import glob
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Import settings first to disable tqdm before akshare import
from src.settings import configure_environment
configure_environment()

import akshare as ak
import pandas as pd

from src.utilities.logger import get_logger
from src.utilities.retry import API_RETRY_CONFIG

if TYPE_CHECKING:
    from rich.progress import Progress

# Initialize logger for this module
logger = get_logger("holding_stock_analyzer")


class HoldingStockAnalyzer:
    """
    A class to encapsulate holding stock analysis functionality.

    This class manages the analysis of individual stocks for holding reports,
    providing comprehensive financial metrics, fund flow analysis, and
    performance tracking for investment portfolios.
    """

    # Class constants for analysis parameters
    DAYS_ANALYSIS_PERIOD = 29  # Default analysis period in days
    REPORT_DIR = "data/holding_stocks/reports"
    HOLDING_STOCKS_DIR = "data/holding_stocks"

    def __init__(
        self,
        industry_stock_mapping_df: pd.DataFrame,
        stock_zh_a_spot_em_df: pd.DataFrame,
    ):
        """
        Initialize the HoldingStockAnalyzer with market data.

        Args:
            industry_stock_mapping_df: DataFrame containing industry-stock mapping
            stock_zh_a_spot_em_df: DataFrame containing stock market data
        """
        self.industry_stock_mapping_df = industry_stock_mapping_df
        self.stock_zh_a_spot_em_df = stock_zh_a_spot_em_df

    def _get_analysis_columns(self, days: int) -> List[str]:
        """
        Generate analysis column names with dynamic days parameter.

        Args:
            days: Number of days for fund flow analysis

        Returns:
            List of column names for analysis results
        """
        return [
            "账户",
            "行业",
            "代码",
            "名称",
            "总市值(亿)",
            "流通市值(亿)",
            "市盈率-动态",
            "市净率",
            "收盘价",
            f"{days}日主力净流入-总净额(亿)",
            f"{days}日涨跌幅(%)",
            "60日涨跌幅(%)",
            "年初至今涨跌幅(%)",
        ]

    def _get_market_by_stock_code(self, stock_code: str) -> str:
        """
        Determine the market based on stock code prefix.

        Args:
            stock_code: Stock code (e.g., "000001", "600001", "301001")

        Returns:
            Market identifier: "sh" for Shanghai, "sz" for Shenzhen, "bj" for Beijing
        """
        if stock_code.startswith("6"):
            return "sh"  # Shanghai Stock Exchange
        elif stock_code.startswith("0") or stock_code.startswith("3"):
            return "sz"  # Shenzhen Stock Exchange
        else:
            return "bj"  # Beijing Stock Exchange

    def validate_stock_name(self, stock_code: str, stock_name: str) -> None:
        """
        Validate that the stock name matches the stock code in the dataset.

        Args:
            stock_code: Stock code to validate (e.g., "000001")
            stock_name: Expected stock name

        Raises:
            ValueError: If stock name doesn't match or stock code not found
        """
        try:
            actual_name = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ]["名称"].values[0]
            if actual_name != stock_name:
                raise ValueError(
                    f"Stock name mismatch for {stock_code}: {stock_name} != {actual_name}"
                )
        except (IndexError, KeyError):
            raise ValueError(f"Stock code {stock_code} not found")

    def load_holding_stocks_from_files(self, dir_path: str = None) -> Dict[str, Dict[str, str]]:
        """
        Load holding stocks data from JSON files in the specified directory.

        Args:
            dir_path: Directory path containing JSON files (default: class constant)

        Returns:
            Dictionary with account names as keys and {stock_code: stock_name} 
            dictionaries as values

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If JSON files are malformed
        """
        if dir_path is None:
            dir_path = self.HOLDING_STOCKS_DIR

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Holding stocks directory not found: {dir_path}")

        holding_stocks_data = {}
        json_files = glob.glob(os.path.join(dir_path, "*.json"))
        
        if not json_files:
            logger.warning("No JSON files found in directory: %s", dir_path)
            return holding_stocks_data

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    account_name = os.path.splitext(os.path.basename(file_path))[0]
                    holding_stocks = json.load(f)
                    
                    # Validate JSON structure
                    if not isinstance(holding_stocks, dict):
                        raise ValueError(f"JSON file {file_path} should contain a dictionary")
                    
                    for stock_code, stock_name in holding_stocks.items():
                        if not isinstance(stock_code, str) or not isinstance(stock_name, str):
                            raise ValueError(f"Invalid stock data in {file_path}: {stock_code} -> {stock_name}")
                    
                    holding_stocks_data[account_name] = holding_stocks
                    logger.info("Loaded %d stocks for account '%s'", len(holding_stocks), account_name)
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Error loading JSON file %s: %s", file_path, str(e))
                raise ValueError(f"Malformed JSON file {file_path}: {str(e)}")
            except Exception as e:
                logger.error("Error reading file %s: %s", file_path, str(e))
                raise

        logger.info("Loaded holding stocks for %d accounts", len(holding_stocks_data))
        return holding_stocks_data

    def _fetch_stock_fund_flow_sync(self, stock_code: str, market: str) -> pd.DataFrame:
        """
        Fetch stock individual fund flow data with retry mechanism.

        Args:
            stock_code: Stock code (e.g., "000001")
            market: Market identifier (e.g., "sz", "sh", "bj")

        Returns:
            DataFrame containing historical fund flow data for the specified stock
        """
        return API_RETRY_CONFIG.retry(
            ak.stock_individual_fund_flow, stock=stock_code, market=market
        )

    def _fetch_sector_fund_flow_sync(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock sector fund flow historical data with retry mechanism.

        Args:
            symbol: Sector symbol identifier

        Returns:
            DataFrame containing historical sector fund flow data
        """
        return API_RETRY_CONFIG.retry(ak.stock_sector_fund_flow_hist, symbol=symbol)

    async def analyze_single_stock(
        self,
        industry_name: str,
        stock_code: str,
        stock_name: str,
        days: int = None,
    ) -> Optional[List[Any]]:
        """
        Perform comprehensive analysis of a single stock including fund flow and performance metrics.

        This method analyzes a stock's financial performance, fund flow patterns,
        and calculates key metrics for investment decision making.

        Args:
            industry_name: Industry classification of the stock
            stock_code: Stock code (e.g., "000001")
            stock_name: Stock name for validation and display
            days: Number of days to analyze (default: class constant)

        Returns:
            List containing analysis results with financial metrics, or None if
            analysis fails or stock doesn't meet criteria
        """
        if days is None:
            days = self.DAYS_ANALYSIS_PERIOD

        logger.debug(
            "Processing %s (%s) in %s industry", stock_name, stock_code, industry_name
        )

        # Determine the market based on the stock code
        market = self._get_market_by_stock_code(stock_code)

        try:
            # Extract the stock's market data
            stock_data = self.stock_zh_a_spot_em_df[
                self.stock_zh_a_spot_em_df["代码"] == stock_code
            ].iloc[0]

            stock_total_market_value = round(stock_data["总市值"] / 1e8, 0)
            stock_circulating_market_value = round(stock_data["流通市值"] / 1e8, 0)
            stock_pe_dynamic = stock_data["市盈率-动态"]
            stock_pb = stock_data["市净率"]
            stock_60d_change = stock_data["60日涨跌幅"]
            stock_ytd_change = stock_data["年初至今涨跌幅"]

            # Extract the historical data of the stock (async with retry)
            stock_individual_fund_flow_df = await asyncio.to_thread(
                self._fetch_stock_fund_flow_sync, stock_code, market
            )

            if len(stock_individual_fund_flow_df) < days:
                logger.warning(
                    "Skipping %s (%s) due to insufficient data for the last %d days",
                    stock_name,
                    stock_code,
                    days,
                )
                return None

            stock_individual_fund_flow_df = stock_individual_fund_flow_df.iloc[-days:]

            # Get the main net inflow data
            stock_main_net_flow = round(
                stock_individual_fund_flow_df["主力净流入-净额"].sum() / 1e8, 2
            )

            # Calculate change percentage with division by zero protection
            stock_1st_price = stock_individual_fund_flow_df.iloc[0]["收盘价"]
            stock_last_price = stock_individual_fund_flow_df.iloc[-1]["收盘价"]

            if stock_1st_price == 0:
                logger.warning(
                    "First price is zero for %s (%s), skipping price change calculation",
                    stock_name,
                    stock_code,
                )
                return None

            stock_price_change_percentage = round(
                (stock_last_price - stock_1st_price) / stock_1st_price * 100, 2
            )

            return [
                industry_name,
                stock_code,
                stock_name,
                stock_total_market_value,
                stock_circulating_market_value,
                stock_pe_dynamic,
                stock_pb,
                stock_last_price,
                stock_main_net_flow,
                stock_price_change_percentage,
                stock_60d_change,
                stock_ytd_change,
            ]

        except Exception as e:
            logger.error("Error processing %s (%s): %s", stock_name, stock_code, str(e))
            return None

    def _save_report(self, df: pd.DataFrame, last_date_str: str) -> None:
        """
        Save holding analysis report to CSV file.

        Args:
            df: DataFrame containing analysis results
            last_date_str: Date string for report naming
        """
        try:
            report_path = f"{self.REPORT_DIR}/持股报告-{last_date_str}.csv"
            df.to_csv(report_path, index=True)
            logger.info("Report saved to %s", report_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to save holding report: %s", str(e))
            raise

    async def run_analysis(
        self,
        holding_stocks_data: Dict[str, Dict[str, str]],
        days: int = None,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> None:
        """
        Run the complete holding stock analysis pipeline.

        This method orchestrates the entire holding stock analysis process including
        stock validation, analysis, and report generation.

        Args:
            holding_stocks_data: Dictionary with account names as keys and
                                {stock_code: stock_name} dictionaries as values
            days: Number of days to analyze (default: class constant)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        if days is None:
            days = self.DAYS_ANALYSIS_PERIOD

        # Initialize DataFrame with analysis columns
        columns = self._get_analysis_columns(days)
        df = pd.DataFrame(columns=columns)

        # Process each account's holdings
        for account_name, holding_stocks in holding_stocks_data.items():
            for stock_code, stock_name in holding_stocks.items():
                try:
                    # Validate stock name
                    self.validate_stock_name(stock_code, stock_name)

                    # Get industry name
                    industry_name = self.industry_stock_mapping_df[
                        self.industry_stock_mapping_df["代码"] == stock_code
                    ]["行业"].values[0]

                    # Analyze the stock
                    result = await self.analyze_single_stock(
                        industry_name=industry_name,
                        stock_code=stock_code,
                        stock_name=stock_name,
                        days=days,
                    )

                    if result is not None:
                        df.loc[len(df)] = [account_name] + result

                except Exception as e:
                    logger.error(
                        "Error processing %s (%s) for account %s: %s",
                        stock_name,
                        stock_code,
                        account_name,
                        str(e),
                    )
                    continue

        # Get report date with retry mechanism
        stock_sector_data = await asyncio.to_thread(
            self._fetch_sector_fund_flow_sync, "证券"
        )
        last_date = stock_sector_data.iloc[-1]["日期"]
        last_date_str = last_date.strftime("%Y%m%d")

        # Save the report
        self._save_report(df, last_date_str)

    async def run_analysis_from_files(
        self,
        dir_path: str = None,
        days: int = None,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional[int] = None,
        batch_task_id: Optional[int] = None,
    ) -> None:
        """
        Load holding stocks from JSON files and run the complete analysis pipeline.

        This is a convenience method that combines loading JSON files and running analysis.

        Args:
            dir_path: Directory path containing JSON files (default: class constant)
            days: Number of days to analyze (default: class constant)
            progress: Optional Rich Progress instance for hierarchical progress tracking
            parent_task_id: Optional parent task ID for hierarchical progress structure
            batch_task_id: Optional pre-created batch task ID for proper hierarchy display
        """
        # Load holding stocks data from JSON files
        holding_stocks_data = self.load_holding_stocks_from_files(dir_path)
        
        if not holding_stocks_data:
            logger.warning("No holding stocks data loaded, skipping analysis")
            return
        
        # Run the analysis with loaded data
        await self.run_analysis(
            holding_stocks_data=holding_stocks_data,
            days=days,
            progress=progress,
            parent_task_id=parent_task_id,
            batch_task_id=batch_task_id,
        )


async def main(
    industry_stock_mapping_df: pd.DataFrame,
    stock_zh_a_spot_em_df: pd.DataFrame,
    holding_stocks_data: Dict[str, Dict[str, str]],
    progress: Optional["Progress"] = None,
    parent_task_id: Optional[int] = None,
    batch_task_id: Optional[int] = None,
) -> None:
    """
    Main function to execute holding stock analysis and generate reports.

    This function creates a HoldingStockAnalyzer instance and runs the complete analysis.
    Maintained for backward compatibility.

    Args:
        industry_stock_mapping_df: DataFrame containing industry-stock mapping
        stock_zh_a_spot_em_df: DataFrame containing stock market data
        holding_stocks_data: Dictionary with account names as keys and
                            {stock_code: stock_name} dictionaries as values
        progress: Optional Rich Progress instance for hierarchical progress tracking
        parent_task_id: Optional parent task ID for hierarchical progress structure
        batch_task_id: Optional pre-created batch task ID for proper hierarchy display
    """
    holding_stock_analyzer = HoldingStockAnalyzer(
        industry_stock_mapping_df, stock_zh_a_spot_em_df
    )
    await holding_stock_analyzer.run_analysis(
        holding_stocks_data=holding_stocks_data,
        progress=progress,
        parent_task_id=parent_task_id,
        batch_task_id=batch_task_id,
    )
