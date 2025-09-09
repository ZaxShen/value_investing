"""
Akshare API integration modules.

This package provides centralized interfaces for akshare API calls,
eliminating code duplication across different analysis modules.
"""

from .stock_individual_fund_flow import (
    StockIndividualFundFlowAPI,
    StockIndividualFundFlowConfig,
    fetch_stock_fund_flow_async,
    fetch_stock_fund_flow_sync,
    get_market_by_stock_code,
    process_fund_flow_for_periods,
)

__all__ = [
    "StockIndividualFundFlowAPI",
    "StockIndividualFundFlowConfig", 
    "fetch_stock_fund_flow_async",
    "fetch_stock_fund_flow_sync",
    "get_market_by_stock_code",
    "process_fund_flow_for_periods",
]