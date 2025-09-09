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
from .stock_board_industry import (
    StockBoardIndustryAPI,
    StockBoardIndustryHistConfig,
    fetch_industry_names_sync,
    fetch_industry_names_async,
    fetch_industry_hist_sync,
    fetch_industry_hist_async,
    resolve_date_range,
    calculate_price_changes_for_periods,
    date_converter,
)
from .stock_sector_fund_flow import (
    StockSectorFundFlowAPI,
    StockSectorFundFlowConfig,
    fetch_sector_fund_flow_sync,
    fetch_sector_fund_flow_async,
    validate_fund_flow_data_availability,
    create_empty_fund_flow_data,
    process_fund_flow_for_periods as process_sector_fund_flow_for_periods,
)
from .stock_market_data import (
    StockMarketDataAPI,
    StockMarketDataConfig,
    fetch_shanghai_spot_sync,
    fetch_shenzhen_spot_sync,
    fetch_beijing_spot_sync,
    fetch_all_a_shares_spot_sync,
    fetch_shanghai_spot_async,
    fetch_shenzhen_spot_async,
    fetch_beijing_spot_async,
    fetch_all_a_shares_spot_async,
    fetch_multiple_markets_async,
    combine_market_data,
)
from .stock_board_constituents import (
    StockBoardConstituentsAPI,
    StockBoardConstituentsConfig,
    fetch_industry_constituents_sync,
    fetch_industry_constituents_async,
    fetch_multiple_industries_constituents_async,
    create_industry_stock_mapping,
    get_stocks_by_industries,
    get_all_stocks_from_industries,
)

__all__ = [
    # Stock individual fund flow
    "StockIndividualFundFlowAPI",
    "StockIndividualFundFlowConfig", 
    "fetch_stock_fund_flow_async",
    "fetch_stock_fund_flow_sync",
    "get_market_by_stock_code",
    "process_fund_flow_for_periods",
    # Stock board industry
    "StockBoardIndustryAPI",
    "StockBoardIndustryHistConfig",
    "fetch_industry_names_sync",
    "fetch_industry_names_async",
    "fetch_industry_hist_sync",
    "fetch_industry_hist_async",
    "resolve_date_range",
    "calculate_price_changes_for_periods",
    "date_converter",
    # Stock sector fund flow
    "StockSectorFundFlowAPI",
    "StockSectorFundFlowConfig",
    "fetch_sector_fund_flow_sync",
    "fetch_sector_fund_flow_async",
    "validate_fund_flow_data_availability",
    "create_empty_fund_flow_data",
    "process_sector_fund_flow_for_periods",
    # Stock market data
    "StockMarketDataAPI",
    "StockMarketDataConfig",
    "fetch_shanghai_spot_sync",
    "fetch_shenzhen_spot_sync",
    "fetch_beijing_spot_sync",
    "fetch_all_a_shares_spot_sync",
    "fetch_shanghai_spot_async",
    "fetch_shenzhen_spot_async",
    "fetch_beijing_spot_async",
    "fetch_all_a_shares_spot_async",
    "fetch_multiple_markets_async",
    "combine_market_data",
    # Stock board constituents
    "StockBoardConstituentsAPI",
    "StockBoardConstituentsConfig",
    "fetch_industry_constituents_sync",
    "fetch_industry_constituents_async",
    "fetch_multiple_industries_constituents_async",
    "create_industry_stock_mapping",
    "get_stocks_by_industries",
    "get_all_stocks_from_industries",
]