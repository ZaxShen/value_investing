#!/usr/bin/env python3
"""
Test script for the refactored FHPS filter.

This script demonstrates how to use the FhpsFilter class with market data.
It's designed to be run independently without being part of the main pipeline.
"""

import asyncio
import os
from datetime import datetime

from src.filters.fhps_filter import FhpsFilter
from src.utilities.market_data_fetcher import (
    get_market_data, 
    get_industry_stock_mapping_data
)


async def test_fhps_filter():
    """
    Test the FHPS filter with real market data.
    """
    print("🚀 Testing FHPS Filter")
    print("=" * 50)
    
    try:
        # Step 1: Fetch market data
        print("📊 Fetching market data...")
        stock_zh_a_spot_em_df = await get_market_data()
        industry_stock_mapping_df = await get_industry_stock_mapping_data()
        
        print(f"✅ Market data: {len(stock_zh_a_spot_em_df)} stocks")
        print(f"✅ Industry mapping: {len(industry_stock_mapping_df)} mappings")
        
        # Step 2: Initialize FHPS filter
        print("\n🔧 Initializing FHPS filter...")
        fhps_filter = FhpsFilter(
            industry_stock_mapping_df, 
            stock_zh_a_spot_em_df, 
            config_name="test"
        )
        print("✅ FHPS filter initialized")
        
        # Step 3: Run analysis (with limited scope for testing)
        print("\n📈 Running FHPS analysis...")
        print("Note: This will fetch real FHPS data and process a limited number of stocks")
        
        # Run the analysis
        await fhps_filter.run_analysis()
        
        # Step 4: Check output
        output_dir = fhps_filter.REPORT_DIR
        today_str = datetime.now().strftime("%Y%m%d")
        expected_file = os.path.join(
            output_dir, 
            fhps_filter.OUTPUT_FILENAME_TEMPLATE.format(date=today_str)
        )
        
        if os.path.exists(expected_file):
            print(f"✅ Report generated: {expected_file}")
            
            # Show basic stats
            import pandas as pd
            df = pd.read_csv(expected_file)
            print(f"📋 Report contains {len(df)} stocks")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"📋 Sample data:")
            print(df.head(3)[['代码', '名称', '行业', '除权除息日', '自除权出息日起涨跌幅%']])
        else:
            print(f"⚠️  Expected report file not found: {expected_file}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fhps_filter())