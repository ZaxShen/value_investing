# FHPS Filter - Refactored Implementation

## Overview

The FHPS Filter has been completely refactored from a Jupyter notebook-style script to follow industrial standards, matching the pattern used in `watchlist_analyzer.py` and other components in the codebase.

## What FHPS Filter Does

FHPS (分红派息送股) Filter analyzes Chinese stocks with dividend/split plans by:

1. **Fetching stocks with ex-dividend dates** from akshare API
2. **Enriching with comprehensive data**:
   - Industry information (from industry mapping)
   - Market data (market cap, P/E, P/B ratios, etc.)
   - Historical price data (ex-dividend date vs current price)
   - Fund flow data (1/5/29-day periods)
   - Price change data (multiple timeframes)
3. **Generating detailed reports** with complete analysis

## Key Improvements

### ✅ **Removed from Original**
- All print statements and debug code
- Hardcoded values and direct execution
- Jupyter notebook style structure
- Fixed date and filter parameters
- Simple CSV output with limited columns

### ✅ **Added New Features**
- **Class-based architecture** following industrial standards
- **Configuration-driven approach** with YAML configs
- **Async/await processing** with batch control and rate limiting
- **Progress tracking integration** for Rich progress bars
- **Comprehensive data enrichment**:
  - Industry mapping from market data
  - Fund flow data (1/5/29-day periods)
  - Extended price change analysis (60-day, YTD)
  - Market fundamentals (P/E, P/B, market cap)
- **Enhanced output format** with 18+ columns vs original 7
- **Proper error handling** and logging
- **Configurable parameters** (dates, ratios, batch sizes)

## Usage

### Configuration

Edit `input/config/filters/fhps_filter/test.yml`:

```yaml
# FHPS specific settings
fhps_filter:
  fhps_date: "20241231"  # Date for FHPS data query (YYYYMMDD)
  min_transfer_ratio: 1.0  # Minimum transfer ratio to filter
  batch_size: 5  # Concurrent processing batch size
  report_dir: "output/reports/filters/fhps_filter"
  output_filename_template: "除权除息股票-{date}.csv"

# Fund flow settings  
akshare:
  stock_individual_fund_flow:
    period_count: [1, 5, 29]  # Days for fund flow analysis
```

### Programmatic Usage

```python
import asyncio
from src.filters.fhps_filter import FhpsFilter
from src.utilities.market_data_fetcher import get_market_data, get_industry_stock_mapping_data

async def run_fhps_analysis():
    # Fetch market data
    stock_data = await get_market_data()
    industry_data = await get_industry_stock_mapping_data()
    
    # Initialize filter
    fhps_filter = FhpsFilter(industry_data, stock_data, config_name="test")
    
    # Run analysis
    await fhps_filter.run_analysis()

# Execute
asyncio.run(run_fhps_analysis())
```

### Standalone Testing

Use the provided test script:

```bash
uv run test_fhps_filter.py
```

## Output Format

The generated CSV includes these columns:

| Column | Description |
|--------|-------------|
| (empty) | Row number starting from 0 |
| 行业 | Industry from mapping data |
| 代码 | Stock code |
| 名称 | Stock name |
| 总市值(亿) | Total market cap (100M RMB) |
| 流通市值(亿) | Float market cap (100M RMB) |
| 市盈率-动态 | Dynamic P/E ratio |
| 市净率 | P/B ratio |
| 送转股份-送转总比例 | Dividend/split ratio |
| 除权除息日 | Ex-dividend date |
| 除权除息日股价 | Stock price on ex-dividend date |
| {YYYYMMDD}股价 | Current stock price |
| 自除权出息日起涨跌幅% | Price change since ex-dividend |
| 1日主力净流入-总净额(亿) | 1-day main fund net inflow |
| 5日主力净流入-总净额(亿) | 5-day main fund net inflow |
| 29日主力净流入-总净额(亿) | 29-day main fund net inflow |
| 1日涨跌幅(%) | 1-day price change |
| 5日涨跌幅(%) | 5-day price change |
| 29日涨跌幅(%) | 29-day price change |
| 60日涨跌幅(%) | 60-day price change |
| 年初至今涨跌幅(%) | Year-to-date price change |

## Integration Notes

- **Not included in main.py** - Runs independently (as specified in requirements)
- **Uses same fund flow API** as `watchlist_analyzer.py` for consistency
- **Follows same config pattern** as other filters
- **Compatible with progress tracking** when integrated into larger workflows
- **Thread-safe and async-compatible** for concurrent use

## File Locations

- **Main class**: `src/filters/fhps_filter.py`
- **Config**: `input/config/filters/fhps_filter/test.yml`
- **Output**: `output/reports/filters/fhps_filter/除权除息股票-{YYYYMMDD}.csv`
- **Test**: `test_fhps_filter.py`

## Status

✅ **Completed and tested** - Ready for production use