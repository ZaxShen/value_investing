# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a value investing analysis project with separate components for different markets:

- `China/` - Chinese stock market analysis using akshare library
- `USA/` - US stock market analysis using various APIs
- `shared/` - Common utilities and shared resources

## Environment Setup

The project uses uv as the package manager with workspace configuration:

- Root `pyproject.toml` defines workspace with China as a member
- China subproject has its own `pyproject.toml` with specific dependencies
- USA subproject has its own `pyproject.toml`

### Running the Project

To run the Chinese stock analysis:

```bash
cd China
uv run python stock_filter.py
```

To run specific analysis modules:

```bash
cd China
uv run python main.py
uv run python stock_analysis_claude.py
```

## Key Components

### Chinese Stock Analysis (China/)

Main libraries used:

- `akshare` - Chinese stock market data
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Data visualization
- `asyncio` - Asynchronous processing for API calls

Key modules:

- `stock_filter.py` - Main stock filtering and analysis logic with async processing
- `stock_analysis_claude.py` - Improved version with better code structure and caching
- `utilities/get_stock_data.py` - Data fetching utilities with caching mechanism
- `utilities/tools.py` - Common utilities including timer decorator

### Data Management

- Stock data is cached in `data/stocks/` directory with date-based filenames
- Reports are generated in `data/stocks/reports/` directory
- Cached data includes market data and industry mapping to avoid repeated API calls

### Async Processing Architecture

The code uses async/await patterns with:

- Semaphore-based rate limiting for API calls (`REQUEST_SEMAPHORE`)
- Batch processing of industries to avoid overwhelming APIs
- Executor pattern for running blocking akshare calls in thread pool

### Stock Filtering Logic

Default filters applied:

- Market cap < 20 billion RMB
- P/E ratio between 0 and 50  
- Main net inflow > 100 million RMB (configurable)
- Price change < 10% over analysis period

## Development Notes

- The timer decorator in `utilities/tools.py` works with both sync and async functions
- API rate limiting is implemented to respect akshare service limits
- Data caching prevents unnecessary API calls and improves performance
- Error handling is implemented for individual stock processing failures

## Report Generation

Reports are generated with date stamps in format `YYYYMMDD`:

- Raw reports: `股票筛选报告-raw-{date}.csv`
- Filtered reports: `股票筛选报告-{date}.csv`
