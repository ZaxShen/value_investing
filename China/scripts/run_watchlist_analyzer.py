#!/usr/bin/env python3
"""
Runner script for watchlist analyzer operations.

This script runs the watchlist analysis functionality independently with progress tracking.
It fetches market data, loads watchlist files (based on config), and generates analysis reports.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.analyzers.watchlist_analyzer import WatchlistAnalyzer
from src.utilities.market_data_fetcher import (
    get_industry_stock_mapping_data,
    get_market_data,
)

console = Console()


async def run_watchlist_analyzer(config_name: str = "config.yml"):
    """
    Run the watchlist analyzer with market data and progress tracking.
    
    Args:
        config_name: Configuration file name to use (without .yml extension if not provided)
    """
    console.print("[bold green]🚀 Running Watchlist Analyzer[/bold green]")

    try:
        # Ensure config name has .yml extension
        if not config_name.endswith('.yml'):
            config_name = f"{config_name}.yml"

        # Fetch market data with shared progress context
        console.print("📊 Fetching market data...")
        with Progress(console=console) as data_progress:
            stock_zh_a_spot_em_df, industry_stock_mapping_df = await asyncio.gather(
                get_market_data(progress=data_progress),
                get_industry_stock_mapping_data(progress=data_progress),
            )

        console.print(f"✅ Market data: {len(stock_zh_a_spot_em_df)} stocks")
        console.print(f"✅ Industry mapping: {len(industry_stock_mapping_df)} mappings")

        # Initialize watchlist analyzer with specified config
        console.print(f"📋 Loading configuration: {config_name}")
        analyzer = WatchlistAnalyzer(
            industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name=config_name
        )

        console.print("✅ Watchlist analyzer initialized with config:")
        console.print(f"   - Watchlist directory: {analyzer.WATCHLIST_DIR}")
        console.print(f"   - Report directory: {analyzer.REPORT_DIR}")
        console.print(f"   - Days lookback period: {analyzer.DAYS_LOOKBACK_PERIOD}")
        
        # Show file selection mode
        if analyzer.analyzer_config.watchlist_files == "*":
            console.print(f"   - File mode: Scan all JSON files")
        else:
            file_count = len(analyzer.analyzer_config.watchlist_files)
            console.print(f"   - File mode: Process {file_count} specific files")

        # Run analysis with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            main_task = progress.add_task("🔍 Watchlist Analysis", total=100)

            await analyzer.run_analysis_from_files(
                _progress=progress, _parent_task_id=main_task
            )

        console.print(
            "[bold green]🎉 Watchlist analysis completed successfully![/bold green]"
        )

    except FileNotFoundError as e:
        console.print(f"[bold red]❌ Configuration file not found: {e}[/bold red]")
        console.print("[yellow]Available configs in input/config/analyzers/watchlist_analyzer/:[/yellow]")
        config_dir = Path("input/config/analyzers/watchlist_analyzer")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yml"):
                console.print(f"  - {config_file.stem}")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"[bold red]❌ Watchlist analysis failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the watchlist analyzer runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run watchlist analyzer operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_watchlist_analyzer.py                    # Use default config
  python scripts/run_watchlist_analyzer.py --config config    # Use config.yml
  python scripts/run_watchlist_analyzer.py --config test      # Use test.yml
  python scripts/run_watchlist_analyzer.py -c my_config       # Use my_config.yml
        """
    )
    
    parser.add_argument(
        "-c", "--config", 
        default="config",
        help="Configuration file name (without .yml extension, default: config)"
    )
    
    args = parser.parse_args()
    
    # Run the analysis process
    asyncio.run(run_watchlist_analyzer(args.config))


if __name__ == "__main__":
    main()