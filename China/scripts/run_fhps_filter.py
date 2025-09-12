#!/usr/bin/env python3
"""
Runner script for FHPS filter operations.

This script runs the FHPS filtering functionality independently with progress tracking.
It requires cached FHPS data from the caching script to be available.
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.filters.fhps_filter import FhpsFilter
from src.utilities.market_data_fetcher import (
    get_industry_stock_mapping_data,
    get_market_data,
)

console = Console()


async def run_fhps_filter(config_name: str = "filter_config"):
    """
    Run the FHPS filter with market data and progress tracking.
    
    Args:
        config_name: Configuration file name to use (without .yml extension)
    """
    console.print("[bold green]🚀 Running FHPS Filter[/bold green]")

    try:
        # Fetch market data with shared progress context
        console.print("📊 Fetching market data...")
        with Progress(console=console) as data_progress:
            stock_zh_a_spot_em_df, industry_stock_mapping_df = await asyncio.gather(
                get_market_data(progress=data_progress),
                get_industry_stock_mapping_data(progress=data_progress),
            )

        console.print(f"✅ Market data: {len(stock_zh_a_spot_em_df)} stocks")
        console.print(f"✅ Industry mapping: {len(industry_stock_mapping_df)} mappings")

        # Initialize FHPS filter with specified config
        console.print(f"📋 Loading configuration: {config_name}")
        fhps_filter = FhpsFilter(
            industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name=config_name
        )

        console.print("✅ FHPS filter initialized with config:")
        console.print(f"   - Max price change: {fhps_filter.MAX_PRICE_CHANGE_PERCENT}%")
        console.print(f"   - Min transfer ratio: {fhps_filter.MIN_TRANSFER_RATIO}")
        console.print(f"   - Max circulating market cap: {fhps_filter.MAX_CIRCULATING_MARKET_CAP_YI}亿")
        console.print(f"   - Min P/E ratio: > {fhps_filter.MIN_PE_RATIO}")
        console.print(f"   - Batch size: {fhps_filter.BATCH_SIZE}")

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
            main_task = progress.add_task("🔍 FHPS Filter Analysis", total=100)

            await fhps_filter.run_analysis(
                _progress=progress, _parent_task_id=main_task
            )

        console.print(
            "[bold green]🎉 FHPS analysis completed successfully![/bold green]"
        )

    except FileNotFoundError as e:
        console.print(f"[bold red]❌ Configuration file not found: {e}[/bold red]")
        console.print("[yellow]Available configs in input/config/filters/fhps_filter/:[/yellow]")
        config_dir = Path("input/config/filters/fhps_filter")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yml"):
                console.print(f"  - {config_file.stem}")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"[bold red]❌ FHPS filter failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the FHPS filter runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run FHPS filter operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_fhps_filter.py                        # Use default config
  python scripts/run_fhps_filter.py --config filter_config # Use filter_config
  python scripts/run_fhps_filter.py -c my_config          # Use custom config
        """
    )
    
    parser.add_argument(
        "-c", "--config", 
        default="filter_config",
        help="Configuration file name (without .yml extension, default: filter_config)"
    )
    
    args = parser.parse_args()
    
    # Run the filter process
    asyncio.run(run_fhps_filter(args.config))


if __name__ == "__main__":
    main()
