#!/usr/bin/env python3
"""
Simple runner script for FHPS filter.

This script demonstrates how to run the FHPS filter independently.
"""

import asyncio

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


async def run_fhps_filter():
    """
    Run the FHPS filter with market data and progress tracking.
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

        # Initialize FHPS filter
        fhps_filter = FhpsFilter(
            industry_stock_mapping_df, stock_zh_a_spot_em_df, config_name="test"
        )

        console.print("✅ FHPS filter initialized with config:")
        console.print(f"   - FHPS date: {fhps_filter.FHPS_DATE}")
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

    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_fhps_filter())
