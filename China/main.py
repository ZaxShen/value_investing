"""
Main entry point for the China stock analysis pipeline.

This module orchestrates the execution of all analysis components including
stock filtering, stock analysis, and industry analysis. It provides both
sequential and parallel execution modes with beautiful progress bars.
"""

# Configure environment settings before importing any other modules
import src.settings  # This configures the environment

import asyncio
import os
import shutil
import glob
import re
from typing import Optional
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from src.stock_filter import main as stock_filter_main
from src.stock_analysis import main as stock_analysis_main
from src.industry_filter import main as industry_filter_main
from src.utilities.logger import get_logger, set_console_log_level
from src.utilities.tools import timer


# Initialize logger and console
logger = get_logger("main")
console = Console()


@timer
def get_latest_file(pattern: str) -> Optional[str]:
    """
    Get the latest file matching the pattern based on date in filename.

    Args:
        pattern: Glob pattern to match files (e.g., "data/stocks/report-*.csv")

    Returns:
        Path to the latest file matching the pattern, or None if no files found
    """
    files = glob.glob(pattern)
    if not files:
        return None

    # Extract date from filename (format: YYYYMMDD)
    def extract_date(filename: str) -> str:
        # Look for 8-digit date pattern (YYYYMMDD)
        match = re.search(r"(\d{8})", os.path.basename(filename))
        if match:
            return match.group(1)
        # Fallback to modification time if no date pattern found
        return str(int(os.path.getmtime(filename)))

    # Sort by date in filename, get the latest
    latest_file = max(files, key=extract_date)
    return latest_file


@timer
async def copy_latest_reports() -> None:
    """
    Copy the latest reports to data/today directory.

    This function finds the most recent report files and copies them to a
    standardized 'today' directory for easy access.
    """
    logger.info("Starting report copying process")

    # Create data/today directory if it doesn't exist
    today_dir = "data/today"
    os.makedirs(today_dir, exist_ok=True)

    # Define report patterns to copy
    report_patterns = [
        {
            "pattern": "data/holding_stocks/reports/持股报告-[0-9]*.csv",
            "description": "持股报告",
        },
        {
            "pattern": "data/stocks/reports/股票筛选报告-[0-9]*.csv",  # Exclude raw reports
            "description": "股票筛选报告",
        },
        {
            "pattern": "data/stocks/reports/行业筛选报告-raw-[0-9]*.csv",  # Exclude raw reports
            "description": "行业筛选报告",
        },
    ]

    # Copy each report type
    for report_info in report_patterns:
        latest_file = get_latest_file(report_info["pattern"])
        if latest_file:
            original_filename = os.path.basename(latest_file)
            target_path = os.path.join(today_dir, original_filename)
            shutil.copy2(latest_file, target_path)
            logger.info(
                "Successfully copied %s: %s",
                report_info["description"],
                original_filename,
            )
        else:
            logger.warning(
                "No %s found matching pattern: %s",
                report_info["description"],
                report_info["pattern"],
            )

    logger.info("All latest reports copied to %s/", today_dir)


@timer
async def run_all_scripts() -> None:
    """
    Run all analysis scripts sequentially with progress tracking.

    This function executes stock filtering, stock analysis, and industry
    analysis in sequence, providing detailed progress feedback.
    """
    logger.info("Starting sequential execution of all scripts")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Sequential Stock Analysis Pipeline", total=4)

        # Run stock_filter.py
        progress.update(task, description="Running stock filter")
        await stock_filter_main()
        progress.advance(task)

        # Run stock_analysis.py
        progress.update(task, description="Running stock analysis")
        await stock_analysis_main()
        progress.advance(task)

        # Run industry_filter.py
        progress.update(task, description="Running industry filter")
        await industry_filter_main()
        progress.advance(task)

        # Copy latest reports to data/today
        progress.update(task, description="Copying latest reports")
        await copy_latest_reports()
        progress.advance(task)

        progress.update(task, description="Sequential analysis completed!")

    logger.info("All scripts completed!")


@timer
async def run_all_scripts_parallel() -> None:
    """
    Run all analysis scripts in parallel with progress tracking.

    This function executes stock filtering, stock analysis, and industry
    analysis concurrently for better performance, providing real-time
    progress feedback for each component.
    """
    logger.info("Starting parallel execution of all scripts")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        main_task = progress.add_task("Parallel Stock Analysis Pipeline", total=2)

        # Run all async scripts in parallel
        progress.update(
            main_task, description="Running all analysis scripts in parallel"
        )

        # Create individual tasks for each script
        stock_filter_task = progress.add_task("Stock Filter", total=1)
        stock_analysis_task = progress.add_task("Stock Analysis", total=1)
        industry_filter_task = progress.add_task("Industry Filter", total=1)

        # Run tasks in parallel
        async def run_with_progress(coro, task_id, name):
            progress.update(task_id, description=f"Running {name}")
            await coro
            progress.advance(task_id)
            progress.update(task_id, description=f"{name} completed")

        await asyncio.gather(
            run_with_progress(stock_filter_main(), stock_filter_task, "Stock Filter"),
            run_with_progress(
                stock_analysis_main(), stock_analysis_task, "Stock Analysis"
            ),
            run_with_progress(
                industry_filter_main(), industry_filter_task, "Industry Filter"
            ),
        )

        progress.advance(main_task)

        # Copy latest reports to data/today
        progress.update(main_task, description="Copying latest reports")
        await copy_latest_reports()
        progress.advance(main_task)

        progress.update(main_task, description="Parallel analysis completed!")

    logger.info("All scripts completed!")


@timer
def main() -> None:
    """
    Main entry point for the stock analysis pipeline.

    This function configures logging, initializes the progress display,
    and orchestrates the execution of all analysis components. It provides
    error handling and completion notifications.
    """
    logger.info("=== Starting China Stock Analysis Pipeline ===")

    # Set console logging to only show errors to avoid interfering with progress bars
    set_console_log_level("ERROR")

    # Option to set log level dynamically for detailed file logging
    # set_log_level("DEBUG")  # Uncomment for verbose logging

    console.print("[bold green]🚀 Starting China Stock Analysis Pipeline[/bold green]")

    try:
        # Choose one of these approaches:

        # Option 1: Run sequentially
        # asyncio.run(run_all_scripts())

        # Option 2: Run in parallel (current default)
        asyncio.run(run_all_scripts_parallel())

        logger.info("=== China Stock Analysis Pipeline Completed ===")
        console.print(
            "[bold green]✅ All analysis completed! Check logs/ directory for detailed logs.[/bold green]"
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        console.print(f"[bold red]❌ Pipeline failed: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()
