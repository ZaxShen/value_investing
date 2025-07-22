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
from tqdm import tqdm
import time
from src.stock_filter import main as stock_filter_main
from src.stock_analysis import main as stock_analysis_main
from src.industry_filter import main as industry_filter_main
from src.utilities.logger import get_logger, set_console_log_level
from src.utilities.tools import timer
from src.utilities.akshare_checker import (
    check_akshare_connectivity,
    get_akshare_health_status,
    log_connectivity_status,
    ConnectivityStatus,
)


# Initialize logger
logger = get_logger("main")


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

    # Sequential execution with tqdm progress bar
    tasks = [
        ("Running stock filter", stock_filter_main),
        ("Running stock analysis", stock_analysis_main),
        ("Running industry filter", industry_filter_main),
        ("Copying latest reports", copy_latest_reports),
    ]

    with tqdm(
        total=len(tasks),
        desc="Sequential Stock Analysis Pipeline",
        unit="task",
        leave=True,
    ) as pbar:
        for desc, task_func in tasks:
            pbar.set_description(desc)
            await task_func()
            pbar.update(1)

        pbar.set_description("Sequential analysis completed!")

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

    # Parallel execution - run main scripts concurrently
    print("🔄 Running all analysis scripts in parallel...")

    # Run the three main scripts concurrently
    await asyncio.gather(
        stock_filter_main(),
        stock_analysis_main(),
        industry_filter_main(),
    )

    print("✅ All parallel scripts completed!")

    # Copy latest reports
    print("📋 Copying latest reports...")
    await copy_latest_reports()

    print("🎉 Parallel analysis completed!")

    logger.info("All scripts completed!")


@timer
def startup_connectivity_check() -> None:
    """
    Perform initial akshare connectivity check before starting the pipeline.

    This ensures that akshare services are accessible before beginning
    any data-intensive operations.
    """
    logger.info("🔍 Performing startup akshare connectivity check...")
    print("🔍 Checking akshare connectivity...")

    try:
        # Perform comprehensive health check
        status, details = get_akshare_health_status()

        # Log detailed status
        log_connectivity_status(status, details)

        # Handle different status levels
        if status == ConnectivityStatus.UNAVAILABLE:
            error_msg = (
                "❌ Akshare services are completely unavailable. Cannot start pipeline."
            )
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print("Please check your internet connection and try again.")
            raise ConnectionError("Akshare services unavailable")

        elif status == ConnectivityStatus.DEGRADED:
            warning_msg = "⚠️  Akshare services are degraded but operational."
            logger.warning(warning_msg)
            print(f"⚠️ {warning_msg}")
            print("Some features may be slower than usual.")

        else:  # HEALTHY
            success_msg = "✅ Akshare connectivity verified successfully!"
            logger.info(success_msg)
            print(f"✅ {success_msg}")

    except Exception as e:
        error_msg = f"❌ Startup connectivity check failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        raise


@timer
def main() -> None:
    """
    Main entry point for the stock analysis pipeline.

    This function performs startup checks, configures logging, initializes
    the progress display, and orchestrates the execution of all analysis
    components. It provides error handling and completion notifications.
    """
    logger.info("=== Starting China Stock Analysis Pipeline ===")

    # Set console logging to only show errors to avoid interfering with progress bars
    set_console_log_level("ERROR")

    # Option to set log level dynamically for detailed file logging
    # set_log_level("DEBUG")  # Uncomment for verbose logging

    print("🚀 Starting China Stock Analysis Pipeline")

    # Perform startup connectivity check first
    # startup_connectivity_check()  # Comment to temporarily disabled for testing
    print("⚠️ Connectivity check disabled - proceeding with analysis")

    try:
        # Choose one of these approaches:

        # Option 1: Run sequentially
        # asyncio.run(run_all_scripts())

        # Option 2: Run in parallel (current default)
        asyncio.run(run_all_scripts_parallel())

        logger.info("=== China Stock Analysis Pipeline Completed ===")
        print("✅ All analysis completed! Check logs/ directory for detailed logs.")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        print(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
