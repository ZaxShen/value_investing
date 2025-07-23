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
import argparse
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

    # Copy each report type with progress tracking
    with tqdm(total=len(report_patterns), desc="Copying reports", unit="report", leave=False) as copy_pbar:
        successful_copies = 0
        
        for report_info in report_patterns:
            copy_pbar.set_description(f"Copying {report_info['description']}")
            
            latest_file = get_latest_file(report_info["pattern"])
            if latest_file:
                original_filename = os.path.basename(latest_file)
                target_path = os.path.join(today_dir, original_filename)
                shutil.copy2(latest_file, target_path)
                successful_copies += 1
                logger.info(
                    "✅ Successfully copied %s: %s",
                    report_info["description"],
                    original_filename,
                )
            else:
                logger.warning(
                    "⚠️ No %s found matching pattern: %s",
                    report_info["description"],
                    report_info["pattern"],
                )
            
            copy_pbar.update(1)
        
        copy_pbar.set_description(f"✅ Copied {successful_copies}/{len(report_patterns)} reports")

    logger.info("Report copying completed: %d/%d reports copied to %s/", 
                successful_copies, len(report_patterns), today_dir)


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
        for i, (desc, task_func) in enumerate(tasks, 1):
            pbar.set_description(f"[{i}/{len(tasks)}] {desc}")
            logger.info("Starting task %d/%d: %s", i, len(tasks), desc)
            
            try:
                await task_func()
                pbar.update(1)
                logger.info("✅ Completed task %d/%d: %s", i, len(tasks), desc)
                
            except Exception as e:
                logger.error("❌ Error in task %d/%d (%s): %s", i, len(tasks), desc, str(e))
                raise

        pbar.set_description("🎉 Sequential analysis completed successfully!")

    print("🎉 Sequential analysis pipeline completed successfully!")
    logger.info("=== All sequential scripts completed successfully ===")


@timer
async def run_all_scripts_parallel() -> None:
    """
    Run all analysis scripts in parallel with progress tracking.

    This function executes stock filtering, stock analysis, and industry
    analysis concurrently for better performance, providing real-time
    progress feedback for each component.
    """
    logger.info("Starting parallel execution of all scripts")

    # Create a progress bar for the main stages
    main_stages = [
        "🔄 Running analysis scripts in parallel",
        "📋 Copying latest reports",
        "🎉 Pipeline completed"
    ]
    
    with tqdm(total=len(main_stages), desc="Parallel Stock Analysis Pipeline", 
              unit="stage", leave=True, position=0) as main_pbar:
        
        # Stage 1: Run the three main scripts concurrently
        main_pbar.set_description("🔄 Running analysis scripts in parallel")
        logger.info("Executing stock filter, stock analysis, and industry filter in parallel")
        
        try:
            await asyncio.gather(
                stock_filter_main(),
                stock_analysis_main(),
                industry_filter_main(),
            )
            main_pbar.update(1)
            logger.info("✅ All parallel analysis scripts completed successfully")
            
        except Exception as e:
            logger.error("❌ Error in parallel script execution: %s", str(e))
            raise

        # Stage 2: Copy latest reports
        main_pbar.set_description("📋 Copying latest reports")
        logger.info("Starting report copying process")
        
        try:
            await copy_latest_reports()
            main_pbar.update(1)
            logger.info("✅ Report copying completed successfully")
            
        except Exception as e:
            logger.error("❌ Error in report copying: %s", str(e))
            raise

        # Stage 3: Completion
        main_pbar.set_description("🎉 Pipeline completed successfully!")
        main_pbar.update(1)

    print("🎉 Parallel analysis pipeline completed successfully!")
    logger.info("=== All parallel scripts completed successfully ===")


@timer
def startup_connectivity_check(skip_check: bool = False) -> None:
    """
    Perform initial akshare connectivity check before starting the pipeline.

    This ensures that akshare services are accessible before beginning
    any data-intensive operations.
    
    Args:
        skip_check: If True, skip the connectivity check entirely
    """
    if skip_check:
        logger.info("⚠️ Akshare connectivity check disabled via command line argument")
        print("⚠️ Connectivity check disabled - proceeding with analysis")
        return
    
    logger.info("🔍 Performing startup akshare connectivity check...")
    
    try:
        with tqdm(total=1, desc="🔍 Checking akshare connectivity", unit="check", leave=False) as conn_pbar:
            # Perform comprehensive health check
            status, details = get_akshare_health_status()
            conn_pbar.update(1)
            
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


def parse_arguments():
    """
    Parse command line arguments for the stock analysis pipeline.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="China Stock Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with connectivity check (default)
  python main.py --skip-check       # Skip connectivity check
  python main.py --sequential       # Run scripts sequentially instead of parallel
  python main.py --skip-check --sequential  # Skip check and run sequentially
        """
    )
    
    parser.add_argument(
        "--skip-check", 
        action="store_true",
        help="Skip akshare connectivity check at startup"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true", 
        help="Run analysis scripts sequentially instead of in parallel"
    )
    
    return parser.parse_args()


@timer
def main() -> None:
    """
    Main entry point for the stock analysis pipeline.

    This function performs startup checks, configures logging, initializes
    the progress display, and orchestrates the execution of all analysis
    components. It provides error handling and completion notifications.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=== Starting China Stock Analysis Pipeline ===")

    # Set console logging to only show errors to avoid interfering with progress bars
    set_console_log_level("ERROR")

    # Option to set log level dynamically for detailed file logging
    # set_log_level("DEBUG")  # Uncomment for verbose logging

    print("🚀 Starting China Stock Analysis Pipeline")

    # Perform startup connectivity check (unless disabled)
    startup_connectivity_check(skip_check=args.skip_check)

    try:
        # Choose execution mode based on arguments
        if args.sequential:
            print("📋 Running scripts sequentially...")
            asyncio.run(run_all_scripts())
        else:
            print("⚡ Running scripts in parallel...")
            asyncio.run(run_all_scripts_parallel())

        logger.info("=== China Stock Analysis Pipeline Completed ===")
        print("✅ All analysis completed! Check logs/ directory for detailed logs.")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        print(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
