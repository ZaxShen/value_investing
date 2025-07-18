import asyncio
import os
import shutil
import glob
from datetime import datetime
from src.stock_filter import main as stock_filter_main
from src.stock_analysis import main as stock_analysis_main
from src.industry_filter import main as industry_filter_main
from src.utilities.logger import get_logger, set_log_level
from src.utilities.tools import logged, verbose, verbose_tracker, enable_verbose_tracking, print_verbose_summary


# Initialize logger
logger = get_logger("main")

@verbose
def get_latest_file(pattern):
    """Get the latest file matching the pattern based on date in filename"""
    import re

    files = glob.glob(pattern)
    if not files:
        return None

    # Extract date from filename (format: YYYYMMDD)
    def extract_date(filename):
        # Look for 8-digit date pattern (YYYYMMDD)
        match = re.search(r"(\d{8})", os.path.basename(filename))
        if match:
            return match.group(1)
        # Fallback to modification time if no date pattern found
        return str(int(os.path.getmtime(filename)))

    # Sort by date in filename, get the latest
    latest_file = max(files, key=extract_date)
    return latest_file


@verbose
async def copy_latest_reports():
    """Copy the latest reports to data/today directory"""
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
            logger.info(f"Successfully copied {report_info['description']}: {original_filename}")
        else:
            logger.warning(f"No {report_info['description']} found matching pattern: {report_info['pattern']}")

    logger.info(f"All latest reports copied to {today_dir}/")


@verbose
async def run_all_scripts():
    """Run all async scripts sequentially"""
    logger.info("Starting sequential execution of all scripts")

    # Start process tracking
    verbose_tracker.start_process("sequential_analysis", "Sequential Stock Analysis Pipeline", 4)

    # Run stock_filter.py
    verbose_tracker.update_process("sequential_analysis", "Running stock filter")
    await stock_filter_main()

    # Run stock_analysis.py
    verbose_tracker.update_process("sequential_analysis", "Running stock analysis")
    await stock_analysis_main()  # Now this is an async function

    # Run industry_filter.py
    verbose_tracker.update_process("sequential_analysis", "Running industry filter")
    await industry_filter_main()

    logger.info("All scripts completed!")

    # Copy latest reports to data/today
    verbose_tracker.update_process("sequential_analysis", "Copying latest reports")
    await copy_latest_reports()
    
    # Complete process tracking
    verbose_tracker.complete_process("sequential_analysis", True)


@verbose
async def run_all_scripts_parallel():
    """Run all async scripts in parallel (if they don't depend on each other)"""
    logger.info("Starting parallel execution of all scripts")

    # Start process tracking
    verbose_tracker.start_process("parallel_analysis", "Parallel Stock Analysis Pipeline", 2)

    # Run all async scripts in parallel
    verbose_tracker.update_process("parallel_analysis", "Running all analysis scripts in parallel")
    tasks = [
        stock_filter_main(),
        stock_analysis_main(),
        industry_filter_main(),
    ]

    await asyncio.gather(*tasks)
    logger.info("All scripts completed!")

    # Copy latest reports to data/today
    verbose_tracker.update_process("parallel_analysis", "Copying latest reports")
    await copy_latest_reports()
    
    # Complete process tracking
    verbose_tracker.complete_process("parallel_analysis", True)


@verbose
def main():
    """Main entry point for the stock analysis pipeline"""
    logger.info("=== Starting China Stock Analysis Pipeline ===")

    # Enable verbose tracking for detailed progress monitoring
    enable_verbose_tracking()

    # Option to set log level dynamically
    # set_log_level("DEBUG")  # Uncomment for verbose logging

    # Start overall pipeline tracking
    verbose_tracker.start_process("main_pipeline", "China Stock Analysis Pipeline", 3)

    try:
        # Choose one of these approaches:
        verbose_tracker.update_process("main_pipeline", "Initializing analysis pipeline")

        # Option 1: Run sequentially
        # asyncio.run(run_all_scripts())

        # Option 2: Run in parallel (current default)
        verbose_tracker.update_process("main_pipeline", "Running analysis scripts")
        asyncio.run(run_all_scripts_parallel())
        
        verbose_tracker.update_process("main_pipeline", "Generating summary report")
        logger.info("=== China Stock Analysis Pipeline Completed ===")
        
        # Complete main pipeline
        verbose_tracker.complete_process("main_pipeline", True)
        
        # Print comprehensive summary
        print_verbose_summary()
        
        # Log completion message
        logger.info("🎉 All analysis completed! Check logs/ directory for detailed logs.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        verbose_tracker.complete_process("main_pipeline", False)
        raise


if __name__ == "__main__":
    main()
