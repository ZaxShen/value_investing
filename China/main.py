import asyncio
import os
import shutil
import glob
from datetime import datetime
from src.stock_filter import main as stock_filter_main
from src.stock_analysis import main as stock_analysis_main
from src.industry_filter import main as industry_filter_main


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


async def copy_latest_reports():
    """Copy the latest reports to data/today directory"""
    print("\n=== Copying latest reports to data/today ===")

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
            print(f"✅ Copied {report_info['description']}: {original_filename}")
        else:
            print(
                f"❌ No {report_info['description']} found matching pattern: {report_info['pattern']}"
            )

    print(f"📁 All latest reports copied to {today_dir}/")


async def run_all_scripts():
    """Run all async scripts sequentially"""
    print("Starting stock analysis pipeline...")

    # Run stock_filter.py
    print("\n=== Running stock_filter.py ===")
    await stock_filter_main()

    # Run stock_analysis.py
    print("\n=== Running stock_analysis.py ===")
    await stock_analysis_main()  # Now this is an async function

    # Run industry_filter.py
    print("\n=== Running industry_filter.py ===")
    await industry_filter_main()

    print("\nAll scripts completed!")

    # Copy latest reports to data/today
    await copy_latest_reports()


async def run_all_scripts_parallel():
    """Run all async scripts in parallel (if they don't depend on each other)"""
    print("Starting stock analysis pipeline in parallel...")

    # Run all async scripts in parallel
    tasks = [
        stock_filter_main(),
        stock_analysis_main(),
        industry_filter_main(),
    ]

    await asyncio.gather(*tasks)
    print("All scripts completed!")

    # Copy latest reports to data/today
    await copy_latest_reports()


def main():
    print("Hello from china!")

    # Choose one of these approaches:

    # Option 1: Run sequentially
    # asyncio.run(run_all_scripts())

    # Option 2: Run in parallel (uncomment if preferred)
    asyncio.run(run_all_scripts_parallel())


if __name__ == "__main__":
    main()
