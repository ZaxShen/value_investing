"""
Main entry point for the China stock analysis pipeline.

This module orchestrates the execution of all analysis components including
stock filtering, stock analysis, and industry analysis. It provides both
sequential and parallel execution modes with beautiful progress bars.
"""

# Configure environment settings before importing any other modules
# import src.settings  # This configures the environment

import asyncio
import os
import shutil
import glob
import re
from typing import Optional
# Rich Progress imports for detailed hierarchical progress tracking
from rich.progress import (
    Progress,        # Main progress container that manages multiple progress bars
    SpinnerColumn,   # Rotating spinner animation for active tasks
    TextColumn,      # Dynamic text descriptions that update during execution
    BarColumn,       # Visual progress bars with customizable width and styling
    TimeElapsedColumn,  # Shows elapsed time since task started
)
from rich.console import Console  # Rich console for styled output and progress display
from src.stock_filter import main as stock_filter_main
from src.stock_analysis import main as stock_analysis_main
from src.industry_filter import main as industry_filter_main
from src.utilities.logger import get_logger, set_console_log_level
from src.utilities.tools import timer


# Initialize logger and Rich console for styled output
# The logger provides file and console logging for the main pipeline
logger = get_logger("main")
# Rich console provides styled terminal output and progress bar rendering
console = Console()


# @timer
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


# @timer
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


# @timer
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


# @timer
async def run_all_scripts_parallel() -> None:
    """
    Run all analysis scripts in parallel with detailed hierarchical progress tracking.

    This function executes stock filtering, stock analysis, and industry analysis 
    concurrently for better performance. It uses Rich Progress to provide real-time
    progress feedback with a hierarchical structure:
    - Main pipeline progress (high-level stages)
    - Individual script progress bars that appear during execution
    - Detailed sub-task progress for complex operations
    - Automatic cleanup of completed progress bars to reduce clutter

    The progress display includes:
    - Spinner animations for active tasks
    - Progress bars showing completion percentage
    - Elapsed time for each operation
    - Dynamic task descriptions that update as work progresses
    - Automatic removal of completed sub-tasks to keep display clean

    Rich Progress Features Used:
    - SpinnerColumn: Shows rotating spinner for active tasks
    - TextColumn: Displays dynamic task descriptions 
    - BarColumn: Shows visual progress bars
    - TimeElapsedColumn: Shows elapsed time for each task
    - Dynamic task management: Add/remove/update tasks as needed
    """
    logger.info("Starting parallel execution of all scripts")

    # Create Rich Progress instance with detailed column configuration
    # Each column serves a specific purpose in the progress display
    with Progress(
        SpinnerColumn(),  # Rotating spinner indicates active tasks
        TextColumn("[progress.description]{task.description}"),  # Dynamic task descriptions
        BarColumn(bar_width=40),  # Progress bars with fixed width for consistent display
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # Completion percentage
        TimeElapsedColumn(),  # Time elapsed since task started
        console=console,  # Use our configured Rich console instance
        transient=False,  # Keep completed progress bars visible briefly before cleanup
    ) as progress:

        # Create main pipeline progress tracker
        # This tracks the overall progress of the parallel execution pipeline
        main_task = progress.add_task(
            "🚀 Parallel Stock Analysis Pipeline", 
            total=2  # 2 main stages: parallel scripts + report copying
        )

        # Stage 1: Execute all analysis scripts in parallel
        # Create individual progress bars for each script to show detailed progress
        logger.info("Creating individual progress trackers for parallel script execution")
        
        # Each script gets its own progress bar that will show detailed sub-progress
        stock_filter_task = progress.add_task("📈 Stock Filter", total=100, visible=True)
        # Pre-create batch processing task for Stock Filter (initially hidden)
        stock_filter_batch_task = progress.add_task("    📊 Stock Filter batches", total=100, visible=False)
        
        stock_analysis_task = progress.add_task("💼 Stock Analysis", total=100, visible=True)
        # Pre-create batch processing task for Stock Analysis (initially hidden) 
        stock_analysis_batch_task = progress.add_task("    💼 Stock Analysis processing", total=100, visible=False)
        
        industry_filter_task = progress.add_task("🏭 Industry Filter", total=100, visible=True)
        # Pre-create batch processing task for Industry Filter (initially hidden)
        industry_filter_batch_task = progress.add_task("    🏭 Industry Filter batches", total=100, visible=False)

        # Enhanced progress wrapper that provides detailed progress updates
        # This function wraps each main script and provides progress feedback
        # It implements the tqdm-like behavior requested by the user:
        # - Detailed low-level task progress bars for batch processing
        # - Hierarchical progress structure with subtasks under main tasks
        # - Subtask progress bars that disappear when finished
        # - Top-level tasks remain visible (don't disappear)
        async def run_with_detailed_progress(script_func, task_id, name, progress_instance, batch_task_id=None):
            """
            Execute a coroutine with detailed progress tracking and cleanup.
            
            This wrapper function:
            1. Updates the task description to show it's starting
            2. Simulates detailed progress updates during execution
            3. Marks the task as completed when done
            4. Removes the completed task from display to reduce clutter
            
            Args:
                coro: The coroutine to execute (main script function)
                task_id: Rich Progress task ID for this operation
                name: Human-readable name for the operation
                progress_instance: Rich Progress instance for updates
            """
            try:
                # Mark task as starting with dynamic description
                progress_instance.update(task_id, description=f"🔄 Starting {name}...")
                
                # Execute the actual script with progress tracking
                # Pass progress instance and task_id to enable hierarchical progress
                await script_func(progress=progress_instance, parent_task_id=task_id, batch_task_id=batch_task_id)
                # Mark top-level task as completed but DON'T remove it
                # According to TODO.md: "Don't make the progress bar disappear for the top level tasks"
                progress_instance.update(task_id, completed=100, description=f"✅ {name} completed")
                
                logger.info(f"✅ {name} completed (top-level task kept visible)")
                
            except Exception as e:
                # Handle errors and update progress accordingly
                progress_instance.update(task_id, description=f"❌ {name} failed: {str(e)}")
                logger.error(f"❌ Error in {name}: {str(e)}")
                raise

        # Execute all scripts in parallel with hierarchical progress tracking
        logger.info("🚀 Launching parallel script execution with hierarchical progress tracking")
        await asyncio.gather(
            # FIXME: tqdm data preprocessing issue
            run_with_detailed_progress(stock_filter_main, stock_filter_task, "Stock Filter", progress, stock_filter_batch_task),
            # FIXME: tqdm data preprocessing issue
            run_with_detailed_progress(stock_analysis_main, stock_analysis_task, "Stock Analysis", progress, stock_analysis_batch_task),
            # TODO: Uncomment
            run_with_detailed_progress(industry_filter_main, industry_filter_task, "Industry Filter", progress, industry_filter_batch_task),
        )

        # Update main progress after parallel execution
        progress.advance(main_task)
        progress.update(main_task, description="📋 All analysis scripts completed, copying reports...")

        # Stage 2: Copy latest reports with progress tracking
        logger.info("Starting report copying with progress feedback")
        copy_task = progress.add_task("📋 Copying latest reports", total=100)
        
        # Simulate progress for report copying
        copy_progress_task = asyncio.create_task(simulate_copy_progress(progress, copy_task))
        
        # Execute actual report copying
        await copy_latest_reports()
        
        # Complete copy progress - this is a subtask so it can disappear
        copy_progress_task.cancel()
        progress.update(copy_task, completed=100, description="✅ Report copying completed")
        await asyncio.sleep(0.5)
        progress.remove_task(copy_task)  # Subtask disappears when finished

        # Complete main pipeline - top-level task stays visible
        progress.advance(main_task)
        progress.update(main_task, description="🎉 Parallel analysis pipeline completed successfully!")
        # Note: main_task is NOT removed - it stays visible as per TODO.md requirements

    logger.info("🎉 All parallel scripts completed successfully!")


# simulate_script_progress function removed - now using real progress from batch processing


async def simulate_copy_progress(progress, task_id):
    """
    Simulate progress for report copying operation.
    
    This function provides detailed progress feedback for the report copying
    phase, which involves finding and copying multiple report files. It gives
    users visibility into what files are being processed during the copy operation.
    
    Args:
        progress: Rich Progress instance for updating progress display
        task_id: Task ID to update with copy operation progress
    """
    try:
        copy_stages = [
            "Finding latest reports...",
            "Copying stock reports...",
            "Copying analysis reports...", 
            "Copying industry reports...",
            "Verifying copied files..."
        ]
        
        # Progress through each copy stage with detailed descriptions
        # Each stage represents a specific file type being copied
        for i, stage in enumerate(copy_stages):
            # Calculate progress in 20% increments for 5 stages
            progress.update(task_id, completed=(i + 1) * 20, description=f"📋 {stage}")
            await asyncio.sleep(0.5)  # Brief delay to show copy progress
            
    except asyncio.CancelledError:
        pass


# @timer
def main() -> None:
    """
    Main entry point for the stock analysis pipeline.

    This function configures logging, initializes the progress display,
    and orchestrates the execution of all analysis components. It provides
    error handling and completion notifications.
    """
    logger.info("=== Starting China Stock Analysis Pipeline ===")

    # Set console logging to only show errors to avoid interfering with Rich progress bars
    # This prevents log messages from disrupting the clean progress bar display
    set_console_log_level("ERROR")

    # Option to set log level dynamically for detailed file logging
    # set_log_level("DEBUG")  # Uncomment for verbose logging

    # Display startup message using Rich console for styled output
    console.print("[bold green]🚀 Starting China Stock Analysis Pipeline[/bold green]")

    try:
        # Choose one of these approaches:

        # Option 1: Run sequentially
        # asyncio.run(run_all_scripts())

        # Option 2: Run in parallel (current default)
        asyncio.run(run_all_scripts_parallel())

        # Log and display completion messages
        logger.info("=== China Stock Analysis Pipeline Completed ===")
        # Use Rich console for styled completion message
        console.print(
            "[bold green]✅ All analysis completed! Check logs/ directory for detailed logs.[/bold green]"
        )

    except Exception as e:
        # Log and display error messages using both logger and Rich console
        logger.error("Pipeline failed: %s", str(e))
        # Use Rich console for styled error message
        console.print(f"[bold red]❌ Pipeline failed: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()
