"""
Main entry point for the China stock analysis pipeline.

This module orchestrates the execution of all analysis components including
stock filtering, stock analysis, and industry analysis. It provides both
sequential and parallel execution modes with beautiful progress bars.
"""

# Configure environment settings before importing any other modules
# import src.settings  # This configures the environment

import asyncio
import glob
import os
import re
import shutil
from typing import Optional

from rich.console import Console  # Rich console for styled output and progress display

# Rich Progress imports for detailed hierarchical progress tracking
from rich.progress import (
    BarColumn,  # Visual progress bars with customizable width and styling
    Progress,  # Main progress container that manages multiple progress bars
    SpinnerColumn,  # Rotating spinner animation for active tasks
    TextColumn,  # Dynamic text descriptions that update during execution
    TimeElapsedColumn,  # Shows elapsed time since task started
)

from src.industry_filter import IndustryFilter
from src.holding_stock_analyzer import HoldingStockAnalyzer
from src.stock_filter import StockFilter
from src.utilities.get_stock_data import get_stock_market_data, get_industry_stock_mapping_data
from src.utilities.logger import get_logger, set_console_log_level

# Initialize logger and Rich console for styled output
# The logger provides file and console logging for the main pipeline
logger = get_logger("main")
# Rich console provides styled terminal output and progress bar rendering
console = Console()


class StockAnalysisPipeline:
    """
    A class to orchestrate the complete China stock analysis pipeline.
    
    This class manages data fetching, analysis execution, and report generation
    for all components of the stock analysis system including stock filtering,
    industry analysis, and holding stock analysis.
    """
    
    def __init__(self, data_dir: str = "data/stocks"):
        """
        Initialize the StockAnalysisPipeline.
        
        Args:
            data_dir: Directory for storing cached data files
        """
        self.data_dir = data_dir
        self.industry_stock_mapping_df = None
        self.stock_zh_a_spot_em_df = None
        self.logger = get_logger("pipeline")
    
    async def fetch_market_data(self, progress: Optional[Progress] = None, task_id: Optional[int] = None) -> None:
        """
        Fetch and cache market data required for all analysis modules.
        
        This method must be run before any analysis modules as it provides
        the core data that all other modules depend on.
        
        Args:
            progress: Optional Rich Progress instance for progress tracking
            task_id: Optional task ID for progress updates
        """
        self.logger.info("Starting market data fetching")
        
        if progress and task_id:
            progress.update(task_id, description="📊 Fetching market data...", completed=0, total=100)
        
        # Create a local progress context if none provided
        if progress:
            # Use provided progress instance
            shared_progress = progress
        else:
            # Create temporary progress for standalone use
            from rich.progress import Progress
            shared_progress = Progress(console=console)
            shared_progress.start()
        
        try:
            # Update progress at start
            if progress and task_id:
                progress.update(task_id, completed=10, description="📊 Loading market data...")
            
            # Fetch both datasets in parallel for better performance
            self.stock_zh_a_spot_em_df, self.industry_stock_mapping_df = await asyncio.gather(
                get_stock_market_data(self.data_dir, progress=shared_progress),
                get_industry_stock_mapping_data(self.data_dir, progress=shared_progress)
            )
            
            # Update progress after data is loaded
            if progress and task_id:
                progress.update(task_id, completed=90, description="📊 Processing market data...")
            
            self.logger.info("Stock market data fetched: %d stocks", len(self.stock_zh_a_spot_em_df))
            self.logger.info("Industry mapping data fetched: %d mappings", len(self.industry_stock_mapping_df))
        finally:
            # Clean up temporary progress if we created it
            if not progress:
                shared_progress.stop()
        
        if progress and task_id:
            progress.update(task_id, completed=100, description="✅ Market data ready")
        
        self.logger.info("Market data fetching completed successfully")
    
    async def run_stock_filter(self, progress: Optional[Progress] = None, **kwargs) -> None:
        """Run stock filter analysis with the fetched data."""
        stock_filter = StockFilter(
            self.industry_stock_mapping_df,
            self.stock_zh_a_spot_em_df
        )
        await stock_filter.run_analysis(progress=progress, **kwargs)
    
    async def run_industry_filter(self, progress: Optional[Progress] = None, **kwargs) -> None:
        """Run industry filter analysis."""
        industry_filter = IndustryFilter()
        await industry_filter.run_analysis(progress=progress, **kwargs)
    
    async def run_holding_stock_analyzer(self, holding_stocks_data: dict = None, progress: Optional[Progress] = None, **kwargs) -> None:
        """Run holding stock analysis with the fetched data."""
        analyzer = HoldingStockAnalyzer(
            self.industry_stock_mapping_df,
            self.stock_zh_a_spot_em_df
        )
        
        if holding_stocks_data:
            # Use provided data
            await analyzer.run_analysis(
                holding_stocks_data,
                progress=progress,
                **kwargs
            )
        else:
            # Use file-based approach (load JSON files)
            await analyzer.run_analysis_from_files(progress=progress, **kwargs)


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
            "pattern": "data/stocks/reports/股票筛选报告-[0-9]*.csv",  # Filtered stock reports
            "description": "股票筛选报告",
        },
        {
            "pattern": "data/stocks/reports/行业筛选报告-[0-9]*.csv",  # Filtered industry reports
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
    Run all analysis scripts sequentially with progress tracking using the pipeline class.

    This function executes data fetching, stock filtering, holding stock analysis, 
    and industry analysis in sequence, providing detailed progress feedback.
    """
    logger.info("Starting sequential execution of all scripts")
    
    # Initialize the pipeline
    pipeline = StockAnalysisPipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sequential Stock Analysis Pipeline", total=5)

        # Step 1: Fetch market data (must be first)
        progress.update(task, description="Fetching market data")
        await pipeline.fetch_market_data(progress, task)
        progress.advance(task)

        # Step 2: Run stock_filter
        progress.update(task, description="Running stock filter")
        await pipeline.run_stock_filter()
        progress.advance(task)

        # Step 3: Run holding stock analyzer
        progress.update(task, description="Running holding stock analyzer")
        await pipeline.run_holding_stock_analyzer()
        progress.advance(task)

        # Step 4: Run industry_filter
        progress.update(task, description="Running industry filter")
        await pipeline.run_industry_filter()
        progress.advance(task)

        # Step 5: Copy latest reports to data/today
        progress.update(task, description="Copying latest reports")
        await copy_latest_reports()
        progress.advance(task)

        progress.update(task, description="Sequential analysis completed!")

    logger.info("All scripts completed!")


# @timer
async def run_all_scripts_parallel() -> None:
    """
    Run all analysis scripts in parallel with detailed hierarchical progress tracking using the pipeline class.

    This function first fetches market data, then executes stock filtering, holding stock analysis, 
    and industry analysis concurrently for better performance. It uses Rich Progress to provide 
    real-time progress feedback with a hierarchical structure:
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
    logger.info("Starting parallel execution of all scripts with data fetching")
    
    # Initialize the pipeline
    pipeline = StockAnalysisPipeline()

    # Create Rich Progress instance with detailed column configuration
    # Each column serves a specific purpose in the progress display
    with Progress(
        SpinnerColumn(),  # Rotating spinner indicates active tasks
        TextColumn(
            "[progress.description]{task.description}"
        ),  # Dynamic task descriptions
        BarColumn(
            bar_width=40
        ),  # Progress bars with fixed width for consistent display
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%"
        ),  # Completion percentage
        TimeElapsedColumn(),  # Time elapsed since task started
        console=console,  # Use our configured Rich console instance
        transient=False,  # Keep completed progress bars visible briefly before cleanup
    ) as progress:
        # Create main pipeline progress tracker
        # This tracks the overall progress of the parallel execution pipeline
        main_task = progress.add_task(
            "🚀 Parallel Stock Analysis Pipeline",
            total=3,  # 3 main stages: data fetching + parallel scripts + report copying
        )

        # Stage 1: Fetch market data (must be first, before any analysis)
        logger.info("Fetching market data before parallel script execution")
        data_fetch_task = progress.add_task("📊 Fetching Market Data", total=100, visible=True)
        
        await pipeline.fetch_market_data(progress, data_fetch_task)
        
        # Mark data fetching as complete but keep visible
        progress.update(data_fetch_task, completed=100, description="✅ Market Data Ready")
        progress.advance(main_task)
        
        # Stage 2: Execute all analysis scripts in parallel
        # Create individual progress bars for each script to show detailed progress
        logger.info(
            "Creating individual progress trackers for parallel script execution"
        )

        # Each script gets its own progress bar that will show detailed sub-progress
        stock_filter_task = progress.add_task(
            "📈 Stock Filter", total=100, visible=True
        )
        # Pre-create batch processing task for Stock Filter (initially hidden)
        stock_filter_batch_task = progress.add_task(
            "    📊 Stock Filter batches", total=100, visible=False
        )

        holding_stock_task = progress.add_task(
            "💼 Holding Stock Analyzer", total=100, visible=True
        )
        # Pre-create batch processing task for Holding Stock Analyzer (initially hidden)
        holding_stock_batch_task = progress.add_task(
            "    💼 Holding Stock processing", total=100, visible=False
        )

        industry_filter_task = progress.add_task(
            "🏭 Industry Filter", total=100, visible=True
        )
        # Pre-create batch processing task for Industry Filter (initially hidden)
        industry_filter_batch_task = progress.add_task(
            "    🏭 Industry Filter batches", total=100, visible=False
        )

        # Enhanced progress wrapper that provides detailed progress updates
        # This function wraps each main script and provides progress feedback
        # It implements the tqdm-like behavior requested by the user:
        # - Detailed low-level task progress bars for batch processing
        # - Hierarchical progress structure with subtasks under main tasks
        # - Subtask progress bars that disappear when finished
        # - Top-level tasks remain visible (don't disappear)
        async def run_with_detailed_progress(
            script_func, task_id, name, progress_instance, batch_task_id=None
        ):
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
                await script_func(
                    progress=progress_instance,
                    parent_task_id=task_id,
                    batch_task_id=batch_task_id,
                )
                # Mark top-level task as completed but DON'T remove it
                # According to TODO.md: "Don't make the progress bar disappear for the top level tasks"
                progress_instance.update(
                    task_id, completed=100, description=f"✅ {name} completed"
                )

                logger.info(f"✅ {name} completed (top-level task kept visible)")

            except Exception as e:
                # Handle errors and update progress accordingly
                progress_instance.update(
                    task_id, description=f"❌ {name} failed: {str(e)}"
                )
                logger.error(f"❌ Error in {name}: {str(e)}")
                raise

        # Execute all scripts in parallel with hierarchical progress tracking
        logger.info(
            "🚀 Launching parallel script execution with hierarchical progress tracking"
        )
        await asyncio.gather(
            run_with_detailed_progress(
                pipeline.run_stock_filter,
                stock_filter_task,
                "Stock Filter",
                progress,
                stock_filter_batch_task,
            ),
            run_with_detailed_progress(
                pipeline.run_holding_stock_analyzer,
                holding_stock_task,
                "Holding Stock Analyzer",
                progress,
                holding_stock_batch_task,
            ),
            run_with_detailed_progress(
                pipeline.run_industry_filter,
                industry_filter_task,
                "Industry Filter",
                progress,
                industry_filter_batch_task,
            ),
        )

        # Update main progress after parallel execution
        progress.advance(main_task)
        progress.update(
            main_task,
            description="📋 All analysis scripts completed, copying reports...",
        )

        # Stage 2: Copy latest reports with progress tracking
        logger.info("Starting report copying with progress feedback")
        copy_task = progress.add_task("📋 Copying latest reports", total=100)

        # Simulate progress for report copying
        copy_progress_task = asyncio.create_task(
            simulate_copy_progress(progress, copy_task)
        )

        # Execute actual report copying
        await copy_latest_reports()

        # Complete copy progress - this is a subtask so it can disappear
        copy_progress_task.cancel()
        progress.update(
            copy_task, completed=100, description="✅ Report copying completed"
        )
        await asyncio.sleep(0.5)
        progress.remove_task(copy_task)  # Subtask disappears when finished

        # Complete main pipeline - top-level task stays visible
        progress.advance(main_task)
        progress.update(
            main_task,
            description="🎉 Parallel analysis pipeline completed successfully!",
        )
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
            "Verifying copied files...",
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
