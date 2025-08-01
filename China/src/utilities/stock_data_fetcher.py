"""
Stock data fetching and caching for Chinese equity markets.

This module provides a StockDataFetcher class that encapsulates asynchronous functions
to fetch and cache stock market data and industry-stock mapping data from akshare API.
It implements intelligent caching, retry mechanisms, and comprehensive logging.
"""

import asyncio
import glob
import os
import sys
import logging
import io
import contextlib
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

# Handle imports for both module and standalone execution
try:
    # Import settings first to disable tqdm before akshare import
    from src.settings import configure_environment
    configure_environment()
    from src.utilities.logger import get_logger
    from src.utilities.retry import API_RETRY_CONFIG, retry_call
except ModuleNotFoundError:
    # When running as standalone script, add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.settings import configure_environment
    configure_environment()
    from src.utilities.logger import get_logger
    from src.utilities.retry import API_RETRY_CONFIG, retry_call

import akshare as ak
import pandas as pd
from rich.console import Console
from rich.progress import Progress

if TYPE_CHECKING:
    from rich.progress import TaskID

# Initialize logger for this module
logger = get_logger("stock_data_fetcher")

# Semaphore to control concurrent API requests
REQUEST_SEMAPHORE = asyncio.Semaphore(10)


class StockDataFetcher:
    """
    A class to encapsulate stock data fetching and caching functionality.
    
    This class manages the fetching and caching of stock market data and 
    industry-stock mapping data from akshare API with comprehensive logging
    and retry mechanisms.
    """
    
    # Class constants
    DEFAULT_DATA_DIR = "data/stocks"
    BATCH_SIZE = 10
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        """
        Initialize the StockDataFetcher.
        
        Args:
            data_dir: Directory to store cached data files
        """
        self.data_dir = data_dir
        self.logger = get_logger("stock_data_fetcher")
        self.console = Console()
        
        # Setup dedicated detailed logger
        self._setup_detailed_logger()
        
        # Thread lock for stderr redirection to prevent conflicts
        self._stderr_lock = threading.Lock()
    
    def _setup_detailed_logger(self):
        """Setup a dedicated logger for detailed stock data operations."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create a dedicated logger for stock data
        self.detailed_logger = logging.getLogger("stock_data_detailed")
        self.detailed_logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not self.detailed_logger.handlers:
            # Create file handler with detailed format
            log_file = logs_dir / f"stock_data_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.detailed_logger.addHandler(file_handler)
            
            # Also add console handler for real-time monitoring
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.detailed_logger.addHandler(console_handler)

    class TqdmToLogger:
        """Redirect tqdm output to logger."""
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
            self.buffer = io.StringIO()

        def write(self, buf):
            self.buffer.write(buf)
            # Log each complete line
            if '\n' in buf:
                lines = self.buffer.getvalue().split('\n')
                for line in lines[:-1]:  # All lines except the last incomplete one
                    if line.strip():  # Only log non-empty lines
                        self.logger.log(self.level, f"TQDM: {line.strip()}")
                # Keep the last incomplete line in buffer
                self.buffer = io.StringIO()
                if lines[-1]:
                    self.buffer.write(lines[-1])

        def flush(self):
            # Log any remaining content
            content = self.buffer.getvalue()
            if content.strip():
                self.logger.log(self.level, f"TQDM: {content.strip()}")
            self.buffer = io.StringIO()

    @contextlib.contextmanager
    def _capture_tqdm_to_logger(self, logger):
        """Context manager to capture tqdm output and send it to logger."""
        # Use thread lock to prevent concurrent stderr redirection
        with self._stderr_lock:
            # Store original stderr
            original_stderr = sys.stderr
            
            try:
                # Create our custom writer
                tqdm_logger = self.TqdmToLogger(logger)
                # Redirect stderr to our logger
                sys.stderr = tqdm_logger
                logger.debug("Started stderr redirection for thread %s", threading.current_thread().name)
                yield
            finally:
                # Flush any remaining content
                if hasattr(sys.stderr, 'flush'):
                    sys.stderr.flush()
                # Restore original stderr
                sys.stderr = original_stderr
                logger.debug("Restored stderr redirection for thread %s", threading.current_thread().name)

    async def get_stock_market_data(
        self, progress: Optional["Progress"] = None
    ) -> pd.DataFrame:
        """
        Fetch stock market data with caching and progress tracking.

        Args:
            progress: Optional Rich Progress instance for progress tracking

        Returns:
            DataFrame containing stock market data with columns including
            stock codes, names, prices, and financial metrics
        """
        self.detailed_logger.info("=== Starting get_stock_market_data() ===")
        self.detailed_logger.info("Data directory: %s", self.data_dir)
        
        today = datetime.now().strftime("%Y%m%d")
        file_path = f"{self.data_dir}/stock_zh_a_spot_em_df-{today}.csv"
        self.detailed_logger.info("Target cache file: %s", file_path)

        if os.path.exists(file_path):
            self.detailed_logger.info("Cache file exists, loading cached stock market data")
            self.logger.info("Loading cached stock market data from %s", file_path)
            df = pd.read_csv(file_path, dtype={"代码": str})
            self.detailed_logger.info("Loaded cached data: %d rows, %d columns", len(df), len(df.columns))
            return df

        # Delete outdated files
        self.detailed_logger.info("Checking for outdated cache files to remove")
        outdated_files = glob.glob(f"{self.data_dir}/stock_zh_a_spot_em_df-*.csv")
        self.detailed_logger.info("Found %d outdated files to remove: %s", len(outdated_files), outdated_files)
        self.logger.info("Removing outdated stock market data files")
        for f in outdated_files:
            self.detailed_logger.debug("Removing file: %s", f)
            os.remove(f)

        # Fetch and save new data with retry mechanism and progress tracking
        use_own_progress = progress is None
        if use_own_progress:
            progress = Progress(console=self.console)
            progress.start()

        task = progress.add_task(
            "[cyan]Fetching stock market data from akshare API...",
            total=None,  # Indeterminate progress for single API call
        )

        self.detailed_logger.info("Starting fresh API call to akshare stock_zh_a_spot_em")
        self.logger.info("Fetching new stock market data from akshare API")
        try:
            # Wrap the akshare call to capture tqdm output
            def fetch_with_logging():
                self.detailed_logger.info("Executing akshare API call with retry mechanism")
                with self._capture_tqdm_to_logger(self.detailed_logger):
                    return retry_call(ak.stock_zh_a_spot_em, timeout=None)
            
            self.detailed_logger.info("Starting threaded execution of akshare API call")
            stock_df = await asyncio.to_thread(fetch_with_logging)
            self.detailed_logger.info("API call completed successfully, received DataFrame with %d rows", len(stock_df))

            # Update progress to show completion
            progress.update(
                task,
                completed=1,
                total=1,
                description="    [green]✓ Stock market data fetched successfully",
            )
            await asyncio.sleep(0.5)  # Brief pause to show completion

            # Save the data
            self.detailed_logger.info("Ensuring data directory exists: %s", self.data_dir)
            os.makedirs(self.data_dir, exist_ok=True)
            self.detailed_logger.info("Saving stock market data to: %s", file_path)
            stock_df.to_csv(file_path, index=False)
            self.detailed_logger.info("Successfully saved %d rows to cache file", len(stock_df))
            self.logger.info("Successfully saved stock market data to %s", file_path)
            self.detailed_logger.info("=== get_stock_market_data() completed successfully ===")
            return stock_df
        except Exception as e:
            self.detailed_logger.error("Exception occurred in get_stock_market_data(): %s", str(e))
            self.detailed_logger.error("Exception type: %s", type(e).__name__)
            progress.update(task, description="[red]✗ Failed to fetch stock market data")
            self.logger.error("Failed to fetch stock market data: %s", str(e))
            raise
        finally:
            if use_own_progress:
                progress.stop()

    async def _fetch_industry_stocks(self, industry_name: str) -> List[tuple]:
        """
        Fetch stocks for a specific industry using akshare API.

        Args:
            industry_name: Name of the industry to fetch stocks for

        Returns:
            List of (industry_name, stock_code) tuples
        """
        async with REQUEST_SEMAPHORE:
            try:
                self.detailed_logger.debug("Fetching stocks for industry: %s", industry_name)
                
                def fetch_industry_stocks():
                    self.detailed_logger.debug("Executing industry stocks API call for: %s", industry_name)
                    with self._capture_tqdm_to_logger(self.detailed_logger):
                        return API_RETRY_CONFIG.retry(ak.stock_board_industry_cons_em)(symbol=industry_name)
                
                industry_stocks = await asyncio.to_thread(fetch_industry_stocks)
                self.detailed_logger.debug("Industry %s returned %d stocks", industry_name, len(industry_stocks))
                
                result = [(industry_name, code) for code in industry_stocks["代码"]]
                self.detailed_logger.debug("Created %d mappings for industry %s", len(result), industry_name)
                return result
            except Exception as e:
                self.detailed_logger.error("Exception fetching industry %s: %s (%s)", industry_name, str(e), type(e).__name__)
                self.logger.error(
                    "Error fetching data for industry %s: %s", industry_name, str(e)
                )
                return []

    async def get_industry_stock_mapping_data(
        self, progress: Optional["Progress"] = None
    ) -> pd.DataFrame:
        """
        Fetch industry-stock mapping data with caching and optimized concurrent processing.

        Args:
            progress: Optional Rich Progress instance for progress tracking

        Returns:
            DataFrame containing industry-stock mapping with columns for
            industry names and corresponding stock codes
        """
        self.detailed_logger.info("=== Starting get_industry_stock_mapping_data() ===")
        self.detailed_logger.info("Data directory: %s", self.data_dir)
        
        today = datetime.now().strftime("%Y%m%d")
        file_path = f"{self.data_dir}/industry_stock_mapping_df-{today}.csv"
        self.detailed_logger.info("Target cache file: %s", file_path)

        if os.path.exists(file_path):
            self.detailed_logger.info("Cache file exists, loading cached industry mapping data")
            self.logger.info("Loading cached industry mapping data from %s", file_path)
            df = pd.read_csv(file_path, dtype={"代码": str})
            self.detailed_logger.info("Loaded cached data: %d rows, %d columns", len(df), len(df.columns))
            return df

        # Delete outdated files
        self.logger.info("Removing outdated industry mapping data files")
        for f in glob.glob(f"{self.data_dir}/industry_stock_mapping_df-*.csv"):
            os.remove(f)

        # Fetch industry names with retry mechanism
        self.detailed_logger.info("Starting industry names API call")
        self.logger.info("Fetching industry names from akshare API")
        try:
            def fetch_industry_names():
                self.detailed_logger.info("Executing akshare industry names API call")
                with self._capture_tqdm_to_logger(self.detailed_logger):
                    return API_RETRY_CONFIG.retry(ak.stock_board_industry_name_em)()
            
            self.detailed_logger.info("Starting threaded execution of industry names API call")
            industry_data = await asyncio.to_thread(fetch_industry_names)
            self.detailed_logger.info("Industry names API call completed, received %d industries", len(industry_data))
            
            industry_names = industry_data["板块名称"]
            self.detailed_logger.info("Extracted industry names: %s", industry_names.tolist()[:10] if len(industry_names) > 10 else industry_names.tolist())
            self.logger.info("Found %d industries to process", len(industry_names))
        except Exception as e:
            self.logger.error("Failed to fetch industry names: %s", str(e))
            raise

        # Process industries concurrently with batching and progress tracking
        batch_size = self.BATCH_SIZE
        all_mappings = []

        use_own_progress = progress is None
        if use_own_progress:
            progress = Progress(console=self.console)
            progress.start()

        # Create main progress task
        main_task = progress.add_task(
            "[magenta]Processing industries for stock mapping...",
            total=len(industry_names)
        )

        try:
            # Process industries in batches to control concurrency
            for i in range(0, len(industry_names), batch_size):
                batch = industry_names[i:i + batch_size]
                self.logger.info("Processing batch %d/%d with %d industries",
                            i // batch_size + 1,
                            (len(industry_names) + batch_size - 1) // batch_size,
                            len(batch))

                # Process batch concurrently
                batch_results = await asyncio.gather(
                    *[self._fetch_industry_stocks(name) for name in batch],
                    return_exceptions=True
                )

                # Collect results and handle exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error("Batch processing error: %s", str(result))
                    else:
                        all_mappings.extend(result)

                # Update progress
                progress.update(main_task, advance=len(batch))

            # Convert to DataFrame
            if all_mappings:
                mapping_df = pd.DataFrame(all_mappings, columns=["行业", "代码"])
                self.logger.info("Created mapping DataFrame with %d entries", len(mapping_df))
            else:
                self.logger.warning("No industry mappings found, creating empty DataFrame")
                mapping_df = pd.DataFrame(columns=["行业", "代码"])

            # Save to cache
            os.makedirs(self.data_dir, exist_ok=True)
            mapping_df.to_csv(file_path, index=False)
            self.logger.info("Successfully saved industry mapping data to %s", file_path)

            return mapping_df

        finally:
            if use_own_progress:
                progress.stop()

    async def run_analysis(
        self,
        progress: Optional["Progress"] = None,
        parent_task_id: Optional["TaskID"] = None,
        batch_task_id: Optional["TaskID"] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete data fetching analysis.
        
        Args:
            progress: Optional Rich Progress instance for progress tracking
            parent_task_id: Optional parent task ID for progress updates
            batch_task_id: Optional batch task ID for progress updates
            
        Returns:
            Tuple of (industry_stock_mapping_df, stock_zh_a_spot_em_df)
        """
        self.logger.info("Starting complete stock data fetching analysis")
        
        # Use single Progress context for both operations
        use_own_progress = progress is None
        if use_own_progress:
            progress = Progress(console=self.console)
            progress.start()

        try:
            # Fetch both datasets sequentially to avoid threading conflicts
            self.logger.info("Fetching industry mapping data...")
            industry_df = await self.get_industry_stock_mapping_data(progress=progress)
            
            self.logger.info("Fetching stock market data...")
            stock_df = await self.get_stock_market_data(progress=progress)

            # Display basic statistics
            self.logger.info("Data fetching completed successfully!")
            self.logger.info(
                "Industry mapping data: %d rows, %d columns",
                len(industry_df),
                len(industry_df.columns),
            )
            self.logger.info(
                "Stock market data: %d rows, %d columns",
                len(stock_df),
                len(stock_df.columns),
            )

            return industry_df, stock_df

        except Exception as e:
            self.logger.error("Data fetching failed: %s", str(e))
            raise
        finally:
            if use_own_progress:
                progress.stop()


# Maintain backward compatibility by providing the original function interfaces
async def get_stock_market_data(
    data_dir: str = "data/stocks", progress: Optional["Progress"] = None
) -> pd.DataFrame:
    """
    Legacy function interface for backward compatibility.
    
    Args:
        data_dir: Directory to store cached data files
        progress: Optional Rich Progress instance for progress tracking
        
    Returns:
        DataFrame containing stock market data
    """
    fetcher = StockDataFetcher(data_dir)
    return await fetcher.get_stock_market_data(progress)


async def get_industry_stock_mapping_data(
    data_dir: str = "data/stocks", progress: Optional["Progress"] = None
) -> pd.DataFrame:
    """
    Legacy function interface for backward compatibility.
    
    Args:
        data_dir: Directory to store cached data files
        progress: Optional Rich Progress instance for progress tracking
        
    Returns:
        DataFrame containing industry-stock mapping data
    """
    fetcher = StockDataFetcher(data_dir)
    return await fetcher.get_industry_stock_mapping_data(progress)


async def main():
    """
    Main function to test data fetching functionality.

    This function fetches both industry mapping and stock market data
    using the new class-based approach.

    Returns:
        Tuple of (industry_stock_mapping_df, stock_zh_a_spot_em_df)
    """
    logger.info("Starting data fetching test using class-based approach...")
    
    # Create fetcher instance
    fetcher = StockDataFetcher()
    
    # Run complete analysis
    return await fetcher.run_analysis()


if __name__ == "__main__":
    """
    When running this script directly, execute the main function.
    
    Usage:
        uv run python src/utilities/stock_data_fetcher.py
    
    Note: Use 'uv run' to ensure all dependencies are available.
    """
    asyncio.run(main())