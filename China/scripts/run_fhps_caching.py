#!/usr/bin/env python3
"""
Runner script for FHPS data caching operations.

This script runs the FHPS caching functionality independently with progress tracking.
It handles both Phase 1 (historical data) and Phase 2 (target year with prices) caching.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.caching.fhps_caching import FhpsCaching

console = Console()


async def run_fhps_caching(config_name: str = "config"):
    """
    Run FHPS caching operations with progress tracking.
    
    Args:
        config_name: Configuration file name to use (without .yml extension)
    """
    console.print("[bold green]🚀 Starting FHPS Data Caching[/bold green]")
    
    try:
        # Initialize caching system
        console.print(f"📋 Loading configuration: {config_name}")
        caching = FhpsCaching(config_name)
        
        # Display configuration info
        console.print(f"📊 Historical years: {caching.config.phase1_caching.historical_years}")
        console.print(f"🎯 Target years: {caching.config.phase2_caching.target_years}")
        
        # Create progress context
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            
            # Main caching task
            main_task = progress.add_task(
                "🔄 Running FHPS caching operations...", 
                total=100
            )
            
            try:
                # Update progress for Phase 1
                progress.update(main_task, completed=10, description="📊 Starting Phase 1 caching (historical data)...")
                
                # Run Phase 1 caching
                await caching.phase1_caching()
                progress.update(main_task, completed=50, description="✅ Phase 1 completed")
                
                # Update progress for Phase 2
                progress.update(main_task, completed=60, description="🎯 Starting Phase 2 caching (target year with prices)...")
                
                # Run Phase 2 caching
                await caching.phase2_caching()
                progress.update(main_task, completed=90, description="✅ Phase 2 completed")
                
                # Final completion
                progress.update(main_task, completed=100, description="🎉 All FHPS caching operations completed successfully!")
                
            except Exception as e:
                progress.update(main_task, description=f"❌ Caching failed: {str(e)}")
                raise
        
        console.print("[bold green]✅ FHPS caching completed successfully![/bold green]")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]❌ Configuration file not found: {e}[/bold red]")
        console.print("[yellow]Available configs in input/config/caching/fhps/:[/yellow]")
        config_dir = Path("input/config/caching/fhps")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yml"):
                console.print(f"  - {config_file.stem}")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"[bold red]❌ FHPS caching failed: {e}[/bold red]")
        sys.exit(1)


def main():
    """Main entry point for the FHPS caching runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run FHPS data caching operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_fhps_caching.py                    # Use default config
  python scripts/run_fhps_caching.py --config test      # Use test config
  python scripts/run_fhps_caching.py -c my_config       # Use custom config
        """
    )
    
    parser.add_argument(
        "-c", "--config", 
        default="config",
        help="Configuration file name (without .yml extension, default: config)"
    )
    
    args = parser.parse_args()
    
    # Run the caching process
    asyncio.run(run_fhps_caching(args.config))


if __name__ == "__main__":
    main()