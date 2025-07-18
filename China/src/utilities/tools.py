import asyncio
import functools
import time
from typing import Callable, Any, Dict, Optional
from datetime import datetime
from .logger import get_logger


def timer(func: Callable) -> Callable:
    """
    Timer decorator that works with both sync and async functions.
    Logs execution time and function details.
    """
    logger = get_logger("timer")
    
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.info(f"Starting async function: {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                logger.info(f"⏱️  Function '{func.__name__}' runtime: {exec_time:.4f} s")

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.info(f"Starting sync function: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                logger.info(f"⏱️  Function '{func.__name__}' runtime: {exec_time:.4f} s")

        return sync_wrapper


def logged(func: Callable) -> Callable:
    """
    Logging decorator that tracks function entry, exit, and arguments.
    """
    logger = get_logger("function_tracker")
    
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Log function entry with arguments
            args_str = ', '.join([str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={str(v)[:30] + '...' if len(str(v)) > 30 else str(v)}" for k, v in kwargs.items()])
            logger.debug(f"→ Entering {func.__name__}({args_str}{', ' + kwargs_str if kwargs_str else ''})")
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"← Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"✗ Exception in {func.__name__}: {str(e)}")
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Log function entry with arguments
            args_str = ', '.join([str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg) for arg in args])
            kwargs_str = ', '.join([f"{k}={str(v)[:30] + '...' if len(str(v)) > 30 else str(v)}" for k, v in kwargs.items()])
            logger.debug(f"→ Entering {func.__name__}({args_str}{', ' + kwargs_str if kwargs_str else ''})")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"← Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"✗ Exception in {func.__name__}: {str(e)}")
                raise

        return sync_wrapper


def timed_and_logged(func: Callable) -> Callable:
    """
    Combined decorator that provides both timing and detailed logging.
    """
    return timer(logged(func))


# Global verbose tracking state
class VerboseTracker:
    """Global tracker for verbose progress monitoring"""
    
    def __init__(self):
        self.processes: Dict[str, Dict] = {}
        self.functions: Dict[str, Dict] = {}
        self.enabled = False
        self.logger = get_logger("verbose_tracker")
        
    def enable(self):
        """Enable verbose tracking"""
        self.enabled = True
        self.logger.info("🔍 Verbose tracking enabled")
        
    def disable(self):
        """Disable verbose tracking"""
        self.enabled = False
        self.logger.info("🔍 Verbose tracking disabled")
        
    def start_process(self, process_name: str, description: str = "", total_steps: int = 0):
        """Start tracking a process"""
        if not self.enabled:
            return
            
        self.processes[process_name] = {
            'description': description,
            'start_time': time.perf_counter(),
            'total_steps': total_steps,
            'current_step': 0,
            'steps': [],
            'status': 'running'
        }
        
        self.logger.info(f"🚀 Started process: {process_name} - {description}")
        if total_steps > 0:
            print(f"🚀 Starting {process_name}: {description} ({total_steps} steps)")
        else:
            print(f"🚀 Starting {process_name}: {description}")
            
    def update_process(self, process_name: str, step_description: str, step_number: Optional[int] = None):
        """Update process progress"""
        if not self.enabled or process_name not in self.processes:
            return
            
        process = self.processes[process_name]
        if step_number is not None:
            process['current_step'] = step_number
        else:
            process['current_step'] += 1
            
        process['steps'].append({
            'step': process['current_step'],
            'description': step_description,
            'time': time.perf_counter() - process['start_time']
        })
        
        if process['total_steps'] > 0:
            progress = (process['current_step'] / process['total_steps']) * 100
            print(f"   📊 [{process['current_step']}/{process['total_steps']}] {progress:.1f}% - {step_description}")
        else:
            print(f"   📊 Step {process['current_step']}: {step_description}")
            
        self.logger.info(f"📊 {process_name} progress: {step_description}")
        
    def complete_process(self, process_name: str, success: bool = True):
        """Complete a process"""
        if not self.enabled or process_name not in self.processes:
            return
            
        process = self.processes[process_name]
        process['status'] = 'completed' if success else 'failed'
        process['end_time'] = time.perf_counter()
        process['duration'] = process['end_time'] - process['start_time']
        
        status_icon = "✅" if success else "❌"
        self.logger.info(f"{status_icon} Process {process_name} {'completed' if success else 'failed'} in {process['duration']:.2f}s")
        print(f"{status_icon} {process_name} {'completed' if success else 'failed'} in {process['duration']:.2f}s")
        
    def track_function(self, func_name: str, execution_time: float, args_info: str = ""):
        """Track individual function execution"""
        if not self.enabled:
            return
            
        if func_name not in self.functions:
            self.functions[func_name] = {
                'calls': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'last_call': None
            }
            
        func_stats = self.functions[func_name]
        func_stats['calls'] += 1
        func_stats['total_time'] += execution_time
        func_stats['min_time'] = min(func_stats['min_time'], execution_time)
        func_stats['max_time'] = max(func_stats['max_time'], execution_time)
        func_stats['last_call'] = datetime.now()
        func_stats['avg_time'] = func_stats['total_time'] / func_stats['calls']
        
        # Log detailed function info for slow functions
        if execution_time > 1.0:  # Functions taking more than 1 second
            self.logger.warning(f"🐌 Slow function: {func_name} took {execution_time:.2f}s {args_info}")
            print(f"🐌 Slow function: {func_name} took {execution_time:.2f}s")
            
    def get_summary(self) -> Dict:
        """Get tracking summary"""
        return {
            'processes': self.processes,
            'functions': self.functions,
            'enabled': self.enabled
        }
        
    def print_summary(self):
        """Print comprehensive summary"""
        if not self.enabled:
            print("🔍 Verbose tracking is disabled")
            return
            
        print("\n" + "="*60)
        print("📊 VERBOSE TRACKING SUMMARY")
        print("="*60)
        
        # Process summary
        if self.processes:
            print("\n🚀 PROCESS SUMMARY:")
            for name, process in self.processes.items():
                status_icon = "✅" if process['status'] == 'completed' else "❌" if process['status'] == 'failed' else "🔄"
                duration = process.get('duration', time.perf_counter() - process['start_time'])
                print(f"   {status_icon} {name}: {process['description']} ({duration:.2f}s)")
                
        # Function summary
        if self.functions:
            print("\n⚡ FUNCTION PERFORMANCE:")
            # Sort by total time descending
            sorted_funcs = sorted(self.functions.items(), key=lambda x: x[1]['total_time'], reverse=True)
            for name, stats in sorted_funcs[:10]:  # Top 10 most time-consuming
                avg_time = stats['avg_time']
                print(f"   📈 {name}: {stats['calls']} calls, avg {avg_time:.3f}s, total {stats['total_time']:.2f}s")
                
        print("="*60)


# Global verbose tracker instance
verbose_tracker = VerboseTracker()


def verbose(func: Callable) -> Callable:
    """
    Verbose decorator that provides detailed tracking for time-consuming functions.
    Combines timing, logging, and progress tracking.
    """
    logger = get_logger("verbose")
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            
            # Create args summary for tracking
            args_summary = f"({len(args)} args)" if args else ""
            if kwargs:
                args_summary += f" {list(kwargs.keys())}"
                
            # Log function start
            logger.info(f"🔍 Starting {func.__name__}{args_summary}")
            if verbose_tracker.enabled:
                print(f"🔍 Starting {func.__name__}{args_summary}")
                
            try:
                result = await func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.perf_counter() - start_time
                
                # Track function performance
                verbose_tracker.track_function(func.__name__, execution_time, args_summary)
                
                # Log completion
                logger.info(f"✅ Completed {func.__name__} in {execution_time:.2f}s")
                if verbose_tracker.enabled:
                    print(f"✅ Completed {func.__name__} in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(f"❌ Failed {func.__name__} after {execution_time:.2f}s: {str(e)}")
                if verbose_tracker.enabled:
                    print(f"❌ Failed {func.__name__} after {execution_time:.2f}s: {str(e)}")
                raise
                
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            
            # Create args summary for tracking
            args_summary = f"({len(args)} args)" if args else ""
            if kwargs:
                args_summary += f" {list(kwargs.keys())}"
                
            # Log function start
            logger.info(f"🔍 Starting {func.__name__}{args_summary}")
            if verbose_tracker.enabled:
                print(f"🔍 Starting {func.__name__}{args_summary}")
                
            try:
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.perf_counter() - start_time
                
                # Track function performance
                verbose_tracker.track_function(func.__name__, execution_time, args_summary)
                
                # Log completion
                logger.info(f"✅ Completed {func.__name__} in {execution_time:.2f}s")
                if verbose_tracker.enabled:
                    print(f"✅ Completed {func.__name__} in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(f"❌ Failed {func.__name__} after {execution_time:.2f}s: {str(e)}")
                if verbose_tracker.enabled:
                    print(f"❌ Failed {func.__name__} after {execution_time:.2f}s: {str(e)}")
                raise
                
        return sync_wrapper


def enable_verbose_tracking():
    """Enable verbose tracking globally"""
    verbose_tracker.enable()


def disable_verbose_tracking():
    """Disable verbose tracking globally"""
    verbose_tracker.disable()


def print_verbose_summary():
    """Print verbose tracking summary"""
    verbose_tracker.print_summary()
