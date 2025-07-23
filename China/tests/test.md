# Testing Documentation for Stock Analysis Logging System

## Overview

This document provides comprehensive testing information for the logging functionality integrated into the China stock analysis project. The test suite uses `pytest` to ensure the reliability and correctness of all logging components.

## Test Structure

```sh
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Shared fixtures and configuration
├── test.md                        # This documentation file
├── unit/                          # Unit tests
│   ├── test_logger.py             # Logger functionality tests
│   └── test_decorators.py         # Decorator functionality tests
├── integration/                   # Integration tests
│   └── test_logging_system.py    # End-to-end logging tests
├── fixtures/                      # Test data and fixtures
└── coverage_html/                # HTML coverage reports (generated)
```

## Quick Start

### Running All Tests

```bash
# Run all tests with coverage
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test categories
uv run pytest -m unit          # Only unit tests
uv run pytest -m integration   # Only integration tests
uv run pytest -m "not slow"    # Exclude slow tests
```

### Installing Test Dependencies

```bash
# Dependencies are automatically installed with:
uv sync
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

#### Logger Tests (`test_logger.py`)

**Purpose**: Test individual logger components in isolation.

**Key Test Classes**:

- `TestSetupLogger`: Tests logger initialization and configuration
- `TestGetLogger`: Tests logger retrieval and module-specific loggers
- `TestSetLogLevel`: Tests dynamic log level changes
- `TestLoggerIntegration`: Basic integration between logger components

**Example Test Cases**:

```python
def test_setup_logger_default_params()
def test_get_logger_with_name()
def test_set_log_level_valid_levels()
def test_logger_writes_to_file()
```

#### Decorator Tests (`test_decorators.py`)

**Purpose**: Test logging decorators (@timer, @logged, @timed_and_logged).

**Key Test Classes**:

- `TestTimerDecorator`: Tests function timing functionality
- `TestLoggedDecorator`: Tests function entry/exit logging
- `TestTimedAndLoggedDecorator`: Tests combined decorator
- `TestDecoratorIntegration`: Tests decorator combinations

**Example Test Cases**:

```python
def test_timer_sync_function()
async def test_timer_async_function()
def test_logged_function_with_arguments()
def test_timed_and_logged_combines_both()
```

### 2. Integration Tests (`tests/integration/`)

#### System Integration (`test_logging_system.py`)

**Purpose**: Test complete logging workflow from setup to file output.

**Key Test Classes**:

- `TestLoggingSystemIntegration`: End-to-end logging scenarios

**Example Test Cases**:

```python
def test_end_to_end_logging_workflow()
async def test_async_logging_integration()
def test_multiple_modules_logging()
def test_exception_handling_across_system()
```

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config", 
    "--verbose",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:tests/coverage_html",
    "--asyncio-mode=auto"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

### Test Dependencies

- `pytest>=8.0.0` - Main testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.0.0` - Code coverage reporting
- `pytest-mock>=3.10.0` - Mocking utilities

## Fixtures

### Shared Fixtures (`conftest.py`)

**`temp_logs_dir`**: Creates temporary directory for log files during testing

```python
@pytest.fixture
def temp_logs_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_logs_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
```

**`clean_loggers`**: Prevents logger interference between tests

```python
@pytest.fixture
def clean_loggers():
    # Stores and restores logger state
```

**`sample_test_data`**: Provides consistent test data

```python
@pytest.fixture
def sample_test_data():
    return {
        'stock_codes': ['000001', '000002', '600519'],
        'stock_names': ['平安银行', '万科A', '贵州茅台'],
        'industries': ['银行', '房地产开发', '白酒']
    }
```

## Test Execution Guide

### 1. Basic Test Execution

#### Run All Tests

```bash
uv run pytest
```

**Expected Output**:

```sh
========================= test session starts =========================
collected 45 items

tests/unit/test_logger.py::TestSetupLogger::test_setup_logger_default_params PASSED
tests/unit/test_logger.py::TestSetupLogger::test_setup_logger_custom_params PASSED
tests/unit/test_decorators.py::TestTimerDecorator::test_timer_sync_function PASSED
...
========================= 45 passed in 2.34s =========================
```

#### Run Specific Test Files

```bash
# Test only logger functionality
uv run pytest tests/unit/test_logger.py

# Test only decorators
uv run pytest tests/unit/test_decorators.py

# Test only integration
uv run pytest tests/integration/
```

#### Run Specific Test Classes or Functions

```bash
# Run specific test class
uv run pytest tests/unit/test_logger.py::TestSetupLogger

# Run specific test function
uv run pytest tests/unit/test_logger.py::TestSetupLogger::test_setup_logger_default_params
```

### 2. Test Categories and Markers

#### Run by Test Type

```bash
# Unit tests only (fast)
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# Exclude slow tests
uv run pytest -m "not slow"

# Run only slow tests
uv run pytest -m slow
```

#### Run with Different Verbosity

```bash
# Minimal output
uv run pytest -q

# Verbose output (shows test names)
uv run pytest -v

# Very verbose (shows test names and output)
uv run pytest -vv
```

### 3. Coverage Analysis

#### Generate Coverage Report

```bash
# Run tests with coverage (default)
uv run pytest

# Generate detailed HTML coverage report
uv run pytest --cov-report=html

# View coverage report
open tests/coverage_html/index.html
```

#### Coverage Interpretation

- **Green lines**: Covered by tests
- **Red lines**: Not covered by tests
- **Target**: Aim for >90% coverage on logging components

### 4. Debugging Failed Tests

#### Show Test Output

```bash
# Show print statements and logs
uv run pytest -s

# Show local variables on failure
uv run pytest --tb=long

# Run with Python debugger on failure
uv run pytest --pdb
```

#### Run Specific Failed Test

```bash
# Re-run only failed tests
uv run pytest --lf

# Re-run failed tests first, then continue
uv run pytest --ff
```

## Test Examples and Expected Behaviors

### 1. Logger Setup Test

```python
def test_setup_logger_default_params():
    logger = setup_logger()
    assert logger.name == "stock_analysis"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 2  # File and console
```

**Expected Behavior**: Creates logger with proper name, level, and handlers.

### 2. Timer Decorator Test

```python
def test_timer_sync_function(capsys):
    @timer
    def sync_function(x, y):
        time.sleep(0.01)
        return x + y
    
    result = sync_function(2, 3)
    assert result == 5
    
    captured = capsys.readouterr()
    assert "⏱️  Function 'sync_function' runtime:" in captured.out
```

**Expected Behavior**: Function executes correctly and timing is logged.

### 3. Logged Decorator Test

```python
def test_logged_function_entry_exit(caplog):
    with caplog.at_level(logging.DEBUG):
        @logged
        def simple_function(a, b):
            return a + b
        
        result = simple_function(1, 2)
        assert result == 3
        
        log_messages = [record.message for record in caplog.records]
        entry_logs = [msg for msg in log_messages if "→ Entering simple_function" in msg]
        exit_logs = [msg for msg in log_messages if "← Exiting simple_function successfully" in msg]
        
        assert len(entry_logs) >= 1
        assert len(exit_logs) >= 1
```

**Expected Behavior**: Function entry and exit are logged with arguments.

### 4. Integration Test

```python
def test_end_to_end_logging_workflow(temp_logs_dir):
    # Setup logger with temporary directory
    logger = setup_logger("integration_workflow", "DEBUG")
    
    @logged
    @timer
    def test_workflow_function(data):
        module_logger = get_logger("test_module")
        module_logger.info(f"Processing data: {data}")
        return f"Processed: {data}"
    
    result = test_workflow_function("test_data")
    assert result == "Processed: test_data"
    
    # Verify log file contains expected messages
    # (Implementation checks log file content)
```

**Expected Behavior**: Complete workflow from function execution to log file output.

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**:

```bash
# Ensure you're in the project root directory
cd /path/to/China/
uv run pytest
```

#### 2. Logger Interference Between Tests

**Problem**: Tests affect each other's logging
**Solution**: Use the `clean_loggers` fixture

```python
def test_my_logger_test(clean_loggers):
    # Test code here
```

#### 3. Async Test Issues

**Problem**: `RuntimeError: asyncio.run() cannot be called from a running event loop`
**Solution**: Use `pytest-asyncio` markers

```python
@pytest.mark.asyncio
async def test_async_function():
    # Async test code
```

#### 4. Coverage Issues

**Problem**: Low coverage on decorated functions
**Solution**: Test both the decorator and the decorated function:

```python
def test_decorator_functionality():
    # Test decorator behavior
    
def test_decorated_function_behavior():
    # Test actual function behavior
```

#### 5. Log File Permissions

**Problem**: `PermissionError: [Errno 13] Permission denied`
**Solution**: Use temporary directories in tests

```python
def test_with_temp_dir(temp_logs_dir):
    # Use temp_logs_dir instead of actual logs directory
```

### Performance Considerations

#### Test Execution Time

- **Unit tests**: Should complete in <1 second each
- **Integration tests**: May take 1-5 seconds each
- **Slow tests**: Marked with `@pytest.mark.slow`, may take >5 seconds

#### Memory Usage

- Tests clean up after themselves using fixtures
- Temporary files are automatically removed
- Logger handlers are properly closed

## Continuous Integration

### Running Tests in CI/CD

```bash
# Install dependencies
uv sync

# Run tests with coverage and JUnit XML output
uv run pytest --cov=src --cov-report=xml --junit-xml=test-results.xml

# Check minimum coverage threshold (optional)
uv run pytest --cov=src --cov-fail-under=85
```

### Test Status Badges

If using CI/CD, you can add badges to your README:

```markdown
![Tests](https://github.com/your-repo/workflows/Tests/badge.svg)
![Coverage](https://codecov.io/gh/your-repo/branch/main/graph/badge.svg)
```

## Best Practices for Adding New Tests

### 1. Test Naming Convention

```python
# Good test names (descriptive and specific)
def test_setup_logger_creates_file_handler()
def test_timer_decorator_preserves_function_metadata()
async def test_logged_decorator_handles_async_exceptions()

# Bad test names (too generic)
def test_logger()
def test_decorator()
```

### 2. Test Structure (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: Set up test data and conditions
    test_data = "sample_input"
    expected_result = "expected_output"
    
    # Act: Execute the function being tested
    actual_result = function_under_test(test_data)
    
    # Assert: Verify the result
    assert actual_result == expected_result
```

### 3. Use Appropriate Fixtures

```python
def test_with_clean_environment(clean_loggers, temp_logs_dir):
    # Test that won't interfere with others
    logger = setup_logger_in_temp_dir(temp_logs_dir)
    # ... test logic
```

### 4. Test Both Success and Failure Cases

```python
def test_function_success_case():
    # Test normal operation
    
def test_function_handles_invalid_input():
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test("invalid_input")
```

### 5. Mock External Dependencies

```python
@patch('src.utilities.logger.logging.FileHandler')
def test_logger_setup_with_mocked_file_handler(mock_file_handler):
    # Test logger setup without actually creating files
```

## Summary

The logging system test suite provides comprehensive coverage of:

✅ **Logger initialization and configuration**  
✅ **Decorator functionality (@timer, @logged, @timed_and_logged)**  
✅ **Async and sync function support**  
✅ **Exception handling and error logging**  
✅ **Multi-module logging coordination**  
✅ **Log level management**  
✅ **File and console output**  
✅ **Performance under load**  

**Key Commands**:

- `uv run pytest` - Run all tests
- `uv run pytest -m unit` - Run unit tests only
- `uv run pytest -v --cov=src` - Verbose with coverage
- `uv run pytest tests/unit/test_logger.py` - Test specific module

**Coverage Target**: >90% for logging components

**Test Execution Time**: ~10-30 seconds for full suite

The test suite ensures that the logging functionality is robust, reliable, and ready for production use in the stock analysis pipeline.
