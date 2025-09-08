# Version History

## v1

### v1.7.0

- :tada: **New Features:**
  - Refactored codebase for scalable architecture:
    - `src/analyzers/`: Analyzer modules for stocks and industries
    - `src/filters/`: Filter modules for screening and analysis
    - Maintained backward compatibility with updated imports and tests
  - Handling uncertain arguments from JSON to analyzers and filters
    - Signifiicantly improved total flexibility from hardcoded paras to flexible paras from `config.yml`
    - Enable automatic data types check
  - Upgraded `industry_filter.py` with Lazy Initialization for better performanc

### v1.6.0

- :tada: **New Features:**
  - AI collaboration framework with `TODO.md` and `AI_TODO.md` guidelines
  - Clear boundaries between human-only and AI-approved tasks
  - Enhanced AI-human workflow coordination

### v1.5.2

- :tada: **New Features:**
  - Complete git history sanitization removing all `.gitignore` matched files
  - Enhanced cybersecurity by eliminating sensitive data from version control
  - Improved repo collaboration readiness

### v1.5.1

- :heavy_check_mark: **Bug Fix:**
  - Fixed progress bar display issues with `akshare` API calls
  - Individual progress tracking for SH, SZ, BJ stock markets
  - Granular progress visibility during concurrent data fetching
  - Proper `tqdm` integration for `akshare` progress metrics

### v1.5.0

- :tada: **New Features:**
  - Codebase refactoring from functional to OOP architecture
  - Enhanced progress bar with real-time updates & hierarchical tracking
  - Robust retry mechanisms with exponential backoff for API reliability
  - Timeout handling & comprehensive error recovery
  - Advanced logging & monitoring for debugging & performance analysis

### v1.4.0

- :tada: **New Features:**
  - Rich library-based progress visualization with hierarchical structure
  - Color-coded progress indicators & detailed status info
  - Improved concurrent operation visibility

### v1.3.0

- :tada: **New Features:**
  - Replaced verbose logging with sophisticated progress bars
  - Real-time feedback during long-running operations
- :warning: **Deprecated Features:**
  - Verbose text-based status reporting

### v1.2.0

- :tada: **New Features:**
  - Comprehensive verbose logging for operational transparency
  - Detailed execution tracking & status reporting

### v1.1.0

- :tada: **New Features:**
  - Enterprise-grade logging infrastructure with structured formatting
  - Comprehensive testing framework (unit & integration tests)
  - Automated testing pipeline for code quality assurance
  - Debugging & monitoring for production environments

### v1.0.0

- :tada: **New Features:**
  - Core functionality with three analysis modules:
    - **Stock Filter**: Advanced filtering based on financial metrics
    - **Industry Filter**: Sector-wide analysis & filtering
    - **Holding Stock Analyzer**: Portfolio analysis for existing positions
  - Performance optimization via `asyncio` concurrency:
    - 15x improvement: 45min → 3min execution time
    - Parallel processing for simultaneous data fetching & analysis
    - Optimized API call patterns maximizing throughput while respecting rate limits
