# Versions

## v1

### v1.6.0

- :tada: New Features:
  - Enabled AI guidelines files `TODO.md` and `AI_TODO.md` for better corporating with AI assistances and project stability.

### v1.5.2

- :tada: New Features:
  - Remove all files in git history, which are defined by `.gitignore`, for better cybersecurity and teamwork


### v1.5.1

- :heavy_check_mark: Bug Fix:
  - Fix progress bar bug. Now it can correctly display `akshare`'s internal progress from `tqdm`

### v1.5.0

- :tada: New Features:
  - Refactored all code to Class
  - Improved progress bar to display more info and update progress status in time
  - Enabled retry features

### v1.4.0

- :tada: New Features:
  - Enabled rich progress bar in hierarchical structure for better visualization


### v1.3.0

- :tada: New Features:
  - Enabled Progress Bar to replace Verbose
- :warning: Deprecated Features:
  - Disabled Verbose

### v1.2.0

- :tada: New Features:
  - Enabled Verbose to track program running status

### v1.1.0

- :tada: New Features:
  - logging
  - unit tests, and integration test


### v1.0.0

- :tada: New Features:
  - Main functionalities are complete, including
    - stock filter
    - industry filter
    - holding stock analyzer
  - Enabled concurency by `asyncio`
    - Performance improved by 15x, from 45 mins to 3 mins
