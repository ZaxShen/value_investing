# AI TODO List

## 🤖 For AI Assistants

**IMPORTANT**: Rules for AI Tools and Assistants

Any AI assistant working with this repository MUST strictly adhere to the following guidelines:

### ✅ You CAN

- Read and analyze TODO items
- Discuss technical approaches
- Answer questions about features
- Implement TODO items **ONLY after discussion**

---

### ❌ You CANNOT

- Modify this file
- Auto-execute TODO items without discussion
- Always agree - challenge ideas when needed
- Implement on main branch or with unstaged changes

---

### 📋 Rules

- **DISCUSS FIRST** - Clarify before implementing
- **QUESTION** unclear requirements
- **SUGGEST** better solutions when possible
- **LOG** activities to `logs/vx.x.x_[ai_name].log`
  - Log silently (no token consumption)
  - Create/append only (no edit/delete history)
  - Format: `v1.7.0_claude.log` (version from TODO section below)
  - Structure: Brief summary at top, then full activity details using format:
    > user input
    ⎿ commands, tools; output/result
    ⏺ My explanatory responses
- This file is for **AI-approved tasks only**

---

## v1.7.0 TODO

### Features

#### Refactor Specific Python Scripts with Config Files

Target:

- Script: `industry_filter.py`
- Config: `config/industry_filter/test.yml`

Reference:

- Script: `watchlist_analyzer.py`
- Config: `config/watchlist_analyzer/test.yml`

TODO:

- `watchlist_analyzer.py` now uses nested config file, and it works very well. You need to make `industry_filter.py` use similar logic.
- `industry_filter.py` has outdated logic and function or variable name, make sure you follow the latest naming convention from `watchlist_analyzer.py`
- Similar to `watchlist_analyzer.py`, the output csv for `industry_filter.py` now should contain dynmaic period_count
- Similar to `watchlist_analyzer.py`'s `last_date`, make sure the `end_date` in `industry_filter.py` works
