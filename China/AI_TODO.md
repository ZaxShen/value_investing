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

Target scripts:

- `watchlists_analyzer.py` (notice this one was renamed from holding_stocks_analyzer.py)

Config file path:

- `config/watchlists_analyzer/config.yml`

TODO:

- `watchlists_analyzer.py` now uses nested config file, you need to make sure it can read the new config file. Use similar new code from `industry_filter.py` to replace with the old code.
- function `_get_analysis_columns` is used to make the output csv's columns. One parameter and variable here is `days`, which is used to define the fund flow of a specif period of days. Now I want to output more period, for exmaple, in the past is 29 days; now I want 1, 5, 29 days, which are from 
