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

- holding_stock_analyzer.py

Before implement:

- Review industry_filter.py and apply similar features/updates to holding_stock_analyzer.py

New Features:

- Add date parameters for backtesting
- Use `*args, **kwargs` with JSON config files
- In `data/input/holding_stock_analyzer/akshare/stock_individual_fund_flow`, create a `config.yml` and `test.yml`. Sample files are in `data/input/industry_filter/akshare/stock_board_industry_hist_em`
