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

## v1.9.0 TODO

### Upgrade file organization

Target dir:

- `data/`
- `input/`
- `output/`
- `cache`

Tasks:

Above are the final file structures. You need to follow those structure to complete the following tasks:

- Scan above four paths to make sure you understand the structure.
- Update all code to make sure all configs in `input/config/` can be correctly read.
- Update `watchlist_analyzer.py` code to make sure it can correctly load watchlist data from `input/watchlists`
- Update `main.py` to make sure it can correctly output csv files to `output/reports`
- Update `main.py` to make sure it can correctly output csv files to `output/today` with latest output

---

### Develop fhps

Target script:

- `src/filters/fhps_filter.py`, with config `config/filters/fhps_filter/test.yml`

Reference:

- `src/analyzers/watchlist_analyzer.py`

Sample output:

- `

Description:

-
