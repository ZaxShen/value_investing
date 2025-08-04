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

### Apply YAML and pydantic to load config files

target file: `src/filters/industry_filter.py`
YAML's path `data/input/akshare/stock_board_industry_hist_em`
    - if no input dir, Python should create one
    - if no YAML file, Python should raise an error
    - functions or class in target file should has a arg to take the path of YAML config, by default the path is `data/input/akshare/stock_board_industry_hist_em/config.yml`
    - YAML's name: `config.yml`

- target API: `ak.stock_board_industry_hist_em`
  - You need to edit code contains `ak.stock_board_industry_hist_em` to load config file from YAML
