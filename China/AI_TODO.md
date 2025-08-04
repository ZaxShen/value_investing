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
YAML's path `data/stocks/input/`
    - if no input dir, Python should create one
    - if no YAML file, Python should create one
    - YAML's name: `stock_board_industry_hist_em_CONFIG.yml`

- target API: `ak.stock_board_industry_hist_em`
  - You need to edit code contains `ak.stock_board_industry_hist_em` to load config file from YAML

Below are the availabel paras
- Each para I want to have a default value
- para is from below's 名称
- data type is from below's 类型
- comment is from below's 描述

名称 类型 描述
symbol str symbol="小金属"; 可以通过调用 ak.stock_board_industry_name_em() 查看东方财富-行业板块的所有行业代码
start_date str start_date="20211201";
end_date str end_date="20220401";
period str period="日k"; 周期; choice of {"日k", "周k", "月k"}
adjust str adjust=""; choice of {'': 不复权, 默认; "qfq": 前复权, "hfq": 后复权}
