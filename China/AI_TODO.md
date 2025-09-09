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

### Upgrade config organization

Old path:

- `China/data/config/`
  - `industry_filter`
  - `stock_filter`
  - `watchlist_analyzer`

New path:

- `China/config/`
  - `filters`
    - `industry_filter`
    - `stock_filter`
    - `fhps_filter` ignore this dir
  - `analyzers`
    - `watchlist_analyzer`

Description:

- I've move `config/` outside `data/` for better organization
- Additionaly, I seperate filters and analzyers configs for better organization

Tasks:

- You need to scan all corresponding scripts or other files who use above three config dir and configs, re-map the old path to new path to make sure every script works.


---

### Upgrade file organization

Target dir:

- `data`



---

### Develop fhps

Target script:

- `src/filters/fhps_filter.py`, with config `data/config/fhps_filter/test.yml`

Reference:

- `src/analyzers/watchlist_analyzer.py`

Sample output:

- `

Description:

- 
