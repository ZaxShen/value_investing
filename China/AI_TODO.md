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

## v1.8.0 TODO

### Improve usability

Target scripts and configs:

- `src/filters/stock_filter.py`, `data/config/stock_filter/test.yml`
- `src/anazlyers/watchlist_analyzer.py`, `data/config/watchlist_analyzer/test.yml`

Description:

- In `stock_filter.py` and `watchlist_analyzer`, and their configs, there are some shared code, functionality, and configs. `ak.stock_individual_fund_flow` is the main API both of them are using.
- I want to make a new script may be in utilities or other dirs, you can suggest it. The new script name is related stock_individual_fund_flow, you can give suggestions.

Tasks:

- Make `stock_filter.py` and `watchlist_analyzer` load the new script, instead of using similar code blocks.
- Make sure all exisiting functionalities will not be influenced.
- Although `stock_filter.py` and `watchlist_analyzer` both use `ak.stock_individual_fund_flow` and will load the new script, there are may be some difference about calling the api. So you may need to leave some flexible space like `*args, **kwargs` in the script for future updating.

Questions:

- Should I make a folder like 'API' or 'akshare_api' or something else to indicate that they are extrenal apis and my promgram will call them in customized ways?
- If do so, suggest the folder name
- If do so, should I do the same thing for all heavy-use external apis?

