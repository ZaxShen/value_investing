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

## v1.9.3 TODO

Target script:

- `src/fhps_caching.py`

Reference script:

- `src/fhps_filter.py`

Description:

- In `src/fhps_filter.py`, currently we have caching and filtering functionalities together. Now I want to split into two scripts, `src/fhps_caching.py` and `src/fhps_filter.py`

Tasks:

- Make `src/fhps_caching.py` for caching only. The output csv should be named (use variable value `stock_fhps_em-latest.csv`) in the config file. You name the variable name. config file should be store in and load from`input/filters/fhps`.
- In `src/fhps_caching.py`'s config, specifiy the phase 1 caching, which are years in list, for example, [2020, 2021, 2022, 2023, 2024], then it should run logic like below pesudo code:
```python
years = [2020, 2021, 2022, 2023, 2024]  # notice dates is tmp variable name, you should make it more readable in config file
for year in years:
  stock_fhps_em_df = ak.stock_fhps_em(date=f'{year}1231')
  df = stock_fhps_em_df.dropna(subset=["送转股份-送转总比例"])
  df.reset_index(drop=True).to_csv(f"data/fhps/stock_fhps_em-{year}.csv")
```
- In `src/fhps_caching.py`'s config, specifiy the phase 2 caching, which is to fetch the 除权除息前日股价, like above, user should define years in list
- Notice for either phase 1 or phase 2 caching, it should first check if there are cached files already, if so, skip.
- Make `src/fhps_filter.py` run the filter functionality only, the input csv file name should be specified in the config file (use variable of `stock_fhps_em-latest.csv`). You name the variable name.
- In `src/fhps_filter.py`'s config there also should be variable like years, but you can name it. Then load the `stock_fhps_em-{year}.csv` one by one. Temperary store their 代码 in a set. Use `stock_fhps_em-latest.csv`'s 代码 to check if it apears in previous years. Becuase I want to filter a stock previous 除权 logic. So a stock must has 除权 in one of previous years and in `stock_fhps_em-latest.csv`
