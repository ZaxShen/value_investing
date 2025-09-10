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
- Make tests to ensure everthing works well

### Update main.py and market_data_fetcher.py with config

Target scripts:

- `main.py`
- `src/utilities/market_data_fetcher.py`

Tasks:

- Scan `main.py` to see if it needs a config for better customize output
- In `main.py`, in class `StockAnalysisPipeline` we no longer use data_dir to assign or receive any raw data source; instead, they should be configured in source data code, in this case, which are in `src/utilities/market_data_fetcher.py`
- In `src/utilities/market_data_fetcher.py`, see if we need a config


### Develop fhps_filter.py

Target script:

- `src/filters/fhps_filter.py`, with config `config/filters/fhps_filter/test.yml`

Description:

- `src/filters/fhps_filter.py` is a new script intended to filter stocks with split plans. But the code was copied and pasted from a jupyter notebook. So there are many code with unnecessary functionality like print or debug purpose.
- Notice fhps_filter doesn't need to be run daily, so we don't include it in `main.py` for now

Tasks:

- Refactor `fhps_filter.py`:
  - Remove unnecessary code
  - make it follow industrial standards like other scripts such as `src/analyzers/watchlist_analyzer.py`
  - make sure asyncio works well
  - add fund flow data, `period_count` from `config.yml`, after column `自除权出息日起涨跌幅%`.
  - the complete columns very like `output/reports/filters/stock_filter`:
    - first col name is blank, but should has a row number starting from 0
    - 行业
    - 代码
    - 名称
    - 总市值(亿)
    - 流通市值(亿)
    - 市盈率-动态
    - 市净率
    - 送转股份-送转总比例
    - 除权除息日
    - 除权除息日股价
    - {today}股价, format YYYYmmdd
    - 自除权出息日起涨跌幅%
    - 1日主力净流入-总净额(亿), which is from period_count in config.yml
    - 5日主力净流入-总净额(亿), which is from period_count in config.yml
    - 29日主力净流入-总净额(亿), which is from period_count in config.yml
    - 1日涨跌幅(%), which is from period_count in config.yml
    - 5日涨跌幅(%), which is from period_count in config.yml
    - 29日涨跌幅(%), which is from period_count in config.yml
    - 60日涨跌幅(%), which is from stock_zh_a_spot_em_df
    - 年初至今涨跌幅(%), which is from stock_zh_a_spot_em_df
- New functionality: current output has no fund flow data. I want to have similar one just like `src/analyzers/watchlist_analyzer.py`. The module is in `src.api.akshare`
- Update config file `test.yml`, which should be similar to the config file for `src/analyzers/watchlist_analyzer.py`
  - setup cached data source if we have
  - setup output csv file to `output/reports/filters/fhps_filter`
  - output file follow this format: `除权除息股票-{today}.csv`, today format is YYYYmmdd

Reference:

- `src/analyzers/watchlist_analyzer.py`, `input/config/analyzers/watchlist_analyzer/config.yml`
- `watchlist_analyzer.py` also use `ak.stock_individual_fund_flow`, you can take it as a reference

Sample output:

- `output/reports/filters/fhps_filter/除权除息股票.csv`
- Notice this sample output csv is incomplete, since it doesn't have fund flow data
