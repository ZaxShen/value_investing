# TODO

## Features

### Stuck issues

When API call get stucks, can the program return some info or logs? Maybe we need to upgrade `retry.py`

```sh
➜  China git:(dev-rich-tqdm) ✗ uv run main.py                                          
🚀 Starting China Stock Analysis Pipeline
⠼ 🚀 Parallel Stock Analysis Pipeline                       ━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━  33% 0:12:25
  ✅ Market Data Ready                                      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:48
  ✓ Stock market data fetched successfully                  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:47
  ✓ Industry data processed successfully                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:19
⠼ 🔄 Starting Stock Filter...                               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:11:37
⠼     📊 Stock Filter: Processing batch 8/29 (3 industries) ━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  24% 0:11:37
  ✅ Holding Stock Analyzer completed                       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:07
  ✅ Industry Filter completed                              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:42
```
