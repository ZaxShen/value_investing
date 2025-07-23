## TODO

### Progress Bar

The hierarchical progress implementation has a little issue:

- `Processing batch` belongs to `stock_filter.py` should under main progress `Stock Filter`, but it indead under `Industry Filter`

➜  china git:(dev-rich) ✗ uv run main.py
⠹ 🚀 Parallel Stock Analysis Pipeline                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:01:24
⠹ 🔄 Starting Stock Filter...                         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:01:24
  ✅ Stock Analysis completed                         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:11
  ✅ Industry Filter completed                        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:45
⠹         Stock Filter: Processing batch 12/29 (3 in… ━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━  38% 0:01:24
