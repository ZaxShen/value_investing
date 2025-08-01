# TODO

## Bug

- When run `uv run main.py` "Fetching stock market data from akshare API..." cannot show actual progress. It just jump from 0% to 100%
- When run the following code, I can see three progress bar displaying three stock maretks. It is correct. But when run `uv run main.py`, those three progress bar are not displayed. I want to see them and when finish, it convert to "xx stock market data fetched successfully", xx should be SH, SZ, or BJ

```python
import asyncio

from src.utilities.get_stock_data import (
    get_industry_stock_mapping_data,
    get_stock_market_data,
)

industry_stock_mapping_df, stock_zh_a_spot_em_df = await asyncio.gather(
    get_industry_stock_mapping_data(),
    get_stock_market_data(),
)
```