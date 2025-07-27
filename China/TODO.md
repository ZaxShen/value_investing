# TODO

Update `src/utilities/get_stock_data`

- apply `retry` mechanism
- don't forget to adjust logging system if needed
- do not update pytest for now

Update `main.py`

- refactor it to class
  - keep the original funcitionality
  - add code about fetching data from `src/utilities/get_stock_data`
  - returned two vars are `industry_stock_mapping_df`, `stock_zh_a_spot_em_df`
  - run the code aysnchoronusly
  - this function must be run before `IndustryFilter`, `StockFilter`, and `HoldingStockAnalyzer`
- Asynchronouly run `IndustryFilter`, `StockFilter`, and `HoldingStockAnalyzer`
- don't forget to adjust logging system if needed
- do not update pytest for now
