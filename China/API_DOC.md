# API

## Data Preprocessing

### src/utilities/get_stock_data.py

|Package|API|Description|Config Path|
|:-|:-|:-|:-|
|[akshare](https://github.com/akfamily/akshare)|[stock_sh_a_spot_em](https://akshare.akfamily.xyz/data/stock/stock.html#id14)|东方财富网-沪 A 股-实时行情数据|None|
|[akshare](https://github.com/akfamily/akshare)|[stock_sz_a_spot_em](https://akshare.akfamily.xyz/data/stock/stock.html#id15)|东方财富网-深 A 股-实时行情数据|None|
|[akshare](https://github.com/akfamily/akshare)|[stock_bj_a_spot_em](https://akshare.akfamily.xyz/data/stock/stock.html#id16)|东方财富网-京 A 股-实时行情数据|None|
|[akshare](https://github.com/akfamily/akshare)|[stock_board_industry_name_em](https://akshare.akfamily.xyz/data/stock/stock.html#id358)|东方财富-沪深京板块-行业板块|None|
|[akshare](https://github.com/akfamily/akshare)|[stock_board_industry_cons_em](https://akshare.akfamily.xyz/data/stock/stock.html#id360)|东方财富-沪深板块-行业板块-板块成份|None|

---

## Analyzers

## Filters

### src/filters/indusstry_filter.py

|Package|API|Description|Config Path|
|:-|:-|:-|:-|
|[akshare](https://github.com/akfamily/akshare)|[stock_board_industry_name_em](https://akshare.akfamily.xyz/data/stock/stock.html#id358)|东方财富-沪深京板块-行业板块|None|
|[akshare](https://github.com/akfamily/akshare)|[stock_board_industry_hist_em](https://akshare.akfamily.xyz/data/stock/stock.html#id361)|东方财富-沪深板块-行业板块-历史行情数据|`data/input/akshare/stock_board_industry_hist_em`|
|[akshare](https://github.com/akfamily/akshare)|[stock_sector_fund_flow_hist](https://akshare.akfamily.xyz/data/stock/stock.html#id171)|东方财富网-数据中心-资金流向-行业资金流-行业历史资金流|None|

### src/filters/stock_filter.py

|Package|API|Description|Config Path|
|:-|:-|:-|:-|
|[akshare](https://github.com/akfamily/akshare)|[stock_individual_fund_flow](https://akshare.akfamily.xyz/data/stock/stock.html#id165)|东方财富网-数据中心-个股资金流向|`data/input/akshare/stock_individual_fund_flow`|
|[akshare](https://github.com/akfamily/akshare)|[stock_sector_fund_flow_hist](https://akshare.akfamily.xyz/data/stock/stock.html#id171)|东方财富网-数据中心-资金流向-行业资金流-行业历史资金流|None|
