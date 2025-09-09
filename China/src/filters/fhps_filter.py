import asyncio
from datetime import datetime

import akshare as ak
import pandas as pd

# Filter stocks with ex-dividend dates before today
stock_fhps_em_df = ak.stock_fhps_em(date="20241231")
df = stock_fhps_em_df.dropna(subset=["送转股份-送转总比例"])
df.loc[:, "除权除息日"] = pd.to_datetime(df["除权除息日"], format="%Y-%m-%d")
today = datetime.today()
filter_past = df.loc[:, "除权除息日"] < today
df_filtered = df[filter_past]
print(f"Stocks with ex-dividend dates before today: {len(df_filtered)}")
df_filtered.head()


# Async function to get stock price for a specific date
async def get_stock_price_async(stock_code, date):
    """Get stock price for a specific date using asyncio.to_thread"""
    try:
        # Use asyncio.to_thread for non-blocking akshare calls
        df_price = await asyncio.to_thread(
            ak.stock_zh_a_hist,
            stock_code,
            period="daily",
            start_date=date.strftime("%Y%m%d"),
            end_date=date.strftime("%Y%m%d"),
        )
        if not df_price.empty:
            return df_price["收盘"].iloc[0]  # Close price
        return None
    except Exception as e:
        print(f"Error fetching price for {stock_code} on {date}: {e}")
        return None


# Simplified batch function that takes the filtered DataFrame directly
async def get_stock_prices_batch(df_filtered):
    """Fetch prices for all stocks in the DataFrame concurrently"""
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
    today_date = datetime.today()

    async def fetch_stock_prices(row):
        async with semaphore:
            stock_code = row["代码"]
            ex_date = row["除权除息日"]
            print(f"Fetching prices for {stock_code}...")

            ex_price = await get_stock_price_async(stock_code, ex_date)
            today_price = await get_stock_price_async(stock_code, today_date)

            return {
                "代码": stock_code,
                "名称": row["名称"],
                "除权除息日": ex_date,
                "送转股份-送转总比例": row["送转股份-送转总比例"],
                "除权除息日股价": ex_price,
                f"{today_date.strftime('%Y%m%d')}股价": today_price,
            }

    # Create tasks for all rows
    tasks = [fetch_stock_prices(row) for _, row in df_filtered.iterrows()]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return valid results
    return [r for r in results if not isinstance(r, Exception)]


# Run async price fetching - much simpler now!
print("Starting async price fetching...")
results = await get_stock_prices_batch(df_filtered)
print(f"Completed fetching prices for {len(results)} stocks")

# Convert results to DataFrame and analyze
price_df = pd.DataFrame(results)
print("Price data summary:")
print(price_df.head())
print(f"\nSuccessful price fetches: {price_df['除权除息日股价'].notna().sum()}")
print(f"Failed price fetches: {price_df['除权除息日股价'].isna().sum()}")

# Convert results to DataFrame and calculate performance
price_df = pd.DataFrame(results)

# Get today's date for dynamic column naming
today_date = datetime.today()
today_price_col = f"{today_date.strftime('%Y%m%d')}股价"

# Calculate price change percentage
price_df["涨跌幅%"] = (
    (price_df[today_price_col] - price_df["除权除息日股价"])
    / price_df["除权除息日股价"]
    * 100
).round(2)

# Filter out rows with missing prices
result_display = price_df.dropna(subset=["除权除息日股价", today_price_col])

print(f"Results with both prices available: {len(result_display)}")
print("\nSample results:")
columns_to_show = [
    "代码",
    "名称",
    "除权除息日",
    "送转股份-送转总比例",
    "除权除息日股价",
    today_price_col,
    "涨跌幅%",
]
print(result_display[columns_to_show].head(10))

filter_price_change = result_display["涨跌幅%"] < 5
df_final = result_display[filter_price_change]
df_final = df_final.sort_values(by=["涨跌幅%"], ascending=True)

df_final.to_csv("除权除息股票.csv", index=False)
