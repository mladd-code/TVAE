"""
fix_dates.py
------------
Snap every date in panel_clean.parquet to the last day of its calendar
month, so the panel has exactly 300 monthly periods.
"""
import pandas as pd

path = "refinitiv_data/panel_clean.parquet"
df = pd.read_parquet(path)

print(f"Before: {df['date'].nunique()} distinct dates")

df["date"] = df["date"] + pd.offsets.MonthEnd(0)
df = df.drop_duplicates(subset=["date", "permno"], keep="last")
df = df.sort_values(["permno", "date"]).reset_index(drop=True)

print(f"After:  {df['date'].nunique()} distinct dates")
print(f"Rows:   {len(df)}")
print(f"Permnos: {df['permno'].nunique()}")
print(f"Range:  {df['date'].min()} to {df['date'].max()}")

df.to_parquet(path, index=False)
print(f"Overwrote {path}")