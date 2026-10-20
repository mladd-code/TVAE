"""
diag_panel.py
-------------
Quick diagnostic on the cleaned panel.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("refinitiv_data/panel_clean.parquet")
print(f"Rows: {len(df)}")
print(f"Permnos: {df['permno'].nunique()}")
print(f"Dates: {df['date'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Distribution of rows per month
counts = df.groupby(df['date'].dt.to_period('M')).size()
print(f"\nDistinct calendar months: {counts.shape[0]}")
print(f"Rows per calendar month: min={counts.min()}, "
      f"median={counts.median():.0f}, max={counts.max()}")

# Find calendar months with multiple distinct date labels
multi = counts[df.groupby(df['date'].dt.to_period('M'))['date'].nunique() > 1]
print(f"\nCalendar months with multiple distinct date labels: {len(multi)}")
if len(multi) > 0:
    print(multi.head(20))

# Show a few example dates that look unusual
df["day"] = df["date"].dt.day
day_counts = df["day"].value_counts().sort_index()
print(f"\nDistribution of day-of-month:")
print(day_counts.head(20))
print(f"...")
print(day_counts.tail(20))

# Row counts by year
print("\nRows per year:")
print(df.groupby(df['date'].dt.year).size())