"""
build_vol_panel.py
------------------
Builds a realized-volatility panel from the daily returns panel.

For each (permno, date), compute log(RV) where RV is the sum of squared
daily returns over the trailing 5 days:

    RV_it = sum_{s=0}^{4} ret_{i,t-s}^2
    logRV_it = 0.5 * log(RV_it)

This panel has the same schema as panel_daily.parquet, so it plugs
directly into reconstruction_r2.py.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("refinitiv_data/panel_daily.parquet")
df = df.sort_values(["permno", "date"]).copy()
print(f"Loaded returns panel: {len(df)} rows, "
      f"{df['permno'].nunique()} permnos, {df['date'].nunique()} dates")

df["ret2"] = df["ret"] ** 2

# 5-day realized variance
df["rv5"] = df.groupby("permno")["ret2"].transform(
    lambda x: x.rolling(5, min_periods=5).sum())

# Log transform. Clip to avoid log(0).
df["log_rv"] = 0.5 * np.log(df["rv5"].clip(lower=1e-10))

# Drop rows without a valid RV
n0 = len(df)
df = df.dropna(subset=["log_rv"]).copy()
n1 = len(df)
print(f"Dropped {n0 - n1} rows without RV ({100*(n0-n1)/n0:.2f}%)")

# Rebuild with same schema as returns panel
out = (df[["date", "permno", "log_rv", "mcap"]]
       .rename(columns={"log_rv": "ret"})
       .sort_values(["date", "permno"])
       .reset_index(drop=True))

out.to_parquet("refinitiv_data/panel_daily_vol.parquet", index=False)
print(f"Saved log-RV panel: {len(out)} rows, "
      f"{out['permno'].nunique()} permnos, {out['date'].nunique()} dates")
print(f"log-RV range: [{out['ret'].min():.3f}, {out['ret'].max():.3f}]")
print(f"log-RV mean: {out['ret'].mean():.3f}, std: {out['ret'].std():.3f}")