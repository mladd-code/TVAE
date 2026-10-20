"""
build_crypto_vol_panel.py
=========================
Builds a log-realized-volatility panel from the crypto returns panel.

    RV_it   = sum_{s=0}^{4} ret_{i,t-s}^2
    logRV_it = 0.5 * log(RV_it)

Same schema as panel_daily_crypto.parquet, so it plugs directly into
reconstruction_r2_crypto_vol.py.
"""
import numpy as np
import pandas as pd

path_in = "refinitiv_data/panel_daily_crypto.parquet"
path_out = "refinitiv_data/panel_daily_crypto_vol.parquet"

df = pd.read_parquet(path_in)
df = df.sort_values(["permno", "date"]).copy()
print(f"Loaded crypto returns panel: {len(df)} rows, "
      f"{df['permno'].nunique()} permnos, {df['date'].nunique()} dates")

df["ret2"] = df["ret"] ** 2

# 5-day realized variance per coin
df["rv5"] = df.groupby("permno")["ret2"].transform(
    lambda x: x.rolling(5, min_periods=5).sum())

# Log transform, clipped at a small floor to avoid log(0)
df["log_rv"] = 0.5 * np.log(df["rv5"].clip(lower=1e-10))

n0 = len(df)
df = df.dropna(subset=["log_rv"]).copy()
n1 = len(df)
print(f"Dropped {n0 - n1} rows without RV ({100*(n0-n1)/max(n0,1):.2f}%)")

out = (df[["date", "permno", "log_rv", "mcap"]]
       .rename(columns={"log_rv": "ret"})
       .sort_values(["date", "permno"])
       .reset_index(drop=True))

out.to_parquet(path_out, index=False)
print(f"Saved to {path_out}: {len(out)} rows, "
      f"{out['permno'].nunique()} permnos, {out['date'].nunique()} dates")
print(f"log-RV: mean {out['ret'].mean():.3f}, std {out['ret'].std():.3f}, "
      f"range [{out['ret'].min():.2f}, {out['ret'].max():.2f}]")

# Coverage report
cov = out.groupby("date").size()
for t in (15, 20, 25, 30, 40):
    if (cov >= t).any():
        first = cov[cov >= t].index.min()
        print(f"  first date with >= {t} coins: {first.date()}")
    else:
        print(f"  first date with >= {t} coins: N/A")