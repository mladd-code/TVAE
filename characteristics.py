"""
characteristics.py
------------------
Builds price-based characteristics from the daily panel.

Input:  df with columns [date, permno, ret, mcap]
Output: df with additional columns for each characteristic.

All characteristics are backward-looking: at date t, they use only
information available up to and including date t-1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Characteristics to compute
CHAR_COLS = [
    "size",         # log market cap
    "mom_12_1",     # 250-day momentum, skipping last 21 days
    "rev_1m",       # past 21-day return
    "vol_3m",       # 63-day return volatility
    "vol_12m",      # 250-day return volatility
    "beta_12m",     # 250-day market beta
    "idio_vol_12m", # residual std from market model
    "skew_12m",     # 250-day return skewness
    "max_ret_1m",   # max daily return over past 21 days
]


def build_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute characteristics for each (permno, date). Backward-looking.
    """
    df = df.sort_values(["permno", "date"]).copy()

    # Market return = cross-sectional mean at each date
    mkt = df.groupby("date")["ret"].mean().rename("mkt_ret")
    df = df.merge(mkt, left_on="date", right_index=True, how="left")

    g = df.groupby("permno", group_keys=False)

    # size = log(mcap)
    df["size"] = np.log(df["mcap"].clip(lower=1e-3))

    # 1-month reversal: past 21-day simple return (using log approx)
    df["rev_1m"] = g["ret"].transform(
        lambda x: (1 + x).rolling(21, min_periods=10).apply(
            np.prod, raw=True) - 1)

    # 12-1 momentum: log return over t-250 to t-21
    logret = np.log1p(df["ret"].clip(lower=-0.99))
    df["_logret"] = logret
    df["mom_12_1"] = (df.groupby("permno")["_logret"].transform(
        lambda x: x.rolling(250, min_periods=100).sum().shift(21)))

    # 3-month and 12-month volatility
    df["vol_3m"] = g["ret"].transform(
        lambda x: x.rolling(63, min_periods=30).std())
    df["vol_12m"] = g["ret"].transform(
        lambda x: x.rolling(250, min_periods=100).std())

    # Rolling market beta and residual std
    def _rolling_beta(group):
        cov = group["ret"].rolling(250, min_periods=100).cov(
            group["mkt_ret"])
        var = group["mkt_ret"].rolling(250, min_periods=100).var()
        return cov / var.replace(0, np.nan)

    df["beta_12m"] = df.groupby("permno", group_keys=False).apply(
        _rolling_beta).reset_index(level=0, drop=True)

    # Residual variance and idiosyncratic vol
    # E[(r - beta*mkt)^2] over 250 days
    df["_resid"] = df["ret"] - df["beta_12m"] * df["mkt_ret"]
    df["idio_vol_12m"] = df.groupby("permno")["_resid"].transform(
        lambda x: x.rolling(250, min_periods=100).std())

    # 12-month skewness
    df["skew_12m"] = g["ret"].transform(
        lambda x: x.rolling(250, min_periods=100).skew())

    # MAX: largest daily return in the past 21 days
    df["max_ret_1m"] = g["ret"].transform(
        lambda x: x.rolling(21, min_periods=10).max())

    df = df.drop(columns=["mkt_ret", "_logret", "_resid"])
    return df


def rank_normalize_char(df: pd.DataFrame,
                        char_cols: list) -> pd.DataFrame:
    """
    Cross-sectional rank normalization to [-1, 1] at each date.
    NaN values are filled with 0 (the cross-sectional median).
    """
    df = df.copy()
    for c in char_cols:
        rank = df.groupby("date")[c].rank(pct=True)
        df[c] = 2 * rank - 1
        df[c] = df[c].fillna(0.0)
    return df


if __name__ == "__main__":
    from mvp_app import Config
    df = pd.read_parquet(Config.PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Panel: {len(df)} rows")

    df = build_characteristics(df)
    print(f"After characteristics: {df.shape}")

    df = rank_normalize_char(df, CHAR_COLS)
    print("After rank normalization: ok")

    # Report coverage
    print("\nCharacteristic non-zero coverage (fraction of rows):")
    for c in CHAR_COLS:
        frac = (df[c].abs() > 1e-6).mean()
        print(f"  {c:<15s}  {frac:.3f}")

    df.to_parquet("refinitiv_data/panel_daily_char.parquet", index=False)
    print(f"\nSaved to refinitiv_data/panel_daily_char.parquet")