"""
compute_rv.py
-------------
Construct two panels from the commodity OHLC data:

  1. commodity_returns.parquet : daily close-to-close log returns
  2. commodity_rv.parquet      : 21-day trailing realized volatility

Realized volatility uses the Garman-Klass estimator, which combines
open, high, low, and close into a single daily variance estimate:

    sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2 ln 2 - 1) * (ln(C/O))^2

The 21-day RV is the sum of daily GK variances over the trailing window:

    RV_t = sum_{s=t-20}^{t} sigma_GK,s^2

Close-to-close squared returns are reported as a fallback estimator
that does not use OHLC. It is noisier but is available for every date.

Output format matches the existing panel conventions:
    date, permno, ret, rv
where "permno" is the RIC string.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


class Config:
    IN_PATH = "refinitiv_data/commodity_ohlc_clean.parquet"
    OUT_DIR = "refinitiv_data"
    RV_WINDOW = 21        # 21 trading days


def compute_rv(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Returns long-form DataFrame with columns:
        date, permno, ret, rv, rv_cc
    where:
        ret   = close-to-close log return
        rv    = trailing-window Garman-Klass realized volatility
        rv_cc = trailing-window close-to-close realized volatility
    """
    df = df.copy()
    df = df.sort_values(["ric", "date"]).reset_index(drop=True)

    # ---- Garman-Klass daily variance -------------------------------------
    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
    if has_ohlc:
        h = df["high"].values
        l = df["low"].values
        o = df["open"].values
        c = df["close"].values
        # Guard against zeros
        valid = (h > 0) & (l > 0) & (o > 0) & (c > 0) & (h >= l)
        gk = np.full_like(c, np.nan)
        gk[valid] = (0.5 * (np.log(h[valid] / l[valid]))**2
                     - (2 * np.log(2) - 1) * (np.log(c[valid] / o[valid]))**2)
        # Floor at zero (GK can occasionally go negative for tiny moves)
        gk = np.maximum(gk, 0.0)
        df["gk_var"] = gk
    else:
        print("OHLC missing; using close-to-close squared returns only.")
        df["gk_var"] = np.nan

    # ---- Close-to-close log return ---------------------------------------
    df["log_ret"] = df.groupby("ric")["close"].transform(
        lambda x: np.log(x / x.shift(1)))
    df["cc_var"] = df["log_ret"] ** 2

    # ---- Trailing rolling sums -------------------------------------------
    rv_list = []
    rvcc_list = []
    for ric, sub in df.groupby("ric", sort=False):
        sub = sub.sort_values("date")
        # Rolling sum of daily GK variance, annualized (×252)
        rv = sub["gk_var"].rolling(window=window, min_periods=window).sum() * 252
        # Fallback: close-to-close
        rv_cc = sub["cc_var"].rolling(window=window, min_periods=window).sum() * 252
        rv_list.append(rv)
        rvcc_list.append(rv_cc)

    df["rv"] = pd.concat(rv_list).sort_index()
    df["rv_cc"] = pd.concat(rvcc_list).sort_index()

    out = df[["date", "ric", "log_ret", "rv", "rv_cc"]].copy()
    out = out.rename(columns={"ric": "permno", "log_ret": "ret"})
    out = out.dropna(subset=["ret"]).reset_index(drop=True)
    return out


def main():
    df = pd.read_parquet(Config.IN_PATH)
    print(f"Input panel: {len(df)} rows, {df['ric'].nunique()} RICs, "
          f"{df['date'].nunique()} dates")
    print(f"Columns: {df.columns.tolist()}")

    panel = compute_rv(df, window=Config.RV_WINDOW)
    print(f"Output panel: {len(panel)} rows, "
          f"{panel['permno'].nunique()} RICs")

    # Diagnostics
    print("\nCoverage per RIC (fraction of dates with non-missing RV):")
    for ric in sorted(panel["permno"].unique()):
        sub = panel[panel["permno"] == ric]
        cov = sub["rv"].notna().mean()
        cov_cc = sub["rv_cc"].notna().mean()
        print(f"  {ric:<8s}  RV: {cov:.3f}  RV_cc: {cov_cc:.3f}  "
              f"rows: {len(sub)}")

    # Report basic stats
    print("\nRealized volatility statistics (annualized):")
    print(panel[["rv", "rv_cc"]].describe())

    # Split into two panels
    returns_panel = panel[["date", "permno", "ret"]].copy()
    rv_panel = panel[["date", "permno", "rv"]].dropna().reset_index(drop=True)
    rv_cc_panel = panel[["date", "permno", "rv_cc"]].dropna().reset_index(drop=True)

    returns_path = os.path.join(Config.OUT_DIR, "commodity_returns.parquet")
    rv_path = os.path.join(Config.OUT_DIR, "commodity_rv.parquet")
    rv_cc_path = os.path.join(Config.OUT_DIR, "commodity_rv_cc.parquet")

    returns_panel.to_parquet(returns_path, index=False)
    rv_panel.to_parquet(rv_path, index=False)
    rv_cc_panel.to_parquet(rv_cc_path, index=False)

    print(f"\nSaved returns to {returns_path}: {len(returns_panel)} rows")
    print(f"Saved RV (Garman-Klass) to {rv_path}: {len(rv_panel)} rows")
    print(f"Saved RV (close-to-close) to {rv_cc_path}: {len(rv_cc_panel)} rows")


if __name__ == "__main__":
    main()