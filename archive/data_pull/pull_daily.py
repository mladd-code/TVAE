"""
pull_daily.py  (v3)
-------------------
Daily returns and S&P 500 universe/benchmark pull for the MVP application.

Modes:
    python pull_daily.py spx_universe   # S&P 500 constituents
    python pull_daily.py spx_index      # S&P 500 index history
    python pull_daily.py daily          # daily returns for constituents

Fixes vs v2:
  - Smaller default batch size (25) to reduce API payload.
  - Progressive fallback (25 -> 5 -> 2 -> 1) with per-RIC retries.
  - Resume support via daily_partial.parquet; safe to rerun after crash.
  - Progress reporting with rate in RICs per minute.
"""

from __future__ import annotations
import sys, os

class _Tee:
    def __init__(self, f):
        self.f = f
    def write(self, d):
        sys.__stdout__.write(d)
        try: self.f.write(d)
        except Exception: pass
    def flush(self):
        try: sys.__stdout__.flush()
        except Exception: pass
        try: self.f.flush()
        except Exception: pass

_tee = _Tee(open("pull_daily_output.txt", "w", encoding="utf-8"))
sys.stdout = _tee
sys.stderr = _tee

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd


class Config:
    OUT_DIR = os.path.abspath("refinitiv_data")
    START_DATE = "2005-01-01"
    END_DATE   = "2024-12-31"

    BATCH_SIZE = 25
    MAX_RETRIES = 2
    SLEEP_SECONDS = 1.0

    PRICING_FIELDS = [
        "TR.TotalReturn",
        "TR.PriceClose",
        "TR.CompanyMarketCap",
        "TR.SharesOutstanding",
    ]


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
def open_session():
    import refinitiv.data as rd
    try:
        rd.open_session()
    except Exception as e:
        print(f"open_session failed: {e}")
        raise
    print("Refinitiv session opened.")
    return rd


def close_session():
    import refinitiv.data as rd
    try:
        rd.close_session()
        print("Refinitiv session closed.")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# S&P 500 universe
# --------------------------------------------------------------------------- #
def build_spx_universe(cfg):
    import refinitiv.data as rd
    print("Pulling S&P 500 constituents...")
    df = rd.get_data(
        universe=["0#.SPX"],
        fields=["TR.CommonName", "TR.CompanyMarketCap",
                "TR.ISIN", "TR.CUSIP", "TR.ExchangeName"],
    )
    rename = {
        "Instrument": "ric",
        "Company Name": "name",
        "Company Common Name": "name",
        "Company Market Cap": "mcap",
        "ISIN": "isin", "CUSIP": "cusip", "Exchange Name": "exchange",
    }
    df = df.rename(columns={k: v for k, v in rename.items()
                            if k in df.columns})
    if "ric" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ric"})
    df = df.dropna(subset=["ric"]).reset_index(drop=True)
    if "exchange" in df.columns:
        df = df[df["exchange"].notna()].reset_index(drop=True)
    print(f"Universe: {len(df)} constituents")

    path = os.path.join(cfg.OUT_DIR, "universe_spx.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved to {path}")
    return df


# --------------------------------------------------------------------------- #
# S&P 500 index
# --------------------------------------------------------------------------- #
def pull_spx_index(cfg):
    import refinitiv.data as rd
    print("Pulling S&P 500 index history...")
    df = rd.get_history(
        universe=[".SPX"],
        fields=["TR.PriceClose"],
        interval="daily",
        start=cfg.START_DATE, end=cfg.END_DATE,
    )
    if df is None or df.empty:
        print("Empty index history.")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{ric}|{field}" for ric, field in df.columns]
    df = df.reset_index()
    for cand in ("Timestamp", "index", "Date"):
        if cand in df.columns:
            df = df.rename(columns={cand: "date"})
            break
    df["date"] = pd.to_datetime(df["date"])
    close_col = [c for c in df.columns if "Price Close" in str(c)][0]
    df = df[["date", close_col]].rename(columns={close_col: "spx_close"})
    df["spx_ret"] = df["spx_close"].pct_change()

    path = os.path.join(cfg.OUT_DIR, "spx_index.parquet")
    df.to_parquet(path, index=False)
    print(f"Saved index history to {path}: {len(df)} rows")
    return df


# --------------------------------------------------------------------------- #
# Daily pull helpers
# --------------------------------------------------------------------------- #
def _pull_daily_batch(rd, rics, cfg):
    """Pull one batch and reshape. Raises on API failure."""
    df = rd.get_history(
        universe=rics,
        fields=cfg.PRICING_FIELDS,
        interval="daily",
        start=cfg.START_DATE, end=cfg.END_DATE,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{ric}|{field}" for ric, field in df.columns]
    elif len(rics) == 1:
        df.columns = [f"{rics[0]}|{c}" for c in df.columns]

    df = df.reset_index()
    for cand in ("Timestamp", "index", "Date"):
        if cand in df.columns:
            df = df.rename(columns={cand: "date"})
            break
    df["date"] = pd.to_datetime(df["date"])

    long = df.melt(id_vars="date", var_name="ric_field", value_name="value")
    split = long["ric_field"].str.split("|", expand=True, n=1)
    long["ric"] = split[0]
    long["field"] = split[1]

    wide = long.pivot_table(
        index=["date", "ric"], columns="field", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for c in ["Total Return", "Price Close",
              "Company Market Cap", "Outstanding Shares"]:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float64")
    return wide


def _pull_batch_with_retry(rd, batch, cfg, label):
    """
    Try a batch; on failure, split into smaller sub-batches and retry.
    Final fallback: pull each RIC individually.
    """
    delay = 5.0
    for attempt in range(cfg.MAX_RETRIES):
        try:
            df = _pull_daily_batch(rd, batch, cfg)
            return df
        except Exception as e:
            err_msg = str(e)[:100].replace("\n", " ")
            print(f"    {label} attempt {attempt+1}/{cfg.MAX_RETRIES} "
                  f"failed: {err_msg}")
            time.sleep(delay)
            delay *= 2

    if len(batch) <= 1:
        print(f"    {label}: giving up on {batch}")
        return pd.DataFrame()

    # Progressive fallback
    if len(batch) > 10:
        chunk = 5
    elif len(batch) > 3:
        chunk = 2
    else:
        chunk = 1

    sub_dfs = []
    for i in range(0, len(batch), chunk):
        sub = batch[i:i + chunk]
        try:
            df_sub = _pull_daily_batch(rd, sub, cfg)
            if not df_sub.empty:
                sub_dfs.append(df_sub)
        except Exception:
            # If chunk > 1 still fails, go one level deeper
            if chunk > 1:
                for ric in sub:
                    try:
                        df_one = _pull_daily_batch(rd, [ric], cfg)
                        if not df_one.empty:
                            sub_dfs.append(df_one)
                    except Exception:
                        print(f"    {label}: skipping {ric}")
            else:
                print(f"    {label}: skipping {sub}")
        time.sleep(1.0)

    if sub_dfs:
        return pd.concat(sub_dfs, ignore_index=True)
    return pd.DataFrame()


def pull_daily_returns(cfg, universe):
    import refinitiv.data as rd

    rics = universe["ric"].tolist()

    # Resume support
    partial_path = os.path.join(cfg.OUT_DIR, "daily_partial.parquet")
    done_rics = set()
    existing = []
    if os.path.exists(partial_path):
        try:
            prev = pd.read_parquet(partial_path)
            done_rics = set(prev["ric"].unique())
            existing.append(prev)
            print(f"Resuming: {len(done_rics)} RICs already pulled")
        except Exception:
            pass

    todo_rics = [r for r in rics if r not in done_rics]
    print(f"Pulling {len(todo_rics)} RICs (batch size {cfg.BATCH_SIZE}, "
          f"individual fallback enabled)")

    pieces = list(existing)
    n_pulled = len(done_rics)
    t_start = time.time()

    for i in range(0, len(todo_rics), cfg.BATCH_SIZE):
        batch = todo_rics[i:i + cfg.BATCH_SIZE]
        t0 = time.time()
        df_b = _pull_batch_with_retry(
            rd, batch, cfg, f"batch {i//cfg.BATCH_SIZE + 1}")

        if not df_b.empty:
            pieces.append(df_b)
            n_pulled += df_b["ric"].nunique()
            # Save progress after every batch
            pd.concat(pieces, ignore_index=True).to_parquet(
                partial_path, index=False)

        elapsed = time.time() - t_start
        rate = (n_pulled - len(done_rics)) / max(elapsed, 1) * 60
        print(f"  batch {i//cfg.BATCH_SIZE + 1}: "
              f"{len(batch)} RICs -> {len(df_b)} rows "
              f"({time.time()-t0:.1f}s). Total: {n_pulled} RICs, "
              f"rate {rate:.1f} RICs/min")
        time.sleep(cfg.SLEEP_SECONDS)

    if not pieces:
        print("No daily data returned.")
        return pd.DataFrame()

    full = pd.concat(pieces, ignore_index=True)
    full = full.drop_duplicates(subset=["date", "ric"], keep="last")
    path = os.path.join(cfg.OUT_DIR, "daily_raw.parquet")
    full.to_parquet(path, index=False)
    print(f"Saved raw daily pull to {path}: {len(full)} rows, "
          f"{full['ric'].nunique()} unique RICs")

    # Clean up partial file
    try:
        os.remove(partial_path)
    except Exception:
        pass
    return full


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_daily(df, cfg):
    df = df.copy()
    rename_map = {
        "Total Return": "ret_raw",
        "Price Close": "price",
        "Company Market Cap": "mcap",
        "Outstanding Shares": "shares",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                            if k in df.columns})

    for c in ["ret_raw", "mcap"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    n0 = len(df)
    df = df.dropna(subset=["ret_raw", "mcap"])
    n1 = len(df)
    df = df[df["mcap"] > 0]
    n2 = len(df)
    print(f"Cleaning: {n0} -> {n1} (dropna) -> {n2} (mcap > 0)")

    # Percent vs decimal detection: use 99.5th percentile of |ret_raw|.
    # If data is in percent, |ret_raw| for a 5% daily move would be 5.0.
    # If data is in decimal, it would be 0.05. So a threshold of 5
    # cleanly separates the two conventions.
    q995 = df["ret_raw"].abs().quantile(0.995)
    print(f"  99.5th pct |ret_raw|: {q995:.4f}")
    if q995 > 5.0:
        df["ret"] = df["ret_raw"] / 100.0
        print("  -> dividing by 100 (percent convention detected)")
    else:
        df["ret"] = df["ret_raw"]
        print("  -> keeping as-is (decimal convention detected)")

    # Drop implausible returns (|ret| >= 50% in a single day)
    n3 = len(df)
    df = df[df["ret"].abs() < 0.5]
    n4 = len(df)
    print(f"  dropped {n3 - n4} rows with |ret| >= 0.5 "
          f"({100*(n3-n4)/max(n3,1):.2f}%)")

    df = df.rename(columns={"ric": "permno"})
    df["mcap"] = df["mcap"] / 1e6
    out = df[["date", "permno", "ret", "mcap"]].copy()
    out["permno"] = out["permno"].astype("string")
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(mode: str = "spx_universe"):
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    print(f"Mode: {mode}")
    open_session()
    try:
        if mode == "spx_universe":
            build_spx_universe(Config)
            return
        if mode == "spx_index":
            pull_spx_index(Config)
            return
        if mode == "daily":
            u_path = os.path.join(Config.OUT_DIR, "universe_spx.parquet")
            if not os.path.exists(u_path):
                raise FileNotFoundError("Run 'spx_universe' first")
            universe = pd.read_parquet(u_path)
            print(f"Loaded universe: {len(universe)} RICs")

            raw_path = os.path.join(Config.OUT_DIR, "daily_raw.parquet")
            if not os.path.exists(raw_path):
                raw = pull_daily_returns(Config, universe)
            else:
                print(f"Using cached daily pull from {raw_path}")
                raw = pd.read_parquet(raw_path)
                print(f"  {len(raw)} rows, "
                      f"{raw['ric'].nunique()} unique RICs")

            if raw.empty:
                print("No data; aborting.")
                return
            clean = clean_daily(raw, Config)
            c_path = os.path.join(Config.OUT_DIR, "panel_daily.parquet")
            clean.to_parquet(c_path, index=False)
            print(f"Saved cleaned daily panel to {c_path}")
            print(f"  rows: {len(clean)}, "
                  f"permnos: {clean['permno'].nunique()}, "
                  f"dates: {clean['date'].nunique()}")
            return
        raise ValueError(f"Unknown mode: {mode}")
    finally:
        close_session()


if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "spx_universe"
    try:
        main(mode_arg)
    finally:
        _tee.flush()