"""
refinitiv_pull.py
-----------------
Refinitiv data acquisition for the TV-AE empirical application.

Modes:
    python refinitiv_pull.py smoke      # 5 stocks, 24 months, verify
    python refinitiv_pull.py universe   # build screener universe (chunked)
    python refinitiv_pull.py full       # pull monthly panel

All console output is also written to refinitiv_pull_output.txt.
"""

from __future__ import annotations
import sys
import os

# ---- Tee logging ----
class _Tee:
    def __init__(self, file_handle):
        self.file = file_handle
    def write(self, data):
        sys.__stdout__.write(data)
        try:
            self.file.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

_LOG_PATH = os.path.abspath("refinitiv_pull_output.txt")
_tee = _Tee(open(_LOG_PATH, "w", encoding="utf-8"))
sys.stdout = _tee
sys.stderr = _tee

# ---- Suppress noisy pandas warnings ----
import warnings
warnings.filterwarnings("ignore")

# ---- Rest of imports ----
import time
import math
from typing import List, Optional
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class Config:
    OUT_DIR = os.path.abspath("refinitiv_data")

    START_DATE = "2000-01-01"
    END_DATE   = "2024-12-31"

    EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]
    MIN_PRICE = 5.0
    MIN_MCAP_MM = 100.0
    EXCLUDE_FINANCIALS = True

    BATCH_SIZE = 100
    SLEEP_SECONDS = 0.5

    PRICING_FIELDS = [
        "TR.TotalReturn",
        "TR.PriceClose",
        "TR.CompanyMarketCap",
        "TR.SharesOutstanding",
    ]

    REFERENCE_FIELDS = [
        "TR.ISIN",
        "TR.CUSIP",
        "TR.ExchangeName",
        "TR.AssetCategory",
        "TR.GICSIndustryCode",
        "TR.TRBCEconomicSector",
    ]


# --------------------------------------------------------------------------- #
# Session management
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
# Universe
# --------------------------------------------------------------------------- #
def build_universe(cfg: Config) -> pd.DataFrame:
    import refinitiv.data as rd
    from refinitiv.data.discovery import Screener

    mcap_buckets = [
        (100,    200),
        (200,    400),
        (400,    800),
        (800,    1600),
        (1600,   3200),
        (3200,   6400),
        (6400,   12800),
        (12800,  25600),
        (25600,  51200),
        (51200,  10**9),
    ]

    fields = [
        "TR.CommonName",
        "TR.ExchangeName",
        "TR.CompanyMarketCap",
        "TR.ISIN",
        "TR.CUSIP",
    ]

    all_dfs = []
    for (lo, hi) in mcap_buckets:
        query_str = (
            'U(IN(Equity(active,public,primary))/*UNV:Public*/), '
            f'TR.CompanyMarketCap(Scale=6)>={lo}, '
            f'TR.CompanyMarketCap(Scale=6)<{hi}, '
            'TR.HQCountryCode="US", '
            'CURN=USD'
        )
        try:
            screener = Screener(query_str)
            df_b = rd.get_data(universe=screener, fields=fields)
            n = 0 if df_b is None else len(df_b)
            print(f"  bucket [{lo:>6}, {hi:>10}): {n} instruments")
            if df_b is not None and not df_b.empty:
                all_dfs.append(df_b)
        except Exception as e:
            print(f"  bucket [{lo}, {hi}) FAILED: {e}")
        time.sleep(0.5)

    if not all_dfs:
        print("Screener returned no data.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["Instrument"]).reset_index(drop=True)

    rename = {
        "Instrument": "ric",
        "Company Name": "name",
        "Company Common Name": "name",
        "Exchange Name": "exchange",
        "Company Market Cap": "mcap",
        "ISIN": "isin",
        "CUSIP": "cusip",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "ric" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ric"})

    df = df.dropna(subset=["ric"]).reset_index(drop=True)

    print(f"\nTotal before exchange filter: {len(df)} instruments")
    if "exchange" in df.columns:
        vc = df["exchange"].value_counts()
        print(f"  exchange names (top 10):")
        for name, count in vc.head(10).items():
            print(f"    {count:>6d}  {name}")

    if "exchange" in df.columns:
        keep = (
            df["exchange"].str.contains("New York", case=False, na=False) |
            df["exchange"].str.contains("Nasdaq", case=False, na=False) |
            df["exchange"].str.contains("American Stock", case=False, na=False)
        )
        df = df[keep].reset_index(drop=True)

    print(f"\nTotal universe after filter: {len(df)} instruments")
    if "exchange" in df.columns:
        print(f"  exchanges: {df['exchange'].value_counts().to_dict()}")
    return df


# --------------------------------------------------------------------------- #
# Pricing history
# --------------------------------------------------------------------------- #
def _pull_history_batch(rd, rics: List[str], cfg: Config) -> pd.DataFrame:
    """
    Returns long-form DataFrame with columns:
      date, ric,
      Total Return, Price Close, Company Market Cap, Outstanding Shares

    Numeric columns are cast to float64 before returning so that the
    downstream Parquet write does not fail on empty strings.
    """
    df = rd.get_history(
        universe=rics,
        fields=cfg.PRICING_FIELDS,
        interval="monthly",
        start=cfg.START_DATE,
        end=cfg.END_DATE,
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
        index=["date", "ric"],
        columns="field",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Cast numeric columns to float64 to strip empty strings
    numeric_cols = ["Total Return", "Price Close",
                    "Company Market Cap", "Outstanding Shares"]
    for c in numeric_cols:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float64")

    return wide


def pull_pricing(cfg, universe, save_path=None):
    import refinitiv.data as rd
    rics = universe["ric"].tolist()
    n_batches = math.ceil(len(rics) / cfg.BATCH_SIZE)
    print(f"Pulling pricing for {len(rics)} RICs in {n_batches} batches...")

    pieces = []
    for i in range(n_batches):
        batch = rics[i * cfg.BATCH_SIZE:(i + 1) * cfg.BATCH_SIZE]
        t0 = time.time()
        try:
            df_b = _pull_history_batch(rd, batch, cfg)
            if not df_b.empty:
                pieces.append(df_b)
            print(f"  batch {i+1}/{n_batches}: {len(batch)} RICs -> "
                  f"{len(df_b)} rows ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  batch {i+1}/{n_batches} FAILED: {e}")
        time.sleep(cfg.SLEEP_SECONDS)

    if not pieces:
        print("No pricing data returned.")
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True)
    if save_path:
        full.to_parquet(save_path, index=False)
        print(f"Saved {len(full)} rows to {save_path}")
    return full


# --------------------------------------------------------------------------- #
# Reference snapshot
# --------------------------------------------------------------------------- #
def pull_reference(cfg, universe, save_path=None):
    import refinitiv.data as rd
    rics = universe["ric"].tolist()
    n_batches = math.ceil(len(rics) / cfg.BATCH_SIZE)
    print(f"Pulling reference for {len(rics)} RICs in {n_batches} batches...")
    pieces = []
    for i in range(n_batches):
        batch = rics[i * cfg.BATCH_SIZE:(i + 1) * cfg.BATCH_SIZE]
        try:
            df_b = rd.get_data(universe=batch, fields=cfg.REFERENCE_FIELDS)
            rename = {
                "Instrument": "ric",
                "ISIN": "isin",
                "CUSIP": "cusip",
                "Exchange Name": "exchange",
                "Asset Category": "asset_category",
                "GICS Industry Code": "gics",
                "TRBC Economic Sector": "trbc_sector",
            }
            df_b = df_b.rename(columns={k: v for k, v in rename.items()
                                        if k in df_b.columns})
            # Force string columns so Parquet does not choke on mixed dtypes
            for c in ["ric", "isin", "cusip", "exchange",
                      "asset_category", "gics", "trbc_sector"]:
                if c in df_b.columns:
                    df_b[c] = df_b[c].astype("string")
            pieces.append(df_b)
            print(f"  reference batch {i+1}/{n_batches}: {len(df_b)} rows")
        except Exception as e:
            print(f"  reference batch {i+1}/{n_batches} FAILED: {e}")
        time.sleep(cfg.SLEEP_SECONDS)
    if not pieces:
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True)
    if save_path:
        full.to_parquet(save_path, index=False)
        print(f"Saved reference to {save_path}")
    return full


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_pricing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Total Return":       "ret_raw",
        "Price Close":        "price",
        "Company Market Cap": "mcap",
        "Outstanding Shares": "shares",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                            if k in df.columns})

    for col in ["ret_raw", "mcap"]:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found. "
                f"Available: {df.columns.tolist()}")

    for col in ["ret_raw", "mcap", "price", "shares"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    n0 = len(df)
    df = df.dropna(subset=["ret_raw", "mcap"])
    n1 = len(df)
    df = df[df["mcap"] > 0]
    n2 = len(df)
    print(f"Cleaning: {n0} -> {n1} (dropna) -> {n2} (mcap > 0)")

    if df["ret_raw"].abs().median() > 1.0:
        df["ret"] = df["ret_raw"] / 100.0
        print(f"  converted Total Return from percent to decimal")
    else:
        df["ret"] = df["ret_raw"]

    n3 = len(df)
    df = df[df["ret"].abs() < 1.0]
    n4 = len(df)
    print(f"  dropped {n3 - n4} rows with |ret| >= 1.0")

    df = df.rename(columns={"ric": "permno"})
    out = df[["date", "permno", "ret", "mcap"]].copy()
    out["permno"] = out["permno"].astype("string")
    out["ret"] = out["ret"].astype("float64")
    out["mcap"] = out["mcap"].astype("float64")
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(mode: str = "smoke"):
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    print(f"Mode: {mode}")
    print(f"Log file: {_LOG_PATH}")

    open_session()
    try:
        # ---------------- SMOKE ----------------
        if mode == "smoke":
            import refinitiv.data as rd
            rics = ["AAPL.O", "MSFT.O", "JPM.N", "XOM.N", "PG.N"]

            df_raw = rd.get_history(
                universe=rics,
                fields=Config.PRICING_FIELDS,
                interval="monthly",
                start="2020-01-01",
                end="2021-12-31",
            )
            print("\n=== Raw get_history output ===")
            print(df_raw.head(3))
            print(f"Columns: {df_raw.columns.tolist()}")
            print(f"Shape: {df_raw.shape}")

            print("\n=== Reshaped via _pull_history_batch ===")
            df_wide = _pull_history_batch(rd, rics, Config)
            print(df_wide.head(10))
            print(f"Reshaped columns: {df_wide.columns.tolist()}")
            print(f"Reshaped shape: {df_wide.shape}")
            print(f"Reshaped dtypes:\n{df_wide.dtypes}")

            print("\n=== After clean_pricing ===")
            df_clean = clean_pricing(df_wide, Config)
            print(df_clean.head(10))
            print(f"Clean columns: {df_clean.columns.tolist()}")
            print(f"Clean shape: {df_clean.shape}")
            return

        # ---------------- UNIVERSE ----------------
        if mode == "universe":
            df_u = build_universe(Config)
            if df_u.empty:
                print("Universe is empty; not saving.")
                return
            path = os.path.join(Config.OUT_DIR, "universe.parquet")
            df_u.to_parquet(path, index=False)
            print(f"Saved universe to {path}")
            return

        # ---------------- FULL ----------------
        if mode == "full":
            u_path = os.path.join(Config.OUT_DIR, "universe.parquet")
            if not os.path.exists(u_path):
                raise FileNotFoundError(
                    f"Run mode='universe' first to create {u_path}")
            universe = pd.read_parquet(u_path)
            print(f"Loaded universe: {len(universe)} RICs")

            p_path = os.path.join(Config.OUT_DIR, "pricing_raw.parquet")
            if not os.path.exists(p_path):
                raw = pull_pricing(Config, universe, save_path=p_path)
            else:
                print(f"Using cached pricing from {p_path}")
                raw = pd.read_parquet(p_path)
            if raw.empty:
                print("No pricing data; aborting.")
                return

            r_path = os.path.join(Config.OUT_DIR, "reference.parquet")
            if not os.path.exists(r_path):
                ref = pull_reference(Config, universe, save_path=r_path)
            else:
                print(f"Using cached reference from {r_path}")
                ref = pd.read_parquet(r_path)

            if not ref.empty and "ric" in ref.columns:
                merge_cols = [c for c in ["ric", "isin", "cusip"]
                              if c in ref.columns]
                raw = raw.merge(
                    ref[merge_cols].drop_duplicates("ric"),
                    on="ric", how="left")

            clean = clean_pricing(raw, Config)
            c_path = os.path.join(Config.OUT_DIR, "panel_clean.parquet")
            clean.to_parquet(c_path, index=False)
            print(f"Saved cleaned panel to {c_path}")
            print(f"  rows: {len(clean)}, "
                  f"permnos: {clean['permno'].nunique()}, "
                  f"dates: {clean['date'].nunique()}")
            return

        raise ValueError(f"Unknown mode: {mode}")
    finally:
        close_session()


if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    try:
        main(mode_arg)
    finally:
        _tee.flush()
        # Do not close the log file explicitly; Python will close it at exit