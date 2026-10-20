"""
pull_crypto.py  (v5)
====================
Pulls daily mid-price history for crypto pairs, one RIC at a time.

Why one at a time
-----------------
Batching crypto RICs with mixed suffix conventions (=BTSP, =CCCL, bare =)
often fails with "Unable to resolve all requested identifiers" even
though each RIC resolves alone. Batching also fails entirely when any
single RIC in the batch is unentitled or delisted. Individual pulls
avoid both problems.

Progress is saved incrementally to crypto_partial.parquet after every
successful pull, so rerunning the script resumes where it left off.

Modes
-----
    python pull_crypto.py pull
    python pull_crypto.py clean
"""

from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

class Config:
    OUT_DIR = os.path.abspath("refinitiv_data")
    RIC_MAP = "crypto_ric_map.json"
    START_DATE = "2015-01-01"
    END_DATE   = "2024-12-31"
    SLEEP_SECONDS = 1.0
    PRICING_FIELDS = ["TR.MidPrice"]


def load_ric_map():
    if not os.path.exists(Config.RIC_MAP):
        raise FileNotFoundError(
            f"{Config.RIC_MAP} not found. Run diag_crypto_rics_full.py first.")
    with open(Config.RIC_MAP) as f:
        data = json.load(f)
    resolved = data["resolved"]
    print(f"Loaded {len(resolved)} resolved RICs")
    return resolved


def open_session():
    import refinitiv.data as rd
    rd.open_session()
    print("Session opened.")
    return rd


def close_session():
    import refinitiv.data as rd
    try: rd.close_session()
    except Exception: pass


def _pull_one(rd, ric):
    """Pull a single RIC. Returns a DataFrame or None on failure."""
    df = rd.get_history(
        universe=[ric],
        fields=Config.PRICING_FIELDS,
        interval="daily",
        start=Config.START_DATE, end=Config.END_DATE)
    if df is None or df.empty:
        return None

    # Column layout: for a single RIC, columns are the field display names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{ric}|{field}" for ric, field in df.columns]
    else:
        df.columns = [f"{ric}|{c}" for c in df.columns]

    df = df.reset_index()
    for cand in ("Timestamp", "index", "Date"):
        if cand in df.columns:
            df = df.rename(columns={cand: "date"})
            break
    df["date"] = pd.to_datetime(df["date"])

    # Locate the Mid Price column
    mid_col = None
    for c in df.columns:
        if "Mid Price" in str(c):
            mid_col = c
            break
    if mid_col is None:
        return None

    out = pd.DataFrame({
        "date": df["date"],
        "ric": ric,
        "price": pd.to_numeric(df[mid_col], errors="coerce"),
    })
    return out


def pull():
    import refinitiv.data as rd
    ric_map = load_ric_map()
    rics = list(ric_map.values())

    partial_path = os.path.join(Config.OUT_DIR, "crypto_partial.parquet")
    done = set()
    pieces = []
    if os.path.exists(partial_path):
        prev = pd.read_parquet(partial_path)
        done = set(prev["ric"].unique())
        pieces.append(prev)
        print(f"Resuming: {len(done)} RICs already pulled")

    todo = [r for r in rics if r not in done]
    print(f"Pulling {len(todo)} RICs one at a time...")
    n_ok, n_fail = 0, 0
    t0 = time.time()

    for i, ric in enumerate(todo):
        try:
            df = _pull_one(rd, ric)
            if df is not None and not df.empty:
                pieces.append(df)
                n_ok += 1
                # Save progress after every success
                pd.concat(pieces, ignore_index=True).to_parquet(
                    partial_path, index=False)
                n_rows = len(df)
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(todo)}] {ric:<15s} -> "
                      f"{n_rows} rows ({elapsed:.0f}s elapsed)")
            else:
                n_fail += 1
                print(f"  [{i+1}/{len(todo)}] {ric:<15s} -> empty")
        except Exception as e:
            n_fail += 1
            print(f"  [{i+1}/{len(todo)}] {ric:<15s} -> "
                  f"FAILED: {str(e)[:80]}")
        time.sleep(Config.SLEEP_SECONDS)

    if not pieces:
        print("No data pulled.")
        return None

    full = pd.concat(pieces, ignore_index=True).drop_duplicates(
        subset=["date", "ric"], keep="last")

    path = os.path.join(Config.OUT_DIR, "crypto_raw.parquet")
    full.to_parquet(path, index=False)
    print(f"\nSaved to {path}: {len(full)} rows, "
          f"{full['ric'].nunique()} unique RICs "
          f"(ok={n_ok}, failed={n_fail})")

    try:
        os.remove(partial_path)
    except Exception:
        pass
    return full


def clean():
    path = os.path.join(Config.OUT_DIR, "crypto_raw.parquet")
    if not os.path.exists(path):
        print(f"{path} not found. Run 'pull' first.")
        return
    df = pd.read_parquet(path).copy()
    df = df.sort_values(["ric", "date"]).copy()
    df["ret"] = df.groupby("ric")["price"].pct_change(fill_method=None)
    df["mcap"] = 1.0

    n0 = len(df)
    df = df.dropna(subset=["ret"]).copy()
    df = df[df["ret"].abs() < 3.0]
    n1 = len(df)
    print(f"Cleaning: {n0} -> {n1} rows "
          f"({100*(n0-n1)/max(n0,1):.2f}% dropped)")

    cov = df.groupby("date").size()
    print(f"  rows per date: min={cov.min()}, median={int(cov.median())}, "
          f"max={cov.max()}")
    for t in (15, 20, 25, 30, 35):
        if (cov >= t).any():
            first = cov[cov >= t].index.min()
            print(f"  first date with >= {t} coins: {first.date()}")
        else:
            print(f"  first date with >= {t} coins: N/A")

    df = df.rename(columns={"ric": "permno"})
    out = df[["date", "permno", "ret", "mcap"]].copy()
    out["permno"] = out["permno"].astype("string")
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    out_path = os.path.join(Config.OUT_DIR, "panel_daily_crypto.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"  rows: {len(out)}, permnos: {out['permno'].nunique()}, "
          f"dates: {out['date'].nunique()}")


def main(mode="pull"):
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    print(f"Mode: {mode}")
    if mode == "clean":
        clean(); return
    open_session()
    try:
        if mode == "pull":
            pull()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    finally:
        close_session()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pull")