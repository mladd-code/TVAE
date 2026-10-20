"""
run_tvaec.py
------------
Runs TV-AE-C on the sliced Refinitiv panel, alongside EW and PCA
benchmarks, and reports Total and Predictive R².
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch

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

_tee = _Tee(open("run_tvaec_output.txt", "w", encoding="utf-8"))
sys.stdout = _tee
sys.stderr = _tee

import warnings
warnings.filterwarnings("ignore")

from tvaec_app1 import prepare_data, rolling_fit_tvaec, CHAR_COLS


# ------------------------- Config -------------------------
PANEL_PATH = "refinitiv_data/panel_clean.parquet"
N_TOP_PERMNOS = 500
DATE_START = "2010-01-01"
DATE_END   = "2024-12-31"

K1 = 3
ENCODER_WIDTH = 32
BETA_WIDTH    = 32
N_SEEDS = 3
N_EPOCHS = 300
LR = 1e-2
WEIGHT_DECAY = 1e-4
PATIENCE = 30

VAL_MONTHS = 12
PURGE_MONTHS = 1
EMBARGO_MONTHS = 1
OOS_START_YEAR = 2016
OOS_END_YEAR   = 2024


# ------------------------- Benchmarks -------------------------
def rolling_fit_ew(df, oos_start_year, oos_end_year):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    mkt = df.groupby('date')['ret'].mean().sort_index()
    rows = []
    for d in sorted(df['date'].unique()):
        yr = pd.Timestamp(d).year
        if yr < oos_start_year or yr > oos_end_year:
            continue
        prev = mkt[mkt.index < d]
        lam = prev.mean() if len(prev) else 0.0
        mkt_t = mkt.loc[d] if d in mkt.index else 0.0
        sub = df[df['date'] == d]
        for _, r in sub.iterrows():
            rows.append((d, int(r['permno']), float(lam), float(mkt_t)))
    return pd.DataFrame(rows, columns=['date','permno','pred_ret','fitted_ret'])


def rolling_fit_pca(df, N_max, K1, oos_start_year, oos_end_year,
                    val_months=12, purge_months=1, embargo_months=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df_train = df[df['date'] < pd.Timestamp(f"{oos_start_year}-01-01")]
    avg_mcap = (df_train.groupby('permno')['mcap'].mean()
                .sort_values(ascending=False))
    top_permnos = avg_mcap.head(N_max).index.tolist()
    permno_map = {p: i for i, p in enumerate(top_permnos)}

    rows = []
    for year in range(oos_start_year, oos_end_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        val_end = test_start - pd.DateOffset(months=embargo_months + 1)
        val_start = val_end - pd.DateOffset(months=val_months - 1)
        train_end = val_start - pd.DateOffset(months=purge_months + 1)

        train_df = df[df['date'] <= train_end]
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Build dense arrays
        train_dates = np.sort(train_df['date'].unique())
        test_dates = np.sort(test_df['date'].unique())
        N = N_max
        Xtr = np.zeros((len(train_dates), N), dtype=np.float32)
        Mtr = np.zeros((len(train_dates), N), dtype=np.float32)
        Xte = np.zeros((len(test_dates), N), dtype=np.float32)
        Mte = np.zeros((len(test_dates), N), dtype=np.float32)
        idx_te = [None] * len(test_dates)

        def fill(dfsub, Xarr, Marr, idx_arr=None, keep_idx=False):
            by = {d: g for d, g in dfsub.groupby('date')}
            for t, d in enumerate(np.sort(dfsub['date'].unique())):
                sub = by[d]
                sub = sub[sub['permno'].isin(permno_map)]
                if len(sub) == 0: continue
                slots = np.array([permno_map[p] for p in sub['permno'].values],
                                 dtype=np.int64)
                Xarr[t, slots] = sub['ret'].values
                Marr[t, slots] = 1.0
                if keep_idx and idx_arr is not None:
                    idx_arr[t] = slots

        fill(train_df, Xtr, Mtr)
        fill(test_df, Xte, Mte, idx_te, keep_idx=True)

        # Per-stock training mean
        mask_sum = Mtr.sum(axis=0)
        mask_sum = np.where(mask_sum > 0, mask_sum, 1.0)
        mu = (Xtr * Mtr).sum(axis=0) / mask_sum

        # SVD on demeaned returns
        Xc = (Xtr - mu[None, :]) * Mtr
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        F_tr = U[:, :K1] * S[:K1]
        Lambda = Vt[:K1, :].T

        # Factors on test data
        F_te = (Xte - mu[None, :]) @ Lambda

        # Expanding mean over F_tr ++ F_te
        F_all = np.vstack([F_tr, F_te])
        F_cummean = (np.cumsum(F_all, axis=0)
                     / np.arange(1, F_all.shape[0] + 1)[:, None])

        T_train = F_tr.shape[0]
        for k, d in enumerate(test_dates):
            if idx_te[k] is None: continue
            i_all = T_train + k
            lam = F_cummean[i_all - 1] if i_all > 0 else np.zeros(K1)
            F_t = F_te[k]
            slots = idx_te[k]
            r_pred = mu[slots] + Lambda[slots] @ lam
            r_fit  = mu[slots] + Lambda[slots] @ F_t
            slot_permnos = [top_permnos[j] for j in slots]
            for j, p in enumerate(slot_permnos):
                rows.append((d, p, float(r_pred[j]), float(r_fit[j])))

    return pd.DataFrame(rows, columns=['date','permno','pred_ret','fitted_ret'])


# ------------------------- Evaluation -------------------------
def compute_r2(pred, true):
    m = pred.merge(true, on=['date','permno'], how='inner')
    if len(m) == 0:
        return {"n": 0, "total_r2": np.nan, "pred_r2": np.nan}
    sse_tot = ((m['ret'] - m['fitted_ret'])**2).sum()
    sse_pred = ((m['ret'] - m['pred_ret'])**2).sum()
    sst = (m['ret']**2).sum()
    return {"n": len(m),
            "total_r2": 1 - sse_tot/sst,
            "pred_r2":  1 - sse_pred/sst}


def summarize(results, true_df):
    print("\n" + "="*76)
    print(f"{'Model':<12} {'N':>10} {'Total R^2':>18} {'Predictive R^2':>18}")
    print("-"*76)
    for name, pred in results.items():
        s = compute_r2(pred, true_df)
        print(f"{name:<12} {s['n']:>10} "
              f"{s['total_r2']:>18.6f} {s['pred_r2']:>18.6f}")

    print("\nPredictive R^2 by year:")
    years = sorted(true_df['date'].dt.year.unique())
    years = [y for y in years if y >= OOS_START_YEAR]
    header = f"{'Year':<6} " + " ".join(f"{n:>10}" for n in results)
    print(header)
    print("-"*len(header))
    for y in years:
        row = f"{y:<6} "
        for name, pred in results.items():
            sub_pred = pred[pred['date'].dt.year == y]
            sub_true = true_df[true_df['date'].dt.year == y]
            s = compute_r2(sub_pred, sub_true)
            row += f"{s['pred_r2']:>10.5f} "
        print(row)


# ------------------------- Main -------------------------
def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Slice: top {N_TOP_PERMNOS} permnos, "
          f"{DATE_START} to {DATE_END}")

    df_raw = pd.read_parquet(PANEL_PATH)
    df, permno_codes = prepare_data(
        df_raw, n_top=N_TOP_PERMNOS,
        date_start=DATE_START, date_end=DATE_END)
    N_max = len(permno_codes)
    print(f"Sliced panel: {len(df)} rows, {N_max} permnos, "
          f"{df['date'].nunique()} dates")

    true_df = df[['date', 'permno', 'ret']].copy()

    results = {}

    print("\n" + "="*60)
    print("Running EW benchmark")
    print("="*60)
    results['EW'] = rolling_fit_ew(df, OOS_START_YEAR, OOS_END_YEAR)

    print("\n" + "="*60)
    print("Running PCA benchmark")
    print("="*60)
    results['PCA'] = rolling_fit_pca(
        df, N_max, K1, OOS_START_YEAR, OOS_END_YEAR,
        VAL_MONTHS, PURGE_MONTHS, EMBARGO_MONTHS)

    print("\n" + "="*60)
    print("Running TV-AE-C")
    print("="*60)
    res = rolling_fit_tvaec(
        df, N_max=N_max, K1=K1, char_cols=CHAR_COLS,
        encoder_width=ENCODER_WIDTH, beta_width=BETA_WIDTH,
        n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
        lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
        val_months=VAL_MONTHS,
        purge_months=PURGE_MONTHS, embargo_months=EMBARGO_MONTHS,
        oos_start_year=OOS_START_YEAR, oos_end_year=OOS_END_YEAR,
        device=device, verbose=True)
    results['TVAEC'] = res['X_hat']

    # Factor diagnostics
    fac = res['factors']
    print(f"\nFactor diagnostics:")
    for c in ['F_1', 'F_2', 'F_3']:
        if c in fac.columns:
            print(f"  std({c}) = {fac[c].std():.6f}  "
                  f"mean({c}) = {fac[c].mean():.6f}")

    summarize(results, true_df)

    for name, pred in results.items():
        pred.to_parquet(f"app1_oos_{name}.parquet", index=False)
    print(f"\nSaved to app1_oos_*.parquet")
    print(f"Total time: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    finally:
        _tee.flush()