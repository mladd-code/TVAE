"""
run_app1_slice.py  (v4)
-----------------------
Four-model comparison on the sliced Refinitiv panel.

All four models use per-stock training-sample mean demeaning:

  1. Equal-weighted market  (trivial benchmark)
  2. Static PCA             (Bai 2003 / SW 2017 linear benchmark)
  3. Static AE              (SX-style nonlinear, no time input)
  4. TV-AE-G                (this paper)

The per-stock mean mu_i is computed on the training sample of each
rolling fold, subtracted from both the encoder input and the target,
and added back at prediction time. This forces the neural models to
learn factor structure rather than collapse to per-stock constants.

Total R^2      : uses contemporaneous factor fit  r_hat = mu_i + beta_t' F_t
Predictive R^2 : uses expanding-mean risk premium r_hat = mu_i + beta_t' lambda_t
"""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Tee logging
# --------------------------------------------------------------------------- #
class _Tee:
    def __init__(self, f):
        self.f = f
    def write(self, d):
        sys.__stdout__.write(d)
        try:
            self.f.write(d)
        except Exception:
            pass
    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        try:
            self.f.flush()
        except Exception:
            pass

_tee = _Tee(open("run_app1_slice_output.txt", "w", encoding="utf-8"))
sys.stdout = _tee
sys.stderr = _tee

import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Imports from the model file
# --------------------------------------------------------------------------- #
from tvae_global_app_1 import (
    build_panel, MonthlyPanel, TVEGApp1, _train_one, predict_for_month,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PANEL_PATH = "refinitiv_data/panel_clean.parquet"

N_TOP_PERMNOS = 500
DATE_START = "2010-01-01"
DATE_END   = "2024-12-31"

K1 = 3
ENCODER_WIDTH = 16      # was 64
BETA_WIDTH = 8          # was 16
N_SEEDS = 3             # was 2
N_EPOCHS = 500          # was 200
LR = 1e-2               # was 1e-3
WEIGHT_DECAY = 1e-4     # was 1e-3
PATIENCE = 50           # was 30

INITIAL_TRAIN_YEARS = 6
VAL_MONTHS = 12
PURGE_MONTHS = 1
EMBARGO_MONTHS = 1
OOS_START_YEAR = 2016
OOS_END_YEAR   = 2024


# --------------------------------------------------------------------------- #
# Static AE (no time input, no free alpha)
# --------------------------------------------------------------------------- #
class StaticApp1(nn.Module):
    """
    Same as TVEGApp1 but without the tau input. No free intercept;
    demeaning by mu_i happens outside the model.

    Encoder : Linear(2*N_max, encoder_width) -> ReLU -> Linear(., K1)
    Beta    : per-permno constant vector  beta_i in R^{K1}
    """
    def __init__(self, N_max: int, N_total: int, K1: int,
                 encoder_width: int = 64, beta_width: int = 16):
        super().__init__()
        self.N_max = N_max
        self.N_total = N_total
        self.K1 = K1

        self.encoder = nn.Sequential(
            nn.Linear(2 * N_max, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )

        self.beta_vec = nn.Parameter(torch.zeros(N_total, K1))
        nn.init.normal_(self.beta_vec, std=0.01)

    def encode(self, X_pad: torch.Tensor, M_pad: torch.Tensor,
               tau: torch.Tensor = None) -> torch.Tensor:
        inp = torch.cat([X_pad * M_pad, M_pad])
        return self.encoder(inp)

    def beta(self, permno_idx: torch.Tensor,
             tau: torch.Tensor = None) -> torch.Tensor:
        return self.beta_vec[permno_idx]

    def forward(self, X_pad, M_pad, permno_idx, tau):
        F_t = self.encode(X_pad, M_pad)
        beta = self.beta(permno_idx)
        return beta @ F_t, F_t, beta


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #
def slice_panel(panel_path: str, n_top: int,
                date_start: str, date_end: str):
    df = pd.read_parquet(panel_path)
    print(f"Full panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")

    df = df[(df["date"] >= date_start) & (df["date"] <= date_end)].copy()
    print(f"After date filter: {len(df)} rows, "
          f"{df['permno'].nunique()} permnos")

    avg_mcap = df.groupby("permno")["mcap"].mean().sort_values(ascending=False)
    top = avg_mcap.head(n_top).index.tolist()
    df = df[df["permno"].isin(top)].copy()
    print(f"After top-{n_top} filter: {len(df)} rows, "
          f"{df['permno'].nunique()} permnos")

    coverage = df.groupby("date").size()
    print(f"Rows per month: min={coverage.min()}, "
          f"median={int(coverage.median())}, max={coverage.max()}")

    permno_codes = {p: i for i, p in enumerate(sorted(df["permno"].unique()))}
    df["permno"] = df["permno"].map(permno_codes).astype("int64")

    df["mcap"] = df["mcap"] / 1e6

    df = df[["date", "permno", "ret", "mcap"]].copy()
    df["ret"] = df["ret"].astype("float64")
    df["mcap"] = df["mcap"].astype("float64")

    return df, permno_codes


# --------------------------------------------------------------------------- #
# Generic rolling-fit for the neural models (with per-stock demeaning)
# --------------------------------------------------------------------------- #
def rolling_fit_generic(df: pd.DataFrame,
                        model_ctor,
                        N_max: int,
                        K1: int,
                        encoder_width: int,
                        beta_width: int,
                        n_seeds: int,
                        n_epochs: int,
                        lr: float,
                        weight_decay: float,
                        patience: int,
                        initial_train_years: int,
                        val_months: int,
                        purge_months: int,
                        embargo_months: int,
                        oos_start_year: int,
                        oos_end_year: int,
                        device: str = 'cpu',
                        verbose: bool = True,
                        label: str = 'model') -> pd.DataFrame:
    """
    Rolling annual refit with per-stock demeaning.

    Each fold:
        1. Compute mu_i = mean return of stock i in the training sample.
        2. Subtract mu_i from X_pad and y in every split (train, val, test).
        3. Train the model on the demeaned panel.
        4. Predict on test months, then add mu_i back.

    Returns DataFrame with columns
        ['date', 'permno', 'pred_ret', 'fitted_ret'].
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Global permno map from the pre-OOS training sample
    df_train_all = df[df['date'] < pd.Timestamp(f"{oos_start_year}-01-01")]
    avg_mcap = (df_train_all.groupby('permno')['mcap'].mean()
                .sort_values(ascending=False))
    top_permnos = avg_mcap.head(N_max).index.tolist()
    permno_map = {p: i for i, p in enumerate(top_permnos)}
    N_total = len(permno_map)
    if verbose:
        print(f"  [{label}] Permno map: N_total={N_total}, "
              f"cutoff={avg_mcap.iloc[N_total-1]:.2f}")

    all_dates = np.sort(df['date'].unique())
    T_all = len(all_dates)
    date_to_tau = {d: i / (T_all - 1) for i, d in enumerate(all_dates)}

    oos_rows = []

    for year in range(oos_start_year, oos_end_year + 1):
        if verbose:
            print(f"\n  [{label}] Year {year}")
        t0 = time.time()

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        val_end = test_start - pd.DateOffset(months=embargo_months + 1)
        val_start = val_end - pd.DateOffset(months=val_months - 1)
        train_end = val_start - pd.DateOffset(months=purge_months + 1)

        train_df = df[df['date'] <= train_end]
        val_df   = df[(df['date'] >= val_start) & (df['date'] <= val_end)]
        test_df  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            if verbose:
                print(f"    skipping year {year}: empty split")
            continue

        # Per-stock mean from training sample
        mu_arr = np.zeros(N_max, dtype=np.float32)
        for slot, p in enumerate(top_permnos):
            sub = train_df[train_df['permno'] == p]
            if len(sub) > 0:
                mu_arr[slot] = sub['ret'].mean()

        Xp_tr, Mp_tr, idx_tr, _, y_tr, w_tr, dates_tr = build_panel(
            train_df, permno_map, N_max, mu_arr=mu_arr)
        Xp_va, Mp_va, idx_va, _, y_va, w_va, dates_va = build_panel(
            val_df, permno_map, N_max, mu_arr=mu_arr)
        Xp_te, Mp_te, idx_te, _, _, _, dates_te = build_panel(
            test_df, permno_map, N_max, mu_arr=mu_arr)

        tau_tr = np.array([date_to_tau[d] for d in dates_tr], dtype=np.float32)
        tau_va = np.array([date_to_tau[d] for d in dates_va], dtype=np.float32)
        tau_te = np.array([date_to_tau[d] for d in dates_te], dtype=np.float32)

        train_ds = MonthlyPanel(Xp_tr, Mp_tr, idx_tr, tau_tr, y_tr, w_tr)
        val_ds   = MonthlyPanel(Xp_va, Mp_va, idx_va, tau_va, y_va, w_va)

        hist_df = df[df['date'] <= test_end]
        Xp_h, Mp_h, _, _, _, _, dates_h = build_panel(
            hist_df, permno_map, N_max, mu_arr=mu_arr)
        tau_h = np.array([date_to_tau[d] for d in dates_h], dtype=np.float32)
        date_to_idx = {d: i for i, d in enumerate(dates_h)}

        F_seeds = []
        B_seeds = [None] * len(dates_te)
        P_seeds = [None] * len(dates_te)   # pred (demeaned)
        Q_seeds = [None] * len(dates_te)   # fit (demeaned)

        for seed in range(n_seeds):
            model = model_ctor(N_max, N_total, K1, encoder_width, beta_width)
            model, best_val = _train_one(
                model, train_ds, val_ds,
                n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay,
                patience=patience, device=device, seed=seed,
                value_weighted=False, verbose=False)

            model.eval()
            F_hist = np.zeros((len(dates_h), K1), dtype=np.float64)
            with torch.no_grad():
                for i in range(len(dates_h)):
                    Xp_t = torch.tensor(Xp_h[i],
                                        dtype=torch.float32).to(device)
                    Mp_t = torch.tensor(Mp_h[i],
                                        dtype=torch.float32).to(device)
                    tau_t = torch.tensor(tau_h[i],
                                         dtype=torch.float32).to(device)
                    F_hist[i] = model.encode(Xp_t, Mp_t, tau_t).cpu().numpy()
            F_seeds.append(F_hist)

            with torch.no_grad():
                for k, d in enumerate(dates_te):
                    if len(idx_te[k]) == 0:
                        continue
                    i_h = date_to_idx[d]
                    lambda_t = torch.tensor(
                        np.cumsum(F_hist, axis=0)[i_h]
                        / (i_h + 1),
                        dtype=torch.float32).to(device)

                    Xp_t = torch.tensor(Xp_te[k],
                                        dtype=torch.float32).to(device)
                    Mp_t = torch.tensor(Mp_te[k],
                                        dtype=torch.float32).to(device)
                    idx_t = torch.tensor(idx_te[k],
                                         dtype=torch.long).to(device)
                    tau_t = torch.tensor(tau_te[k],
                                         dtype=torch.float32).to(device)

                    r_pred, r_fit, F_t, beta = predict_for_month(
                        model, Xp_t, Mp_t, idx_t, tau_t, lambda_t)

                    if B_seeds[k] is None:
                        B_seeds[k] = [beta.astype(np.float64)]
                        P_seeds[k] = [r_pred.astype(np.float64)]
                        Q_seeds[k] = [r_fit.astype(np.float64)]
                    else:
                        B_seeds[k].append(beta.astype(np.float64))
                        P_seeds[k].append(r_pred.astype(np.float64))
                        Q_seeds[k].append(r_fit.astype(np.float64))

        F_avg = np.mean(F_seeds, axis=0)
        F_cummean = (np.cumsum(F_avg, axis=0)
                     / np.arange(1, len(dates_h) + 1)[:, None])

        for k, d in enumerate(dates_te):
            if len(idx_te[k]) == 0 or B_seeds[k] is None:
                continue
            beta_t = np.mean(B_seeds[k], axis=0)             # (N_t, K1)
            pred_t = np.mean(P_seeds[k], axis=0)             # (N_t,)
            fit_t  = np.mean(Q_seeds[k], axis=0)             # (N_t,)

            # Add mu_i back
            mu_t = mu_arr[idx_te[k]]
            pred_t = pred_t + mu_t
            fit_t  = fit_t + mu_t

            slot_permnos = [top_permnos[j] for j in idx_te[k]]
            for j, p in enumerate(slot_permnos):
                oos_rows.append((d, p, float(pred_t[j]), float(fit_t[j])))

        if verbose:
            n_preds = sum(len(idx_te[k]) for k in range(len(dates_te)))
            print(f"    {n_preds} preds in {(time.time()-t0)/60:.2f} min")

    return pd.DataFrame(
        oos_rows, columns=['date', 'permno', 'pred_ret', 'fitted_ret'])


# --------------------------------------------------------------------------- #
# Static PCA benchmark (with per-stock demeaning)
# --------------------------------------------------------------------------- #
def rolling_fit_pca(df: pd.DataFrame,
                    N_max: int,
                    K1: int,
                    initial_train_years: int,
                    val_months: int,
                    purge_months: int,
                    embargo_months: int,
                    oos_start_year: int,
                    oos_end_year: int,
                    verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df_train_all = df[df['date'] < pd.Timestamp(f"{oos_start_year}-01-01")]
    avg_mcap = (df_train_all.groupby('permno')['mcap'].mean()
                .sort_values(ascending=False))
    top_permnos = avg_mcap.head(N_max).index.tolist()
    permno_map = {p: i for i, p in enumerate(top_permnos)}

    oos_rows = []

    for year in range(oos_start_year, oos_end_year + 1):
        if verbose:
            print(f"\n  [PCA] Year {year}")
        t0 = time.time()

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        val_end = test_start - pd.DateOffset(months=embargo_months + 1)
        val_start = val_end - pd.DateOffset(months=val_months - 1)
        train_end = val_start - pd.DateOffset(months=purge_months + 1)

        train_df = df[df['date'] <= train_end]
        test_df  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Per-stock mean from training sample
        mu_arr = np.zeros(N_max, dtype=np.float32)
        for slot, p in enumerate(top_permnos):
            sub = train_df[train_df['permno'] == p]
            if len(sub) > 0:
                mu_arr[slot] = sub['ret'].mean()

        Xp_tr, Mp_tr, _, _, _, _, dates_tr = build_panel(
            train_df, permno_map, N_max, mu_arr=mu_arr)
        Xp_te, Mp_te, idx_te, _, _, _, dates_te = build_panel(
            test_df, permno_map, N_max, mu_arr=mu_arr)

        # SVD on the demeaned training panel
        Xc = Xp_tr * Mp_tr
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        F_tr = U[:, :K1] * S[:K1]
        Lambda = Vt[:K1, :].T

        F_te = Xp_te @ Lambda

        F_all = np.vstack([F_tr, F_te])
        F_cummean = (np.cumsum(F_all, axis=0)
                     / np.arange(1, F_all.shape[0] + 1)[:, None])

        T_train = F_tr.shape[0]
        for k, d in enumerate(dates_te):
            if len(idx_te[k]) == 0:
                continue
            idx_all = T_train + k
            lam = F_cummean[idx_all - 1] if idx_all > 0 else np.zeros(K1)
            F_t = F_te[k]

            mu_t = mu_arr[idx_te[k]]
            r_pred = mu_t + Lambda[idx_te[k]] @ lam
            r_fit  = mu_t + Lambda[idx_te[k]] @ F_t

            slot_permnos = [top_permnos[j] for j in idx_te[k]]
            for j, p in enumerate(slot_permnos):
                oos_rows.append((d, p, float(r_pred[j]), float(r_fit[j])))

        if verbose:
            print(f"    {(time.time()-t0):.2f} s")

    return pd.DataFrame(
        oos_rows, columns=['date', 'permno', 'pred_ret', 'fitted_ret'])


# --------------------------------------------------------------------------- #
# Equal-weight market benchmark
# --------------------------------------------------------------------------- #
def rolling_fit_equal_weight(df: pd.DataFrame,
                             val_months: int,
                             purge_months: int,
                             embargo_months: int,
                             oos_start_year: int,
                             oos_end_year: int,
                             verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    mkt = df.groupby('date')['ret'].mean().sort_index()

    oos_rows = []
    for year in range(oos_start_year, oos_end_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

        for d in sorted(test_df['date'].unique()):
            prev = mkt[mkt.index < d]
            lam = prev.mean() if len(prev) else 0.0
            mkt_t = mkt.loc[d] if d in mkt.index else 0.0
            sub = test_df[test_df['date'] == d]
            for _, row in sub.iterrows():
                oos_rows.append((d, int(row['permno']),
                                 float(lam), float(mkt_t)))

    return pd.DataFrame(
        oos_rows, columns=['date', 'permno', 'pred_ret', 'fitted_ret'])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def compute_r2(pred_df: pd.DataFrame, true_df: pd.DataFrame):
    m = pred_df.merge(true_df, on=['date', 'permno'], how='inner')
    if len(m) == 0:
        return {"n": 0, "total_r2": np.nan, "pred_r2": np.nan}
    sse_tot = ((m['ret'] - m['fitted_ret']) ** 2).sum()
    sse_pred = ((m['ret'] - m['pred_ret']) ** 2).sum()
    sst = (m['ret'] ** 2).sum()
    return {
        "n": len(m),
        "total_r2": 1 - sse_tot / sst,
        "pred_r2": 1 - sse_pred / sst,
    }


def summarize(results: Dict[str, pd.DataFrame], true_df: pd.DataFrame):
    print("\n" + "=" * 76)
    print(f"{'Model':<12} {'N':>10} {'Total R^2':>18} {'Predictive R^2':>18}")
    print("-" * 76)
    for name, pred in results.items():
        s = compute_r2(pred, true_df)
        print(f"{name:<12} {s['n']:>10} "
              f"{s['total_r2']:>18.6f} {s['pred_r2']:>18.6f}")

    print("\nPredictive R^2 by year:")
    years = sorted(true_df['date'].dt.year.unique())
    years = [y for y in years if y >= OOS_START_YEAR]
    header = f"{'Year':<6} " + " ".join(f"{n:>10}" for n in results)
    print(header)
    print("-" * len(header))
    for y in years:
        row = f"{y:<6} "
        for name, pred in results.items():
            pred_y = pred[pred['date'].dt.year == y]
            true_y = true_df[true_df['date'].dt.year == y]
            s = compute_r2(pred_y, true_y)
            row += f"{s['pred_r2']:>10.5f} "
        print(row)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Slice: top {N_TOP_PERMNOS} permnos, {DATE_START} to {DATE_END}")

    df, permno_codes = slice_panel(
        PANEL_PATH, N_TOP_PERMNOS, DATE_START, DATE_END)
    N_max = len(permno_codes)

    true_df = df[['date', 'permno', 'ret']].copy()

    results = {}

    print("\n" + "=" * 60)
    print("Running Equal-Weight Market benchmark")
    print("=" * 60)
    results['EW'] = rolling_fit_equal_weight(
        df, VAL_MONTHS, PURGE_MONTHS, EMBARGO_MONTHS,
        OOS_START_YEAR, OOS_END_YEAR)

    print("\n" + "=" * 60)
    print("Running Static PCA benchmark")
    print("=" * 60)
    results['PCA'] = rolling_fit_pca(
        df, N_max, K1, INITIAL_TRAIN_YEARS, VAL_MONTHS,
        PURGE_MONTHS, EMBARGO_MONTHS,
        OOS_START_YEAR, OOS_END_YEAR)

    print("\n" + "=" * 60)
    print("Running Static AE benchmark")
    print("=" * 60)
    results['StaticAE'] = rolling_fit_generic(
        df, StaticApp1,
        N_max=N_max, K1=K1,
        encoder_width=ENCODER_WIDTH, beta_width=BETA_WIDTH,
        n_seeds=N_SEEDS, n_epochs=N_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE,
        initial_train_years=INITIAL_TRAIN_YEARS,
        val_months=VAL_MONTHS,
        purge_months=PURGE_MONTHS, embargo_months=EMBARGO_MONTHS,
        oos_start_year=OOS_START_YEAR, oos_end_year=OOS_END_YEAR,
        device=device, verbose=True, label='StaticAE')

    print("\n" + "=" * 60)
    print("Running TV-AE-G")
    print("=" * 60)
    results['TVAEG'] = rolling_fit_generic(
        df, TVEGApp1,
        N_max=N_max, K1=K1,
        encoder_width=ENCODER_WIDTH, beta_width=BETA_WIDTH,
        n_seeds=N_SEEDS, n_epochs=N_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE,
        initial_train_years=INITIAL_TRAIN_YEARS,
        val_months=VAL_MONTHS,
        purge_months=PURGE_MONTHS, embargo_months=EMBARGO_MONTHS,
        oos_start_year=OOS_START_YEAR, oos_end_year=OOS_END_YEAR,
        device=device, verbose=True, label='TVAEG')

    summarize(results, true_df)

    for name, pred in results.items():
        pred.to_parquet(f"app1_oos_{name}.parquet", index=False)
    print(f"\nSaved per-model predictions to app1_oos_*.parquet")
    print(f"Total wall time: {(time.time() - t0)/60:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    finally:
        _tee.flush()