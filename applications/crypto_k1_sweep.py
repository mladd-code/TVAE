"""
reconstruction_r2_crypto_k1sweep.py
===================================
Crypto reconstruction R^2 across bottleneck widths K1 in {1,2,3,5,8,12}.

For each K1, runs static_pca (rank K1), static_ae (bottleneck K1),
tvaeg (bottleneck K1) and reports pooled R^2. The aim is to see whether
the AE's advantage over PCA grows, saturates, or collapses with more
capacity.

Runtime: ~12-15 minutes.
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import torch

from crypto_returns import (
    Config, DEVICE, load_panel, select_universe, build_matrices,
    fit_pca, fit_static_ae, fit_tvaeg,
)


K1_VALUES = [1, 2, 3, 5, 8, 12]


def evaluate_fit(fit_fn, K, R, M, dates):
    oos_start = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    refit_dates = list(range(oos_start, oos_end, Config.REBAL))
    sse = sst = 0.0
    n_obs = 0
    for t in refit_dates:
        if t < Config.T_EST:
            continue
        R_tr = R[t - Config.T_EST:t, :]
        M_tr = M[t - Config.T_EST:t, :]
        hold_end = min(t + Config.REBAL, len(dates))
        R_te = R[t:hold_end, :]
        M_te = M[t:hold_end, :]
        if M_tr.sum() < 50:
            continue
        try:
            predict = fit_fn(R_tr, M_tr, K)
            R_hat = predict(R_te, M_te)
        except Exception:
            continue
        sse += (((R_te - R_hat) ** 2) * M_te).sum()
        sst += ((R_te ** 2) * M_te).sum()
        n_obs += int(M_te.sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return r2, n_obs


def main():
    print(f"Device: {DEVICE}")
    df = load_panel()
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    R, M = build_matrices(df, universe, dates)
    print(f"Panel: {R.shape}, coverage {M.mean():.3f}, "
          f"universe {len(universe)} coins")
    print(f"OOS: {Config.OOS_START} to {Config.OOS_END}")

    rows = []
    t_start = time.time()
    for K1 in K1_VALUES:
        print(f"\n=== K1 = {K1} ===")
        t0 = time.time()
        try:
            r2_pca, n_pca = evaluate_fit(fit_pca, K1, R, M, dates)
            r2_ae, n_ae = evaluate_fit(fit_static_ae, K1, R, M, dates)
            r2_tv, n_tv = evaluate_fit(fit_tvaeg, K1, R, M, dates)
            print(f"  static_pca = {r2_pca:.4f}  "
                  f"static_ae = {r2_ae:.4f}  "
                  f"tvaeg = {r2_tv:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            rows.append({
                "K1": K1,
                "static_pca": r2_pca,
                "static_ae": r2_ae,
                "tvaeg": r2_tv,
                "tvaeg_minus_pca": r2_tv - r2_pca,
                "tvaeg_minus_ae": r2_tv - r2_ae,
                "ae_minus_pca": r2_ae - r2_pca,
                "n_obs": n_tv,
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Summary table ===")
    header = (f"{'K1':>4} {'PCA':>10} {'StaticAE':>10} {'TVAEG':>10} "
              f"{'TVAEG-PCA':>12} {'TVAEG-AE':>12} {'AE-PCA':>12}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['K1']:>4} {r['static_pca']:>10.4f} "
              f"{r['static_ae']:>10.4f} {r['tvaeg']:>10.4f} "
              f"{r['tvaeg_minus_pca']:>+12.4f} "
              f"{r['tvaeg_minus_ae']:>+12.4f} "
              f"{r['ae_minus_pca']:>+12.4f}")

    os.makedirs("recon_results_crypto", exist_ok=True)
    pd.DataFrame(rows).to_csv(
        "recon_results_crypto/k1_sweep.csv", index=False)
    print(f"\nSaved to recon_results_crypto/k1_sweep.csv")
    print(f"Total time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()