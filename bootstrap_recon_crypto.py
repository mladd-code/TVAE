"""
bootstrap_recon_crypto.py
=========================
Paired bootstrap over refit windows for the crypto reconstruction R^2.

Loads the fitted models? No -- refits everything, but only the four
methods. To keep it fast we cache per-window SSE and SST for each method
in a single pass, then bootstrap over windows.

Reports:
  - pooled R^2 per method
  - per-window R^2 distribution per method
  - paired difference in pooled R^2 between methods, with 95% CI
  - by-year R^2 comparison
"""

from __future__ import annotations
import sys, os, time
import numpy as np
import pandas as pd
import torch

from reconstruction_r2_crypto import (
    Config, DEVICE, load_panel, select_universe, build_matrices,
    METHODS, fit_pca, fit_lpca, fit_static_ae, fit_tvaeg,
)


def collect_per_window(R, M, dates, method_name):
    """
    Returns per-window (sse, sst, n_obs) arrays and the window start indices.
    """
    fit_fn, K = METHODS[method_name]
    oos_start = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    refit_dates = list(range(oos_start, oos_end, Config.REBAL))

    sse_list, sst_list, n_list, t_list = [], [], [], []

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
        except Exception as e:
            print(f"  [{method_name}] t={t} FAILED: {e}")
            continue
        sse = (((R_te - R_hat) ** 2) * M_te).sum()
        sst = ((R_te ** 2) * M_te).sum()
        sse_list.append(sse)
        sst_list.append(sst)
        n_list.append(int(M_te.sum()))
        t_list.append(dates[t])
    return (np.array(sse_list), np.array(sst_list),
            np.array(n_list), np.array(t_list))


def pooled_r2(sse, sst):
    return 1 - sse.sum() / sst.sum()


def bootstrap_diff(sse_a, sst_a, sse_b, sst_b, n_boot=2000, seed=42):
    """
    Paired bootstrap over windows. Returns (mean_diff, ci_low, ci_high)
    where diff = R2(a) - R2(b).
    """
    n = len(sse_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2_a = 1 - sse_a[idx].sum() / sst_a[idx].sum()
        r2_b = 1 - sse_b[idx].sum() / sst_b[idx].sum()
        diffs[b] = r2_a - r2_b
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), diffs


def main():
    print(f"Device: {DEVICE}")
    df = load_panel()
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    R, M = build_matrices(df, universe, dates)
    print(f"Panel: {R.shape}, coverage {M.mean():.3f}")
    print(f"Universe: {len(universe)} coins")

    method_list = ["static_pca", "static_ae", "tvaeg"]
    # LPCA is broken; skip it
    per_window = {}

    for m in method_list:
        print(f"\n=== Collecting per-window SSE/SST: {m} ===")
        sse, sst, n, t = collect_per_window(R, M, dates, m)
        per_window[m] = {"sse": sse, "sst": sst, "n": n, "t": t}
        print(f"  windows: {len(sse)}, "
              f"pooled R^2 = {pooled_r2(sse, sst):.4f}")

    # ----- Pooled comparison -----
    print("\n=== Pooled R^2 ===")
    for m in method_list:
        pw = per_window[m]
        print(f"  {m:<12} {pooled_r2(pw['sse'], pw['sst']):.4f}")

    # ----- Paired bootstrap on differences -----
    print("\n=== Paired bootstrap (2000 draws over windows) ===")
    pairs = [("tvaeg", "static_pca"),
             ("tvaeg", "static_ae"),
             ("static_ae", "static_pca")]
    for a, b in pairs:
        mean_d, lo, hi, _ = bootstrap_diff(
            per_window[a]["sse"], per_window[a]["sst"],
            per_window[b]["sse"], per_window[b]["sst"])
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  R^2({a}) - R^2({b}) = {mean_d:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")

    # ----- By year -----
    print("\n=== By year ===")
    years = sorted(set(pd.to_datetime(per_window["tvaeg"]["t"]).year))
    header = f"{'year':<6} " + " ".join(f"{m:>12}" for m in method_list)
    print(header)
    print("-" * len(header))
    for y in years:
        row = f"{y:<6} "
        for m in method_list:
            mask = pd.to_datetime(per_window[m]["t"]).year == y
            r2 = pooled_r2(per_window[m]["sse"][mask],
                           per_window[m]["sst"][mask])
            row += f"{r2:>12.4f} "
        print(row)

    # ----- Per-window R^2 distribution -----
    print("\n=== Per-window R^2 summary (across refit windows) ===")
    print(f"{'method':<12} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8}")
    for m in method_list:
        pw = per_window[m]
        r2s = 1 - pw["sse"] / pw["sst"]
        print(f"{m:<12} {r2s.mean():>8.4f} {np.median(r2s):>8.4f} "
              f"{r2s.std():>8.4f} {r2s.min():>8.4f} {r2s.max():>8.4f}")


if __name__ == "__main__":
    main()