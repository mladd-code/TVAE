"""
mvp_app_fix_2.py
================
Minimum variance portfolio application — FIX 2.

Difference from mvp_app.py
--------------------------
TV-AE-C now takes raw returns as the encoder input, not managed
portfolios.

    OLD (mvp_app.py):
        F_t = encoder(x_t, tau_t)          [x_t = managed portfolios]
        beta_i = beta_net(Z_i, tau_t)

    NEW (this file):
        F_t = encoder(X_t, tau_t)          [X_t = raw cross-section]
        beta_i = beta_net(Z_i, tau_t)

    Encoder still has a PCA warm-start linear path:

        F_t = W_pca @ (X_t * M_t) + MLP(X_t * M_t, M_t, tau_t)

Why
---
The previous design used x_t = (Z'Z)^{-1} Z' r as the encoder input.
Those managed portfolios span only 1.92% of return variance (measured
on the panel). The encoder therefore cannot reconstruct more than
1.92% of the returns in the best case. Fix: feed the encoder the raw
cross-section, which carries the full signal.

Not included
------------
static_ae and tvaeg are NOT in this file. Those require Fix 1 (PCA warm-
start for return-only AEs). Use mvp_app_fix_1.py for those methods.

Usage
-----
    python mvp_app_fix_2.py ew
    python mvp_app_fix_2.py tvaec_k5
    python mvp_app_fix_2.py tvaec_k3
    python mvp_app_fix_2.py all
"""

from __future__ import annotations
import sys, os
import time
import math
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from sklearn.covariance import LedoitWolf, OAS
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


MVP_CHAR_COLS = ["size", "mom_12_1", "rev_1m",
                 "vol_12m", "beta_12m", "idio_vol_12m"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    PANEL_PATH = "refinitiv_data/panel_daily.parquet"
    CHAR_PATH = "refinitiv_data/panel_daily_char.parquet"
    INDEX_PATH = "refinitiv_data/spx_index.parquet"
    FWYZ_R_SCRIPT = "fwyz_benchmark.R"
    OUT_DIR = "mvp_results"

    T_EST = 250
    REBAL = 21
    P = 100

    K1 = 5
    AE_N_EPOCHS = 200
    AE_LR = 1e-2
    AE_WEIGHT_DECAY = 1e-4

    TX_COST_BPS = 10.0
    OOS_START = "2010-01-01"
    OOS_END   = "2024-12-31"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_daily_panel(path):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def load_char_panel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run characteristics.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def select_universe(df, p, burn_end):
    burn = df[df["date"] < pd.Timestamp(burn_end)]
    avg_mcap = burn.groupby("permno")["mcap"].mean().sort_values(ascending=False)
    return avg_mcap.head(p).index.tolist()


def build_returns_matrix(df, universe, dates):
    df = df.copy()
    df["date_ns"] = df["date"].values.astype("datetime64[ns]").astype(np.int64)
    dates_ns = dates.astype("datetime64[ns]").astype(np.int64)
    date_to_row = {int(d): i for i, d in enumerate(dates_ns)}
    permno_to_col = {p: j for j, p in enumerate(universe)}
    sub = df[df["permno"].isin(universe)].copy()
    sub["row_idx"] = sub["date_ns"].map(date_to_row)
    sub["col_idx"] = sub["permno"].map(permno_to_col)
    sub = sub.dropna(subset=["row_idx", "col_idx"])
    sub["row_idx"] = sub["row_idx"].astype(int)
    sub["col_idx"] = sub["col_idx"].astype(int)
    T, p = len(dates_ns), len(universe)
    R = np.zeros((T, p)); M = np.zeros((T, p))
    R[sub["row_idx"].values, sub["col_idx"].values] = sub["ret"].values
    M[sub["row_idx"].values, sub["col_idx"].values] = 1.0
    return R, M


def build_char_matrix(char_df, universe, dates, char_cols):
    char_df = char_df.copy()
    char_df["date_ns"] = char_df["date"].values.astype("datetime64[ns]").astype(np.int64)
    dates_ns = dates.astype("datetime64[ns]").astype(np.int64)
    date_to_row = {int(d): i for i, d in enumerate(dates_ns)}
    permno_to_col = {p: j for j, p in enumerate(universe)}
    sub = char_df[char_df["permno"].isin(universe)].copy()
    sub["row_idx"] = sub["date_ns"].map(date_to_row)
    sub["col_idx"] = sub["permno"].map(permno_to_col)
    sub = sub.dropna(subset=["row_idx", "col_idx"])
    sub["row_idx"] = sub["row_idx"].astype(int)
    sub["col_idx"] = sub["col_idx"].astype(int)
    T, p = len(dates_ns), len(universe); P = len(char_cols)
    C = np.zeros((T, p, P), dtype=np.float32)
    C[sub["row_idx"].values, sub["col_idx"].values, :] = sub[char_cols].values.astype(np.float32)
    return C


# --------------------------------------------------------------------------- #
# Classical covariance estimators (same as fix 1)
# --------------------------------------------------------------------------- #
def _psd_repair(Sigma, floor=1e-12):
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, floor)
    S = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (S + S.T)


def cov_sample(R, M):
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    return _psd_repair((Rc.T @ Rc) / R.shape[0])


def cov_lw_linear(R, M):
    if not HAS_SKLEARN: raise RuntimeError("sklearn needed")
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    return _psd_repair(LedoitWolf().fit(Rc).covariance_)


def cov_lw_nonlinear(R, M):
    if not HAS_SKLEARN: raise RuntimeError("sklearn needed")
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    return _psd_repair(OAS().fit(Rc).covariance_)


def cov_poet(R, M, K=3, C=1.0):
    T, p = R.shape
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    F = U[:, :K] * S[:K]; Lambda = Vt[:K, :].T
    E = (Rc - F @ Lambda.T) * M
    Sigma_e = (E.T @ E) / T
    diag = np.diag(Sigma_e).copy()
    sd = np.sqrt(np.maximum(diag, 1e-12))
    thresh = C * np.sqrt(np.log(p) / T) * np.outer(sd, sd)
    Sigma_e = np.sign(Sigma_e) * np.maximum(np.abs(Sigma_e) - thresh, 0.0)
    np.fill_diagonal(Sigma_e, diag)
    return _psd_repair(Lambda @ ((F.T @ F) / T) @ Lambda.T + Sigma_e)


def cov_fwyz(R, M, m=3):
    R_filled = R * M
    with tempfile.TemporaryDirectory() as tmp:
        in_file = os.path.join(tmp, "returns.csv")
        out_file = os.path.join(tmp, "cov.csv")
        np.savetxt(in_file, R_filled, delimiter=",", fmt="%.10f")
        result = subprocess.run(
            ["Rscript", Config.FWYZ_R_SCRIPT, in_file, out_file, str(m)],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FWYZ R failed: {result.stderr[:200]}")
        Sigma = np.loadtxt(out_file, delimiter=",")
    return _psd_repair(Sigma)


# --------------------------------------------------------------------------- #
# FIX 2: TV-AE-C with raw-return encoder
# --------------------------------------------------------------------------- #
def _pca_warm_start(Rc, M, K1):
    _, _, Vt = np.linalg.svd(Rc * M, full_matrices=False)
    return Vt[:K1, :].T.astype(np.float32)


class _TVAECCovRaw(nn.Module):
    """Encoder takes raw returns; beta conditions on characteristics and tau."""
    def __init__(self, p, P_char, K1, W_pca_init,
                 encoder_width=16, beta_width=16):
        super().__init__()
        self.p = p
        self.P = P_char
        self.K1 = K1

        # PCA warm-start linear path
        self.W_pca = nn.Parameter(
            torch.tensor(W_pca_init, dtype=torch.float32))

        # Nonlinear encoder residual
        self.enc_mlp = nn.Sequential(
            nn.Linear(2 * p + 1, encoder_width), nn.ReLU(),
            nn.Linear(encoder_width, K1))
        nn.init.zeros_(self.enc_mlp[-1].weight)
        nn.init.zeros_(self.enc_mlp[-1].bias)

        # Beta network on characteristics + tau
        self.beta_net = nn.Sequential(
            nn.Linear(P_char + 1, beta_width), nn.ReLU(),
            nn.Linear(beta_width, K1))

    def encode(self, X, M, tau):
        F_pca = (X * M) @ self.W_pca
        F_mlp = self.enc_mlp(torch.cat([X * M, M, tau], dim=1))
        return F_pca + F_mlp

    def beta_all(self, Z, tau):
        T, p, P = Z.shape
        tau_exp = tau.unsqueeze(1).expand(T, p, 1)
        inp = torch.cat([Z, tau_exp], dim=2)
        return self.beta_net(inp.reshape(T * p, P + 1)).reshape(T, p, self.K1)

    def forward(self, X, M, Z, tau):
        F = self.encode(X, M, tau)
        B = self.beta_all(Z, tau)
        return torch.einsum('tk,tpk->tp', F, B), F


def cov_tvaec_raw(R, M, char_arr, K1=5, device="cpu"):
    T, p = R.shape
    P = char_arr.shape[2]
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M

    W_pca_init = _pca_warm_start(Rc, M, K1)
    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)

    X_t = torch.tensor(Rc, dtype=torch.float32).to(device)
    M_t = torch.tensor(M, dtype=torch.float32).to(device)
    Z_t = torch.tensor(char_arr, dtype=torch.float32).to(device)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(device)

    model = _TVAECCovRaw(p, P, K1, W_pca_init).to(device)
    optimizer = torch.optim.Adam([
        {'params': [model.W_pca], 'weight_decay': 0.0},
        {'params': model.enc_mlp.parameters(),
         'weight_decay': Config.AE_WEIGHT_DECAY},
        {'params': model.beta_net.parameters(),
         'weight_decay': Config.AE_WEIGHT_DECAY},
    ], lr=Config.AE_LR)

    for _ in range(Config.AE_N_EPOCHS):
        optimizer.zero_grad()
        x_hat, F = model(X_t, M_t, Z_t, tau_t)
        loss = ((x_hat - X_t) ** 2 * M_t).sum() / M_t.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_hat, F = model(X_t, M_t, Z_t, tau_t)
        F = F.cpu().numpy()
        X_hat = x_hat.cpu().numpy()
        B = model.beta_all(Z_t, tau_t).cpu().numpy()

    Sigma_F = (F.T @ F) / T
    E = (Rc - X_hat) * M
    diag = np.diag((E.T @ E) / T).copy()
    sd = np.sqrt(np.maximum(diag, 1e-12))
    thresh = 1.0 * np.sqrt(np.log(p) / T) * np.outer(sd, sd)
    Sigma_e = (E.T @ E) / T
    Sigma_e = np.sign(Sigma_e) * np.maximum(np.abs(Sigma_e) - thresh, 0.0)
    np.fill_diagonal(Sigma_e, diag)

    B_t = B[-1]
    return _psd_repair(B_t @ Sigma_F @ B_t.T + Sigma_e)


# --------------------------------------------------------------------------- #
# MVP weights, backtest
# --------------------------------------------------------------------------- #
def mvp_weights(Sigma):
    inv = np.linalg.pinv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = inv @ ones
    return w / w.sum()


def backtest_method(df, universe, dates, char_arr, method_name,
                    device="cpu", verbose=True):
    R, M = build_returns_matrix(df, universe, dates)
    oos_start = np.searchsorted(dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end = np.searchsorted(dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    rebal = list(range(oos_start, oos_end, Config.REBAL))
    live = (M[oos_start:oos_end + 1] > 0).any(axis=0)
    live_idx = np.where(live)[0]
    p_live = len(live_idx)
    if p_live < 10:
        raise RuntimeError(f"Not enough live assets: {p_live}")

    returns_oos, weights_record = [], []
    prev_w = np.zeros(p_live)

    for k, t in enumerate(rebal):
        if t < Config.T_EST:
            continue
        R_win = R[t - Config.T_EST:t, :][:, live_idx]
        M_win = M[t - Config.T_EST:t, :][:, live_idx]
        C_win = char_arr[t - Config.T_EST:t, :][:, live_idx, :]
        t0 = time.time()
        try:
            if method_name == "ew":
                w = np.ones(p_live) / p_live
            else:
                Sigma = _estimate_cov(method_name, R_win, M_win,
                                      C_win, device=device)
                w = mvp_weights(Sigma)
        except Exception as e:
            if verbose: print(f"  [{method_name}] t={t} failed: {e}")
            continue
        hold_end = min(t + Config.REBAL, len(dates))
        R_hold = R[t:hold_end, :][:, live_idx]
        M_hold = M[t:hold_end, :][:, live_idx]
        port_ret = (R_hold * w[None, :] * M_hold).sum(axis=1)
        turnover = np.abs(w - prev_w).sum()
        cost = turnover * (Config.TX_COST_BPS / 1e4)
        returns_oos.extend((port_ret - cost / max(len(port_ret), 1)).tolist())
        weights_record.append(w.copy())
        prev_w = w
        if verbose and (k % 12 == 0):
            print(f"  [{method_name}] rebal {k}/{len(rebal)} ({time.time()-t0:.1f}s)")

    return {"method": method_name, "returns": np.array(returns_oos),
            "weights": np.array(weights_record)}


def _estimate_cov(method_name, R_win, M_win, C_win, device="cpu"):
    if method_name == "sample": return cov_sample(R_win, M_win)
    if method_name == "lw_lin": return cov_lw_linear(R_win, M_win)
    if method_name == "lw_nl": return cov_lw_nonlinear(R_win, M_win)
    if method_name == "poet": return cov_poet(R_win, M_win, K=Config.K1)
    if method_name == "fwyz": return cov_fwyz(R_win, M_win, m=Config.K1)
    if method_name == "tvaec_k3":
        return cov_tvaec_raw(R_win, M_win, C_win, K1=3, device=device)
    if method_name == "tvaec_k5":
        return cov_tvaec_raw(R_win, M_win, C_win, K1=5, device=device)
    raise ValueError(f"Unknown: {method_name}")


def performance_metrics(returns, annualization=252.0):
    if len(returns) == 0:
        return {"N": 0, "sd": np.nan, "sharpe": np.nan, "cumret": np.nan}
    sd = float(np.std(returns, ddof=1)); mean = float(np.mean(returns))
    return {"N": len(returns), "sd": sd,
            "sharpe": mean / sd * np.sqrt(annualization) if sd > 0 else np.nan,
            "cumret": float(np.sum(returns))}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(method="all"):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    print(f"Device: {device}")
    print(f"Method: {method}")
    print(f"File: mvp_app_fix_2.py (raw-return encoder for TV-AE-C)")

    df = load_daily_panel(Config.PANEL_PATH)
    print(f"Panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")
    char_df = load_char_panel(Config.CHAR_PATH)
    print(f"Char panel: {len(char_df)} rows, "
          f"{len(MVP_CHAR_COLS)} characteristics")
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    print(f"Universe: {len(universe)} permnos")
    dates = np.sort(df["date"].unique())
    char_arr = build_char_matrix(char_df, universe, dates, MVP_CHAR_COLS)
    print(f"Char array: {char_arr.shape}")

    all_methods = ["ew", "sample", "lw_lin", "lw_nl", "poet", "fwyz",
                   "tvaec_k3", "tvaec_k5"]
    methods = all_methods if method == "all" else [method]

    results = {}
    for m in methods:
        print(f"\n=== Running {m} ===")
        try:
            out = backtest_method(df, universe, dates, char_arr, m,
                                  device=device, verbose=True)
            results[m] = out
            s = performance_metrics(out["returns"])
            print(f"  Sharpe: {s['sharpe']:.3f}  CumRet: {s['cumret']:.4f}  "
                  f"AnnVol: {s['sd']*np.sqrt(252):.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Summary ===")
    print(f"{'method':<12} {'N':>6} {'Sharpe':>10} {'AnnVol':>10} {'CumRet':>10}")
    print("-" * 60)
    for name, out in results.items():
        s = performance_metrics(out["returns"])
        print(f"{name:<12} {s['N']:>6} {s['sharpe']:>10.3f} "
              f"{s['sd']*np.sqrt(252):>10.4f} {s['cumret']:>10.4f}")

    if os.path.exists(Config.INDEX_PATH):
        idx = pd.read_parquet(Config.INDEX_PATH)
        idx = idx[(idx["date"] >= Config.OOS_START) &
                  (idx["date"] <= Config.OOS_END)]
        s = performance_metrics(idx["spx_ret"].dropna().values)
        print(f"\nS&P 500: Sharpe {s['sharpe']:.3f}  "
              f"CumRet {s['cumret']:.4f}  AnnVol {s['sd']*np.sqrt(252):.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")