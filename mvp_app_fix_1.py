"""
mvp_app_fix_1.py
================
Minimum variance portfolio application — FIX 1.

Difference from mvp_app.py
--------------------------
The return-only autoencoders (static_ae, tvaeg) now use a PCA warm-start
in the encoder. The encoder is decomposed as:

    F_t = W_pca @ X_t + MLP(X_t)

with W_pca initialized to the top-K1 right singular vectors of the
training window (i.e., PCA loadings), and the MLP output layer zero-
initialized. The network therefore starts at exactly PCA. Training can
only improve on it. W_pca has zero weight decay so the linear backbone
is not shrunk.

Why
---
The original architecture had a randomly initialized encoder trained on
250 days of returns. With a signal-to-noise ratio of ~1e-4, the encoder
collapsed to a constant (factor std = 0). Warm-starting at PCA gives the
network a strong, known-good initialization.

Not included
------------
tvaec_k3 and tvaec_k5 are NOT in this file. Those require Fix 2 (raw-
return encoder for the characteristic-conditioned AE). Use mvp_app_fix_2.py
for those methods.

Usage
-----
    python mvp_app_fix_1.py ew
    python mvp_app_fix_1.py static_ae
    python mvp_app_fix_1.py tvaeg
    python mvp_app_fix_1.py all
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


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    PANEL_PATH = "refinitiv_data/panel_daily.parquet"
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
def load_daily_panel(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def select_universe(df: pd.DataFrame, p: int, burn_end: str) -> List[str]:
    burn = df[df["date"] < pd.Timestamp(burn_end)]
    avg_mcap = (burn.groupby("permno")["mcap"].mean()
                .sort_values(ascending=False))
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


# --------------------------------------------------------------------------- #
# Classical covariance estimators
# --------------------------------------------------------------------------- #
def _psd_repair(Sigma, floor=1e-12):
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, floor)
    return 0.5 * ((eigvecs * eigvals) @ eigvecs.T + (eigvecs * eigvals) @ eigvecs.T).T


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
# FIX 1: PCA warm-start encoder
# --------------------------------------------------------------------------- #
def _pca_warm_start(Rc: np.ndarray, M: np.ndarray, K1: int) -> np.ndarray:
    """Top-K1 right singular vectors of the masked demeaned panel."""
    _, _, Vt = np.linalg.svd(Rc * M, full_matrices=False)
    return Vt[:K1, :].T.astype(np.float32)


class _StaticAECovWarm(nn.Module):
    """Encoder = PCA linear path + small MLP residual."""
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.W_pca = nn.Parameter(
            torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * p, 16), nn.ReLU(),
            nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.beta = nn.Parameter(torch.randn(p, K1) * 0.01)

    def encode(self, X, M):
        return (X * M) @ self.W_pca + self.mlp(torch.cat([X * M, M], dim=1))

    def forward(self, X, M):
        F = self.encode(X, M)
        return F @ self.beta.T, F


class _TVEGCovWarm(nn.Module):
    """Time-varying: encoder = PCA path + small MLP; beta network on tau."""
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.W_pca = nn.Parameter(
            torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * p + 1, 16), nn.ReLU(),
            nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.W1 = nn.Parameter(torch.empty(p, 1, 8))
        self.b1 = nn.Parameter(torch.zeros(p, 8))
        self.W2 = nn.Parameter(torch.empty(p, 8, K1))
        self.b2 = nn.Parameter(torch.zeros(p, K1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X, M, tau):
        return ((X * M) @ self.W_pca
                + self.mlp(torch.cat([X * M, M, tau], dim=1)))

    def beta_all(self, tau):
        tau_b = tau.squeeze(1)
        H = torch.relu(torch.einsum('t,pw->tpw', tau_b, self.W1.squeeze(1))
                       + self.b1.unsqueeze(0))
        return torch.einsum('tpw,pwk->tpk', H, self.W2) + self.b2.unsqueeze(0)

    def forward(self, X, M, tau):
        F = self.encode(X, M, tau)
        B = self.beta_all(tau)
        return torch.einsum('tk,tpk->tp', F, B), F


def _cov_via_ae_warm(R, M, K1, model_kind, device="cpu"):
    T, p = R.shape
    if T < 50:
        return cov_sample(R, M)
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    W_pca = _pca_warm_start(Rc, M, K1)
    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)

    X = torch.tensor(Rc, dtype=torch.float32).to(device)
    Mu = torch.tensor(M, dtype=torch.float32).to(device)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(device)

    if model_kind == "static":
        model = _StaticAECovWarm(p, K1, W_pca).to(device)
        param_groups = [
            {'params': [model.W_pca], 'weight_decay': 0.0},
            {'params': model.mlp.parameters(),
             'weight_decay': Config.AE_WEIGHT_DECAY},
            {'params': [model.beta], 'weight_decay': Config.AE_WEIGHT_DECAY},
        ]
        def fwd():
            return model(X, Mu)
    elif model_kind == "tvaeg":
        model = _TVEGCovWarm(p, K1, W_pca).to(device)
        param_groups = [
            {'params': [model.W_pca], 'weight_decay': 0.0},
            {'params': model.mlp.parameters(),
             'weight_decay': Config.AE_WEIGHT_DECAY},
            {'params': [model.W1, model.b1, model.W2, model.b2],
             'weight_decay': Config.AE_WEIGHT_DECAY},
        ]
        def fwd():
            return model(X, Mu, tau_t)
    else:
        raise ValueError(model_kind)

    optimizer = torch.optim.Adam(param_groups, lr=Config.AE_LR)
    for _ in range(Config.AE_N_EPOCHS):
        optimizer.zero_grad()
        x_hat, F = fwd()
        loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_hat, F = fwd()
        F = F.cpu().numpy()
        X_hat = x_hat.cpu().numpy()
        if model_kind == "static":
            B = model.beta.unsqueeze(0).expand(T, -1, -1).cpu().numpy()
        else:
            B = model.beta_all(tau_t).cpu().numpy()

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


def cov_static_ae(R, M, K1=5, device="cpu"):
    return _cov_via_ae_warm(R, M, K1, "static", device)


def cov_tvaeg(R, M, K1=5, device="cpu"):
    return _cov_via_ae_warm(R, M, K1, "tvaeg", device)


# --------------------------------------------------------------------------- #
# MVP weights
# --------------------------------------------------------------------------- #
def mvp_weights(Sigma):
    inv = np.linalg.pinv(Sigma)
    ones = np.ones(Sigma.shape[0])
    w = inv @ ones
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest_method(df, universe, dates, method_name, device="cpu", verbose=True):
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
        t0 = time.time()
        try:
            if method_name == "ew":
                w = np.ones(p_live) / p_live
            else:
                Sigma = _estimate_cov(method_name, R_win, M_win, device=device)
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


def _estimate_cov(method_name, R_win, M_win, device="cpu"):
    if method_name == "sample": return cov_sample(R_win, M_win)
    if method_name == "lw_lin": return cov_lw_linear(R_win, M_win)
    if method_name == "lw_nl": return cov_lw_nonlinear(R_win, M_win)
    if method_name == "poet": return cov_poet(R_win, M_win, K=Config.K1)
    if method_name == "fwyz": return cov_fwyz(R_win, M_win, m=Config.K1)
    if method_name == "static_ae": return cov_static_ae(R_win, M_win, K1=Config.K1, device=device)
    if method_name == "tvaeg": return cov_tvaeg(R_win, M_win, K1=Config.K1, device=device)
    raise ValueError(f"Unknown: {method_name}")


def performance_metrics(returns, annualization=252.0):
    if len(returns) == 0:
        return {"N": 0, "sd": np.nan, "sharpe": np.nan, "cumret": np.nan}
    sd = float(np.std(returns, ddof=1))
    mean = float(np.mean(returns))
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
    print(f"File: mvp_app_fix_1.py (PCA warm-start)")

    df = load_daily_panel(Config.PANEL_PATH)
    print(f"Panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    print(f"Universe: {len(universe)} permnos")
    dates = np.sort(df["date"].unique())

    all_methods = ["ew", "sample", "lw_lin", "lw_nl", "poet", "fwyz",
                   "static_ae", "tvaeg"]
    methods = all_methods if method == "all" else [method]

    results = {}
    for m in methods:
        print(f"\n=== Running {m} ===")
        try:
            out = backtest_method(df, universe, dates, m,
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