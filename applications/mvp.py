"""
mvp_app.py
----------
Minimum variance portfolio application on the S&P 500 daily panel.

Estimators compared
-------------------
  1. Equal weight (1/N)
  2. Sample covariance
  3. Ledoit-Wolf linear shrinkage
  4. Ledoit-Wolf nonlinear shrinkage (via OAS)
  5. POET (static, Fan-Liao-Mincheva)
  6. FWYZ local PCA + POET (via R subprocess)
  7. Static AE + POET
  8. TV-AE-G + POET
  9. TV-AE-C + POET (K1 = 3)
 10. TV-AE-C + POET (K1 = 5)

Backtest
--------
  - Rolling window of T_EST trading days.
  - Monthly rebalancing.
  - p stocks chosen by average market cap over the burn-in period.
  - OOS evaluation with Sharpe, annualized volatility, cumulative return.
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

from tvae_global_app_1 import build_panel, MonthlyPanel, TVEGApp1, _train_one
from characteristics import CHAR_COLS, build_characteristics, rank_normalize_char
from tvaec_mvp import cov_tvaec, _TVAECCov


# Characteristics used for the MVP application
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
    T_BURN = 250
    REBAL = 21
    P = 100

    K1 = 5
    AE_ENCODER_WIDTH = 16
    AE_BETA_WIDTH = 16
    AE_N_SEEDS = 2
    AE_N_EPOCHS = 100
    AE_LR = 1e-2
    AE_WEIGHT_DECAY = 1e-5

    TX_COST_BPS = 10.0

    OOS_START = "2010-01-01"
    OOS_END   = "2024-12-31"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_daily_panel(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)
    return df


def load_char_panel(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"Characteristics panel not found at {path}.")
        print("Run 'python characteristics.py' first.")
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def select_universe(df: pd.DataFrame, p: int,
                    burn_end: str) -> List[str]:
    burn = df[df["date"] < pd.Timestamp(burn_end)]
    avg_mcap = (burn.groupby("permno")["mcap"].mean()
                .sort_values(ascending=False))
    return avg_mcap.head(p).index.tolist()


def build_returns_matrix(df: pd.DataFrame, universe: List[str],
                         dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    R = np.zeros((T, p), dtype=np.float64)
    M = np.zeros((T, p), dtype=np.float64)
    R[sub["row_idx"].values, sub["col_idx"].values] = sub["ret"].values
    M[sub["row_idx"].values, sub["col_idx"].values] = 1.0
    return R, M


def build_char_matrix(char_df: pd.DataFrame, universe: List[str],
                      dates: np.ndarray, char_cols: List[str]
                      ) -> np.ndarray:
    """
    Returns C: (T, p, P) characteristics matrix, aligned with the returns.
    Missing entries are 0 (the cross-sectional median after rank
    normalization).
    """
    char_df = char_df.copy()
    char_df["date_ns"] = (char_df["date"].values
                          .astype("datetime64[ns]").astype(np.int64))
    dates_ns = dates.astype("datetime64[ns]").astype(np.int64)

    date_to_row = {int(d): i for i, d in enumerate(dates_ns)}
    permno_to_col = {p: j for j, p in enumerate(universe)}

    sub = char_df[char_df["permno"].isin(universe)].copy()
    sub["row_idx"] = sub["date_ns"].map(date_to_row)
    sub["col_idx"] = sub["permno"].map(permno_to_col)
    sub = sub.dropna(subset=["row_idx", "col_idx"])
    sub["row_idx"] = sub["row_idx"].astype(int)
    sub["col_idx"] = sub["col_idx"].astype(int)

    T, p = len(dates_ns), len(universe)
    P = len(char_cols)
    C = np.zeros((T, p, P), dtype=np.float32)
    vals = sub[char_cols].values.astype(np.float32)
    C[sub["row_idx"].values, sub["col_idx"].values, :] = vals
    return C


# --------------------------------------------------------------------------- #
# Classical covariance estimators
# --------------------------------------------------------------------------- #
def cov_sample(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    R = R.copy()
    T, p = R.shape
    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    S = (Rc.T @ Rc) / T
    return _psd_repair(S)


def cov_ledoit_wolf_linear(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for LW shrinkage")
    R = R.copy()
    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    lw = LedoitWolf().fit(Rc)
    return _psd_repair(lw.covariance_)


def cov_ledoit_wolf_nonlinear(R: np.ndarray, M: np.ndarray) -> np.ndarray:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for nonlinear shrinkage")
    R = R.copy()
    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M
    oas = OAS().fit(Rc)
    return _psd_repair(oas.covariance_)


def cov_poet(R: np.ndarray, M: np.ndarray,
             K: int = 3, C: float = 1.0) -> np.ndarray:
    R = R.copy()
    T, p = R.shape
    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M

    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    F = U[:, :K] * S[:K]
    Lambda = Vt[:K, :].T

    R_hat = F @ Lambda.T
    E = (Rc - R_hat) * M
    Sigma_e = (E.T @ E) / T

    diag = np.diag(Sigma_e).copy()
    sd = np.sqrt(np.maximum(diag, 1e-12))
    thresh = C * np.sqrt(np.log(p) / T) * np.outer(sd, sd)

    S_sign = np.sign(Sigma_e)
    S_abs = np.abs(Sigma_e) - thresh
    Sigma_e_thr = S_sign * np.maximum(S_abs, 0.0)
    np.fill_diagonal(Sigma_e_thr, diag)

    Sigma_F = (F.T @ F) / T
    Sigma = Lambda @ Sigma_F @ Lambda.T + Sigma_e_thr
    return _psd_repair(Sigma)


def cov_fwyz(R: np.ndarray, M: np.ndarray, m: int = 3) -> np.ndarray:
    R_filled = R * M
    with tempfile.TemporaryDirectory() as tmp:
        in_file = os.path.join(tmp, "returns.csv")
        out_file = os.path.join(tmp, "cov.csv")
        np.savetxt(in_file, R_filled, delimiter=",", fmt="%.10f")
        try:
            result = subprocess.run(
                ["Rscript", Config.FWYZ_R_SCRIPT, in_file, out_file, str(m)],
                capture_output=True, text=True, timeout=600)
        except FileNotFoundError:
            raise RuntimeError("Rscript not found. Install R and ensure it "
                               "is on PATH.")
        if result.returncode != 0:
            print(f"[FWYZ R error]\nSTDOUT:\n{result.stdout}\n"
                  f"STDERR:\n{result.stderr}")
            raise RuntimeError("FWYZ R script failed.")
        Sigma = np.loadtxt(out_file, delimiter=",")
    return _psd_repair(Sigma)


# --------------------------------------------------------------------------- #
# AE covariance dispatch
# --------------------------------------------------------------------------- #
def cov_static_ae(R, M, K1=3, device="cpu"):
    return _cov_via_ae(R, M, K1, model_kind="static", device=device)


def cov_tvaeg(R, M, K1=3, device="cpu"):
    return _cov_via_ae(R, M, K1, model_kind="tvaeg", device=device)


# --------------------------------------------------------------------------- #
# AE covariance helper (return-only models)
# --------------------------------------------------------------------------- #
def _cov_via_ae(R: np.ndarray, M: np.ndarray, K1: int,
                model_kind: str, device: str = "cpu") -> np.ndarray:
    T, p = R.shape
    if T < 50:
        return cov_sample(R, M)

    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M

    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)

    if model_kind == "static":
        model = _StaticAECov(p, K1).to(device)
        X = torch.tensor(Rc, dtype=torch.float32).to(device)
        Mu = torch.tensor(M, dtype=torch.float32).to(device)
        tau_t = torch.tensor(tau, dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=Config.AE_LR,
                                     weight_decay=Config.AE_WEIGHT_DECAY)
        for ep in range(Config.AE_N_EPOCHS):
            optimizer.zero_grad()
            x_hat, F = model(X, Mu)
            loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x_hat, F = model(X, Mu)
            F = F.detach().cpu().numpy()
            X_hat = x_hat.detach().cpu().numpy()
            B = model.beta.unsqueeze(0).expand(T, -1, -1).detach().cpu().numpy()

    elif model_kind == "tvaeg":
        model = _TVEGCov(p, K1).to(device)
        X = torch.tensor(Rc, dtype=torch.float32).to(device)
        Mu = torch.tensor(M, dtype=torch.float32).to(device)
        tau_t = torch.tensor(tau, dtype=torch.float32).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=Config.AE_LR,
                                     weight_decay=Config.AE_WEIGHT_DECAY)
        for ep in range(Config.AE_N_EPOCHS):
            optimizer.zero_grad()
            x_hat, F = model(X, Mu, tau_t)
            loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x_hat, F = model(X, Mu, tau_t)
            F = F.detach().cpu().numpy()
            X_hat = x_hat.detach().cpu().numpy()
            B = model.beta_all(tau_t).detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    Sigma_F = (F.T @ F) / T
    E = (Rc - X_hat) * M
    diag = np.diag((E.T @ E) / T).copy()
    sd = np.sqrt(np.maximum(diag, 1e-12))
    thresh = 1.0 * np.sqrt(np.log(p) / T) * np.outer(sd, sd)
    Sigma_e = (E.T @ E) / T
    S_sign = np.sign(Sigma_e)
    S_abs = np.abs(Sigma_e) - thresh
    Sigma_e = S_sign * np.maximum(S_abs, 0.0)
    np.fill_diagonal(Sigma_e, diag)

    B_t = B[-1]
    Sigma = B_t @ Sigma_F @ B_t.T + Sigma_e
    return _psd_repair(Sigma)


# --------------------------------------------------------------------------- #
# Return-only AE architectures
# --------------------------------------------------------------------------- #
class _StaticAECov(nn.Module):
    def __init__(self, p, K1):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.encoder = nn.Sequential(
            nn.Linear(2 * p, 16), nn.ReLU(),
            nn.Linear(16, K1))
        self.beta = nn.Parameter(torch.randn(p, K1) * 0.01)

    def forward(self, X, M):
        inp = torch.cat([X * M, M], dim=1)
        F = self.encoder(inp)
        x_hat = F @ self.beta.T
        return x_hat, F


class _TVEGCov(nn.Module):
    def __init__(self, p, K1):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.encoder = nn.Sequential(
            nn.Linear(2 * p + 1, 16), nn.ReLU(),
            nn.Linear(16, K1))
        self.W1 = nn.Parameter(torch.empty(p, 1, 8))
        self.b1 = nn.Parameter(torch.zeros(p, 8))
        self.W2 = nn.Parameter(torch.empty(p, 8, K1))
        self.b2 = nn.Parameter(torch.zeros(p, K1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X, M, tau):
        inp = torch.cat([X * M, M, tau], dim=1)
        return self.encoder(inp)

    def beta_all(self, tau):
        tau_b = tau.squeeze(1)
        W1_b = self.W1.squeeze(1)
        H = torch.relu(
            torch.einsum('t,pw->tpw', tau_b, W1_b) + self.b1.unsqueeze(0))
        return torch.einsum('tpw,pwk->tpk', H, self.W2) + self.b2.unsqueeze(0)

    def forward(self, X, M, tau):
        F = self.encode(X, M, tau)
        B = self.beta_all(tau)
        x_hat = torch.einsum('tk,tpk->tp', F, B)
        return x_hat, F


# --------------------------------------------------------------------------- #
# PSD repair
# --------------------------------------------------------------------------- #
def _psd_repair(Sigma: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, floor)
    Sigma = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (Sigma + Sigma.T)


# --------------------------------------------------------------------------- #
# MVP weights
# --------------------------------------------------------------------------- #
def mvp_weights(Sigma: np.ndarray) -> np.ndarray:
    ones = np.ones(Sigma.shape[0])
    inv = np.linalg.pinv(Sigma)
    w = inv @ ones
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest_method(df: pd.DataFrame, universe: List[str],
                    dates: np.ndarray,
                    method_name: str,
                    char_arr: Optional[np.ndarray] = None,
                    device: str = "cpu",
                    verbose: bool = True) -> Dict[str, Any]:
    R, M = build_returns_matrix(df, universe, dates)
    T, p = R.shape

    oos_start_idx = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end_idx = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    rebal_idx = list(range(oos_start_idx, oos_end_idx, Config.REBAL))

    live = (M[oos_start_idx:oos_end_idx + 1] > 0).any(axis=0)
    live_idx = np.where(live)[0]
    p_live = len(live_idx)
    if p_live < 10:
        raise RuntimeError(f"Not enough live assets: {p_live}")

    returns_oos = []
    weights_record = []
    prev_w = np.zeros(p_live)

    for k, t in enumerate(rebal_idx):
        if t < Config.T_EST:
            continue
        win_start = t - Config.T_EST
        R_win = R[win_start:t, :][:, live_idx]
        M_win = M[win_start:t, :][:, live_idx]
        C_win = None
        if char_arr is not None:
            C_win = char_arr[win_start:t, :][:, live_idx, :]

        t0 = time.time()
        try:
            if method_name == "ew":
                w = np.ones(p_live) / p_live
            else:
                Sigma = _estimate_cov(method_name, R_win, M_win,
                                      char_arr=C_win, device=device)
                w = mvp_weights(Sigma)
        except Exception as e:
            if verbose:
                print(f"  [{method_name}] t={t} failed: {e}")
            continue

        hold_end = min(t + Config.REBAL, len(dates))
        R_hold = R[t:hold_end, :][:, live_idx]
        M_hold = M[t:hold_end, :][:, live_idx]
        port_ret = (R_hold * w[None, :] * M_hold).sum(axis=1)
        turnover = np.abs(w - prev_w).sum()
        cost = turnover * (Config.TX_COST_BPS / 1e4)
        port_ret_net = port_ret - cost / max(len(port_ret), 1)
        returns_oos.extend(port_ret_net.tolist())
        weights_record.append(w.copy())
        prev_w = w
        if verbose and (k % 12 == 0):
            print(f"  [{method_name}] rebal {k}/{len(rebal_idx)} "
                  f"({time.time()-t0:.1f}s)")

    returns_oos = np.array(returns_oos)
    return {
        "method": method_name,
        "returns": returns_oos,
        "weights": np.array(weights_record),
        "n_rebal": len(returns_oos) // Config.REBAL,
    }


def _estimate_cov(method_name: str, R_win: np.ndarray, M_win: np.ndarray,
                  char_arr: Optional[np.ndarray] = None,
                  device: str = "cpu") -> np.ndarray:
    if method_name == "sample":
        return cov_sample(R_win, M_win)
    if method_name == "lw_lin":
        return cov_ledoit_wolf_linear(R_win, M_win)
    if method_name == "lw_nl":
        return cov_ledoit_wolf_nonlinear(R_win, M_win)
    if method_name == "poet":
        return cov_poet(R_win, M_win, K=Config.K1)
    if method_name == "fwyz":
        return cov_fwyz(R_win, M_win, m=Config.K1)
    if method_name == "static_ae":
        return cov_static_ae(R_win, M_win, K1=Config.K1, device=device)
    if method_name == "tvaeg":
        return cov_tvaeg(R_win, M_win, K1=Config.K1, device=device)
    if method_name == "tvaec_k3":
        if char_arr is None:
            raise ValueError("char_arr required for tvaec_k3")
        return cov_tvaec(R_win, M_win, char_arr, K1=3,
                         n_epochs=Config.AE_N_EPOCHS,
                         lr=Config.AE_LR,
                         weight_decay=Config.AE_WEIGHT_DECAY,
                         device=device)
    if method_name == "tvaec_k5":
        if char_arr is None:
            raise ValueError("char_arr required for tvaec_k5")
        return cov_tvaec(R_win, M_win, char_arr, K1=5,
                         n_epochs=Config.AE_N_EPOCHS,
                         lr=Config.AE_LR,
                         weight_decay=Config.AE_WEIGHT_DECAY,
                         device=device)
    raise ValueError(f"Unknown method: {method_name}")


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def performance_metrics(returns: np.ndarray,
                        annualization: float = 252.0) -> Dict[str, float]:
    if len(returns) == 0:
        return {"N": 0, "mean": np.nan, "sd": np.nan, "sharpe": np.nan,
                "cumret": np.nan}
    mean = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    sharpe = mean / sd * np.sqrt(annualization) if sd > 0 else np.nan
    cumret = float(np.sum(returns))
    return {"N": int(len(returns)),
            "mean": mean,
            "sd": sd,
            "sharpe": sharpe,
            "cumret": cumret}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(method: str = "all"):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    print(f"Device: {device}")
    print(f"Method: {method}")

    df = load_daily_panel(Config.PANEL_PATH)
    print(f"Panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")

    char_df = load_char_panel(Config.CHAR_PATH)
    if char_df is not None:
        print(f"Char panel: {len(char_df)} rows, "
              f"{len(MVP_CHAR_COLS)} characteristics")

    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    print(f"Universe (top {Config.P} by mcap): {len(universe)} permnos")

    dates = np.sort(df["date"].unique())

    char_arr = None
    if char_df is not None:
        char_arr = build_char_matrix(char_df, universe, dates,
                                     MVP_CHAR_COLS)
        print(f"Char array: {char_arr.shape}")

    all_methods = ["ew", "sample", "lw_lin", "lw_nl", "poet", "fwyz",
                   "static_ae", "tvaeg", "tvaec_k3", "tvaec_k5"]
    if method == "all":
        methods = all_methods
    else:
        methods = [method]

    results = {}
    for m in methods:
        print(f"\n=== Running {m} ===")
        try:
            out = backtest_method(df, universe, dates, m,
                                  char_arr=char_arr,
                                  device=device, verbose=True)
            results[m] = out
            s = performance_metrics(out["returns"])
            print(f"  Sharpe: {s['sharpe']:.3f}  "
                  f"Cumulative: {s['cumret']:.4f}  "
                  f"Volatility (ann.): {s['sd']*np.sqrt(252):.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Summary ===")
    print(f"{'method':<12} {'N':>6} {'Sharpe':>10} {'AnnVol':>10} "
          f"{'CumRet':>10}")
    print("-" * 60)
    for name, out in results.items():
        s = performance_metrics(out["returns"])
        ann_vol = s["sd"] * np.sqrt(252)
        print(f"{name:<12} {s['N']:>6} {s['sharpe']:>10.3f} "
              f"{ann_vol:>10.4f} {s['cumret']:>10.4f}")
        np.save(os.path.join(Config.OUT_DIR, f"{name}_returns.npy"),
                out["returns"])
        np.save(os.path.join(Config.OUT_DIR, f"{name}_weights.npy"),
                out["weights"])

    if os.path.exists(Config.INDEX_PATH):
        idx = pd.read_parquet(Config.INDEX_PATH)
        idx = idx[(idx["date"] >= Config.OOS_START) &
                  (idx["date"] <= Config.OOS_END)]
        spx_ret = idx["spx_ret"].dropna().values
        s_spx = performance_metrics(spx_ret)
        print(f"\nS&P 500 (benchmark):")
        print(f"  Sharpe: {s_spx['sharpe']:.3f}  "
              f"Cumulative: {s_spx['cumret']:.4f}  "
              f"AnnVol: {s_spx['sd']*np.sqrt(252):.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    method_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(method_arg)