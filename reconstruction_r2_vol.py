"""
reconstruction_r2_vol.py
========================
Out-of-sample reconstruction R^2 on the realized-volatility panel.

Same methodology as reconstruction_r2.py, but the target variable is
log realized volatility instead of returns. This panel has:
  - Higher signal-to-noise (RV averages out microstructure noise)
  - Stronger nonlinearity (vol-of-vol, leverage effects)
  - More cross-sectional heterogeneity (vol betas vary more than return betas)

Usage:
  python reconstruction_r2_vol.py
  python reconstruction_r2_vol.py static_pca static_ae tvaeg
"""

from __future__ import annotations
import sys, os, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Config:
    PANEL_PATH = "refinitiv_data/panel_daily_vol.parquet"
    CHAR_PATH  = None             # no characteristics for this panel yet
    OUT_DIR    = "recon_results_vol"

    T_EST = 250
    REBAL = 21

    P = 100                       # 100 largest by market cap

    AE_N_EPOCHS = 200
    AE_LR = 1e-2
    AE_WEIGHT_DECAY = 1e-4

    OOS_START = "2010-01-01"
    OOS_END   = "2024-12-31"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_panel():
    df = pd.read_parquet(Config.PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def select_universe(df, p, burn_end):
    burn = df[df["date"] < pd.Timestamp(burn_end)]
    avg_mcap = burn.groupby("permno")["mcap"].mean().sort_values(ascending=False)
    return avg_mcap.head(p).index.tolist()


def build_matrices(df, universe, dates):
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


def _demean(R, M):
    counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    return mu, (R - mu[None, :]) * M


# --------------------------------------------------------------------------- #
# Architectures (with PCA warm-start)
# --------------------------------------------------------------------------- #
def _pca_warm_start(Rc, M, K1):
    _, _, Vt = np.linalg.svd(Rc * M, full_matrices=False)
    return Vt[:K1, :].T.astype(np.float32)


class _StaticAE(nn.Module):
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p; self.K1 = K1
        self.W_pca = nn.Parameter(torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(2*p, 16), nn.ReLU(),
                                 nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.beta = nn.Parameter(torch.randn(p, K1) * 0.01)
    def encode(self, X, M):
        return (X * M) @ self.W_pca + self.mlp(torch.cat([X*M, M], dim=1))
    def forward(self, X, M):
        F = self.encode(X, M)
        return F @ self.beta.T, F


class _TVEG(nn.Module):
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p; self.K1 = K1
        self.W_pca = nn.Parameter(torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(2*p+1, 16), nn.ReLU(),
                                 nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.W1 = nn.Parameter(torch.empty(p, 1, 8))
        self.b1 = nn.Parameter(torch.zeros(p, 8))
        self.W2 = nn.Parameter(torch.empty(p, 8, K1))
        self.b2 = nn.Parameter(torch.zeros(p, K1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))
    def encode(self, X, M, tau):
        return ((X * M) @ self.W_pca
                + self.mlp(torch.cat([X*M, M, tau], dim=1)))
    def beta_all(self, tau):
        tau_b = tau.squeeze(1)
        H = torch.relu(torch.einsum('t,pw->tpw', tau_b, self.W1.squeeze(1))
                       + self.b1.unsqueeze(0))
        return torch.einsum('tpw,pwk->tpk', H, self.W2) + self.b2.unsqueeze(0)
    def forward(self, X, M, tau):
        F = self.encode(X, M, tau)
        B = self.beta_all(tau)
        return torch.einsum('tk,tpk->tp', F, B), F


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #
def fit_pca(R_tr, M_tr, K):
    mu, Rc = _demean(R_tr, M_tr)
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    L = Vt[:K, :].T
    def predict(R_te, M_te):
        Rc_te = (R_te - mu[None, :]) * M_te
        return (Rc_te @ L) @ L.T + mu[None, :]
    return predict


def fit_lpca(R_tr, M_tr, K):
    from local_pca import silverman, local_pca
    mu, Rc = _demean(R_tr, M_tr)
    R_filled = Rc * M_tr
    bw = silverman(R_filled)
    res = local_pca(R_filled, r=R_filled.shape[0]-1, bandwidth=bw, m=K)
    L = res["loadings"]
    def predict(R_te, M_te):
        Rc_te = (R_te - mu[None, :]) * M_te
        return (Rc_te @ L) @ L.T + mu[None, :]
    return predict


def fit_static_ae(R_tr, M_tr, K):
    T, p = R_tr.shape
    mu, Rc = _demean(R_tr, M_tr)
    W_pca = _pca_warm_start(Rc, M_tr, K)
    model = _StaticAE(p, K, W_pca).to(DEVICE)
    X = torch.tensor(Rc, dtype=torch.float32).to(DEVICE)
    Mu = torch.tensor(M_tr, dtype=torch.float32).to(DEVICE)
    opt = torch.optim.Adam([
        {'params': [model.W_pca], 'weight_decay': 0.0},
        {'params': model.mlp.parameters(), 'weight_decay': Config.AE_WEIGHT_DECAY},
        {'params': [model.beta], 'weight_decay': Config.AE_WEIGHT_DECAY},
    ], lr=Config.AE_LR)
    for _ in range(Config.AE_N_EPOCHS):
        opt.zero_grad()
        x_hat, _ = model(X, Mu)
        loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
        loss.backward(); opt.step()
    model.eval()
    def predict(R_te, M_te):
        with torch.no_grad():
            Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                              dtype=torch.float32).to(DEVICE)
            Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
            F = model.encode(Xt, Mt)
            return (F @ model.beta.T).cpu().numpy() + mu[None, :]
    return predict


def fit_tvaeg(R_tr, M_tr, K):
    T, p = R_tr.shape
    mu, Rc = _demean(R_tr, M_tr)
    W_pca = _pca_warm_start(Rc, M_tr, K)
    model = _TVEG(p, K, W_pca).to(DEVICE)
    X = torch.tensor(Rc, dtype=torch.float32).to(DEVICE)
    Mu = torch.tensor(M_tr, dtype=torch.float32).to(DEVICE)
    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(DEVICE)
    opt = torch.optim.Adam([
        {'params': [model.W_pca], 'weight_decay': 0.0},
        {'params': model.mlp.parameters(), 'weight_decay': Config.AE_WEIGHT_DECAY},
        {'params': [model.W1, model.b1, model.W2, model.b2],
         'weight_decay': Config.AE_WEIGHT_DECAY},
    ], lr=Config.AE_LR)
    for _ in range(Config.AE_N_EPOCHS):
        opt.zero_grad()
        x_hat, _ = model(X, Mu, tau_t)
        loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
        loss.backward(); opt.step()
    model.eval()
    def predict(R_te, M_te):
        T_te = R_te.shape[0]
        with torch.no_grad():
            Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                              dtype=torch.float32).to(DEVICE)
            Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
            tau_te = np.full((T_te, 1), 1.0, dtype=np.float32)
            tau_te_t = torch.tensor(tau_te, dtype=torch.float32).to(DEVICE)
            F = model.encode(Xt, Mt, tau_te_t)
            B = model.beta_all(tau_te_t)
            return torch.einsum('tk,tpk->tp', F, B).cpu().numpy() + mu[None, :]
    return predict


METHODS = {
    "static_pca": (fit_pca, 5),
    "lpca": (fit_lpca, 5),
    "static_ae": (fit_static_ae, 5),
    "tvaeg": (fit_tvaeg, 5),
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_method(R, M, dates, method_name):
    fit_fn, K = METHODS[method_name]
    oos_start = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    refit_dates = list(range(oos_start, oos_end, Config.REBAL))

    sse = sst = 0.0
    n_obs = 0
    t0 = time.time()
    n_done = 0

    for t in refit_dates:
        if t < Config.T_EST:
            continue
        R_tr = R[t-Config.T_EST:t, :]; M_tr = M[t-Config.T_EST:t, :]
        hold_end = min(t + Config.REBAL, len(dates))
        R_te = R[t:hold_end, :]; M_te = M[t:hold_end, :]

        try:
            predict = fit_fn(R_tr, M_tr, K)
            R_hat = predict(R_te, M_te)
        except Exception as e:
            print(f"  [{method_name}] t={t} FAILED: {e}")
            continue
        sse += (((R_te - R_hat) ** 2) * M_te).sum()
        sst += ((R_te ** 2) * M_te).sum()
        n_obs += int(M_te.sum())
        n_done += 1
        if n_done % 12 == 0:
            print(f"  [{method_name}] refit {n_done}/{len(refit_dates)} "
                  f"({time.time()-t0:.1f}s)")

    r2 = 1 - sse / sst if sst > 0 else np.nan
    return {"method": method_name, "n_obs": n_obs, "r2": r2,
            "n_refit": n_done, "elapsed": time.time() - t0}


def main(methods=None):
    print(f"Device: {DEVICE}")
    print(f"Panel: {Config.PANEL_PATH}")
    df = load_panel()
    print(f"Panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    print(f"Universe: {len(universe)} permnos")
    R, M = build_matrices(df, universe, dates)
    print(f"R: {R.shape}, coverage: {M.mean():.3f}")
    print(f"Target std: {R[M > 0].std():.4f}")

    if methods is None:
        methods = list(METHODS.keys())

    results = []
    for m in methods:
        print(f"\n=== {m} ===")
        try:
            res = evaluate_method(R, M, dates, m)
            results.append(res)
            print(f"  R^2 = {res['r2']:.4f}  "
                  f"n_obs = {res['n_obs']}  "
                  f"({res['elapsed']:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Summary: Reconstruction R^2 on log-RV panel ===")
    print(f"{'method':<14} {'R^2':>10} {'n_obs':>10} {'sec':>8}")
    print("-" * 46)
    for res in sorted(results, key=lambda r: -r["r2"]):
        print(f"{res['method']:<14} {res['r2']:>10.4f} "
              f"{res['n_obs']:>10} {res['elapsed']:>8.1f}")

    os.makedirs(Config.OUT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(Config.OUT_DIR, "recon_r2_vol.csv"), index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(methods=args if args else None)