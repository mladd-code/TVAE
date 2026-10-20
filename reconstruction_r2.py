"""
reconstruction_r2.py
====================
Out-of-sample reconstruction R^2 for factor models.

For each rolling window:
  1. Fit each model's factors and loadings on the training window.
  2. Predict the next T_TEST days of returns.
  3. Accumulate squared errors.

R^2 = 1 - SSE / SST, where SSE and SST are pooled over all OOS (i, t).

Methods
-------
  static_pca  : PCA on the training window
  lpca        : local PCA (Su-Wang 2017, FWYZ 2024)
  static_ae   : static AE with PCA warm-start
  tvaeg       : time-conditioned AE with PCA warm-start
  tvaec_k5    : TV-AE-C with raw-return encoder (K1 = 5)
  tvaec_k3    : TV-AE-C with raw-return encoder (K1 = 3)

Usage
-----
  python reconstruction_r2.py
  python reconstruction_r2.py static_pca lpca tvaeg
"""

from __future__ import annotations
import sys, os, time
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from local_pca import silverman, local_pca


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config:
    PANEL_PATH = "refinitiv_data/panel_daily.parquet"
    CHAR_PATH = "refinitiv_data/panel_daily_char.parquet"
    OUT_DIR = "recon_results"

    T_EST = 250
    T_TEST = 21
    REBAL = 21

    P = 100
    AE_N_EPOCHS = 200
    AE_LR = 1e-2
    AE_WEIGHT_DECAY = 1e-4

    OOS_START = "2010-01-01"
    OOS_END = "2024-12-31"


CHAR_COLS = ["size", "mom_12_1", "rev_1m",
             "vol_12m", "beta_12m", "idio_vol_12m"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(Config.PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def load_char() -> Optional[pd.DataFrame]:
    if not os.path.exists(Config.CHAR_PATH):
        return None
    df = pd.read_parquet(Config.CHAR_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    # Lag characteristics by one day to avoid same-day look-ahead
    for c in CHAR_COLS:
        df[c] = df.groupby("permno")[c].shift(1)
    return df


def select_universe(df: pd.DataFrame, p: int, burn_end: str) -> List[str]:
    burn = df[df["date"] < pd.Timestamp(burn_end)]
    avg_mcap = burn.groupby("permno")["mcap"].mean().sort_values(ascending=False)
    return avg_mcap.head(p).index.tolist()


def build_matrices(df, char_df, universe, dates):
    """
    Build R (T, p), M (T, p), C (T, p, P).
    """
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

    C = None
    if char_df is not None:
        char_df = char_df.copy()
        char_df["date_ns"] = char_df["date"].values.astype("datetime64[ns]").astype(np.int64)
        sub_c = char_df[char_df["permno"].isin(universe)].copy()
        sub_c["row_idx"] = sub_c["date_ns"].map(date_to_row)
        sub_c["col_idx"] = sub_c["permno"].map(permno_to_col)
        sub_c = sub_c.dropna(subset=["row_idx", "col_idx"])
        sub_c["row_idx"] = sub_c["row_idx"].astype(int)
        sub_c["col_idx"] = sub_c["col_idx"].astype(int)
        P = len(CHAR_COLS)
        C = np.zeros((T, p, P), dtype=np.float32)
        C[sub_c["row_idx"].values, sub_c["col_idx"].values, :] = \
            sub_c[CHAR_COLS].fillna(0).values.astype(np.float32)
    return R, M, C


def _demean(R: np.ndarray, M: np.ndarray):
    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    return mu, (R - mu[None, :]) * M


# --------------------------------------------------------------------------- #
# Model architectures (PCA warm-start)
# --------------------------------------------------------------------------- #
def _pca_warm_start(Rc, M, K1):
    _, _, Vt = np.linalg.svd(Rc * M, full_matrices=False)
    return Vt[:K1, :].T.astype(np.float32)


class _StaticAE(nn.Module):
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.W_pca = nn.Parameter(torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * p, 16), nn.ReLU(),
            nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.beta = nn.Parameter(torch.randn(p, K1) * 0.01)

    def encode(self, X, M):
        return (X * M) @ self.W_pca + self.mlp(torch.cat([X * M, M], dim=1))

    def forward(self, X, M):
        F = self.encode(X, M)
        return F @ self.beta.T, F


class _TVEG(nn.Module):
    def __init__(self, p, K1, W_pca_init):
        super().__init__()
        self.p = p
        self.K1 = K1
        self.W_pca = nn.Parameter(torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * p + 1, 16), nn.ReLU(),
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


class _TVAEC(nn.Module):
    def __init__(self, p, P_char, K1, W_pca_init):
        super().__init__()
        self.p = p
        self.P = P_char
        self.K1 = K1
        self.W_pca = nn.Parameter(torch.tensor(W_pca_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * p + 1, 16), nn.ReLU(),
            nn.Linear(16, K1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.beta_net = nn.Sequential(
            nn.Linear(P_char + 1, 16), nn.ReLU(),
            nn.Linear(16, K1))

    def encode(self, X, M, tau):
        return ((X * M) @ self.W_pca
                + self.mlp(torch.cat([X * M, M, tau], dim=1)))

    def beta_all(self, Z, tau):
        T, p, P = Z.shape
        tau_exp = tau.unsqueeze(1).expand(T, p, 1)
        inp = torch.cat([Z, tau_exp], dim=2)
        return self.beta_net(inp.reshape(T * p, P + 1)).reshape(T, p, self.K1)

    def forward(self, X, M, Z, tau):
        F = self.encode(X, M, tau)
        B = self.beta_all(Z, tau)
        return torch.einsum('tk,tpk->tp', F, B), F


# --------------------------------------------------------------------------- #
# Fit / predict closures
# --------------------------------------------------------------------------- #
def fit_pca(R_tr, M_tr, C_tr, K):
    mu, Rc = _demean(R_tr, M_tr)
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    Lambda = Vt[:K, :].T

    def predict(R_te, M_te, C_te):
        Rc_te = (R_te - mu[None, :]) * M_te
        F_te = Rc_te @ Lambda
        Rc_hat = F_te @ Lambda.T
        return Rc_hat + mu[None, :]
    return predict


def fit_lpca(R_tr, M_tr, C_tr, K):
    mu, Rc = _demean(R_tr, M_tr)
    R_filled = Rc * M_tr
    bw = silverman(R_filled)
    res = local_pca(R_filled, r=R_filled.shape[0] - 1, bandwidth=bw, m=K)
    Lambda = res["loadings"]  # (p, K)

    def predict(R_te, M_te, C_te):
        Rc_te = (R_te - mu[None, :]) * M_te
        F_te = Rc_te @ Lambda
        Rc_hat = F_te @ Lambda.T
        return Rc_hat + mu[None, :]
    return predict


def fit_static_ae(R_tr, M_tr, C_tr, K):
    mu, Rc = _demean(R_tr, M_tr)
    W_pca = _pca_warm_start(Rc, M_tr, K)
    p = R_tr.shape[1]
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

    def predict(R_te, M_te, C_te):
        with torch.no_grad():
            Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                              dtype=torch.float32).to(DEVICE)
            Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
            F = model.encode(Xt, Mt)
            Rc_hat = F @ model.beta.T
        return Rc_hat.cpu().numpy() + mu[None, :]
    return predict


def fit_tvaeg(R_tr, M_tr, C_tr, K):
    T = R_tr.shape[0]; p = R_tr.shape[1]
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

    def predict(R_te, M_te, C_te):
        T_te = R_te.shape[0]
        with torch.no_grad():
            Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                              dtype=torch.float32).to(DEVICE)
            Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
            tau_te = np.full((T_te, 1), 1.0, dtype=np.float32)
            tau_te_t = torch.tensor(tau_te, dtype=torch.float32).to(DEVICE)
            F = model.encode(Xt, Mt, tau_te_t)
            B = model.beta_all(tau_te_t)
            Rc_hat = torch.einsum('tk,tpk->tp', F, B)
        return Rc_hat.cpu().numpy() + mu[None, :]
    return predict


def fit_tvaec(R_tr, M_tr, C_tr, K):
    if C_tr is None:
        raise ValueError("TV-AE-C requires characteristics")
    T = R_tr.shape[0]; p = R_tr.shape[1]; P = C_tr.shape[2]
    mu, Rc = _demean(R_tr, M_tr)
    W_pca = _pca_warm_start(Rc, M_tr, K)
    model = _TVAEC(p, P, K, W_pca).to(DEVICE)
    X = torch.tensor(Rc, dtype=torch.float32).to(DEVICE)
    Mu = torch.tensor(M_tr, dtype=torch.float32).to(DEVICE)
    Z = torch.tensor(C_tr, dtype=torch.float32).to(DEVICE)
    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(DEVICE)
    opt = torch.optim.Adam([
        {'params': [model.W_pca], 'weight_decay': 0.0},
        {'params': model.mlp.parameters(), 'weight_decay': Config.AE_WEIGHT_DECAY},
        {'params': model.beta_net.parameters(),
         'weight_decay': Config.AE_WEIGHT_DECAY},
    ], lr=Config.AE_LR)
    for _ in range(Config.AE_N_EPOCHS):
        opt.zero_grad()
        x_hat, _ = model(X, Mu, Z, tau_t)
        loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
        loss.backward(); opt.step()
    model.eval()

    def predict(R_te, M_te, C_te):
        T_te = R_te.shape[0]
        with torch.no_grad():
            Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                              dtype=torch.float32).to(DEVICE)
            Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
            Zt = torch.tensor(C_te, dtype=torch.float32).to(DEVICE)
            tau_te = np.full((T_te, 1), 1.0, dtype=np.float32)
            tau_te_t = torch.tensor(tau_te, dtype=torch.float32).to(DEVICE)
            F = model.encode(Xt, Mt, tau_te_t)
            B = model.beta_all(Zt, tau_te_t)
            Rc_hat = torch.einsum('tk,tpk->tp', F, B)
        return Rc_hat.cpu().numpy() + mu[None, :]
    return predict


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #
METHODS = {
    "static_pca": (fit_pca, 5),
    "lpca": (fit_lpca, 5),
    "static_ae": (fit_static_ae, 5),
    "tvaeg": (fit_tvaeg, 5),
    "tvaec_k5": (fit_tvaec, 5),
    "tvaec_k3": (fit_tvaec, 3),
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_method(R, M, C, dates, method_name):
    fit_fn, K = METHODS[method_name]
    oos_start_idx = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end_idx = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))

    refit_dates = list(range(oos_start_idx, oos_end_idx, Config.REBAL))

    sse = 0.0
    sst = 0.0
    n_obs = 0
    t0 = time.time()
    n_done = 0

    for t in refit_dates:
        if t < Config.T_EST:
            continue
        R_tr = R[t - Config.T_EST:t, :]
        M_tr = M[t - Config.T_EST:t, :]
        C_tr = C[t - Config.T_EST:t, :, :] if C is not None else None

        hold_end = min(t + Config.T_REBAL if hasattr(Config, "T_REBAL")
                       else t + Config.REBAL, len(dates))
        R_te = R[t:hold_end, :]
        M_te = M[t:hold_end, :]
        C_te = C[t:hold_end, :, :] if C is not None else None

        try:
            predict = fit_fn(R_tr, M_tr, C_tr, K)
            R_hat = predict(R_te, M_te, C_te)
        except Exception as e:
            print(f"  [{method_name}] t={t} FAILED: {e}")
            continue

        err = (R_te - R_hat) ** 2 * M_te
        sse += err.sum()
        sst += (R_te ** 2 * M_te).sum()
        n_obs += int(M_te.sum())
        n_done += 1
        if n_done % 12 == 0:
            print(f"  [{method_name}] refit {n_done}/{len(refit_dates)} "
                  f"({time.time()-t0:.1f}s)")

    r2 = 1 - sse / sst if sst > 0 else np.nan
    return {"method": method_name,
            "n_obs": n_obs,
            "sse": sse,
            "sst": sst,
            "r2": r2,
            "n_refit": n_done,
            "elapsed": time.time() - t0}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(methods=None):
    print(f"Device: {DEVICE}")
    df = load_panel()
    char_df = load_char()
    print(f"Panel: {len(df)} rows, {df['permno'].nunique()} permnos, "
          f"{df['date'].nunique()} dates")
    if char_df is not None:
        print(f"Char panel: {len(char_df)} rows, "
              f"{len(CHAR_COLS)} characteristics")

    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    print(f"Universe: {len(universe)} permnos")
    R, M, C = build_matrices(df, char_df, universe, dates)
    print(f"R: {R.shape}, M coverage: {M.mean():.3f}")
    if C is not None:
        print(f"C: {C.shape}")

    if methods is None:
        methods = list(METHODS.keys())

    results = []
    for m in methods:
        print(f"\n=== {m} ===")
        try:
            res = evaluate_method(R, M, C, dates, m)
            results.append(res)
            print(f"  R^2 = {res['r2']:.4f}  "
                  f"n_obs = {res['n_obs']}  "
                  f"refits = {res['n_refit']}  "
                  f"({res['elapsed']:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n=== Summary: Out-of-sample reconstruction R^2 ===")
    print(f"{'method':<14} {'R^2':>10} {'n_obs':>10} {'refits':>8} {'sec':>8}")
    print("-" * 54)
    for res in sorted(results, key=lambda r: -r["r2"]):
        print(f"{res['method']:<14} {res['r2']:>10.4f} "
              f"{res['n_obs']:>10} {res['n_refit']:>8} "
              f"{res['elapsed']:>8.1f}")

    os.makedirs(Config.OUT_DIR, exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(Config.OUT_DIR, "recon_r2.csv"), index=False)
    print(f"\nSaved to {Config.OUT_DIR}/recon_r2.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        main(methods=args)
    else:
        main()