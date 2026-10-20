"""
diag_ae.py
----------
Trains one static AE and one TV-AE-G on a single estimation window.
Reports:
  - training loss (initial, final, ratio)
  - reconstructed R^2 on the training window
  - factor std, residual std
  - condition number of the resulting MVP covariance
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mvp_app import (
    Config, load_daily_panel, select_universe, build_returns_matrix,
    _StaticAECov, _TVEGCov, cov_sample, mvp_weights, _psd_repair,
)


def train_and_diagnose(model, kind, Rc, M, tau, device):
    X = torch.tensor(Rc, dtype=torch.float32).to(device)
    Mu = torch.tensor(M, dtype=torch.float32).to(device)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=Config.AE_LR,
                                 weight_decay=Config.AE_WEIGHT_DECAY)
    losses = []
    t0 = time.time()
    for ep in range(Config.AE_N_EPOCHS):
        optimizer.zero_grad()
        if kind == "static":
            x_hat, F = model(X, Mu)
        else:
            x_hat, F = model(X, Mu, tau_t)
        loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()
        if ep % 50 == 0 or ep == Config.AE_N_EPOCHS - 1:
            losses.append((ep, loss.item()))
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        if kind == "static":
            x_hat, F = model(X, Mu)
        else:
            x_hat, F = model(X, Mu, tau_t)

    x_hat = x_hat.cpu().numpy()
    F = F.cpu().numpy()
    Xc = Rc * M
    sse = ((x_hat - Xc) ** 2 * M).sum()
    sst = (Xc ** 2 * M).sum()
    r2 = 1 - sse / sst
    resid = (Xc - x_hat) * M
    res_std = resid.std()
    F_std = F.std(axis=0)

    print(f"  [{kind}]")
    print(f"    training time: {elapsed:.2f}s")
    print(f"    loss trajectory (ep, loss):")
    for ep, l in losses:
        print(f"      {ep:>4d}  {l:.6f}")
    print(f"    initial/final ratio: {losses[0][1]/losses[-1][1]:.2f}x")
    print(f"    reconstruction R^2 (in-sample): {r2:.4f}")
    print(f"    residual std: {res_std:.6f}")
    print(f"    factor std per dim: {np.round(F_std, 5).tolist()}")
    return F, x_hat


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    df = load_daily_panel(Config.PANEL_PATH)
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    R, M = build_returns_matrix(df, universe, dates)

    # One window: the most recent T_EST days before OOS_END
    end_idx = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    t = min(end_idx, len(dates) - 1)
    R_win = R[t - Config.T_EST:t, :]
    M_win = M[t - Config.T_EST:t, :]
    T, p = R_win.shape
    print(f"Window: T={T}, p={p}")

    counts = M_win.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R_win * M_win).sum(axis=0) / counts
    Rc = (R_win - mu[None, :]) * M_win
    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)

    print("\n--- Sample covariance baseline ---")
    S_sample = cov_sample(R_win, M_win)
    w_s = mvp_weights(S_sample)
    print(f"  MVP variance: {w_s @ S_sample @ w_s:.6e}")
    print(f"  Sample cov condition number: {np.linalg.cond(S_sample):.2e}")

    print("\n--- Static AE ---")
    model_s = _StaticAECov(p, Config.K1).to(device)
    train_and_diagnose(model_s, "static", Rc, M_win, tau, device)

    print("\n--- TV-AE-G ---")
    model_t = _TVEGCov(p, Config.K1).to(device)
    train_and_diagnose(model_t, "tvaeg", Rc, M_win, tau, device)


if __name__ == "__main__":
    main()