"""
reconstruction_r2_crypto_tau.py
===============================
Tests the effect of τ handling in TV-AE-G on reconstruction R^2.

Currently, `fit_tvaeg` freezes τ = 1.0 for the entire test window. This
is a conservative choice: the network never extrapolates beyond its
training range, but it also cannot update the loading function during
the test window. Two alternatives are tested here:

  frozen      : τ_te = 1.0 (current)
  ramp_small  : τ_te = linspace(1.0, 1.05, T_te) — 5% extension
  ramp_full   : τ_te = linspace(1.0, 1.0 + T_te/T_tr, T_te)

The "ramp_full" mode matches the natural extrapolation of τ = t/T_tr
into the test window. Both ramps are controlled extrapolations of the
ReLU network.

Runtime: ~5-7 minutes for three variants.
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from reconstruction_r2_crypto import (
    Config, DEVICE, load_panel, select_universe, build_matrices,
    _demean, _pca_warm_start, _TVEG,
)


def make_fit_tvaeg_ramp(ramp_coef: float):
    """
    ramp_coef = 0.0 : frozen τ at 1.0
    ramp_coef = 0.05: extend τ by 5% into test
    ramp_coef = 1.0 : extend τ by T_te / T_tr (natural extrapolation)
    """
    def fit(R_tr, M_tr, K):
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
            {'params': model.mlp.parameters(),
             'weight_decay': Config.AE_WEIGHT_DECAY},
            {'params': [model.W1, model.b1, model.W2, model.b2],
             'weight_decay': Config.AE_WEIGHT_DECAY},
        ], lr=Config.AE_LR)
        for _ in range(Config.AE_N_EPOCHS):
            opt.zero_grad()
            x_hat, _ = model(X, Mu, tau_t)
            loss = ((x_hat - X) ** 2 * Mu).sum() / Mu.sum().clamp_min(1e-12)
            loss.backward()
            opt.step()
        model.eval()

        def predict(R_te, M_te):
            T_te = R_te.shape[0]
            with torch.no_grad():
                Xt = torch.tensor((R_te - mu[None, :]) * M_te,
                                  dtype=torch.float32).to(DEVICE)
                Mt = torch.tensor(M_te, dtype=torch.float32).to(DEVICE)
                if ramp_coef == 0.0:
                    tau_te = np.full((T_te, 1), 1.0, dtype=np.float32)
                else:
                    end = 1.0 + ramp_coef * (T_te / T)
                    tau_te = np.linspace(1.0, end, T_te,
                                         dtype=np.float32).reshape(-1, 1)
                tau_te_t = torch.tensor(tau_te,
                                        dtype=torch.float32).to(DEVICE)
                F = model.encode(Xt, Mt, tau_te_t)
                B = model.beta_all(tau_te_t)
                return (torch.einsum('tk,tpk->tp', F, B).cpu().numpy()
                        + mu[None, :])
        return predict
    return fit


VARIANTS = {
    "frozen":     make_fit_tvaeg_ramp(0.0),
    "ramp_small": make_fit_tvaeg_ramp(0.05),
    "ramp_full":  make_fit_tvaeg_ramp(1.0),
}


def evaluate_fit(fit_fn, K, R, M, dates):
    oos_start = np.searchsorted(
        dates, np.datetime64(Config.OOS_START).astype("datetime64[ns]"))
    oos_end = np.searchsorted(
        dates, np.datetime64(Config.OOS_END).astype("datetime64[ns]"))
    refit_dates = list(range(oos_start, oos_end, Config.REBAL))
    sse = sst = 0.0
    n_obs = 0
    per_window = []
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
        sse_w = (((R_te - R_hat) ** 2) * M_te).sum()
        sst_w = ((R_te ** 2) * M_te).sum()
        sse += sse_w
        sst += sst_w
        n_obs += int(M_te.sum())
        per_window.append((dates[t], sse_w, sst_w))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return r2, n_obs, per_window


def main():
    print(f"Device: {DEVICE}")
    df = load_panel()
    universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
    dates = np.sort(df["date"].unique())
    R, M = build_matrices(df, universe, dates)
    print(f"Panel: {R.shape}, coverage {M.mean():.3f}")

    results = {}
    per_window = {}
    t_start = time.time()
    for name, fit_fn in VARIANTS.items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        r2, n_obs, pw = evaluate_fit(fit_fn, Config.K1 if False else 5,
                                     R, M, dates)
        print(f"  R^2 = {r2:.4f}  n_obs = {n_obs}  "
              f"({time.time()-t0:.0f}s)")
        results[name] = r2
        per_window[name] = pw

    # Paired bootstrap between ramp variants and frozen
    print("\n=== Paired bootstrap over windows (frozen as baseline) ===")
    frozen_pw = per_window["frozen"]
    frozen_arr = np.array([[w[1], w[2]] for w in frozen_pw])
    for name in ["ramp_small", "ramp_full"]:
        var_arr = np.array([[w[1], w[2]] for w in per_window[name]])
        n = len(frozen_arr)
        rng = np.random.default_rng(42)
        diffs = np.empty(1000)
        for b in range(1000):
            idx = rng.integers(0, n, size=n)
            r2_f = 1 - frozen_arr[idx, 0].sum() / frozen_arr[idx, 1].sum()
            r2_v = 1 - var_arr[idx, 0].sum() / var_arr[idx, 1].sum()
            diffs[b] = r2_v - r2_f
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  R^2({name}) - R^2(frozen) = {diffs.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")

    # By year
    print("\n=== By year ===")
    years = sorted(set(pd.to_datetime(
        [w[0] for w in per_window["frozen"]]).year))
    header = f"{'year':<6} " + " ".join(f"{n:>12}" for n in VARIANTS)
    print(header)
    print("-" * len(header))
    for y in years:
        row = f"{y:<6} "
        for name in VARIANTS:
            pw = per_window[name]
            mask = pd.to_datetime([w[0] for w in pw]).year == y
            sse = sum(w[1] for w, m in zip(pw, mask) if m)
            sst = sum(w[2] for w, m in zip(pw, mask) if m)
            row += f"{1 - sse/sst:>12.4f} "
        print(row)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()