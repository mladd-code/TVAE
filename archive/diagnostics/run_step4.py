"""
run_step4.py
------------
Compare PCA, LPCA, static AE, and TV-AE-L across the DGP grid.

Verbose diagnostics are printed only for the first DGP to keep the log short.
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X
from static_ae import fit_static_ae
from tvae_local import fit_tvae_local


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T


def main():
    N, T, K = 100, 200, 2
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # TV-AE-L settings. Increase for final tables:
    #   M_GRID=20, N_SEEDS_TVAE=3, N_EPOCHS_TVAE=500
    M_GRID = 10
    N_SEEDS_TVAE = 3
    N_EPOCHS_TVAE = 500

    configs = [
        ("linear",      "constant"),
        ("linear",      "logistic"),
        ("linear",      "single_break"),
        ("linear",      "multiple_logistic"),
        ("exponential", "constant"),
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    header = (f"{'DGP':<28} "
              f"{'PCA':>9} {'LPCA':>9} {'StaticAE':>9} {'TVAE-L':>9} "
              f"| {'TVAE-NN':>9}")
    print(header)
    print("-" * len(header))

    for idx, (nonlin, tv) in enumerate(configs):
        verbose = (idx == 0)
        if verbose:
            print(f"\n[verbose diagnostics for DGP 1: {nonlin}, {tv}]")

        cfg = DGPConfig(N=N, T=T, K=K,
                        nonlinearity=nonlin,
                        time_variation=tv,
                        error="iid", seed=seed)
        sim = generate_dgp(cfg)
        X, X_star, F_true = sim["X"], sim["X_star"], sim["F"]

        m_pca = evaluate(pca_baseline(X, K)["X_hat"], X_star)
        m_lpca = evaluate(run_local_pca(X, K), X_star)

        ae_res = fit_static_ae(X, K1=3,
                               n_seeds=3, n_epochs=500,
                               device=device, verbose=False)
        m_ae = evaluate(ae_res["X_hat"], X_star)

        tvae_res = fit_tvae_local(X, K1=3,
                                  m=M_GRID,
                                  n_seeds=N_SEEDS_TVAE,
                                  n_epochs=N_EPOCHS_TVAE,
                                  device=device, verbose=verbose)
        m_tvae = evaluate(tvae_res["X_hat"], X_star)
        m_tvae_nn = evaluate(tvae_res["X_hat_nn"], X_star)

        if verbose:
            print(f"\n[summary for DGP 1]")

        label = f"{nonlin}, {tv}"
        print(f"{label:<28} "
              f"{m_pca['mse_common']:>9.4f} "
              f"{m_lpca['mse_common']:>9.4f} "
              f"{m_ae['mse_common']:>9.4f} "
              f"{m_tvae['mse_common']:>9.4f} | "
              f"{m_tvae_nn['mse_common']:>9.4f}")


if __name__ == "__main__":
    main()