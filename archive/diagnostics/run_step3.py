"""
run_step3.py (refined)
----------------------
Compare static PCA, LPCA (SW 2017), and the SX-style static AE
with more seeds/epochs, and over a range of bottleneck widths K1.
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X
from static_ae import fit_static_ae


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T, res["f_hat"].T


def main():
    N, T, K = 100, 200, 2
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    configs = [
        ("linear",      "constant"),
        ("linear",      "logistic"),
        ("linear",      "single_break"),
        ("linear",      "multiple_logistic"),
        ("exponential", "constant"),
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    for K1 in [2, 3, 4]:
        print(f"\n================ K1 = {K1} ================")
        header = (f"{'DGP':<28} "
                  f"{'PCA':>9} {'LPCA':>9} {'StaticAE':>9}")
        print(header)
        print("-" * len(header))

        for nonlin, tv in configs:
            cfg = DGPConfig(N=N, T=T, K=K,
                            nonlinearity=nonlin,
                            time_variation=tv,
                            error="iid", seed=seed)
            sim = generate_dgp(cfg)
            X, X_star, F_true = sim["X"], sim["X_star"], sim["F"]

            pca_res = pca_baseline(X, K)
            m_pca = evaluate(pca_res["X_hat"], X_star,
                             pca_res["F_hat"], F_true)

            X_hat_lpca, F_hat_lpca = run_local_pca(X, K)
            m_lpca = evaluate(X_hat_lpca, X_star, F_hat_lpca, F_true)

            ae_res = fit_static_ae(X, K1=K1,
                                   n_seeds=5, n_epochs=500,
                                   device=device, verbose=False)
            m_ae = evaluate(ae_res["X_hat"], X_star)

            label = f"{nonlin}, {tv}"
            print(f"{label:<28} "
                  f"{m_pca['mse_common']:>9.4f} "
                  f"{m_lpca['mse_common']:>9.4f} "
                  f"{m_ae['mse_common']:>9.4f}")


if __name__ == "__main__":
    main()