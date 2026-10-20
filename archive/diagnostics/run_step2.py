"""
run_step2.py
------------
Compare static PCA and Local PCA (SW 2017) across a small grid of DGPs.

Primary factor metric: subspace distance (rotation-, sign-, scale-invariant).
Secondary factor metric: Procrustes distance (diagnostic only).
"""

import numpy as np
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X


def run_local_pca(X_NT: np.ndarray, K: int):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    X_hat_NT = reconstruct_X(res).T
    F_hat_KT = res["f_hat"].T
    return X_hat_NT, F_hat_KT


def main():
    N, T, K = 100, 200, 2
    seed = 42

    configs = [
        ("linear",      "constant"),
        ("linear",      "logistic"),
        ("linear",      "single_break"),
        ("linear",      "multiple_logistic"),
        ("exponential", "constant"),
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    header = (f"{'DGP':<30} "
              f"{'PCA mse':>9} {'LPCA mse':>9} "
              f"{'PCA fdist':>10} {'LPCA fdist':>11} "
              f"{'PCA pdist':>10} {'LPCA pdist':>11}")
    print(header)
    print("-" * len(header))

    for nonlin, tv in configs:
        cfg = DGPConfig(N=N, T=T, K=K,
                        nonlinearity=nonlin,
                        time_variation=tv,
                        error="iid",
                        seed=seed)
        sim = generate_dgp(cfg)
        X, X_star, F_true = sim["X"], sim["X_star"], sim["F"]

        pca_res = pca_baseline(X, K)
        m_pca = evaluate(pca_res["X_hat"], X_star,
                         pca_res["F_hat"], F_true)

        X_hat_lpca, F_hat_lpca = run_local_pca(X, K)
        m_lpca = evaluate(X_hat_lpca, X_star, F_hat_lpca, F_true)

        label = f"{nonlin}, {tv}"
        print(f"{label:<30} "
              f"{m_pca['mse_common']:>9.4f} "
              f"{m_lpca['mse_common']:>9.4f} "
              f"{m_pca['factor_distance']:>10.4f} "
              f"{m_lpca['factor_distance']:>11.4f} "
              f"{m_pca['factor_procrustes']:>10.4f} "
              f"{m_lpca['factor_procrustes']:>11.4f}")


if __name__ == "__main__":
    main()