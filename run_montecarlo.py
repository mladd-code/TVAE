"""
run_montecarlo.py
-----------------
Monte Carlo replication of the Step 5 comparison.

For each of R replications and each DGP, this runs
  PCA, LPCA, StaticAE, TVAE-G
and records the pooled MSE of the common component.

Outputs:
  mc_results.csv  : long-format results, one row per (rep, dgp, method)
  Printed summary : mean ± std of MSE per (dgp, method)
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import time
import numpy as np
import pandas as pd
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X
from static_ae import fit_static_ae
from tvae_global import fit_tvae_global


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T


def main():
    N, T, K = 100, 200, 2
    n_reps = 10          # start small; raise to 30 for the paper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}, n_reps = {n_reps}")

    configs = [
        ("linear",      "constant"),
        ("linear",      "logistic"),
        ("linear",      "single_break"),
        ("linear",      "multiple_logistic"),
        ("exponential", "constant"),
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    rows = []
    t0 = time.time()

    for rep in range(n_reps):
        seed_dgp = 1000 + rep
        for (nonlin, tv) in configs:
            cfg = DGPConfig(N=N, T=T, K=K,
                            nonlinearity=nonlin,
                            time_variation=tv,
                            error="iid", seed=seed_dgp)
            sim = generate_dgp(cfg)
            X, X_star = sim["X"], sim["X_star"]

            # PCA
            mse_pca = evaluate(pca_baseline(X, K)["X_hat"], X_star)["mse_common"]
            rows.append({'rep': rep, 'dgp': f"{nonlin}, {tv}",
                         'method': 'PCA', 'mse': mse_pca})

            # LPCA
            mse_lpca = evaluate(run_local_pca(X, K), X_star)["mse_common"]
            rows.append({'rep': rep, 'dgp': f"{nonlin}, {tv}",
                         'method': 'LPCA', 'mse': mse_lpca})

            # Static AE
            ae_res = fit_static_ae(X, K1=3, n_seeds=2, n_epochs=300,
                                   device=device, verbose=False)
            mse_ae = evaluate(ae_res["X_hat"], X_star)["mse_common"]
            rows.append({'rep': rep, 'dgp': f"{nonlin}, {tv}",
                         'method': 'StaticAE', 'mse': mse_ae})

            # TV-AE-G
            tvae_res = fit_tvae_global(X, K1=3, n_seeds=2, n_epochs=300,
                                       device=device, verbose=False)
            mse_tvae = evaluate(tvae_res["X_hat"], X_star)["mse_common"]
            rows.append({'rep': rep, 'dgp': f"{nonlin}, {tv}",
                         'method': 'TVAE-G', 'mse': mse_tvae})

            elapsed = time.time() - t0
            print(f"  rep {rep+1}/{n_reps}, {nonlin:12s}, {tv:18s}  "
                  f"[{elapsed/60:.1f} min elapsed]")

    df = pd.DataFrame(rows)
    df.to_csv("mc_results.csv", index=False)

    # Summary table
    print("\n=== Mean MSE (std in parentheses) over "
          f"{n_reps} reps ===")
    methods = ['PCA', 'LPCA', 'StaticAE', 'TVAE-G']
    header = f"{'DGP':<28} " + " ".join(f"{m:>16}" for m in methods)
    print(header)
    print("-" * len(header))
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        row = f"{key:<28} "
        for m in methods:
            vals = sub[sub['method'] == m]['mse'].values
            row += f"{vals.mean():>7.4f} ({vals.std():>5.3f}) "
        print(row)


if __name__ == "__main__":
    main()