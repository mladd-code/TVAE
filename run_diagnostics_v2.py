"""
run_diagnostic_v2.py
--------------------
Side-by-side comparison of two TV-AE-G variants on relevant DGPs:

  - TVAE-G (no time)   : encoder sees X_t only (previous version)
  - TVAE-G (time)      : encoder sees (X_t, t/T) concatenated (proposed)

Also reports StaticAE and LPCA as anchors.

Runs 10 replications per DGP to give mean and standard deviation.
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

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
    n_reps = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}, n_reps = {n_reps}")

    configs = [
        ("linear",      "constant"),
        ("linear",      "logistic"),
        ("linear",      "single_break"),
        ("exponential", "constant"),
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    rows = []
    for rep in range(n_reps):
        seed_dgp = 1000 + rep
        for (nonlin, tv) in configs:
            cfg = DGPConfig(N=N, T=T, K=K,
                            nonlinearity=nonlin,
                            time_variation=tv,
                            error="iid", seed=seed_dgp)
            sim = generate_dgp(cfg)
            X, X_star = sim["X"], sim["X_star"]

            mse_lpca = evaluate(run_local_pca(X, K), X_star)["mse_common"]

            ae_res = fit_static_ae(X, K1=3, n_seeds=2, n_epochs=300,
                                   device=device, verbose=False)
            mse_ae = evaluate(ae_res["X_hat"], X_star)["mse_common"]

            res_no_time = fit_tvae_global(
                X, K1=3, n_seeds=2, n_epochs=300,
                encoder_time=False, device=device, verbose=False)
            mse_no_time = evaluate(res_no_time["X_hat"], X_star)["mse_common"]

            res_time = fit_tvae_global(
                X, K1=3, n_seeds=2, n_epochs=300,
                encoder_time=True, device=device, verbose=False)
            mse_time = evaluate(res_time["X_hat"], X_star)["mse_common"]

            key = f"{nonlin}, {tv}"
            rows.append({'rep': rep, 'dgp': key, 'method': 'LPCA',
                         'mse': mse_lpca})
            rows.append({'rep': rep, 'dgp': key, 'method': 'StaticAE',
                         'mse': mse_ae})
            rows.append({'rep': rep, 'dgp': key, 'method': 'TVAEG-decoder',
                         'mse': mse_no_time})
            rows.append({'rep': rep, 'dgp': key, 'method': 'TVAEG-both',
                         'mse': mse_time})
            print(f"  rep {rep+1}/{n_reps}, {key}")

    df = pd.DataFrame(rows)
    df.to_csv("mc_diagnostic_v2.csv", index=False)

    print("\n=== Mean MSE (std in parentheses) ===")
    methods = ['LPCA', 'StaticAE', 'TVAEG-decoder', 'TVAEG-both']
    header = f"{'DGP':<28} " + " ".join(f"{m:>18}" for m in methods)
    print(header)
    print("-" * len(header))
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        row = f"{key:<28} "
        for m in methods:
            vals = sub[sub['method'] == m]['mse'].values
            row += f"{vals.mean():>9.4f} ({vals.std():>5.3f}) "
        print(row)


if __name__ == "__main__":
    main()