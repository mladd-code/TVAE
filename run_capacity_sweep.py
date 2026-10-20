"""
run_capacity_sweep.py
---------------------
Sweep decoder width, weight decay, and bottleneck width for TV-AE-G.

Two nonlinear time-varying DGPs are used, since those are where the model
has the largest theoretical advantage and where the previous diagnostic
showed overfitting.

Outputs:
  capacity_sweep.csv   : long-format results, one row per (rep, dgp, config)
  Printed summary      : mean MSE per (dgp, config), sorted
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import time
import itertools
import numpy as np
import pandas as pd
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate
from tvae_global import fit_tvae_global


def main():
    N, T, K = 100, 200, 2
    n_reps = 10
    n_seeds = 1         # single seed per fit; raise later for final tables
    n_epochs = 300
    patience = 30
    encoder_width = 64
    encoder_time = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}, n_reps = {n_reps}")

    # --- Sweep grid ---
    decoder_widths = [8, 16, 32]
    weight_decays  = [0.0, 1e-4, 1e-3]
    K1_values      = [2, 3]

    # --- DGPs to sweep over ---
    configs = [
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    # --- Loop ---
    rows = []
    grid = list(itertools.product(decoder_widths, weight_decays, K1_values))
    total_fits = len(grid) * len(configs) * n_reps
    fit_counter = 0
    t0 = time.time()

    for (nonlin, tv) in configs:
        for rep in range(n_reps):
            seed_dgp = 1000 + rep
            cfg = DGPConfig(N=N, T=T, K=K, nonlinearity=nonlin,
                            time_variation=tv, error="iid", seed=seed_dgp)
            sim = generate_dgp(cfg)
            X, X_star = sim["X"], sim["X_star"]

            for (dw, wd, K1) in grid:
                fit_counter += 1
                res = fit_tvae_global(
                    X, K1=K1,
                    n_seeds=n_seeds,
                    n_epochs=n_epochs,
                    lr=1e-3,
                    batch_size=16,
                    patience=patience,
                    encoder_width=encoder_width,
                    decoder_width=dw,
                    encoder_time=encoder_time,
                    weight_decay=wd,
                    device=device, verbose=False,
                )
                mse = evaluate(res["X_hat"], X_star)["mse_common"]
                rows.append({
                    'rep': rep,
                    'dgp': f"{nonlin}, {tv}",
                    'decoder_width': dw,
                    'weight_decay': wd,
                    'K1': K1,
                    'mse': mse,
                })
                elapsed = time.time() - t0
                avg = elapsed / fit_counter
                eta = avg * (total_fits - fit_counter)
                print(f"  [{fit_counter}/{total_fits}] "
                      f"{nonlin[:4]}, {tv[:9]}, dw={dw:2d}, "
                      f"wd={wd:.0e}, K1={K1}  MSE={mse:.4f}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv("capacity_sweep.csv", index=False)

    # --- Summary tables ---
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        summary = (sub.groupby(['decoder_width', 'weight_decay', 'K1'])
                      ['mse'].agg(['mean', 'std'])
                      .sort_values('mean'))
        print(f"\n=== Top 10 configs for {key} (n_reps = {n_reps}) ===")
        print(summary.head(10).to_string())

    # --- Marginal effects ---
    print("\n=== Marginal effect of decoder_width ===")
    print(df.groupby('decoder_width')['mse'].agg(['mean', 'std']).to_string())
    print("\n=== Marginal effect of weight_decay ===")
    print(df.groupby('weight_decay')['mse'].agg(['mean', 'std']).to_string())
    print("\n=== Marginal effect of K1 ===")
    print(df.groupby('K1')['mse'].agg(['mean', 'std']).to_string())


if __name__ == "__main__":
    main()