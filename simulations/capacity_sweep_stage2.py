"""
run_capacity_sweep_v2.py
------------------------
Extended capacity sweep for TV-AE-G.

Sweeps:
  weight_decay   in {0, 1e-3, 5e-3, 1e-2}
  decoder_width  in {16, 32, 64}
  encoder_width  in {32, 64, 128}
with K1 = 3 fixed (from previous sweep: K1=3 beats K1=2 consistently).

Runs on the two nonlinear time-varying DGPs where TV-AE-G should excel.

n_reps = 5, n_seeds = 2 to keep runtime manageable. The final comparison
will use the chosen config with n_seeds = 3.
"""

import sys
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

# ---- Tee logging ----
class _Tee:
    def __init__(self, f):
        self.f = f
    def write(self, d):
        sys.__stdout__.write(d)
        self.f.write(d)
    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

def _setup_tee(path):
    return _Tee(open(path, "w"))

sys.stdout = _setup_tee("run_capacity_sweep_v2.txt")

import time
import itertools
import numpy as np
import pandas as pd
import torch
from tvae.dgp import DGPConfig, generate_dgp, evaluate
from tvae.models.tvae_global import fit_tvae_global


def main():
    N, T, K = 100, 200, 2
    n_reps = 10
    n_seeds = 2
    n_epochs = 300
    patience = 30
    K1 = 3
    encoder_time = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"n_reps = {n_reps}, n_seeds = {n_seeds}, "
          f"K1 = {K1}, n_epochs = {n_epochs}")

    # --- Sweep grid ---
    weight_decays  = [0.0, 1e-3, 5e-3, 1e-2]
    decoder_widths = [16, 32, 64]
    encoder_widths = [32, 64, 128]

    configs = [
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    grid = list(itertools.product(weight_decays, decoder_widths,
                                  encoder_widths))
    total_fits = len(grid) * len(configs) * n_reps
    print(f"Total configurations: {len(grid)}, total fits: {total_fits}")
    print(f"Grid:")
    for (wd, dw, ew) in grid:
        print(f"  wd={wd:.0e}, dw={dw:3d}, ew={ew:3d}")

    rows = []
    fit_counter = 0
    t0 = time.time()

    for (nonlin, tv) in configs:
        print(f"\n{'='*70}\nDGP: {nonlin}, {tv}\n{'='*70}")
        for rep in range(n_reps):
            seed_dgp = 1000 + rep
            cfg = DGPConfig(N=N, T=T, K=K, nonlinearity=nonlin,
                            time_variation=tv, error="iid", seed=seed_dgp)
            sim = generate_dgp(cfg)
            X, X_star = sim["X"], sim["X_star"]

            for (wd, dw, ew) in grid:
                fit_counter += 1
                res = fit_tvae_global(
                    X, K1=K1,
                    n_seeds=n_seeds,
                    n_epochs=n_epochs,
                    lr=1e-3,
                    batch_size=16,
                    patience=patience,
                    encoder_width=ew,
                    decoder_width=dw,
                    encoder_time=encoder_time,
                    weight_decay=wd,
                    device=device, verbose=False,
                )
                mse = evaluate(res["X_hat"], X_star)["mse_common"]
                rows.append({
                    'rep': rep,
                    'dgp': f"{nonlin}, {tv}",
                    'weight_decay': wd,
                    'decoder_width': dw,
                    'encoder_width': ew,
                    'K1': K1,
                    'mse': mse,
                })
                elapsed = time.time() - t0
                avg = elapsed / fit_counter
                eta = avg * (total_fits - fit_counter)
                print(f"  [{fit_counter}/{total_fits}] "
                      f"{nonlin[:4]}, {tv[:9]}, "
                      f"wd={wd:.0e}, dw={dw:3d}, ew={ew:3d}  "
                      f"MSE={mse:.4f}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv("capacity_sweep_v2.csv", index=False)

    # --- Per-DGP top-10 ---
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        summary = (sub.groupby(['weight_decay', 'decoder_width',
                                'encoder_width'])['mse']
                      .agg(['mean', 'std'])
                      .sort_values('mean'))
        print(f"\n=== Top 10 configs for {key} (n_reps={n_reps}) ===")
        print(summary.head(10).to_string())

    # --- Marginal effects ---
    print("\n=== Marginal effect of weight_decay ===")
    print(df.groupby('weight_decay')['mse'].agg(['mean', 'std']).to_string())
    print("\n=== Marginal effect of decoder_width ===")
    print(df.groupby('decoder_width')['mse'].agg(['mean', 'std']).to_string())
    print("\n=== Marginal effect of encoder_width ===")
    print(df.groupby('encoder_width')['mse'].agg(['mean', 'std']).to_string())

    # --- Best overall config ---
    summary_all = (df.groupby(['weight_decay', 'decoder_width',
                               'encoder_width'])['mse']
                     .agg(['mean', 'std'])
                     .sort_values('mean'))
    print("\n=== Best 5 configs averaged across both DGPs ===")
    print(summary_all.head(5).to_string())


if __name__ == "__main__":
    main()