"""
run_scaling.py
--------------
Scaling study: how does TV-AE-G's advantage over StaticAE and LPCA grow
with the time-series length T?

Design:
  - N = 100 fixed
  - T in {200, 500, 1000}
  - Three DGPs: (linear, constant), (exponential, logistic),
    (quadratic, logistic)
  - Four methods: PCA, LPCA, StaticAE, TVAEG
  - Tuned TVAEG config from the capacity sweep:
      K1=3, dw=32, ew=64, wd=1e-3

Replication counts are reduced as T grows, since each fit is costlier:
  T=200  -> 10 reps
  T=500  ->  5 reps
  T=1000 ->  3 reps

Outputs:
  scaling_results.csv  - long format, one row per (rep, dgp, T, method)
  scaling.png          - log-log plot of mean MSE vs T, one panel per DGP
  run_scaling.txt      - console log
"""

import sys
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

# ---------- Tee logging ----------
class _Tee:
    def __init__(self, f):
        self.f = f
    def write(self, d):
        sys.__stdout__.write(d)
        self.f.write(d)
    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

sys.stdout = _Tee(open("run_scaling.txt", "w"))

import time
import numpy as np
import pandas as pd
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X
from static_ae import fit_static_ae
from tvae_global import fit_tvae_global


# ---------- Tuned config (from capacity sweep) ----------
K1 = 3
ENCODER_WIDTH = 64
DECODER_WIDTH = 32
WEIGHT_DECAY = 1e-3
N_SEEDS = 2          # reduced from 3 to keep scaling runtime tractable
N_EPOCHS = 500
PATIENCE = 30
BATCH_SIZE = 16
LR = 1e-3

N_FIXED = 100
K_TRUE = 2

T_VALUES = [200, 500, 1000]
REPS_BY_T = {200: 10, 500: 5, 1000: 3}

CONFIGS = [
    ("linear",      "constant"),
    ("exponential", "logistic"),
    ("quadratic",   "logistic"),
]


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T, bw


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"N = {N_FIXED}, K = {K_TRUE}")
    print(f"T values: {T_VALUES}")
    print(f"Reps by T: {REPS_BY_T}")
    print(f"TVAE config: K1={K1}, dw={DECODER_WIDTH}, "
          f"ew={ENCODER_WIDTH}, wd={WEIGHT_DECAY}, "
          f"n_seeds={N_SEEDS}, n_epochs={N_EPOCHS}")

    rows = []
    t_start = time.time()

    # Count total AE fits for ETA
    total_units = sum(REPS_BY_T[T] for T in T_VALUES) * len(CONFIGS)
    unit_counter = 0

    for T in T_VALUES:
        n_reps = REPS_BY_T[T]
        print(f"\n{'='*70}\nT = {T}, n_reps = {n_reps}\n{'='*70}")

        for (nonlin, tv) in CONFIGS:
            for rep in range(n_reps):
                seed_dgp = 1000 + rep
                cfg = DGPConfig(N=N_FIXED, T=T, K=K_TRUE,
                                nonlinearity=nonlin,
                                time_variation=tv,
                                error="iid", seed=seed_dgp)
                sim = generate_dgp(cfg)
                X, X_star = sim["X"], sim["X_star"]

                # PCA
                mse_pca = evaluate(
                    pca_baseline(X, K_TRUE)["X_hat"], X_star
                )["mse_common"]

                # LPCA
                X_hat_lpca, bw_lpca = run_local_pca(X, K_TRUE)
                mse_lpca = evaluate(X_hat_lpca, X_star)["mse_common"]

                # StaticAE
                ae_res = fit_static_ae(
                    X, K1=K1,
                    n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
                    encoder_width=ENCODER_WIDTH,
                    decoder_width=DECODER_WIDTH,
                    device=device, verbose=False)
                mse_ae = evaluate(ae_res["X_hat"], X_star)["mse_common"]

                # TVAEG
                tvae_res = fit_tvae_global(
                    X, K1=K1,
                    n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
                    lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE,
                    encoder_width=ENCODER_WIDTH,
                    decoder_width=DECODER_WIDTH,
                    encoder_time=True,
                    weight_decay=WEIGHT_DECAY,
                    device=device, verbose=False)
                mse_tvae = evaluate(tvae_res["X_hat"], X_star)["mse_common"]

                key = f"{nonlin}, {tv}"
                for method, mse in [('PCA', mse_pca), ('LPCA', mse_lpca),
                                    ('StaticAE', mse_ae), ('TVAEG', mse_tvae)]:
                    rows.append({
                        'rep': rep, 'dgp': key, 'T': T,
                        'method': method, 'mse': mse,
                    })

                unit_counter += 1
                elapsed = time.time() - t_start
                avg = elapsed / unit_counter
                eta = avg * (total_units - unit_counter)
                print(f"  [{unit_counter}/{total_units}] "
                      f"T={T:>4d} {key:<28} rep={rep:>2d}  "
                      f"PCA={mse_pca:.4f}  LPCA={mse_lpca:.4f}  "
                      f"AE={mse_ae:.4f}  TVAEG={mse_tvae:.4f}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv("scaling_results.csv", index=False)

    # ---------- Summary table ----------
    methods = ['PCA', 'LPCA', 'StaticAE', 'TVAEG']
    print("\n" + "="*100)
    print("Mean MSE by (DGP, T, method)")
    print("="*100)
    print(f"{'DGP':<28} {'T':>5}  " +
          " ".join(f"{m:>12}" for m in methods))
    print("-" * 100)
    for (nonlin, tv) in CONFIGS:
        key = f"{nonlin}, {tv}"
        for T in T_VALUES:
            sub = df[(df['dgp'] == key) & (df['T'] == T)]
            row = f"{key:<28} {T:>5}  "
            for m in methods:
                vals = sub[sub['method'] == m]['mse'].values
                row += f"{vals.mean():>12.4f} "
            print(row)

    # ---------- Improvement of TVAEG over StaticAE, by T ----------
    print("\n" + "="*70)
    print("TVAEG improvement over StaticAE (%, by T)")
    print("="*70)
    for (nonlin, tv) in CONFIGS:
        key = f"{nonlin}, {tv}"
        print(f"\n{key}:")
        for T in T_VALUES:
            sub = df[(df['dgp'] == key) & (df['T'] == T)]
            ae_mean = sub[sub['method'] == 'StaticAE']['mse'].mean()
            tvae_mean = sub[sub['method'] == 'TVAEG']['mse'].mean()
            pct = 100 * (ae_mean - tvae_mean) / ae_mean
            print(f"  T = {T:>4d}: {pct:>+6.2f}%")

    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.2f} minutes")

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(CONFIGS), figsize=(15, 5),
                                 sharey=False)
        colors = {'PCA': '#888888', 'LPCA': '#1f77b4',
                  'StaticAE': '#ff7f0e', 'TVAEG': '#2ca02c'}
        markers = {'PCA': 'o', 'LPCA': 's', 'StaticAE': '^', 'TVAEG': 'D'}

        for ax, (nonlin, tv) in zip(axes, CONFIGS):
            key = f"{nonlin}, {tv}"
            for method in methods:
                means = []
                stds = []
                for T in T_VALUES:
                    vals = df[(df['dgp'] == key) &
                              (df['T'] == T) &
                              (df['method'] == method)]['mse'].values
                    means.append(vals.mean())
                    stds.append(vals.std() / np.sqrt(len(vals)))
                ax.errorbar(T_VALUES, means, yerr=stds,
                            label=method, color=colors[method],
                            marker=markers[method], linewidth=2,
                            markersize=7, capsize=4)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('T')
            ax.set_ylabel('Mean MSE of common component')
            ax.set_title(key)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig('scaling.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to scaling.png")
    except Exception as e:
        print(f"\nPlotting failed (non-fatal): {e}")


if __name__ == "__main__":
    main()