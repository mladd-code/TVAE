"""
run_scaling_v2.py
-----------------
Revised scaling study with four fixes relative to run_scaling.py:

  1. Symmetric weight decay: both StaticAE and TVAEG use weight_decay=1e-3
     (requires the static_ae.py patch described in the accompanying message).
  2. More replications at T=1000 (5 instead of 3).
  3. Matplotlib import and plotting are made non-fatal.
  4. Added T=2000 for the quadratic-logistic DGP only.

Config: K1=3, decoder_width=32, encoder_width=64, weight_decay=1e-3,
n_seeds=2, n_epochs=500, patience=30.

Outputs: run_scaling_v2.txt, scaling_results_v2.csv, scaling_v2.png.
"""

import sys
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


class _Tee:
    def __init__(self, f):
        self.f = f
    def write(self, d):
        sys.__stdout__.write(d)
        self.f.write(d)
    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = _Tee(open("run_scaling_v2.txt", "w"))

import time
import numpy as np
import pandas as pd
import torch
from tvae.dgp import DGPConfig, generate_dgp, evaluate, pca_baseline
from tvae.models.lpca import silverman, local_pca_all, reconstruct_X
from tvae.models.static_ae import fit_static_ae
from tvae.models.tvae_global import fit_tvae_global


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_FIXED = 100
K_TRUE = 2

K1 = 3
ENCODER_WIDTH = 64
DECODER_WIDTH = 32
WEIGHT_DECAY = 1e-3
N_SEEDS = 2
N_EPOCHS = 500
PATIENCE = 30
BATCH_SIZE = 16
LR = 1e-3

# Explicit run specification: (T, dgp, n_reps)
RUN_SPEC = [
    (200,  "linear, constant",        10),
    (200,  "exponential, logistic",   10),
    (200,  "quadratic, logistic",     10),
    (500,  "linear, constant",         5),
    (500,  "exponential, logistic",    5),
    (500,  "quadratic, logistic",      5),
    (1000, "linear, constant",         5),
    (1000, "exponential, logistic",    5),
    (1000, "quadratic, logistic",      5),
    (2000, "quadratic, logistic",      3),
]


def parse_dgp(key):
    parts = key.split(", ")
    return parts[0], parts[1]


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"N = {N_FIXED}, K = {K_TRUE}")
    print(f"Config: K1={K1}, dw={DECODER_WIDTH}, ew={ENCODER_WIDTH}, "
          f"wd={WEIGHT_DECAY}, n_seeds={N_SEEDS}, n_epochs={N_EPOCHS}")
    print(f"Total units: {sum(r for _, _, r in RUN_SPEC)}")
    print("Run spec:")
    for (T, key, n) in RUN_SPEC:
        print(f"  T={T:>4d}  {key:<28}  n_reps={n}")

    rows = []
    t_start = time.time()
    unit_counter = 0
    total_units = sum(r for _, _, r in RUN_SPEC)

    for (T, dgp_key, n_reps) in RUN_SPEC:
        nonlin, tv = parse_dgp(dgp_key)
        print(f"\n{'='*72}\nT = {T}, DGP = {dgp_key}, n_reps = {n_reps}\n{'='*72}")

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
            mse_lpca = evaluate(
                run_local_pca(X, K_TRUE), X_star
            )["mse_common"]

            # StaticAE with weight decay (now symmetric with TVAEG)
            ae_res = fit_static_ae(
                X, K1=K1,
                n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
                encoder_width=ENCODER_WIDTH,
                decoder_width=DECODER_WIDTH,
                weight_decay=WEIGHT_DECAY,
                device=device, verbose=False)
            mse_ae = evaluate(ae_res["X_hat"], X_star)["mse_common"]

            # TVAEG with weight decay
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

            for method, mse in [('PCA', mse_pca),
                                ('LPCA', mse_lpca),
                                ('StaticAE', mse_ae),
                                ('TVAEG', mse_tvae)]:
                rows.append({
                    'rep': rep, 'dgp': dgp_key, 'T': T,
                    'method': method, 'mse': mse,
                })

            unit_counter += 1
            elapsed = time.time() - t_start
            avg = elapsed / unit_counter
            eta = avg * (total_units - unit_counter)
            print(f"  [{unit_counter}/{total_units}] "
                  f"T={T:>4d} {dgp_key:<28} rep={rep:>2d}  "
                  f"PCA={mse_pca:.4f}  LPCA={mse_lpca:.4f}  "
                  f"AE={mse_ae:.4f}  TVAEG={mse_tvae:.4f}  "
                  f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv("scaling_results_v2.csv", index=False)

    # ---------------- Summary table ----------------
    methods = ['PCA', 'LPCA', 'StaticAE', 'TVAEG']
    seen_dgps = []
    for _, key, _ in RUN_SPEC:
        if key not in seen_dgps:
            seen_dgps.append(key)

    print("\n" + "=" * 100)
    print("Mean MSE by (DGP, T, method)")
    print("=" * 100)
    print(f"{'DGP':<28} {'T':>5}  " +
          " ".join(f"{m:>12}" for m in methods))
    print("-" * 100)
    for key in seen_dgps:
        T_list = sorted(set(T for (T, k, _) in RUN_SPEC if k == key))
        for T in T_list:
            sub = df[(df['dgp'] == key) & (df['T'] == T)]
            row = f"{key:<28} {T:>5}  "
            for m in methods:
                vals = sub[sub['method'] == m]['mse'].values
                row += f"{vals.mean():>12.4f} "
            print(row)

    # ---------------- Improvement of TVAEG over StaticAE ----------------
    print("\n" + "=" * 72)
    print("TVAEG improvement over StaticAE (%, positive means TVAEG better)")
    print("=" * 72)
    for key in seen_dgps:
        T_list = sorted(set(T for (T, k, _) in RUN_SPEC if k == key))
        print(f"\n{key}:")
        for T in T_list:
            sub = df[(df['dgp'] == key) & (df['T'] == T)]
            ae_mean = sub[sub['method'] == 'StaticAE']['mse'].mean()
            tvae_mean = sub[sub['method'] == 'TVAEG']['mse'].mean()
            pct = 100 * (ae_mean - tvae_mean) / ae_mean
            print(f"  T = {T:>4d}: {pct:>+6.2f}%")

    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.2f} minutes")

    # ---------------- Plot ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot. "
              "Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, len(seen_dgps),
                             figsize=(5 * len(seen_dgps), 5),
                             sharey=False)
    if len(seen_dgps) == 1:
        axes = [axes]
    colors = {'PCA': '#888888', 'LPCA': '#1f77b4',
              'StaticAE': '#ff7f0e', 'TVAEG': '#2ca02c'}
    markers = {'PCA': 'o', 'LPCA': 's', 'StaticAE': '^', 'TVAEG': 'D'}

    for ax, key in zip(axes, seen_dgps):
        T_list = sorted(set(T for (T, k, _) in RUN_SPEC if k == key))
        for method in methods:
            means, sems = [], []
            for T in T_list:
                vals = df[(df['dgp'] == key) &
                          (df['T'] == T) &
                          (df['method'] == method)]['mse'].values
                means.append(vals.mean())
                sems.append(vals.std() / max(np.sqrt(len(vals)), 1))
            ax.errorbar(T_list, means, yerr=sems,
                        label=method, color=colors[method],
                        marker=markers[method], linewidth=2,
                        markersize=7, capsize=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('T')
        ax.set_ylabel('Mean MSE of common component')
        ax.set_title(key)
        ax.set_xticks(T_list)
        ax.set_xticklabels([str(t) for t in T_list])
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('scaling_v2.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to scaling_v2.png")


if __name__ == "__main__":
    main()