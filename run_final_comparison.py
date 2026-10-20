"""
run_final_comparison.py
-----------------------
Final comparison of PCA, LPCA, StaticAE, and TV-AE-G across the six-DGP
grid, using the tuned TV-AE-G configuration from the capacity sweep.

Tuned config (from run_capacity_sweep_v2.txt):
  K1 = 3, decoder_width = 32, encoder_width = 64, weight_decay = 1e-3

Reports mean and std of the pooled MSE of the common component over
n_reps = 10 replications. Uses n_seeds = 3 for the autoencoders.

Logs all console output to run_final_comparison.txt.
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

sys.stdout = _Tee(open("run_final_comparison.txt", "w"))

import time
import numpy as np
import pandas as pd
import torch
from tvae_sim import DGPConfig, generate_dgp, evaluate, pca_baseline
from local_pca import silverman, local_pca_all, reconstruct_X
from static_ae import fit_static_ae
from tvae_global import fit_tvae_global


# ---------------- Tuned TV-AE-G config ----------------
K1_TVAE = 3
DECODER_WIDTH = 32
ENCODER_WIDTH = 64
WEIGHT_DECAY = 1e-3

# StaticAE uses a slightly smaller config for historical comparability
K1_STATIC = 3
STATIC_ENC_WIDTH = 64
STATIC_DEC_WIDTH = 32

N_SEEDS = 3
N_EPOCHS = 500
PATIENCE = 30
N_REPS = 30
N, T, K = 100, 200, 2


def run_local_pca(X_NT, K):
    returns_Tp = X_NT.T
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, K)
    return reconstruct_X(res).T


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"N={N}, T={T}, K={K}, n_reps={N_REPS}, n_seeds={N_SEEDS}, "
          f"n_epochs={N_EPOCHS}")
    print(f"TVAE config: K1={K1_TVAE}, dw={DECODER_WIDTH}, "
          f"ew={ENCODER_WIDTH}, wd={WEIGHT_DECAY}")
    print(f"StaticAE config: K1={K1_STATIC}, "
          f"enc_w={STATIC_ENC_WIDTH}, dec_w={STATIC_DEC_WIDTH}")

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
    total_units = len(configs) * N_REPS
    unit_counter = 0

    for (nonlin, tv) in configs:
        for rep in range(N_REPS):
            seed_dgp = 1000 + rep
            cfg = DGPConfig(N=N, T=T, K=K,
                            nonlinearity=nonlin,
                            time_variation=tv,
                            error="iid", seed=seed_dgp)
            sim = generate_dgp(cfg)
            X, X_star = sim["X"], sim["X_star"]

            # PCA
            mse_pca = evaluate(pca_baseline(X, K)["X_hat"], X_star)["mse_common"]

            # LPCA
            mse_lpca = evaluate(run_local_pca(X, K), X_star)["mse_common"]

            # StaticAE
            ae_res = fit_static_ae(
                X, K1=K1_STATIC,
                n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
                encoder_width=STATIC_ENC_WIDTH,
                decoder_width=STATIC_DEC_WIDTH,
                device=device, verbose=False)
            mse_ae = evaluate(ae_res["X_hat"], X_star)["mse_common"]

            # TVAEG (tuned)
            tvae_res = fit_tvae_global(
                X, K1=K1_TVAE,
                n_seeds=N_SEEDS, n_epochs=N_EPOCHS,
                lr=1e-3, batch_size=16, patience=PATIENCE,
                encoder_width=ENCODER_WIDTH,
                decoder_width=DECODER_WIDTH,
                encoder_time=True,
                weight_decay=WEIGHT_DECAY,
                device=device, verbose=False)
            mse_tvae = evaluate(tvae_res["X_hat"], X_star)["mse_common"]

            key = f"{nonlin}, {tv}"
            rows.append({'rep': rep, 'dgp': key, 'method': 'PCA',
                         'mse': mse_pca})
            rows.append({'rep': rep, 'dgp': key, 'method': 'LPCA',
                         'mse': mse_lpca})
            rows.append({'rep': rep, 'dgp': key, 'method': 'StaticAE',
                         'mse': mse_ae})
            rows.append({'rep': rep, 'dgp': key, 'method': 'TVAEG',
                         'mse': mse_tvae})

            unit_counter += 1
            elapsed = time.time() - t0
            avg_per_unit = elapsed / unit_counter
            eta = avg_per_unit * (total_units - unit_counter)
            print(f"  [{unit_counter}/{total_units}] {key:<28}  "
                  f"PCA={mse_pca:.4f}  LPCA={mse_lpca:.4f}  "
                  f"AE={mse_ae:.4f}  TVAEG={mse_tvae:.4f}  "
                  f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

    df = pd.DataFrame(rows)
    df.to_csv("final_comparison.csv", index=False)

    # ---------------- Summary table ----------------
    methods = ['PCA', 'LPCA', 'StaticAE', 'TVAEG']
    print("\n" + "=" * 90)
    print(f"Mean MSE (std in parentheses) over {N_REPS} reps")
    print("=" * 90)
    header = f"{'DGP':<28} " + " ".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        row = f"{key:<28} "
        for m in methods:
            vals = sub[sub['method'] == m]['mse'].values
            row += f"{vals.mean():>7.4f}({vals.std():>5.3f}) "
        print(row)

    # ---------------- Improvement of TVAEG over StaticAE ----------------
    print("\n" + "=" * 60)
    print("TVAEG improvement over StaticAE (%, positive means TVAEG better)")
    print("=" * 60)
    for (nonlin, tv) in configs:
        key = f"{nonlin}, {tv}"
        sub = df[df['dgp'] == key]
        ae_mean = sub[sub['method'] == 'StaticAE']['mse'].mean()
        tvae_mean = sub[sub['method'] == 'TVAEG']['mse'].mean()
        pct = 100 * (ae_mean - tvae_mean) / ae_mean
        print(f"  {key:<28}  {pct:>+6.2f}%")

    print(f"\nTotal wall time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    main()