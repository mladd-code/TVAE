"""
crossval_lpca.py
----------------
Generates a panel from a chosen DGP, runs the Python LPCA port, and
saves the inputs and outputs so the R code can be run on the same data.
"""

import os
import numpy as np
from tvae.dgp import DGPConfig, generate_dgp
from tvae.models.lpca import silverman, local_pca_all, reconstruct_X

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Same configuration as the smoke test
    cfg = DGPConfig(
        N=100, T=200, K=2,
        nonlinearity="linear",
        time_variation="logistic",
        error="iid",
        seed=42,
    )
    sim = generate_dgp(cfg)
    X_NT = sim["X"]                       # N x T
    returns_Tp = X_NT.T                   # T x p (matches R orientation)

    # Save the returns matrix as CSV (no header, no row names)
    np.savetxt(os.path.join(OUT_DIR, "crossval_data.csv"),
               returns_Tp, delimiter=",", fmt="%.10f")

    # Run the Python LPCA
    bw = silverman(returns_Tp)
    res = local_pca_all(returns_Tp, bw, cfg.K)
    X_hat_Tp = reconstruct_X(res)         # T x p
    F_hat_Tp = res["f_hat"]               # T x m

    np.savetxt(os.path.join(OUT_DIR, "crossval_Xhat_py.csv"),
               X_hat_Tp, delimiter=",", fmt="%.10f")
    np.savetxt(os.path.join(OUT_DIR, "crossval_Fhat_py.csv"),
               F_hat_Tp, delimiter=",", fmt="%.10f")

    # Save the bandwidth so R uses exactly the same value
    with open(os.path.join(OUT_DIR, "crossval_bw.txt"), "w") as f:
        f.write(f"{bw:.10f}\n")

    print(f"Input (T x p)              : {returns_Tp.shape}")
    print(f"Bandwidth                  : {bw:.10f}")
    print(f"Saved crossval_data.csv    : {returns_Tp.shape}")
    print(f"Saved crossval_Xhat_py.csv : {X_hat_Tp.shape}")
    print(f"Saved crossval_Fhat_py.csv : {F_hat_Tp.shape}")
    print(f"Saved crossval_bw.txt")
    print("\nNext: run crossval_lpca.R (in R), then crossval_compare.py (Python).")


if __name__ == "__main__":
    main()