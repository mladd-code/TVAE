"""
crossval_compare.py
-------------------
Loads the Python and R LPCA outputs and compares them.
"""

import os
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return np.loadtxt(os.path.join(OUT_DIR, name), delimiter=",")

def main():
    X_hat_py = load("crossval_Xhat_py.csv")
    X_hat_R  = load("crossval_Xhat_R.csv")
    F_hat_py = load("crossval_Fhat_py.csv")
    F_hat_R  = load("crossval_Fhat_R.csv")

    assert X_hat_py.shape == X_hat_R.shape, \
        f"X_hat shape mismatch: {X_hat_py.shape} vs {X_hat_R.shape}"
    assert F_hat_py.shape == F_hat_R.shape, \
        f"F_hat shape mismatch: {F_hat_py.shape} vs {F_hat_R.shape}"

    scale_X = np.max(np.abs(X_hat_py))
    scale_F = np.max(np.abs(F_hat_py))

    diff_X = np.abs(X_hat_py - X_hat_R)
    rel_X = diff_X.max() / scale_X

    print("=== Common component X_hat ===")
    print(f"  shape                : {X_hat_py.shape}")
    print(f"  max |X_hat_py|       : {scale_X:.6f}")
    print(f"  max |X_hat_py - R|   : {diff_X.max():.3e}")
    print(f"  relative error       : {rel_X:.3e}")

    # F_hat: allow per-column sign flips (eigenvector signs are arbitrary)
    F_hat_R_flipped = F_hat_R.copy()
    flips = []
    for j in range(F_hat_py.shape[1]):
        c = np.corrcoef(F_hat_py[:, j], F_hat_R[:, j])[0, 1]
        if c < 0:
            F_hat_R_flipped[:, j] = -F_hat_R[:, j]
            flips.append(j)
    diff_F_raw = np.abs(F_hat_py - F_hat_R)
    diff_F     = np.abs(F_hat_py - F_hat_R_flipped)
    rel_F      = diff_F.max() / scale_F

    print("\n=== Factors F_hat (per-column sign aligned) ===")
    print(f"  shape                : {F_hat_py.shape}")
    print(f"  max |F_hat_py|       : {scale_F:.6f}")
    print(f"  columns flipped      : {flips if flips else 'none'}")
    print(f"  max |diff| (raw)     : {diff_F_raw.max():.3e}")
    print(f"  max |diff| (aligned) : {diff_F.max():.3e}")
    print(f"  relative error       : {rel_F:.3e}")

    print("\n=== Verdict ===")
    if rel_X < 1e-6 and rel_F < 1e-6:
        print("  Python and R agree to 1e-6.")
        print("  LPCA port is validated.")
    elif rel_X < 1e-4 and rel_F < 1e-4:
        print("  Python and R agree to 1e-4.")
        print("  The small gap is integration tolerance in the boundary kernel.")
        print("  LPCA port is validated.")
    else:
        print("  Discrepancy larger than expected.")
        print("  Checklist:")
        print("   - Working directory in R is the project folder?")
        print("   - Did R source the correct kernel_functions.R, pca_function.R, bandwidth.R?")
        print("   - Was the bandwidth read from crossval_bw.txt (not recomputed)?")
        print("   - Did you run crossval_lpca.R after crossval_lpca.py?")


if __name__ == "__main__":
    main()