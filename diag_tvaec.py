"""
diag_tvaec.py
-------------
Reports the fraction of return variance spanned by the managed portfolios,
and the fraction of variance the AE can reconstruct in the best case.
"""
import numpy as np
import pandas as pd
from mvp_app import (Config, load_daily_panel, load_char_panel,
                     select_universe, build_returns_matrix,
                     build_char_matrix, MVP_CHAR_COLS)

df = load_daily_panel(Config.PANEL_PATH)
char_df = load_char_panel(Config.CHAR_PATH)
universe = select_universe(df, Config.P, burn_end=Config.OOS_START)
dates = np.sort(df["date"].unique())
R, M = build_returns_matrix(df, universe, dates)
C = build_char_matrix(char_df, universe, dates, MVP_CHAR_COLS)
T, p = R.shape
P = C.shape[2]

# Full-sample demean
counts = M.sum(axis=0); counts = np.where(counts > 0, counts, 1.0)
mu = (R * M).sum(axis=0) / counts
Rc = (R - mu[None, :]) * M

# Fraction of Rc spanned by Z
Rc_flat = Rc.reshape(-1)
Z_flat = C.reshape(-1, P)
M_flat = M.reshape(-1)
mask = M_flat > 0
Z_obs = Z_flat[mask]
Rc_obs = Rc_flat[mask]

coef, *_ = np.linalg.lstsq(Z_obs, Rc_obs, rcond=None)
Rc_proj = Z_obs @ coef
var_explained = 1 - ((Rc_obs - Rc_proj)**2).sum() / (Rc_obs**2).sum()
print(f"Fraction of return variance spanned by characteristics: {var_explained:.4f}")
print(f"Upper bound on AE reconstruction R^2 (TV-AE-C): {var_explained:.4f}")

# Fraction reconstructible by top-K_1 PCA
Rc_mat = Rc * M
U, S, Vt = np.linalg.svd(Rc_mat, full_matrices=False)
for K in [3, 5, 10]:
    Rc_hat = (U[:, :K] * S[:K]) @ Vt[:K, :]
    r2 = 1 - ((Rc_mat - Rc_hat)**2 * M).sum() / (Rc_mat**2 * M).sum()
    print(f"PCA rank-{K} reconstruction R^2: {r2:.4f}")