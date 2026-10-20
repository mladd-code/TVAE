"""
local_pca.py
------------
Python port of the Su-Wang (2017) local PCA functions implemented in the
R codebase of Fan, Wu, Yang, and Zhong (2024), "Time-varying minimum
variance portfolio", Journal of Econometrics 239(2), 105339.

References
----------
Su, L., & Wang, X. (2017). On time-varying factor models: Estimation and
testing. Journal of Econometrics, 198(1), 84-101.

Conventions
-----------
returns : T x p array (time x assets). NOTE this is the TRANSPOSE of the
          N x T convention used in tvae_sim.py. We transpose at the driver
          level, so this module matches the R code exactly.

Indices : 0-based in Python. The R code uses 1-based. We shift internally
          so that the boundary-kernel logic matches the R code literally.

Caveat
------
The R `boundary_kernel` is called as `boundary_kernel(r_ref, t_data, ...)`
but its parameters are named `(t, r)`. As a result, inside the function,
`t` is the reference and `r` is the data point. The boundary correction
therefore checks whether the DATA POINT lies near the sample edge. This
differs from Hong & Li (2005), where the correction is applied at the
REFERENCE point. We have ported the R code literally, keeping the same
argument order, so that our Python results are numerically comparable to
the user's R results. If the user later wants to switch to the
Hong-Li-correct convention, change the check `r < Th_floor` to
`t < Th_floor` inside `boundary_kernel`.
"""

from __future__ import annotations
from typing import Callable, Optional, List, Dict, Any
import numpy as np


# --------------------------------------------------------------------------- #
# Kernel functions
# --------------------------------------------------------------------------- #
def epanechnikov_kernel(u):
    """Epanechnikov kernel K(u) = 0.75 (1 - u^2) if |u| <= 1, else 0."""
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration, version-independent."""
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def _integrate_kernel(kernel_func: Callable, a: float, b: float,
                      n: int = 2001) -> float:
    """Numerical integration of kernel_func over [a, b]."""
    if b <= a:
        return 0.0
    grid = np.linspace(a, b, n)
    return _trapz(kernel_func(grid), grid)


def boundary_kernel(t, r, iT, h, kernel_func):
    """
    Boundary-corrected kernel weight. Ported literally from the R code.
    See module docstring for the caveat on argument order.
    """
    scaled_diff = (t - r) / (iT * h)
    k_val = kernel_func(scaled_diff) / h

    Th_floor = int(np.floor(iT * h))

    if r < Th_floor:
        lower = -r / (iT * h)
        integral_val = _integrate_kernel(kernel_func, lower, 1.0)
        return k_val / integral_val
    elif r > (iT - Th_floor):
        upper = (1.0 - r / iT) / h
        integral_val = _integrate_kernel(kernel_func, -1.0, upper)
        return k_val / integral_val
    else:
        return k_val


# --------------------------------------------------------------------------- #
# Bandwidth: Silverman's rule of thumb, exactly as in the R code
# --------------------------------------------------------------------------- #
def silverman(returns: np.ndarray) -> float:
    iT, ip = returns.shape
    return (2.35 / np.sqrt(12.0)) * iT ** (-0.2) * ip ** (-0.1)


# --------------------------------------------------------------------------- #
# Local PCA at a single reference point
# --------------------------------------------------------------------------- #
def local_pca(returns: np.ndarray,
              r: int,
              bandwidth: float,
              m: int,
              kernel_func: Callable = epanechnikov_kernel,
              prev_F: Optional[np.ndarray] = None
              ) -> Dict[str, np.ndarray]:
    """
    Local PCA at reference time r (0-based).

    Returns dict with:
        'factors'  : T x m  local factors at reference r
        'f_hat'    : 1 x m  second-pass estimate of F_r
        'loadings' : p x m  first-pass loadings
        'w_r'      : T x 1  kernel weights
    """
    iT, ip = returns.shape

    # Kernel weights. The R code uses 1-based indexing internally; shift.
    r_1based = r + 1
    k_h = np.empty(iT, dtype=float)
    for t0 in range(iT):
        k_h[t0] = boundary_kernel(r_1based, t0 + 1,
                                  iT, bandwidth, kernel_func)

    X_r = returns * np.sqrt(k_h)[:, None]

    # Eigen decomposition of X_r X_r'
    M = X_r @ X_r.T
    eigvals, eigvecs = np.linalg.eigh(M)          # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    F_hat_r = np.sqrt(iT) * eigvecs[:, :m]        # T x m

    # Sign alignment
    if prev_F is not None:
        for j in range(m):
            c = np.corrcoef(prev_F[:, j], F_hat_r[:, j])[0, 1]
            if c < 0:
                F_hat_r[:, j] = -F_hat_r[:, j]

    # First-pass loadings
    Lambda_hat = (F_hat_r.T @ X_r) / iT           # m x p
    loadings = Lambda_hat.T                        # p x m

    # Second pass: F_r_hat = (Lambda' Lambda)^{-1} Lambda' R_r
    A = loadings.T @ loadings                      # m x m
    b = loadings.T @ returns[r, :]                 # m
    F_r_hat = np.linalg.solve(A, b)

    return {
        "factors": F_hat_r,
        "f_hat": F_r_hat[None, :],
        "loadings": loadings,
        "w_r": k_h.reshape(-1, 1),
    }


# --------------------------------------------------------------------------- #
# Local PCA at every reference point (R's `localPCA`)
# --------------------------------------------------------------------------- #
def local_pca_all(returns: np.ndarray,
                  bandwidth: float,
                  m: int,
                  kernel_func: Callable = epanechnikov_kernel
                  ) -> Dict[str, Any]:
    iT, ip = returns.shape
    factors_list: List[Optional[np.ndarray]] = [None] * iT
    loadings_list: List[Optional[np.ndarray]] = [None] * iT
    weights_list: List[Optional[np.ndarray]] = [None] * iT
    f_hat = np.empty((iT, m), dtype=float)

    prev_F = None
    for t_i in range(iT):
        res = local_pca(returns, t_i, bandwidth, m, kernel_func, prev_F)
        factors_list[t_i] = res["factors"]
        loadings_list[t_i] = res["loadings"]
        weights_list[t_i] = res["w_r"]
        f_hat[t_i, :] = res["f_hat"][0, :]
        prev_F = res["factors"]

    return {
        "factors": factors_list,
        "loadings": loadings_list,
        "m": m,
        "weights": weights_list,
        "f_hat": f_hat,
    }


# --------------------------------------------------------------------------- #
# Reconstruction: X_hat[t, :] = f_hat[t, :] @ loadings[t].T
# --------------------------------------------------------------------------- #
def reconstruct_X(localPCA_result: Dict[str, Any]) -> np.ndarray:
    f_hat = localPCA_result["f_hat"]           # T x m
    loadings = localPCA_result["loadings"]     # list of T (p x m)
    iT = f_hat.shape[0]
    ip = loadings[0].shape[0]
    X_hat = np.empty((iT, ip), dtype=float)
    for t in range(iT):
        X_hat[t, :] = f_hat[t, :] @ loadings[t].T
    return X_hat