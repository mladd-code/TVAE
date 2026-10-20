"""
tvae_sim.py
-----------
Step 1 of the TV-AE roadmap:
  - Simulated DGP generator for time-varying nonlinear factor models.
  - Evaluation metrics for common components and factor space.
  - Oracle model.
  - Static PCA benchmark.

Model (shape conventions follow Su & Wang 2017):
    X_{it} = phi_i(F_t, t/T) + U_{it},   i = 1,...,N,  t = 1,...,T

    X       : N x T
    F       : K x T
    lambda  : N x K x T  (time-varying loadings)
    phi     : loading function, applied per time t to give X[:, t]

Metric note
-----------
Factor models identify F and loadings only up to an invertible transform.
For linear/static models with the standard normalization F F' / T = I_K,
this transform is orthogonal.  For nonlinear models, it is only invertible.

The primary factor-space metric here is the PROJECTION-MATRIX distance:

    d(F_hat, F_star) = ||P_hat - P_star||_F

where P = Q Q' is the orthogonal projection onto the row space of F, and Q
is a T x K matrix with orthonormal columns spanning that row space.  This
metric is invariant to rotation, sign, and scale, so it isolates genuine
misalignment of the factor subspace.

We also report the Procrustes-aligned Frobenius distance as a secondary,
diagnostic quantity.  It is sensitive to scale mismatch between F_hat and
F_star and should not be used as the primary metric.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NONLINEARITY_CHOICES = ("linear", "exponential", "quadratic", "radial", "sigmoid")
TV_CHOICES = ("constant", "single_break", "multiple_breaks",
              "logistic", "multiple_logistic")
ERROR_CHOICES = ("iid", "heteroskedastic", "cross_sectional")


@dataclass
class DGPConfig:
    N: int = 100
    T: int = 200
    K: int = 2
    nonlinearity: str = "linear"
    time_variation: str = "constant"
    error: str = "iid"
    break_magnitude: float = 1.0
    seed: Optional[int] = None
    normalise: bool = True
    kappa: float = 10.0

    def __post_init__(self):
        assert self.nonlinearity in NONLINEARITY_CHOICES, self.nonlinearity
        assert self.time_variation in TV_CHOICES, self.time_variation
        assert self.error in ERROR_CHOICES, self.error


# --------------------------------------------------------------------------- #
# Time-varying loading paths
# --------------------------------------------------------------------------- #
def _transition(t_grid: np.ndarray, kind: str, kappa: float) -> np.ndarray:
    x = t_grid
    if kind == "constant":
        return np.zeros_like(x)
    if kind == "single_break":
        return (x > 0.5).astype(float)
    if kind == "multiple_breaks":
        g = np.zeros_like(x)
        g[(x > 0.2) & (x <= 0.4)] = -0.5
        g[(x > 0.6) & (x <= 0.8)] = +0.5
        return g
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-kappa * (x - 0.5)))
    if kind == "multiple_logistic":
        g1 = 1.0 / (1.0 + np.exp(-kappa * (x - 0.3)))
        g2 = -1.0 / (1.0 + np.exp(-kappa * (x - 0.7)))
        return g1 + g2
    raise ValueError(kind)


def build_loadings(cfg: DGPConfig, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray]:
    lambda_0 = rng.uniform(-1.0, 1.0, size=(cfg.N, cfg.K))
    t_grid = (np.arange(cfg.T) + 1) / cfg.T
    g = _transition(t_grid, cfg.time_variation, cfg.kappa)
    return lambda_0, g


# --------------------------------------------------------------------------- #
# DGP
# --------------------------------------------------------------------------- #
def generate_dgp(cfg: DGPConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    N, T, K = cfg.N, cfg.T, cfg.K

    F = rng.uniform(-2.0, 2.0, size=(K, T))

    lambda_0, g = build_loadings(cfg, rng)
    lambda_t = lambda_0[:, :, None] + cfg.break_magnitude * g[None, None, :]

    if cfg.nonlinearity == "linear":
        X_star = np.einsum('nkt,kt->nt', lambda_t, F)
    elif cfg.nonlinearity == "exponential":
        X_star = np.exp(np.einsum('nkt,kt->nt', lambda_t, F))
    elif cfg.nonlinearity == "sigmoid":
        X_star = 1.0 / (1.0 + np.exp(-np.einsum('nkt,kt->nt', lambda_t, F)))
    elif cfg.nonlinearity == "radial":
        d2 = (lambda_t ** 2).sum(axis=1) \
             - 2.0 * np.einsum('nkt,kt->nt', lambda_t, F) \
             + (F ** 2).sum(axis=0)[None, :]
        X_star = np.exp(-d2)
    elif cfg.nonlinearity == "quadratic":
        lam1 = lambda_t
        lam2 = rng.uniform(-1.0, 1.0, size=(N, K, K))
        lin = np.einsum('nkt,kt->nt', lam1, F)
        quad = np.einsum('kt,nkl,lt->nt', F, lam2, F)
        X_star = lin + quad
    else:
        raise ValueError(cfg.nonlinearity)

    if cfg.normalise:
        s = X_star.std(axis=1, ddof=1)
        s = np.where(s < 1e-12, 1.0, s)
        X_star = X_star / s[:, None]

    if cfg.error == "iid":
        U = rng.standard_normal((N, T))
    elif cfg.error == "heteroskedastic":
        sigma = rng.uniform(0.5, 1.5, size=N)
        U = sigma[:, None] * rng.standard_normal((N, T))
    elif cfg.error == "cross_sectional":
        idx = np.arange(N)
        Sigma = 0.5 ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((N, T))
        U = L @ Z
    else:
        raise ValueError(cfg.error)

    X = X_star + U
    return {"X": X, "X_star": X_star, "F": F,
            "lambda": lambda_t, "U": U, "cfg": cfg}


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def _orthonormal_row_basis(F_KT: np.ndarray) -> np.ndarray:
    """
    Return a T x K' matrix Q with orthonormal columns spanning the row space
    of F_KT (K x T).  K' = min(K, T).
    """
    # SVD of F_KT: U (K x K), S (K x min(K,T)), Vt (min(K,T) x T).
    # The row space of F_KT is spanned by columns of V = Vt.T.
    _, _, Vt = np.linalg.svd(F_KT, full_matrices=False)
    return Vt.T  # T x min(K,T)


def factor_subspace_distance(F_hat: np.ndarray,
                             F_star: np.ndarray) -> float:
    """
    ||P_hat - P_star||_F, where P = Q Q' and Q is an orthonormal basis for
    the row space of F.

    Invariant to rotation, sign, and scale.  Range: [0, sqrt(2K)].

    Equivalently, sqrt(2) * sqrt(sum_i sin^2(theta_i)) where theta_i are the
    principal angles between the two subspaces.
    """
    Q_hat = _orthonormal_row_basis(F_hat)
    Q_star = _orthonormal_row_basis(F_star)
    K_eff = min(Q_hat.shape[1], Q_star.shape[1])
    # use first K_eff columns of each
    Qh = Q_hat[:, :K_eff]
    Qs = Q_star[:, :K_eff]
    cos_sq = float(np.sum((Qh.T @ Qs) ** 2))
    return float(np.sqrt(max(2.0 * K_eff - 2.0 * cos_sq, 0.0)))


def _procrustes_align(F_hat: np.ndarray, F_star: np.ndarray) -> np.ndarray:
    if F_hat.shape[0] != F_star.shape[0]:
        raise ValueError(
            f"Procrustes needs matching row counts: "
            f"F_hat {F_hat.shape[0]} vs F_star {F_star.shape[0]}"
        )
    M = F_star @ F_hat.T
    U, _, Vt = np.linalg.svd(M)
    H = Vt.T @ U.T
    return H @ F_star


def factor_procrustes_distance(F_hat: np.ndarray,
                               F_star: np.ndarray) -> float:
    """
    Procrustes-aligned Frobenius distance.  Requires F_hat and F_star to
    have the same number of rows.  Returns NaN otherwise.
    """
    if F_hat.shape[0] != F_star.shape[0]:
        return float("nan")
    F_aligned = _procrustes_align(F_hat, F_star)
    num = float(np.sum((F_hat - F_aligned) ** 2))
    den = float(np.sum(F_star ** 2))
    return num / den if den > 0 else np.nan


def evaluate(X_hat, X_star, F_hat=None, F_star=None):
    err = X_hat - X_star
    out = {"mse_common": float(np.mean(err ** 2))}
    if F_hat is not None and F_star is not None:
        out["factor_distance"] = factor_subspace_distance(F_hat, F_star)
        if F_hat.shape[0] == F_star.shape[0]:
            out["factor_procrustes"] = factor_procrustes_distance(F_hat, F_star)
        else:
            out["factor_procrustes"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def oracle(X: np.ndarray, X_star: np.ndarray, F_star: np.ndarray, K: int
           ) -> Dict[str, np.ndarray]:
    return {"X_hat": X_star.copy(), "F_hat": F_star.copy()}


# --------------------------------------------------------------------------- #
# Static PCA baseline
# --------------------------------------------------------------------------- #
def pca_baseline(X: np.ndarray, K: int) -> Dict[str, np.ndarray]:
    Xc = X - X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U_K = U[:, :K]
    S_K = S[:K]
    V_K = Vt[:K, :]
    F_hat = np.sqrt(X.shape[1]) * V_K
    Lambda_hat = (U_K * S_K) / np.sqrt(X.shape[1])
    X_hat = Lambda_hat @ F_hat + X.mean(axis=1, keepdims=True)
    return {"X_hat": X_hat, "F_hat": F_hat, "Lambda_hat": Lambda_hat}


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = DGPConfig(
        N=100, T=200, K=2,
        nonlinearity="linear",
        time_variation="logistic",
        error="iid",
        seed=42,
    )
    sim = generate_dgp(cfg)
    print(f"DGP: N={cfg.N}, T={cfg.T}, K={cfg.K}, "
          f"nonlinearity={cfg.nonlinearity}, tv={cfg.time_variation}")

    res_or = oracle(sim["X"], sim["X_star"], sim["F"], cfg.K)
    m_or = evaluate(res_or["X_hat"], sim["X_star"],
                    res_or["F_hat"], sim["F"])
    print("\nOracle:")
    for k, v in m_or.items():
        print(f"  {k:>20s} = {v:.3e}")

    res_pca = pca_baseline(sim["X"], cfg.K)
    m_pca = evaluate(res_pca["X_hat"], sim["X_star"],
                     res_pca["F_hat"], sim["F"])
    print("\nStatic PCA:")
    for k, v in m_pca.items():
        print(f"  {k:>20s} = {v:.4f}")

    assert m_or["mse_common"] < 1e-20
    assert m_or["factor_distance"] < 1e-12
    print("\n[OK] Smoke test passed.")