"""
tvaec_mvp.py
------------
TV-AE-C architecture and covariance estimator for the MVP application.

Self-contained: does not import from mvp_app. Config values are passed
as function arguments.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class _TVAECCov(nn.Module):
    def __init__(self, P_char: int, K1: int,
                 encoder_width: int = 32,
                 beta_width: int = 32):
        super().__init__()
        self.P = P_char
        self.K1 = K1

        self.encoder = nn.Sequential(
            nn.Linear(P_char + 1, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(P_char + 1, beta_width),
            nn.ReLU(),
            nn.Linear(beta_width, K1),
        )

    def encode(self, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, tau], dim=1)
        return self.encoder(inp)

    def beta_all(self, Z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        T, p, P = Z.shape
        tau_exp = tau.unsqueeze(1).expand(T, p, 1)
        inp = torch.cat([Z, tau_exp], dim=2)
        B_flat = self.beta_net(inp.reshape(T * p, P + 1))
        return B_flat.reshape(T, p, self.K1)

    def forward(self, x_t, Z, tau):
        F = self.encode(x_t, tau)
        B = self.beta_all(Z, tau)
        x_hat = torch.einsum('tk,tpk->tp', F, B)
        return x_hat, F


def _psd_repair(Sigma: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, floor)
    Sigma = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (Sigma + Sigma.T)


def _build_managed_portfolios(Z: np.ndarray, Rc: np.ndarray,
                              M: np.ndarray,
                              ridge: float = 1e-4) -> np.ndarray:
    """
    x_t = (Z_t'Z_t + ridge * I)^{-1} Z_t' r_t
    Z:  (T, p, P)   characteristics
    Rc: (T, p)      demeaned returns
    M:  (T, p)      mask
    Returns x_t: (T, P)
    """
    T, p, P = Z.shape
    x = np.zeros((T, P), dtype=np.float32)
    eye = ridge * np.eye(P, dtype=np.float32)

    for t in range(T):
        mask = M[t] > 0
        if mask.sum() < P + 5:
            continue
        Z_t = Z[t][mask]
        r_t = Rc[t][mask]
        ZtZ = Z_t.T @ Z_t + eye
        Ztr = Z_t.T @ r_t
        try:
            x[t] = np.linalg.solve(ZtZ, Ztr)
        except np.linalg.LinAlgError:
            pass
    return x


def cov_tvaec(R: np.ndarray, M: np.ndarray, char_arr: np.ndarray,
              K1: int = 5,
              n_epochs: int = 500,
              lr: float = 1e-2,
              weight_decay: float = 1e-5,
              device: str = "cpu") -> np.ndarray:
    """
    TV-AE-C + POET covariance estimator.
    """
    T, p = R.shape
    P = char_arr.shape[2]

    counts = M.sum(axis=0)
    counts = np.where(counts > 0, counts, 1.0)
    mu = (R * M).sum(axis=0) / counts
    Rc = (R - mu[None, :]) * M

    tau = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)

    x_t = _build_managed_portfolios(char_arr, Rc, M)

    Z_t = torch.tensor(char_arr, dtype=torch.float32).to(device)
    X_t = torch.tensor(x_t, dtype=torch.float32).to(device)
    Xc_t = torch.tensor(Rc, dtype=torch.float32).to(device)
    M_t = torch.tensor(M, dtype=torch.float32).to(device)
    tau_t = torch.tensor(tau, dtype=torch.float32).to(device)

    model = _TVAECCov(P, K1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        x_hat, F = model(X_t, Z_t, tau_t)
        loss = ((x_hat - Xc_t) ** 2 * M_t).sum() / M_t.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_hat, F = model(X_t, Z_t, tau_t)
        F = F.cpu().numpy()
        x_hat = x_hat.cpu().numpy()
        B = model.beta_all(Z_t, tau_t).cpu().numpy()

    Sigma_F = (F.T @ F) / T

    E = (Rc - x_hat) * M
    diag = np.diag((E.T @ E) / T).copy()
    sd = np.sqrt(np.maximum(diag, 1e-12))
    thresh = 1.0 * np.sqrt(np.log(p) / T) * np.outer(sd, sd)
    Sigma_e = (E.T @ E) / T
    S_sign = np.sign(Sigma_e)
    S_abs = np.abs(Sigma_e) - thresh
    Sigma_e = S_sign * np.maximum(S_abs, 0.0)
    np.fill_diagonal(Sigma_e, diag)

    B_t = B[-1]
    Sigma = B_t @ Sigma_F @ B_t.T + Sigma_e
    return _psd_repair(Sigma)