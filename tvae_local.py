"""
tvae_local.py
-------------
Time-varying local autoencoder (TV-AE-L).

Trains a separate disjoint-output autoencoder at each grid point, using a
kernel-weighted loss on the *effective sample* (times where the kernel
weight is non-negligible). Reconstructs by kernel-weighted averaging across
grid points. See docs/step4_design.md for the full design.

Key implementation notes (v2)
-----------------------------
1. Train/validation split is drawn from the effective sample (times where
   the kernel weight exceeds `w_eff_threshold * max(w)`), not from all T
   times. This is essential: outside the kernel support, the loss weight
   is ~0, so including those times in the validation set makes early
   stopping fire almost immediately.
2. Kernel weights are renormalized within each split so that w.sum() = 1.
3. Model capacity is smaller (encoder_width=32, decoder_width=16) to match
   the reduced effective sample per grid point (~Th).
4. `_train_local` returns diagnostics (epochs used, effective train/val
   sizes) so we can verify the training is actually happening.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from local_pca import epanechnikov_kernel, silverman


# --------------------------------------------------------------------------- #
# Model (same architecture as static_ae.StaticAE, smaller defaults)
# --------------------------------------------------------------------------- #
class LocalAE(nn.Module):
    def __init__(self, N: int, K1: int,
                 encoder_width: int = 32,
                 decoder_width: int = 16):
        super().__init__()
        self.N = N
        self.K1 = K1
        self.encoder = nn.Sequential(
            nn.Linear(N, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )
        self.W1 = nn.Parameter(torch.empty(N, K1, decoder_width))
        self.b1 = nn.Parameter(torch.zeros(N, decoder_width))
        self.W2 = nn.Parameter(torch.empty(N, decoder_width, 1))
        self.b2 = nn.Parameter(torch.zeros(N, 1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, F):
        H = torch.einsum('bk,nkw->bnw', F, self.W1) + self.b1[None]
        A = torch.relu(H)
        out = torch.einsum('bnw,nwo->bno', A, self.W2) + self.b2[None]
        return out.squeeze(-1)

    def forward(self, X):
        F = self.encode(X)
        return self.decode(F), F


# --------------------------------------------------------------------------- #
# Kernel weights (plain Epanechnikov, no boundary correction)
# --------------------------------------------------------------------------- #
def _kernel_weights(T: int, r_j: int, h: float) -> np.ndarray:
    """w[t] = K((t+1 - r_j) / (T*h)), t = 0..T-1."""
    t_1based = np.arange(1, T + 1)
    u = (t_1based - r_j) / (T * h)
    return epanechnikov_kernel(u).astype(np.float64)


# --------------------------------------------------------------------------- #
# Train one local AE at a grid point
# --------------------------------------------------------------------------- #
def _train_local(X_TN: np.ndarray,
                 w: np.ndarray,
                 K1: int,
                 encoder_width: int,
                 decoder_width: int,
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 patience: int,
                 seed: int,
                 device: str,
                 val_frac: float = 0.2,
                 w_eff_threshold: float = 1e-2,
                 ) -> Tuple[LocalAE, np.ndarray, np.ndarray, int, int, int]:
    """
    Train one local AE at a grid point.

    Returns
    -------
    model, mu, sd, epochs_used, n_eff_train, n_eff_val
    """
    T, N = X_TN.shape
    torch.manual_seed(seed)

    # --- Effective sample: times where the kernel weight is non-negligible ---
    w_max = float(w.max()) if w.max() > 0 else 1.0
    eff_mask = w > (w_eff_threshold * w_max)
    eff_idx = np.where(eff_mask)[0]

    if eff_idx.size < 10:
        raise RuntimeError(
            f"Effective sample too small at this grid point: "
            f"{eff_idx.size} times with w/w_max > {w_eff_threshold}. "
            f"Consider a larger bandwidth or fewer grid points."
        )

    # --- Split the effective sample into train / val ---
    rng = np.random.default_rng(seed)
    perm = rng.permutation(eff_idx.size)
    n_val = max(int(val_frac * eff_idx.size), 5)
    val_idx = eff_idx[perm[:n_val]]
    train_idx = eff_idx[perm[n_val:]]

    X_tr = X_TN[train_idx]
    X_va = X_TN[val_idx]
    w_tr = w[train_idx]
    w_va = w[val_idx]

    # Renormalize weights within each split
    w_tr = w_tr / w_tr.sum()
    w_va = w_va / w_va.sum()

    # Per-unit standardization using training statistics only
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_tr_n = (X_tr - mu) / sd
    X_va_n = (X_va - mu) / sd

    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    X_va_t = torch.tensor(X_va_n, dtype=torch.float32).to(device)
    w_tr_t = torch.tensor(w_tr, dtype=torch.float32).to(device)
    w_va_t = torch.tensor(w_va, dtype=torch.float32).to(device)

    model = LocalAE(N, K1, encoder_width, decoder_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    gen = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X_tr_t, w_tr_t),
                        batch_size=batch_size, shuffle=True, generator=gen)

    best_val = float('inf')
    best_state = None
    stale = 0
    epochs_used = 0

    for epoch in range(n_epochs):
        epochs_used = epoch + 1
        model.train()
        for xb, wb in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            per_item = ((x_hat - xb) ** 2).mean(dim=1)
            loss = (per_item * wb).sum()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_hat_va, _ = model(X_va_t)
            per_item = ((x_hat_va - X_va_t) ** 2).mean(dim=1)
            val_loss = (per_item * w_va_t).sum().item()

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mu, sd, epochs_used, train_idx.size, val_idx.size


# --------------------------------------------------------------------------- #
# Fit TV-AE-L
# --------------------------------------------------------------------------- #
def fit_tvae_local(X_NT: np.ndarray,
                   K1: int,
                   m: int = 10,
                   h: Optional[float] = None,
                   n_seeds: int = 3,
                   n_epochs: int = 500,
                   lr: float = 1e-3,
                   batch_size: int = 16,
                   patience: int = 30,
                   encoder_width: int = 32,
                   decoder_width: int = 16,
                   val_frac: float = 0.2,
                   w_eff_threshold: float = 1e-2,
                   device: str = 'cpu',
                   verbose: bool = False,
                   ) -> Dict[str, Any]:
    """
    Fit TV-AE-L on X_NT (N x T).

    Returns dict with:
        'X_hat'      : N x T  kernel-weighted average reconstruction
        'X_hat_nn'   : N x T  nearest-neighbor reconstruction
        'grid_x'     : m-vector of grid points in (0, 1)
        'grid_r'     : m-vector of grid points in 1..T
        'weights'    : m x T  kernel weights
        'h'          : bandwidth used
        'diagnostics': per-(grid, seed) dict of training metrics
    """
    N, T = X_NT.shape
    X_TN = X_NT.T

    if h is None:
        h = silverman(X_TN)

    # Interior grid points
    x_grid = (np.arange(m) + 0.5) / m
    r_grid = np.clip(np.round(x_grid * T).astype(int), 1, T)

    # Weights (m x T)
    W = np.stack([_kernel_weights(T, r_j, h) for r_j in r_grid], axis=0)

    if verbose:
        eff_sizes = [(W[j] > w_eff_threshold * W[j].max()).sum()
                     for j in range(m)]
        print(f"  m = {m}, h = {h:.4f}, "
              f"median effective T per grid = {int(np.median(eff_sizes))}")
        print(f"  grid (1-based): {r_grid.tolist()}")

    # Train one AE per grid point per seed; store reconstructions.
    X_hat_stack = np.zeros((m, n_seeds, T, N))
    diagnostics = []

    for j in range(m):
        for s in range(n_seeds):
            seed_js = s * 1000 + j
            try:
                (model, mu, sd, ep_used, n_tr, n_va) = _train_local(
                    X_TN, W[j], K1, encoder_width, decoder_width,
                    n_epochs, lr, batch_size, patience,
                    seed=seed_js, device=device, val_frac=val_frac,
                    w_eff_threshold=w_eff_threshold,
                )
                if verbose:
                    print(f"    grid {j+1}/{m} (r={r_grid[j]}), "
                          f"seed {s}: epochs={ep_used}, "
                          f"eff_train={n_tr}, eff_val={n_va}")
            except Exception as e:
                if verbose:
                    print(f"    grid {j+1}/{m}, seed {s} FAILED: {e}")
                X_hat_stack[j, s] = np.nan
                diagnostics.append({
                    'grid': j, 'seed': s, 'error': str(e),
                })
                continue

            model.eval()
            with torch.no_grad():
                X_all_t = torch.tensor(
                    (X_TN - mu) / sd, dtype=torch.float32
                ).to(device)
                x_hat, _ = model(X_all_t)
            X_hat_stack[j, s] = x_hat.cpu().numpy() * sd + mu

            diagnostics.append({
                'grid': j, 'seed': s,
                'epochs': ep_used,
                'eff_train': int(n_tr),
                'eff_val': int(n_va),
            })

    # Average over seeds (ignore NaN seeds)
    with np.errstate(invalid='ignore'):
        X_hat_avg = np.nanmean(X_hat_stack, axis=1)   # m x T x N

    # Check that no (grid, t) is entirely NaN
    nan_grids = np.isnan(X_hat_avg).all(axis=(1, 2))
    if nan_grids.any():
        if verbose:
            print(f"  WARNING: grids with no successful seed: "
                  f"{np.where(nan_grids)[0].tolist()}")

    # Normalize weights over j at each t
    col_sum = W.sum(axis=0, keepdims=True)             # 1 x T
    W_norm = np.where(col_sum > 1e-12, W / col_sum, 0.0)

    # Fallback for t where all weights are zero
    zero_t = np.where(col_sum[0] < 1e-12)[0]
    for t in zero_t:
        j_nn = int(np.argmin(np.abs(r_grid - (t + 1))))
        W_norm[:, t] = 0.0
        W_norm[j_nn, t] = 1.0

    # Kernel-weighted average reconstruction
    # Handle possible NaN from failed seeds by treating them as 0-weight
    X_hat_for_avg = np.where(np.isnan(X_hat_avg), 0.0, X_hat_avg)
    nan_mask = np.isnan(X_hat_avg)                      # m x T x N
    if nan_mask.any():
        # If any grid is NaN, renormalize the weights at those (j, t)
        # to avoid biasing the average downward.
        W_eff = W_norm.copy()
        for j in range(m):
            bad = nan_mask[j].any(axis=1)               # T-vector
            if bad.any():
                W_eff[j, bad] = 0.0
        W_eff_sum = W_eff.sum(axis=0, keepdims=True)
        W_eff = np.where(W_eff_sum > 1e-12, W_eff / W_eff_sum, W_norm)
        X_hat_weighted = np.einsum('jt,jtn->tn', W_eff, X_hat_for_avg)
    else:
        X_hat_weighted = np.einsum('jt,jtn->tn', W_norm, X_hat_for_avg)

    # Nearest-neighbor reconstruction
    X_hat_nn = np.zeros((T, N))
    for t in range(T):
        j_nn = int(np.argmin(np.abs(r_grid - (t + 1))))
        x_row = X_hat_avg[j_nn, t]
        # If that grid's row is NaN, fall back to weighted average
        if np.isnan(x_row).any():
            x_row = X_hat_weighted[t]
        X_hat_nn[t] = x_row

    return {
        'X_hat':       X_hat_weighted.T,   # N x T
        'X_hat_nn':    X_hat_nn.T,         # N x T
        'grid_x':      x_grid,
        'grid_r':      r_grid,
        'weights':     W,
        'h':           h,
        'diagnostics': diagnostics,
    }