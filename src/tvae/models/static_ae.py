"""
static_ae.py
------------
SX-style static autoencoder for nonlinear factor models.

Shen & Xiu (2026) "Deep Autoencoders for Nonlinear Factor Models":
  X_{it} = phi_i(F_t) + U_{it}
  phi_i: R^K -> R    (nonlinear, unit-specific)
  F_t  : K-dim latent factor

Architecture (disjoint output AE, SX Fig. 3):
  Encoder: rho: R^N -> R^{K1}  (a small MLP, shared across units)
  Decoder: N separate MLPs phi_i: R^{K1} -> R

This is the SX model with NO time variation and NO characteristics.
It is the static nonlinear benchmark for the TV-AE.

In our panel convention, X is N x T.  We transpose to T x N for batching
over time.
"""

from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class StaticAE(nn.Module):
    """
    Disjoint output autoencoder with K1-neuron bottleneck.
    Encoder width and decoder width are the number of neurons per hidden
    layer (one hidden layer on each side, as in SX's AE1-like architectures).
    """
    def __init__(self, N: int, K1: int,
                 encoder_width: int = 64,
                 decoder_width: int = 32):
        super().__init__()
        self.N = N
        self.K1 = K1

        # Encoder: R^N -> R^K1
        self.encoder = nn.Sequential(
            nn.Linear(N, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )

        # Disjoint decoder: N separate one-hidden-layer MLPs
        #   phi_i(z) = W2_i ReLU(W1_i z + b1_i) + b2_i
        # We keep these as batched 3-tensors so the forward pass is efficient.
        self.W1 = nn.Parameter(torch.empty(N, K1, decoder_width))
        self.b1 = nn.Parameter(torch.zeros(N, decoder_width))
        self.W2 = nn.Parameter(torch.empty(N, decoder_width, 1))
        self.b2 = nn.Parameter(torch.zeros(N, 1))

        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """X: B x N  ->  F: B x K1"""
        return self.encoder(X)

    def decode(self, F: torch.Tensor) -> torch.Tensor:
        """F: B x K1  ->  X_hat: B x N"""
        H = torch.einsum('bk,nkw->bnw', F, self.W1) + self.b1[None]
        A = torch.relu(H)
        out = torch.einsum('bnw,nwo->bno', A, self.W2) + self.b2[None]
        return out.squeeze(-1)

    def forward(self, X: torch.Tensor):
        F = self.encode(X)
        X_hat = self.decode(F)
        return X_hat, F


def _train_one(X_train, X_val, N, K1,
               n_epochs, lr, batch_size, patience,
               encoder_width, decoder_width,
               seed, device, weight_decay=0.0):
    torch.manual_seed(seed)
    model = StaticAE(N, K1, encoder_width, decoder_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    gen = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(TensorDataset(X_train),
                              batch_size=batch_size,
                              shuffle=True, generator=gen)

    best_val = float('inf')
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        for (xb,) in train_loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = nn.functional.mse_loss(x_hat, xb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_hat_val, _ = model(X_val)
            val_loss = nn.functional.mse_loss(x_hat_val, X_val).item()

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
    return model


def fit_static_ae(X_NT, K1,
                  n_seeds=3,
                  n_epochs=300,
                  lr=1e-3,
                  batch_size=32,
                  patience=20,
                  encoder_width=64,
                  decoder_width=32,
                  weight_decay=0.0,
                  val_frac=0.2,
                  device='cpu',
                  verbose=False):
    """
    Fit an ensemble of StaticAEs on X_NT (N x T).
    Returns:
        'X_hat'     : N x T  ensemble-averaged reconstruction
        'F_hat_all' : K1 x T ensemble-averaged encoder output
    """
    N, T = X_NT.shape
    X_TN = X_NT.T  # T x N

    # Per-unit standardisation (does not change the estimand, just helps
    # the optimiser behave; we un-standardise on the way out).
    mu = X_TN.mean(axis=0, keepdims=True)
    sd = X_TN.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_TN_norm = (X_TN - mu) / sd

    # Internal train/val split for early stopping
    T_val = max(int(val_frac * T), 5)
    T_train = T - T_val

    X_train = torch.tensor(X_TN_norm[:T_train], dtype=torch.float32).to(device)
    X_val = torch.tensor(X_TN_norm[T_train:], dtype=torch.float32).to(device)
    X_all = torch.tensor(X_TN_norm, dtype=torch.float32).to(device)

    X_hat_sum = np.zeros((T, N), dtype=np.float64)
    F_hat_sum = np.zeros((T, K1), dtype=np.float64)

    for seed in range(n_seeds):
        if verbose:
            print(f"    static AE: seed {seed}/{n_seeds}...")
        model = _train_one(X_train, X_val, N, K1,
                           n_epochs, lr, batch_size, patience,
                           encoder_width, decoder_width,
                           seed, device, weight_decay=weight_decay)
        model.eval()
        with torch.no_grad():
            x_hat, f_hat = model(X_all)
        X_hat_sum += x_hat.cpu().numpy().astype(np.float64)
        F_hat_sum += f_hat.cpu().numpy().astype(np.float64)

    X_hat_norm = X_hat_sum / n_seeds
    F_hat_avg = F_hat_sum / n_seeds

    X_hat = X_hat_norm * sd + mu  # un-standardise

    return {
        'X_hat': X_hat.T,           # N x T
        'F_hat_all': F_hat_avg.T,   # K1 x T
    }