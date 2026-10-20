"""
tvae_global.py
--------------
Time-conditioned global autoencoder (TV-AE-G).

Architecture:
  Encoder rho  : R^N (+ optional t/T) -> R^{K1}
  Decoder phi_i: R^{K1} x R           -> R        (takes (z_t, t/T) for each unit i)

  X_{it} = phi_i(rho(X_t, t/T), t/T) + U_{it}

The `encoder_time` flag controls whether the encoder also sees t/T:
  - encoder_time = False : encoder sees X_t only (previous version)
  - encoder_time = True  : encoder sees (X_t, t/T) concatenated (recommended)

The decoder always sees t/T. This is the key asymmetry: the decoder must be
able to represent a loading function that varies with t, and the encoder must
be able to invert a loading function that varies with t.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TimeConditionedAE(nn.Module):
    def __init__(self, N: int, K1: int,
                 encoder_width: int = 64,
                 decoder_width: int = 32,
                 encoder_time: bool = True):
        super().__init__()
        self.N = N
        self.K1 = K1
        self.encoder_time = encoder_time

        enc_in = N + 1 if encoder_time else N
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )

        # Disjoint decoders: (K1 + 1) -> 1 per unit (extra dim is t/T)
        self.W1 = nn.Parameter(torch.empty(N, K1 + 1, decoder_width))
        self.b1 = nn.Parameter(torch.zeros(N, decoder_width))
        self.W2 = nn.Parameter(torch.empty(N, decoder_width, 1))
        self.b2 = nn.Parameter(torch.zeros(N, 1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """X: B x N, s: B or B x 1  ->  F: B x K1"""
        if self.encoder_time:
            if s.dim() == 1:
                s = s.unsqueeze(-1)
            X_aug = torch.cat([X, s], dim=-1)
            return self.encoder(X_aug)
        return self.encoder(X)

    def decode(self, F: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """F: B x K1, s: B or B x 1  ->  X_hat: B x N"""
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        y = torch.cat([F, s], dim=-1)
        H = torch.einsum('bk,nkw->bnw', y, self.W1) + self.b1[None]
        A = torch.relu(H)
        out = torch.einsum('bnw,nwo->bno', A, self.W2) + self.b2[None]
        return out.squeeze(-1)

    def forward(self, X: torch.Tensor, s: torch.Tensor):
        F = self.encode(X, s)
        X_hat = self.decode(F, s)
        return X_hat, F


def _train_one(X_train: torch.Tensor, s_train: torch.Tensor,
               X_val: torch.Tensor, s_val: torch.Tensor,
               N: int, K1: int,
               encoder_width: int, decoder_width: int,
               encoder_time: bool,
               n_epochs: int, lr: float, batch_size: int,
               patience: int, seed: int,
               device: str,
               weight_decay: float = 0.0,
               ) -> Tuple[TimeConditionedAE, int]:
    torch.manual_seed(seed)
    model = TimeConditionedAE(N, K1, encoder_width, decoder_width,
                              encoder_time=encoder_time).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    gen = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X_train, s_train),
                        batch_size=batch_size, shuffle=True, generator=gen)

    best_val = float('inf')
    best_state = None
    stale = 0
    epochs_used = 0

    for epoch in range(n_epochs):
        epochs_used = epoch + 1
        model.train()
        for xb, sb in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb, sb)
            loss = ((x_hat - xb) ** 2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_hat_v, _ = model(X_val, s_val)
            val_loss = ((x_hat_v - X_val) ** 2).mean().item()

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
    return model, epochs_used


def fit_tvae_global(X_NT: np.ndarray, K1: int,
                    n_seeds: int = 3,
                    n_epochs: int = 500,
                    lr: float = 1e-3,
                    batch_size: int = 16,
                    patience: int = 30,
                    encoder_width: int = 64,
                    decoder_width: int = 32,
                    encoder_time: bool = True,
                    weight_decay: float = 0.0,
                    val_frac: float = 0.2,
                    device: str = 'cpu',
                    verbose: bool = False
                    ) -> Dict[str, Any]:
    N, T = X_NT.shape
    X_TN = X_NT.T
    s_T = (np.arange(1, T + 1) / T).astype(np.float32)

    mu = X_TN.mean(axis=0, keepdims=True)
    sd = X_TN.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X_norm = (X_TN - mu) / sd

    rng = np.random.default_rng(0)
    perm = rng.permutation(T)
    n_val = max(int(val_frac * T), 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train_t = torch.tensor(X_norm[train_idx], dtype=torch.float32).to(device)
    s_train_t = torch.tensor(s_T[train_idx], dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_norm[val_idx], dtype=torch.float32).to(device)
    s_val_t = torch.tensor(s_T[val_idx], dtype=torch.float32).to(device)

    X_all_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
    s_all_t = torch.tensor(s_T, dtype=torch.float32).to(device)

    X_hat_sum = np.zeros((T, N))
    F_hat_sum = np.zeros((T, K1))
    diagnostics = []

    for seed in range(n_seeds):
        model, ep_used = _train_one(
            X_train_t, s_train_t, X_val_t, s_val_t,
            N, K1, encoder_width, decoder_width, encoder_time,
            n_epochs, lr, batch_size, patience, seed, device,
            weight_decay=weight_decay,
        )
        if verbose:
            print(f"    TV-AE-G (encoder_time={encoder_time}) "
                  f"seed {seed}: epochs={ep_used}")
        diagnostics.append({'seed': seed, 'epochs': ep_used})

        model.eval()
        with torch.no_grad():
            x_hat, f_hat = model(X_all_t, s_all_t)
        X_hat_sum += x_hat.cpu().numpy()
        F_hat_sum += f_hat.cpu().numpy()

    X_hat_norm = X_hat_sum / n_seeds
    F_hat_avg = F_hat_sum / n_seeds
    X_hat = X_hat_norm * sd + mu

    return {
        'X_hat':       X_hat.T,
        'F_hat_all':   F_hat_avg.T,
        'diagnostics': diagnostics,
    }