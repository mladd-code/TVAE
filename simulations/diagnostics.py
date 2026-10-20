"""
run_diagnostic.py
-----------------
Diagnose TV-AE-G training dynamics on nonlinear time-varying DGPs.

For each of exponential+logistic and quadratic+logistic:
  1. Train one TV-AE-G seed with a large epoch budget and no early stopping.
     Record the full train/val loss curve.
  2. Print the curve every 20 epochs.
  3. Report the epoch at which val loss is minimized.
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tvae.dgp import DGPConfig, generate_dgp
from tvae.models.tvae_global import TimeConditionedAE


def train_with_history(X_NT, K1, n_epochs, patience, encoder_width, decoder_width,
                       lr, batch_size, val_frac, seed, device):
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

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32).to(device)
    s_train = torch.tensor(s_T[train_idx], dtype=torch.float32).to(device)
    X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32).to(device)
    s_val = torch.tensor(s_T[val_idx], dtype=torch.float32).to(device)

    torch.manual_seed(seed)
    model = TimeConditionedAE(N, K1, encoder_width, decoder_width).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    gen = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X_train, s_train),
                        batch_size=batch_size, shuffle=True, generator=gen)

    history = []
    best_val = float('inf')
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, sb in loader:
            opt.zero_grad()
            x_hat, _ = model(xb, sb)
            loss = ((x_hat - xb) ** 2).mean()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.shape[0]
            train_n += xb.shape[0]
        train_loss = train_loss_sum / train_n

        model.eval()
        with torch.no_grad():
            x_hat_v, _ = model(X_val, s_val)
            val_loss = ((x_hat_v - X_val) ** 2).mean().item()

        history.append((epoch + 1, train_loss, val_loss))

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch, best_val


def main():
    N, T, K = 100, 200, 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    configs = [
        ("exponential", "logistic"),
        ("quadratic",   "logistic"),
    ]

    for (nonlin, tv) in configs:
        print(f"\n{'='*70}")
        print(f"DGP: {nonlin}, {tv}")
        print(f"{'='*70}")
        cfg = DGPConfig(N=N, T=T, K=K, nonlinearity=nonlin,
                        time_variation=tv, error="iid", seed=42)
        sim = generate_dgp(cfg)
        X, X_star = sim["X"], sim["X_star"]

        # Full training run with effectively no early stopping
        model, history, best_epoch, best_val = train_with_history(
            X, K1=3, n_epochs=1000, patience=1000,
            encoder_width=64, decoder_width=32,
            lr=1e-3, batch_size=16, val_frac=0.2,
            seed=0, device=device,
        )

        print(f"\n{'epoch':>6} {'train':>12} {'val':>12}")
        for (ep, tr, va) in history:
            if ep % 20 == 0 or ep == len(history):
                print(f"{ep:>6} {tr:>12.6f} {va:>12.6f}")
        print(f"\nBest epoch: {best_epoch}, best val loss: {best_val:.6f}")
        print(f"Total epochs run: {len(history)}")

        # Evaluate MSE at best model
        X_TN = X.T
        s_T = (np.arange(1, T + 1) / T).astype(np.float32)
        mu = X_TN.mean(axis=0, keepdims=True)
        sd = X_TN.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X_norm = (X_TN - mu) / sd

        model.eval()
        with torch.no_grad():
            X_all = torch.tensor(X_norm, dtype=torch.float32).to(device)
            s_all = torch.tensor(s_T, dtype=torch.float32).to(device)
            x_hat, _ = model(X_all, s_all)
        X_hat = (x_hat.cpu().numpy() * sd + mu).T
        mse = float(np.mean((X_hat - X_star) ** 2))
        print(f"MSE of common component (full T=200, one seed): {mse:.4f}")

        # Now compare with early-stopped training at different patience
        print(f"\n--- MSE vs patience ---")
        for pat in [10, 30, 100, 1000]:
            m, h, be, bv = train_with_history(
                X, K1=3, n_epochs=1000, patience=pat,
                encoder_width=64, decoder_width=32,
                lr=1e-3, batch_size=16, val_frac=0.2,
                seed=0, device=device,
            )
            m.eval()
            with torch.no_grad():
                x_hat, _ = m(X_all, s_all)
            X_hat = (x_hat.cpu().numpy() * sd + mu).T
            mse = float(np.mean((X_hat - X_star) ** 2))
            print(f"  patience={pat:>4}: epochs={len(h):>4}, "
                  f"best_ep={be:>4}, mse={mse:.4f}")


if __name__ == "__main__":
    main()