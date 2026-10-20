"""
tvae_global_app_1.py
--------------------
Application 1 (Design 1): TV-AE-G for unbalanced equity panels.

Model:
    r_tilde_it   = r_it - mu_i                                    # demeaned return
    F_t          = rho(X_tilde_t, mask_t, tau_t)                  # encoder
    beta_{i,t}   = psi_i(tau_t)  in R^{K1}                        # per-permno loadings
    r_hat_it     = mu_i + beta_{i,t}' F_t                         # fitted (total)
    r_pred_it    = mu_i + beta_{i,t}' lambda_t                    # predictive

mu_i is the per-stock training-sample mean return, computed once per
rolling refit. It is subtracted from both the encoder input and the
target, and added back at prediction time. This forces the model to
learn factor structure rather than collapse to per-stock means.

Unbalanced-panel handling:
    - Fixed max cross-section size N_max, chosen as the top-N_max permnos
      by average market cap over the training sample.
    - Each permno has a fixed slot in the encoder input.
    - For each month, the encoder input is a (N_max,) vector with the
      demeaned returns of present permnos in their slots and 0 elsewhere,
      plus a (N_max,) 0/1 mask indicating presence.

Data interface:
    df must have columns: date (datetime64), permno (int), ret (float),
    mcap (float). One row per (permno, month).

Output:
    A dict with keys:
        'permno_map' : dict {permno -> slot index}
        'X_hat'      : DataFrame (date, permno, pred_ret, fitted_ret)
        'factors'    : DataFrame (date, F_1..F_K1)
        'betas'      : DataFrame (date, permno, beta_1..beta_K1)
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MonthlyPanel(Dataset):
    """
    One item per month:
      X_pad        : (N_max,)          float32, demeaned returns
      M_pad        : (N_max,)          float32
      permno_idx   : (N_t,)            int64
      tau          : scalar            float32
      y            : (N_t,)            float32, demeaned returns
      w            : (N_t,)            float32, cap weights (normalized)
    """

    def __init__(self,
                 X_pad: np.ndarray,
                 M_pad: np.ndarray,
                 permno_idx: List[np.ndarray],
                 tau: np.ndarray,
                 y: List[np.ndarray],
                 w: List[np.ndarray]):

        self.X_pad = torch.tensor(X_pad, dtype=torch.float32)
        self.M_pad = torch.tensor(M_pad, dtype=torch.float32)
        self.permno_idx = [torch.tensor(idx, dtype=torch.long)
                           for idx in permno_idx]
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.y = [torch.tensor(yi, dtype=torch.float32) for yi in y]
        self.w = [torch.tensor(wi, dtype=torch.float32) for wi in w]

    def __len__(self):
        return self.X_pad.shape[0]

    def __getitem__(self, i):
        return (self.X_pad[i], self.M_pad[i], self.permno_idx[i],
                self.tau[i], self.y[i], self.w[i])


# --------------------------------------------------------------------------- #
# Model (no free alpha in the network; mu is precomputed outside)
# --------------------------------------------------------------------------- #
class TVEGApp1(nn.Module):
    """
    Encoder : Linear(2*N_max + 1, encoder_width) -> ReLU -> Linear(., K1)
    Beta    : per-permno MLP, tau -> R^{K1}
              W1 (N_total, 1, beta_width), b1 (N_total, beta_width)
              W2 (N_total, beta_width, K1), b2 (N_total, K1)
    """

    def __init__(self, N_max: int, N_total: int, K1: int,
                 encoder_width: int = 64,
                 beta_width: int = 16):
        super().__init__()
        self.N_max = N_max
        self.N_total = N_total
        self.K1 = K1

        self.encoder = nn.Sequential(
            nn.Linear(2 * N_max + 1, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )

        self.W1 = nn.Parameter(torch.empty(N_total, 1, beta_width))
        self.b1 = nn.Parameter(torch.zeros(N_total, beta_width))
        self.W2 = nn.Parameter(torch.empty(N_total, beta_width, K1))
        self.b2 = nn.Parameter(torch.zeros(N_total, K1))
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def encode(self, X_pad: torch.Tensor, M_pad: torch.Tensor,
               tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([X_pad * M_pad, M_pad, tau.view(1)])
        return self.encoder(inp)

    def beta(self, permno_idx: torch.Tensor,
             tau: torch.Tensor) -> torch.Tensor:
        """
        permno_idx: (N_t,), tau: scalar
        returns beta: (N_t, K1)
        """
        W1_sel = self.W1[permno_idx]
        b1_sel = self.b1[permno_idx]
        W2_sel = self.W2[permno_idx]
        b2_sel = self.b2[permno_idx]
        H = torch.relu(W1_sel.squeeze(1) * tau.view(1, 1) + b1_sel)
        beta = torch.einsum('nw,nwk->nk', H, W2_sel) + b2_sel
        return beta

    def forward(self, X_pad, M_pad, permno_idx, tau):
        F_t = self.encode(X_pad, M_pad, tau)
        beta = self.beta(permno_idx, tau)
        r_hat = beta @ F_t
        return r_hat, F_t, beta


# --------------------------------------------------------------------------- #
# Panel construction (with optional demeaning by mu)
# --------------------------------------------------------------------------- #
def build_panel(df: pd.DataFrame,
                permno_map: Dict[int, int],
                N_max: int,
                mu_arr: Optional[np.ndarray] = None,
                tau_start: float = 0.0,
                tau_end: float = 1.0
                ) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray,
                           List, List, List]:
    """
    If mu_arr is provided (shape (N_max,)), each return is demeaned by
    its slot's value before being stored in X_pad and y.

    Returns:
        X_pad       (T, N_max)  demeaned returns (if mu_arr given)
        M_pad       (T, N_max)
        permno_idx  list of T arrays
        tau         (T,)
        y           list of T arrays
        w           list of T arrays
        dates       list of T dates
    """
    dates = np.sort(df['date'].unique())
    T = len(dates)
    X_pad = np.zeros((T, N_max), dtype=np.float32)
    M_pad = np.zeros((T, N_max), dtype=np.float32)
    permno_idx = [None] * T
    y_list = [None] * T
    w_list = [None] * T
    tau = np.linspace(tau_start, tau_end, T, dtype=np.float32)

    for t, d in enumerate(dates):
        sub = df[df['date'] == d]
        sub = sub[sub['permno'].isin(permno_map)]
        if len(sub) == 0:
            permno_idx[t] = np.array([], dtype=np.int64)
            y_list[t] = np.array([], dtype=np.float32)
            w_list[t] = np.array([], dtype=np.float32)
            continue

        idx = np.array([permno_map[p] for p in sub['permno'].values],
                       dtype=np.int64)
        ret = sub['ret'].values.astype(np.float32)
        mcap = sub['mcap'].values.astype(np.float32)

        if mu_arr is not None:
            ret = ret - mu_arr[idx]

        X_pad[t, idx] = ret
        M_pad[t, idx] = 1.0
        permno_idx[t] = idx
        y_list[t] = ret
        w = mcap / mcap.sum() if mcap.sum() > 0 else \
            np.ones_like(mcap) / len(mcap)
        w_list[t] = w.astype(np.float32)

    return X_pad, M_pad, permno_idx, tau, y_list, w_list, list(dates)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train_one(model: TVEGApp1,
               train_ds: MonthlyPanel,
               val_ds: MonthlyPanel,
               n_epochs: int,
               lr: float,
               weight_decay: float,
               patience: int,
               device: str,
               seed: int,
               value_weighted: bool = False,
               verbose: bool = False):
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    best_val = float('inf')
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for X_pad, M_pad, permno_idx, tau, y, w in train_ds:
            X_pad = X_pad.to(device)
            M_pad = M_pad.to(device)
            permno_idx = permno_idx.to(device)
            tau = tau.to(device)
            y = y.to(device)
            w = w.to(device)
            if permno_idx.numel() == 0:
                continue
            optimizer.zero_grad()
            r_hat, _, _ = model(X_pad, M_pad, permno_idx, tau)
            resid2 = (r_hat - y) ** 2
            if value_weighted:
                loss = (w * resid2).sum() / w.sum().clamp_min(1e-12)
            else:
                loss = resid2.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * permno_idx.numel()
            total_n += permno_idx.numel()
        train_loss = total_loss / max(total_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for X_pad, M_pad, permno_idx, tau, y, w in val_ds:
                X_pad = X_pad.to(device)
                M_pad = M_pad.to(device)
                permno_idx = permno_idx.to(device)
                tau = tau.to(device)
                y = y.to(device)
                if permno_idx.numel() == 0:
                    continue
                r_hat, _, _ = model(X_pad, M_pad, permno_idx, tau)
                resid2 = (r_hat - y) ** 2
                val_loss_sum += resid2.sum().item()
                val_n += permno_idx.numel()
        val_loss = val_loss_sum / max(val_n, 1)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}: train={train_loss:.6f} "
                  f"val={val_loss:.6f}")

        if val_loss < best_val - 1e-9:
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
    return model, best_val


# --------------------------------------------------------------------------- #
# Prediction helper (returns demeaned predictions)
# --------------------------------------------------------------------------- #
def predict_for_month(model: TVEGApp1,
                      X_pad: torch.Tensor,
                      M_pad: torch.Tensor,
                      permno_idx: torch.Tensor,
                      tau: torch.Tensor,
                      lambda_t: torch.Tensor
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (r_hat_pred_demean, r_hat_fit_demean, F_t, beta):
        r_hat_pred_demean : (N_t,)   beta @ lambda_t
        r_hat_fit_demean  : (N_t,)   beta @ F_t
        F_t               : (K1,)
        beta              : (N_t, K1)

    Caller must add the per-stock mean mu back.
    """
    model.eval()
    with torch.no_grad():
        F_t = model.encode(X_pad, M_pad, tau)
        beta = model.beta(permno_idx, tau)
        r_hat_pred = beta @ lambda_t
        r_hat_fit  = beta @ F_t
    return (r_hat_pred.cpu().numpy(),
            r_hat_fit.cpu().numpy(),
            F_t.cpu().numpy(),
            beta.cpu().numpy())


# --------------------------------------------------------------------------- #
# Rolling annual refit (module-level; used by run_app1_slice.py via generic)
# --------------------------------------------------------------------------- #
def rolling_fit(df: pd.DataFrame,
                N_max: int = 2000,
                K1: int = 3,
                encoder_width: int = 64,
                beta_width: int = 16,
                n_seeds: int = 3,
                n_epochs: int = 500,
                lr: float = 1e-3,
                weight_decay: float = 1e-3,
                patience: int = 30,
                initial_train_years: int = 10,
                val_months: int = 12,
                purge_months: int = 1,
                embargo_months: int = 1,
                oos_start_year: int = 2015,
                oos_end_year: int = 2024,
                value_weighted_loss: bool = False,
                device: str = 'cpu',
                verbose: bool = True
                ) -> Dict[str, Any]:
    """
    Rolling annual refit with per-stock demeaning by training-sample mean.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df_train = df[df['date'] < pd.Timestamp(f"{oos_start_year}-01-01")]
    avg_mcap = (df_train.groupby('permno')['mcap'].mean()
                .sort_values(ascending=False))
    top_permnos = avg_mcap.head(N_max).index.tolist()
    permno_map = {p: i for i, p in enumerate(top_permnos)}
    N_total = len(permno_map)
    if verbose:
        print(f"Permno map: N_total = {N_total}, "
              f"avg mcap cutoff = {avg_mcap.iloc[N_total - 1]:.2f}")

    all_dates = np.sort(df['date'].unique())
    T_all = len(all_dates)
    date_to_tau = {d: i / (T_all - 1) for i, d in enumerate(all_dates)}

    oos_preds = []
    oos_factors = []
    oos_betas = []

    for year in range(oos_start_year, oos_end_year + 1):
        if verbose:
            print(f"\n=== Year {year} ===")
        t0 = time.time()

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end   = pd.Timestamp(f"{year}-12-31")

        val_end = test_start - pd.DateOffset(months=embargo_months + 1)
        val_start = val_end - pd.DateOffset(months=val_months - 1)
        train_end = val_start - pd.DateOffset(months=purge_months + 1)

        train_df = df[df['date'] <= train_end]
        val_df   = df[(df['date'] >= val_start) & (df['date'] <= val_end)]
        test_df  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            if verbose:
                print(f"  Skipping year {year}: empty split.")
            continue

        # Per-stock mean from training sample
        mu_arr = np.zeros(N_max, dtype=np.float32)
        for slot, p in enumerate(top_permnos):
            sub = train_df[train_df['permno'] == p]
            if len(sub) > 0:
                mu_arr[slot] = sub['ret'].mean()

        Xp_tr, Mp_tr, idx_tr, _, y_tr, w_tr, dates_tr = build_panel(
            train_df, permno_map, N_max, mu_arr=mu_arr)
        Xp_va, Mp_va, idx_va, _, y_va, w_va, dates_va = build_panel(
            val_df, permno_map, N_max, mu_arr=mu_arr)
        Xp_te, Mp_te, idx_te, _, y_te, w_te, dates_te = build_panel(
            test_df, permno_map, N_max, mu_arr=mu_arr)

        tau_tr = np.array([date_to_tau[d] for d in dates_tr], dtype=np.float32)
        tau_va = np.array([date_to_tau[d] for d in dates_va], dtype=np.float32)
        tau_te = np.array([date_to_tau[d] for d in dates_te], dtype=np.float32)

        train_ds = MonthlyPanel(Xp_tr, Mp_tr, idx_tr, tau_tr, y_tr, w_tr)
        val_ds   = MonthlyPanel(Xp_va, Mp_va, idx_va, tau_va, y_va, w_va)

        ensemble_pred = [None] * len(dates_te)
        ensemble_fit  = [None] * len(dates_te)
        ensemble_F    = np.zeros((len(dates_te), K1), dtype=np.float64)
        ensemble_B    = [None] * len(dates_te)
        n_valid = 0

        for seed in range(n_seeds):
            if verbose:
                print(f"  Training seed {seed + 1}/{n_seeds}...")
            model = TVEGApp1(N_max, N_total, K1, encoder_width, beta_width)
            model, best_val = _train_one(
                model, train_ds, val_ds,
                n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay,
                patience=patience, device=device, seed=seed,
                value_weighted=value_weighted_loss, verbose=False)
            if verbose:
                print(f"    best val loss = {best_val:.6f}")

            hist_df = df[df['date'] <= test_end]
            Xp_h, Mp_h, _, _, _, _, dates_h = build_panel(
                hist_df, permno_map, N_max, mu_arr=mu_arr)
            tau_h = np.array([date_to_tau[d] for d in dates_h],
                             dtype=np.float32)

            model.eval()
            F_hist = np.zeros((len(dates_h), K1), dtype=np.float64)
            with torch.no_grad():
                for i in range(len(dates_h)):
                    Xp_t = torch.tensor(Xp_h[i],
                                        dtype=torch.float32).to(device)
                    Mp_t = torch.tensor(Mp_h[i],
                                        dtype=torch.float32).to(device)
                    tau_t = torch.tensor(tau_h[i],
                                         dtype=torch.float32).to(device)
                    F_hist[i] = model.encode(Xp_t, Mp_t, tau_t).cpu().numpy()

            F_cummean = (np.cumsum(F_hist, axis=0)
                         / np.arange(1, len(dates_h) + 1)[:, None])
            date_to_idx = {d: i for i, d in enumerate(dates_h)}

            with torch.no_grad():
                for k, d in enumerate(dates_te):
                    if len(idx_te[k]) == 0:
                        continue
                    i_h = date_to_idx[d]
                    lambda_t = torch.tensor(F_cummean[i_h],
                                            dtype=torch.float32).to(device)

                    Xp_t = torch.tensor(Xp_te[k],
                                        dtype=torch.float32).to(device)
                    Mp_t = torch.tensor(Mp_te[k],
                                        dtype=torch.float32).to(device)
                    idx_t = torch.tensor(idx_te[k],
                                         dtype=torch.long).to(device)
                    tau_t = torch.tensor(tau_te[k],
                                         dtype=torch.float32).to(device)

                    r_pred, r_fit, F_t, beta = predict_for_month(
                        model, Xp_t, Mp_t, idx_t, tau_t, lambda_t)

                    if ensemble_pred[k] is None:
                        ensemble_pred[k] = r_pred.astype(np.float64)
                        ensemble_fit[k]  = r_fit.astype(np.float64)
                        ensemble_B[k]    = beta.astype(np.float64)
                    else:
                        ensemble_pred[k] += r_pred
                        ensemble_fit[k]  += r_fit
                        ensemble_B[k]    += beta
                    ensemble_F[k] += F_t

            n_valid += 1

        if n_valid == 0:
            continue

        ensemble_F /= n_valid
        for k in range(len(ensemble_B)):
            if ensemble_B[k] is not None:
                ensemble_B[k]    /= n_valid
                ensemble_pred[k] /= n_valid
                ensemble_fit[k]  /= n_valid

        # Add mu back and store
        for k, d in enumerate(dates_te):
            if len(idx_te[k]) == 0 or ensemble_B[k] is None:
                continue
            slot_permnos = [top_permnos[j] for j in idx_te[k]]
            mu_t = mu_arr[idx_te[k]]
            pred_t = ensemble_pred[k] + mu_t
            fit_t  = ensemble_fit[k] + mu_t
            F_t    = ensemble_F[k]
            beta_t = ensemble_B[k]

            for j, p in enumerate(slot_permnos):
                oos_preds.append(
                    (d, p, float(pred_t[j]), float(fit_t[j])))
            oos_factors.append((d, *[float(F_t[c]) for c in range(K1)]))
            for j, p in enumerate(slot_permnos):
                oos_betas.append((d, p,
                                  *[float(beta_t[j, c])
                                    for c in range(K1)]))

        if verbose:
            n_preds = sum(len(idx_te[k]) for k in range(len(dates_te)))
            print(f"  done in {(time.time() - t0) / 60:.2f} min; "
                  f"n_preds={n_preds}")

    pred_df = pd.DataFrame(
        oos_preds,
        columns=['date', 'permno', 'pred_ret', 'fitted_ret'])
    factor_df = pd.DataFrame(
        oos_factors, columns=['date'] + [f'F_{c+1}' for c in range(K1)])
    factor_df = factor_df.groupby('date').first().reset_index()
    beta_cols = ['date', 'permno'] + [f'beta_{c+1}' for c in range(K1)]
    beta_df = pd.DataFrame(oos_betas, columns=beta_cols)

    return {
        'permno_map': permno_map,
        'X_hat': pred_df,
        'factors': factor_df,
        'betas': beta_df,
    }