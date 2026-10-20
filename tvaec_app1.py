"""
tvaec_app1.py
=============
TV-AE-C (Time-Varying Autoencoder with Characteristics) for Application 1.

Model
-----
    F_t          = rho(x_t, tau_t)                    in R^{K1}
    beta_{i,t}   = psi(Z_{i,t-1}, tau_t)              in R^{K1}
    fitted_ret   = mu_i + beta_{i,t}' F_t             (contemporaneous)
    pred_ret     = mu_i + beta_{i,t}' lambda_t        (expanding mean)

    x_t  : managed portfolio returns at month t (P-dim)
    Z_i  : characteristics of stock i at end of previous month
    mu_i : per-stock training-sample mean

Characteristics (all backward-looking):
    size = log(market cap)
    mom  = cumulative log return from t-11 to t-1 (skip most recent month)
    rev  = return in month t
    vol  = 12-month rolling std of returns ending at t

Data interface
--------------
df with columns [date, permno, ret, mcap]
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


CHAR_COLS = ["size", "mom", "rev", "vol"]


# --------------------------------------------------------------------------- #
# Characteristic construction
# --------------------------------------------------------------------------- #
def compute_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["permno", "date"]).copy()
    g = df.groupby("permno")

    df["size"] = np.log(df["mcap"].clip(lower=1e-6))
    df["rev"] = df["ret"]
    df["_logret"] = np.log1p(df["ret"].clip(lower=-0.99))

    df["mom"] = (g["_logret"]
                 .transform(lambda x: x.rolling(11, min_periods=6).sum().shift(1)))
    df["vol"] = (g["ret"]
                 .transform(lambda x: x.rolling(12, min_periods=6).std()))

    df = df.drop(columns=["_logret"])
    return df


def rank_normalize(df: pd.DataFrame, char_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in char_cols:
        rank = df.groupby("date")[c].rank(pct=True)
        df[c] = 2 * rank - 1
        df[c] = df[c].fillna(0.0)
    return df


def prepare_data(df: pd.DataFrame,
                 n_top: Optional[int] = None,
                 date_start: Optional[str] = None,
                 date_end: Optional[str] = None,
                 char_cols: List[str] = CHAR_COLS
                 ) -> Tuple[pd.DataFrame, Dict[int, int]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if date_start is not None:
        df = df[df["date"] >= date_start]
    if date_end is not None:
        df = df[df["date"] <= date_end]
    if n_top is not None:
        avg_mcap = df.groupby("permno")["mcap"].mean().sort_values(ascending=False)
        top = avg_mcap.head(n_top).index.tolist()
        df = df[df["permno"].isin(top)].copy()

    permno_codes = {p: i for i, p in enumerate(sorted(df["permno"].unique()))}
    df["permno"] = df["permno"].map(permno_codes).astype("int64")
    df["mcap"] = df["mcap"] / 1e6

    df = compute_characteristics(df)
    df = rank_normalize(df, char_cols)
    return df, permno_codes


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def build_tvaec_panel(df, permno_map, N_max, char_cols):
    """
    Returns:
        X_mgd (T, P)                    managed portfolio returns
        Z_list list of T                each (N_t, P)
        y_list list of T                each (N_t,) raw returns (NOT demeaned)
        permno_idx_list list of T       each (N_t,) int64
        tau (T,)
        dates (T,) list of datetime
    """
    dates = np.sort(df["date"].unique())
    T = len(dates)
    P = len(char_cols)

    X_mgd = np.zeros((T, P), dtype=np.float32)
    Z_list: List[np.ndarray] = [None] * T
    y_list: List[np.ndarray] = [None] * T
    permno_idx_list: List[np.ndarray] = [None] * T
    tau = np.linspace(0, 1, T, dtype=np.float32)

    df_by_date = {d: g for d, g in df.groupby("date")}

    for t, d in enumerate(dates):
        if t == 0:
            Z_list[t] = np.zeros((0, P), dtype=np.float32)
            y_list[t] = np.zeros(0, dtype=np.float32)
            permno_idx_list[t] = np.zeros(0, dtype=np.int64)
            continue

        d_prev = dates[t - 1]
        ret_df = df_by_date[d]
        chars_prev = df_by_date[d_prev]

        ret_small = ret_df[ret_df["permno"].isin(permno_map)][["permno", "ret"]]
        chars_small = chars_prev[chars_prev["permno"].isin(permno_map)][
            ["permno"] + char_cols]

        merged = ret_small.merge(chars_small, on="permno", how="inner")
        if len(merged) == 0:
            Z_list[t] = np.zeros((0, P), dtype=np.float32)
            y_list[t] = np.zeros(0, dtype=np.float32)
            permno_idx_list[t] = np.zeros(0, dtype=np.int64)
            continue

        Z_mat = merged[char_cols].values.astype(np.float32)
        r_vec = merged["ret"].values.astype(np.float32)

        # Ridge-stabilized managed portfolio
        ZtZ = Z_mat.T @ Z_mat + 1e-4 * np.eye(P, dtype=np.float32)
        Ztr = Z_mat.T @ r_vec
        try:
            x_t = np.linalg.solve(ZtZ, Ztr)
        except np.linalg.LinAlgError:
            x_t = np.zeros(P, dtype=np.float32)

        X_mgd[t] = x_t
        Z_list[t] = Z_mat
        y_list[t] = r_vec
        permno_idx_list[t] = np.array(
            [permno_map[p] for p in merged["permno"].values], dtype=np.int64)

    return X_mgd, Z_list, y_list, permno_idx_list, tau, list(dates)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TVAECApp1(nn.Module):
    def __init__(self, P: int, K1: int,
                 encoder_width: int = 32,
                 beta_width: int = 32):
        super().__init__()
        self.P = P
        self.K1 = K1

        self.encoder = nn.Sequential(
            nn.Linear(P + 1, encoder_width),
            nn.ReLU(),
            nn.Linear(encoder_width, K1),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(P + 1, beta_width),
            nn.ReLU(),
            nn.Linear(beta_width, K1),
        )

    def encode(self, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, tau.view(1)])
        return self.encoder(inp)

    def beta(self, Z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        N = Z.shape[0]
        tau_col = tau.view(1, 1).expand(N, 1)
        inp = torch.cat([Z, tau_col], dim=1)
        return self.beta_net(inp)

    def forward(self, x_t, Z, tau):
        F_t = self.encode(x_t, tau)
        beta = self.beta(Z, tau)
        r_hat = beta @ F_t
        return r_hat, F_t, beta


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TVAECPanel(Dataset):
    def __init__(self, X_mgd, Z_list, tau, y_list):
        self.X_mgd = torch.tensor(X_mgd, dtype=torch.float32)
        self.Z_list = [torch.tensor(z, dtype=torch.float32) for z in Z_list]
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.y_list = [torch.tensor(y, dtype=torch.float32) for y in y_list]
        self.T = len(self.Z_list)

    def __len__(self):
        return self.T

    def __getitem__(self, i):
        return (self.X_mgd[i], self.Z_list[i], self.tau[i], self.y_list[i])


def _single_collate(batch):
    return batch[0]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train_one_tvaec(model, train_loader, val_loader,
                     n_epochs, lr, weight_decay, patience,
                     device, seed, verbose=False):
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float('inf')
    best_state = None
    stale = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for x_t, Z_t, tau_t, y_t in train_loader:
            x_t = x_t.to(device)
            Z_t = Z_t.to(device)
            tau_t = tau_t.to(device)
            y_t = y_t.to(device)
            if Z_t.shape[0] == 0:
                continue
            optimizer.zero_grad()
            r_hat, _, _ = model(x_t, Z_t, tau_t)
            loss = ((r_hat - y_t) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Z_t.shape[0]
            total_n += Z_t.shape[0]

        train_loss = total_loss / max(total_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x_t, Z_t, tau_t, y_t in val_loader:
                x_t = x_t.to(device)
                Z_t = Z_t.to(device)
                tau_t = tau_t.to(device)
                y_t = y_t.to(device)
                if Z_t.shape[0] == 0:
                    continue
                r_hat, _, _ = model(x_t, Z_t, tau_t)
                val_loss_sum += ((r_hat - y_t) ** 2).sum().item()
                val_n += Z_t.shape[0]
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
# Rolling annual refit
# --------------------------------------------------------------------------- #
def rolling_fit_tvaec(df, N_max, K1, char_cols=CHAR_COLS,
                      encoder_width=32, beta_width=32,
                      n_seeds=3, n_epochs=300, lr=1e-2, weight_decay=1e-4,
                      patience=30, val_months=12,
                      purge_months=1, embargo_months=1,
                      oos_start_year=2016, oos_end_year=2024,
                      device='cpu', verbose=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df_train_all = df[df['date'] < pd.Timestamp(f"{oos_start_year}-01-01")]
    avg_mcap = (df_train_all.groupby('permno')['mcap'].mean()
                .sort_values(ascending=False))
    top_permnos = avg_mcap.head(N_max).index.tolist()
    permno_map = {p: i for i, p in enumerate(top_permnos)}
    N_total = len(permno_map)
    if verbose:
        print(f"Permno map: N_total = {N_total}, "
              f"cutoff = {avg_mcap.iloc[N_total-1]:.2f}")

    X_mgd, Z_list, y_list, permno_idx_list, tau, dates_all = build_tvaec_panel(
        df, permno_map, N_max, char_cols)
    T_all = len(dates_all)
    P = len(char_cols)
    if verbose:
        print(f"Panel: T = {T_all}, P = {P}")

    oos_preds = []
    oos_factors = []
    oos_betas = []

    for year in range(oos_start_year, oos_end_year + 1):
        if verbose:
            print(f"\n=== Year {year} ===")
        t0 = time.time()

        test_start = pd.Timestamp(f"{year}-01-01")
        test_end   = pd.Timestamp(f"{year}-12-31")
        val_end    = test_start - pd.DateOffset(months=embargo_months + 1)
        val_start  = val_end - pd.DateOffset(months=val_months - 1)
        train_end  = val_start - pd.DateOffset(months=purge_months + 1)

        train_pidx = [i for i, d in enumerate(dates_all)
                      if d <= train_end and permno_idx_list[i].shape[0] > 0]
        val_pidx   = [i for i, d in enumerate(dates_all)
                      if val_start <= d <= val_end and permno_idx_list[i].shape[0] > 0]
        test_pidx  = [i for i, d in enumerate(dates_all)
                      if test_start <= d <= test_end and permno_idx_list[i].shape[0] > 0]
        hist_pidx  = [i for i, d in enumerate(dates_all) if d <= test_end]

        if not train_pidx or not val_pidx or not test_pidx:
            if verbose:
                print(f"  Skipping year {year}: empty split")
            continue

        train_df = df[df['date'] <= train_end]
        mu_arr = np.zeros(N_max, dtype=np.float32)
        for slot, p in enumerate(top_permnos):
            sub = train_df[train_df['permno'] == p]
            if len(sub) > 0:
                mu_arr[slot] = sub['ret'].mean()

        def make_loader(pidx, shuffle):
            Xs = X_mgd[pidx]
            Zs = [Z_list[i] for i in pidx]
            taus = tau[pidx]
            ys = [y_list[i] - mu_arr[permno_idx_list[i]] for i in pidx]
            ds = TVAECPanel(Xs, Zs, taus, ys)
            return DataLoader(ds, batch_size=1, shuffle=shuffle,
                              collate_fn=_single_collate)

        train_loader = make_loader(train_pidx, True)
        val_loader   = make_loader(val_pidx, False)

        ensemble_pred = [None] * len(test_pidx)
        ensemble_fit  = [None] * len(test_pidx)
        ensemble_F = np.zeros((len(test_pidx), K1), dtype=np.float64)
        ensemble_B = [None] * len(test_pidx)
        n_valid = 0

        for seed in range(n_seeds):
            if verbose:
                print(f"  Training seed {seed+1}/{n_seeds}...")
            model = TVAECApp1(P, K1, encoder_width, beta_width)
            model, best_val = _train_one_tvaec(
                model, train_loader, val_loader,
                n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay, patience=patience,
                device=device, seed=seed, verbose=False)
            if verbose:
                print(f"    best val = {best_val:.6f}")

            model.eval()
            F_hist = np.zeros((T_all, K1), dtype=np.float64)
            with torch.no_grad():
                for i in hist_pidx:
                    x_t = torch.tensor(X_mgd[i], dtype=torch.float32).to(device)
                    tau_t = torch.tensor(tau[i], dtype=torch.float32).to(device)
                    F_hist[i] = model.encode(x_t, tau_t).cpu().numpy()

            F_cummean = (np.cumsum(F_hist, axis=0)
                         / np.arange(1, T_all + 1)[:, None])

            with torch.no_grad():
                for k, i in enumerate(test_pidx):
                    Z_t = torch.tensor(Z_list[i], dtype=torch.float32).to(device)
                    tau_t = torch.tensor(tau[i], dtype=torch.float32).to(device)
                    x_t = torch.tensor(X_mgd[i], dtype=torch.float32).to(device)

                    beta = model.beta(Z_t, tau_t)
                    F_t = model.encode(x_t, tau_t)

                    if i > 0:
                        lambda_t = torch.tensor(F_cummean[i - 1],
                                                dtype=torch.float32).to(device)
                    else:
                        lambda_t = torch.zeros(K1, dtype=torch.float32).to(device)

                    r_pred = (beta @ lambda_t).cpu().numpy().astype(np.float64)
                    r_fit  = (beta @ F_t).cpu().numpy().astype(np.float64)
                    beta_np = beta.cpu().numpy().astype(np.float64)
                    F_np = F_t.cpu().numpy().astype(np.float64)

                    if ensemble_B[k] is None:
                        ensemble_pred[k] = r_pred
                        ensemble_fit[k]  = r_fit
                        ensemble_B[k]    = beta_np
                        ensemble_F[k]    = F_np
                    else:
                        ensemble_pred[k] += r_pred
                        ensemble_fit[k]  += r_fit
                        ensemble_B[k]    += beta_np
                        ensemble_F[k]    += F_np

            n_valid += 1

        if n_valid == 0:
            continue

        ensemble_F /= n_valid
        for k in range(len(test_pidx)):
            if ensemble_B[k] is not None:
                ensemble_pred[k] /= n_valid
                ensemble_fit[k]  /= n_valid
                ensemble_B[k]    /= n_valid

        for k, i in enumerate(test_pidx):
            d = dates_all[i]
            idx = permno_idx_list[i]
            slot_permnos = [top_permnos[j] for j in idx]
            mu_t = mu_arr[idx]
            pred_t = ensemble_pred[k] + mu_t
            fit_t  = ensemble_fit[k] + mu_t
            F_t    = ensemble_F[k]
            beta_t = ensemble_B[k]

            for j, p in enumerate(slot_permnos):
                oos_preds.append((d, p, float(pred_t[j]), float(fit_t[j])))
            oos_factors.append((d, *[float(F_t[c]) for c in range(K1)]))
            for j, p in enumerate(slot_permnos):
                oos_betas.append((d, p,
                                  *[float(beta_t[j, c]) for c in range(K1)]))

        if verbose:
            print(f"  done in {(time.time()-t0)/60:.2f} min")

    pred_df = pd.DataFrame(
        oos_preds, columns=['date', 'permno', 'pred_ret', 'fitted_ret'])
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