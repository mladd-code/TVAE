# Time-Varying Nonlinear Factor Models via Deep Autoencoders

Reference implementation, simulation study, and empirical applications for the
working paper:

    Time-Varying Nonlinear Factor Models via Deep Autoencoders
    Matthew Ladd, University of Toronto, 2026

Working paper: paper/ladd_tvae_2026.pdf
Slides:        paper/ladd_tvae_slides_2026.pdf

---

## Overview

Standard factor models assume the cross-section of observables is a linear
function of a small number of latent factors, with time-invariant loadings.
Both assumptions fail in equity returns, portfolio construction, and
macro-financial forecasting: loading functions are nonlinear in the factors,
and the nonlinearity shifts with the macro-financial regime.

This repository accompanies a paper that develops a theory and estimator for
the natural specification

    X_it = phi*_i(F*_t, t/T) + U_it

where phi*_i(., .) is an unknown smooth loading function of a K-dimensional
latent factor F*_t and the scaled time index t/T. The estimator is a deep
autoencoder: a neural network trained to reconstruct its own input through a
low-dimensional bottleneck, conditioned on time.

### Headline result

The global time-conditioned autoencoder (TV-AE-G) attains the rate

    T^(-2*beta/(2*beta+K+1)) + N^(-1)*K1 + epsilon_N

which strictly dominates the local kernel analogue of Su and Wang (2017),
whose rate is T^(-2*beta/(2*beta+K+2)), and reduces to the Shen and Xiu (2026)
static rate T^(-2*beta/(2*beta+K)) when the loading function is constant in t.
A characteristic-conditioned variant (TV-AE-C) extends the framework to
observable heterogeneity, at a rate of T^(-2*beta/(2*beta+K+P+1)) where P is
the characteristic dimension.

### Methods at a glance

    static_pca    Linear, time-invariant              Bai (2003) benchmark
    lpca          Linear, time-varying                Su and Wang (2017) local PCA
    static_ae     Nonlinear, time-invariant           Shen and Xiu (2026) static AE
    tvae_local    Nonlinear, time-varying             TV-AE-L (theoretical comparison)
    tvae_global   Nonlinear, time-varying             TV-AE-G (this paper, main estimator)
    tvaec         Nonlinear, time- and char-cond.     TV-AE-C (extension)

---

## Install

Requires Python 3.10 or newer and PyTorch 2.0 or newer.

    git clone https://github.com/mladd-code/TVAE.git
    cd TVAE
    python -m venv .venv
    source .venv/bin/activate          # macOS / Linux
    # source .venv/Scripts/activate    # Windows Git Bash
    pip install --upgrade pip
    pip install -e .

For CPU-only PyTorch (optional):

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Verify the install:

    python -c "from tvae import DGPConfig, generate_dgp, fit_tvae_global; print('ok')"

---

## Quick start

Fit TV-AE-G on a synthetic exponential-logistic DGP and compare against PCA:

    from tvae import (
        DGPConfig, generate_dgp, pca_baseline, evaluate, fit_tvae_global,
    )

    cfg = DGPConfig(
        N=100, T=200, K=2,
        nonlinearity="exponential",
        time_variation="logistic",
        error="iid", seed=42,
    )
    sim = generate_dgp(cfg)
    X, X_star = sim["X"], sim["X_star"]

    # Classical benchmark
    pca = pca_baseline(X, K=2)
    mse_pca = evaluate(pca["X_hat"], X_star)["mse_common"]

    # This paper's estimator
    res = fit_tvae_global(
        X, K1=3, n_seeds=3, n_epochs=500,
        encoder_width=64, decoder_width=32, weight_decay=1e-3,
    )
    mse_tvae = evaluate(res["X_hat"], X_star)["mse_common"]

    print(f"PCA mse:   {mse_pca:.4f}")
    print(f"TV-AE-G:   {mse_tvae:.4f}")

---

## Reproduce the paper

All results in the working paper are reproducible from the scripts below.
Simulation outputs go to results/; logs are written to results/logs/.

### Simulations

    cd simulations

    # Headline Monte Carlo (Table 3): six-DGP grid, 30 reps, tuned TV-AE-G
    python montecarlo_final.py

    # Capacity sweep for hyperparameter selection (Table 2)
    python capacity_sweep_stage2.py

    # Sample-size scaling study (Figure 2)
    python scaling_v2.py

    # Training-curve diagnostics
    python diagnostics_v2.py

Runtime for the full set is roughly 6 to 10 hours on a single CUDA GPU; the
individual scripts print an ETA at each step.

### Empirical applications

These require the pre-built panels in data/processed/ (see Data below).

    cd applications

    # S&P 100 out-of-sample reconstruction (Table 4, Panel A)
    python sp100_returns.py

    # S&P 100 realized-volatility panel (Table 4, Panel B)
    python sp100_vol.py

    # Crypto returns (Table 4, Panel C) -- the headline 3.7pp gain over PCA
    python crypto_returns.py

    # Bottleneck-width sensitivity (Figure 3)
    python crypto_k1_sweep.py

    # Paired bootstrap over refit windows (Table 5)
    python bootstrap_crypto_returns.py

    # Time-input sensitivity for tau (Section 5.5)
    python crypto_tau.py

### Validating the LPCA port

The Python local-PCA port was checked against the original R code of
Fan, Wu, Yang and Zhong (2024). To reproduce:

    cd simulations
    python crossval_lpca.py generate    # writes crossval_data.csv + crossval_bw.txt
    # run crossval_lpca.R (in R) in the same folder
    python crossval_compare.py          # reports agreement to 1e-6

---

## Repository structure

    tvae/
    |-- src/tvae/                # importable library
    |   |-- dgp.py               # synthetic DGPs and evaluation metrics
    |   |-- utils.py             # logging, path helpers
    |   |-- models/
    |       |-- static_ae.py     # Shen and Xiu (2026) static autoencoder
    |       |-- tvae_global.py   # TV-AE-G (this paper)
    |       |-- tvae_local.py    # TV-AE-L (theoretical comparison)
    |       |-- lpca.py          # Su and Wang (2017) local PCA (Python port)
    |
    |-- simulations/             # synthetic Monte Carlo studies
    |   |-- montecarlo_final.py
    |   |-- capacity_sweep_stage1.py
    |   |-- capacity_sweep_stage2.py
    |   |-- scaling_v2.py
    |   |-- diagnostics_v2.py
    |   |-- crossval_lpca.py
    |   |-- crossval_compare.py
    |
    |-- applications/            # empirical reconstruction and MVP
    |   |-- sp100_returns.py
    |   |-- sp100_vol.py
    |   |-- crypto_returns.py
    |   |-- crypto_vol.py
    |   |-- crypto_k1_sweep.py
    |   |-- crypto_tau.py
    |   |-- bootstrap_crypto_returns.py
    |   |-- bootstrap_crypto_vol.py
    |   |-- build_sp100_vol.py
    |   |-- build_crypto_vol.py
    |   |-- build_commodity_rv.py
    |   |-- mvp.py
    |
    |-- paper/                   # working paper and slides (PDF)
    |-- data/                    # panel schema; raw data not distributed
    |-- results/                 # generated outputs (gitignored)
    |
    |-- future_work/gkx_app/     # GKX-style characteristic-conditioned app
    |-- archive/                 # one-off data pulls and diagnostics

The future_work/ directory houses an independent extension to the
Gu-Kelly-Xiu (2021) conditional-autoencoder framework with managed-portfolio
predictors. It is not part of the working paper and is provided for reference
and follow-up work.

The archive/ directory contains historical Refinitiv data-pull scripts and
development diagnostics. They are included as documentation of the exact
universe filters and date ranges used in the paper, and are not required to
reproduce any published result.

---

## Data

The paper uses four panels. All are constructed from Refinitiv Tick History,
which is licensed to the author and cannot be redistributed.

    Panel                          Coverage               File
    S&P 100 constituents           2005-2024, daily       panel_daily.parquet
    S&P 100 realized volatility    2005-2024, daily       panel_daily_vol.parquet
    50 cryptocurrency pairs        2015-2024, daily       panel_daily_crypto.parquet
    Crypto realized volatility     2015-2024, daily       panel_daily_crypto_vol.parquet

data/README.md documents the schema (date, permno, ret, mcap) and the exact
filters (market cap threshold, return cleaning, ticker mapping) so the panels
can be rebuilt with independent Refinitiv access. The synthetic DGPs in
src/tvae/dgp.py are fully self-contained and require no external data.

---

## Reproducibility notes

- Synthetic simulations are seeded (DGPConfig.seed) and fully deterministic
  up to GPU nondeterminism; setting torch.use_deterministic_algorithms(True)
  yields bit-identical results on CPU.

- Encoder/decoder ensembles average over n_seeds independent
  initializations. The paper uses n_seeds=3; scripts default to the same.

- Train/val/test splits in empirical applications use purged, vintage-aware
  splits with a 1-month embargo between estimation and evaluation windows,
  following standard practice in the time-series forecasting literature.

- The LPCA port was validated against the R reference of Fan, Wu, Yang and
  Zhong (2024); agreement to 1e-6 on the common component and factor subspace
  is enforced by simulations/crossval_compare.py. A single documented
  deviation from the R code (boundary-kernel argument order) is noted in the
  module docstring of src/tvae/models/lpca.py.

---

## Citation

If you use this code or the working paper in your research, please cite:

    @article{ladd2026tvae,
      title  = {Time-Varying Nonlinear Factor Models via Deep Autoencoders},
      author = {Ladd, Matthew},
      year   = {2026},
      note   = {Working paper, University of Toronto},
    }

Please update the entry with a journal or arXiv identifier once available.

---

## Acknowledgements

The local-PCA implementation is a Python port of the R code accompanying
Fan, Wu, Yang and Zhong (2024), "Time-varying minimum variance portfolio,"
Journal of Econometrics 239(2), 105339. The static autoencoder follows the
disjoint-output architecture of Shen and Xiu (2026). Any errors in the port
or the extensions are the author's.

---

## License

Code released under the MIT License (see LICENSE). The working paper and
slides are copyright Matthew Ladd, all rights reserved.

---

## Contact

Matthew Ladd
matthew.19.ladd@gmail.com
University of Toronto
