"""
diag_factors.py
---------------
Check the time-series variance of the estimated factors and the ratio of
factor-driven prediction variance to per-stock-mean prediction variance.
"""
import pandas as pd
import numpy as np
import os

for name in ["TVAEG", "StaticAE"]:
    pred = pd.read_parquet(f"app1_oos_{name}.parquet")
    # Compute per-stock mean within the test window
    mu = pred.groupby('permno')['pred_ret'].transform('mean')
    factor_component = pred['pred_ret'] - mu
    print(f"\n=== {name} ===")
    print(f"  std(mu_i) across obs:             {mu.std():.6f}")
    print(f"  std(pred_ret - mu_i) across obs:  {factor_component.std():.6f}")
    print(f"  ratio (factor / mean):            "
          f"{factor_component.std() / mu.std():.4f}")
    print(f"  std(fitted_ret - mu_i):           "
          f"{(pred['fitted_ret'] - mu).std():.6f}")
    # Correlation of pred and fitted within the test sample
    print(f"  corr(pred_ret, fitted_ret):       "
          f"{pred[['pred_ret', 'fitted_ret']].corr().iloc[0, 1]:.6f}")
    # Yearly std of factor component
    pred['year'] = pred['date'].dt.year
    yearly = pred.groupby('year').apply(
        lambda g: (g['pred_ret'] - g.groupby('permno')['pred_ret']
                   .transform('mean')).std())
    print(f"  factor-component std by year:")
    print(yearly.to_string())