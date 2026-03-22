"""
Shared covariance estimators used by the optimizer and Streamlit app.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
_ff_cache = {}


def sample_cov(returns_df, **kwargs):
    return returns_df.cov().values


def ledoit_wolf(returns_df, **kwargs):
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(returns_df.values).covariance_


def rmt_clean(returns_df, **kwargs):
    t_obs, n_assets = returns_df.shape
    sample = returns_df.cov().values
    q = n_assets / t_obs

    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    sigma2 = np.mean(eigenvalues)
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    noise_mask = eigenvalues <= lambda_plus

    cleaned = eigenvalues.copy()
    if noise_mask.any():
        cleaned[noise_mask] = eigenvalues[noise_mask].mean()

    cov = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    return 0.5 * (cov + cov.T)


def _load_ff_factors():
    if "df" in _ff_cache:
        return _ff_cache["df"]

    ff_path = ROOT / "data" / "ff_factors.csv"
    ff = pd.read_csv(ff_path)
    date_col = "date" if "date" in ff.columns else ff.columns[0]
    ff[date_col] = pd.to_datetime(ff[date_col])
    ff = ff.rename(columns={date_col: "date"}).set_index("date")
    _ff_cache["df"] = ff
    return ff


def fama_french(returns_df, **kwargs):
    ff_df = _load_ff_factors()
    common = returns_df.index.intersection(ff_df.index)

    if len(common) < 60:
        return returns_df.cov().values

    returns_aligned = returns_df.loc[common]
    factors = ff_df.loc[common]
    excess_returns = returns_aligned.subtract(factors["RF"], axis=0)
    factor_matrix = factors[["Mkt-RF", "SMB", "HML"]].values
    design = np.column_stack([np.ones(len(common)), factor_matrix])

    betas = np.zeros((returns_aligned.shape[1], 3))
    resid_var = np.zeros(returns_aligned.shape[1])

    for idx, column in enumerate(excess_returns.columns):
        coeffs = np.linalg.lstsq(design, excess_returns[column].values, rcond=None)[0]
        betas[idx] = coeffs[1:]
        residuals = excess_returns[column].values - design @ coeffs
        resid_var[idx] = np.var(residuals, ddof=1)

    factor_cov = np.cov(factor_matrix, rowvar=False)
    return betas @ factor_cov @ betas.T + np.diag(resid_var)


def dcc_garch(returns_df, **kwargs):
    x = returns_df.values
    x = x - x.mean(axis=0, keepdims=True)
    t_obs, n_assets = x.shape
    eps = 1e-8

    alpha_garch = 0.05
    beta_garch = 0.90
    h_var = np.zeros((t_obs, n_assets))

    for i in range(n_assets):
        series = x[:, i]
        variance = max(np.var(series), eps)
        h_var[0, i] = variance
        omega = max((1 - alpha_garch - beta_garch) * np.var(series), eps)

        for t in range(1, t_obs):
            variance = omega + alpha_garch * (series[t - 1] ** 2) + beta_garch * variance
            h_var[t, i] = max(variance, eps)

    h_next = np.zeros(n_assets)
    for i in range(n_assets):
        series = x[:, i]
        h_last = h_var[-1, i]
        variance0 = np.var(series)
        omega = max((1 - alpha_garch - beta_garch) * variance0, eps)
        h_next[i] = max(omega + alpha_garch * (series[-1] ** 2) + beta_garch * h_last, eps)

    sigma_t = np.sqrt(h_next)
    d_t = np.diag(sigma_t)
    z = x / np.sqrt(h_var)

    q_bar = np.cov(z.T) + eps * np.eye(n_assets)
    a_dcc = 0.02
    b_dcc = 0.97
    q_t = q_bar.copy()

    for t in range(1, t_obs):
        z_prev = z[t - 1].reshape(-1, 1)
        q_t = (1 - a_dcc - b_dcc) * q_bar + a_dcc * (z_prev @ z_prev.T) + b_dcc * q_t

    z_last = z[-1].reshape(-1, 1)
    q_t = (1 - a_dcc - b_dcc) * q_bar + a_dcc * (z_last @ z_last.T) + b_dcc * q_t

    diag_q = np.sqrt(np.maximum(np.diag(q_t), eps))
    d_inv = np.diag(1.0 / diag_q)
    corr = d_inv @ q_t @ d_inv
    corr = 0.5 * (corr + corr.T)

    cov = d_t @ corr @ d_t
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


ESTIMATORS = {
    "Sample Covariance": sample_cov,
    "Ledoit-Wolf Shrinkage": ledoit_wolf,
    "RMT Eigenvalue Cleaning": rmt_clean,
    "Fama-French 3-Factor Covariance": fama_french,
    "DCC-GARCH Dynamic Covariance": dcc_garch,
}
