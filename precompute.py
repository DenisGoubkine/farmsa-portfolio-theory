"""
precompute.py — run once before deploying.
Saves all expensive computation results to data/precomputed.pkl.
Re-run whenever data or estimators change.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from estimators import ESTIMATORS

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "data" / "precomputed.pkl"


def min_var(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x if res.success else np.ones(n) / n


def rolling_backtest(returns: pd.DataFrame, estimator_key: str,
                     lookback: int = 252, rebal: int = 21) -> tuple:
    T, N = returns.shape
    est_fn = ESTIMATORS[estimator_key]
    ports: dict[str, list] = {"Equal Weight": [], "Sample Cov MVO": [], "Estimator MVO": []}
    idx_list: list = []
    for start in range(lookback, T - rebal, rebal):
        train = returns.iloc[start - lookback : start]
        test  = returns.iloc[start : start + rebal]
        w_eq  = np.ones(N) / N
        w_sc  = min_var(train.cov().values)
        w_est = min_var(est_fn(train))
        for di in range(len(test)):
            r = test.iloc[di].values
            ports["Equal Weight"].append(w_eq @ r)
            ports["Sample Cov MVO"].append(w_sc @ r)
            ports["Estimator MVO"].append(w_est @ r)
            idx_list.append(test.index[di])
    ret_df = pd.DataFrame(ports, index=idx_list)
    pv_df  = (1 + ret_df).cumprod()
    return ret_df, pv_df


def m1_full_diag(returns: pd.DataFrame) -> dict:
    N  = returns.shape[1]
    S  = returns.cov().values
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    rho_bar = float((corr.sum() - N) / (N * (N - 1)))
    F = rho_bar * np.outer(std, std)
    np.fill_diagonal(F, np.diag(S))
    lw_fit = LedoitWolf().fit(returns.values)
    return {
        "alpha":   float(lw_fit.shrinkage_),
        "rho_bar": rho_bar,
        "S": S, "F": F,
        "LW": lw_fit.covariance_,
        "N": N,
    }


def m1_rolling_diag(returns: pd.DataFrame, roll: int = 252, step: int = 5) -> dict:
    T = returns.shape[0]
    dates, alphas, rho_bars, cond_s, cond_lw = [], [], [], [], []
    for i in range(roll, T, step):
        w   = returns.iloc[i - roll : i]
        S_w = w.cov().values
        Nw  = S_w.shape[0]
        std_w = np.sqrt(np.diag(S_w))
        c_w   = S_w / np.outer(std_w, std_w)
        np.fill_diagonal(c_w, 1.0)
        lw = LedoitWolf().fit(w.values)
        dates.append(returns.index[i])
        alphas.append(float(lw.shrinkage_))
        rho_bars.append(float((c_w.sum() - Nw) / (Nw * (Nw - 1))))
        cond_s.append(float(np.linalg.cond(S_w)))
        cond_lw.append(float(np.linalg.cond(lw.covariance_)))
    return {
        "dates":    pd.DatetimeIndex(dates),
        "alphas":   np.array(alphas),
        "rho_bars": np.array(rho_bars),
        "cond_s":   np.array(cond_s),
        "cond_lw":  np.array(cond_lw),
    }


def rolling_backtest_m3(returns: pd.DataFrame, ff_factors: pd.DataFrame,
                        lookback: int = 252, rebal: int = 21) -> tuple:
    """Rolling backtest for M3 — needs FF factors for CAPM/FF3 estimators."""
    from estimators import fama_french, sample_cov

    T, N = returns.shape
    strategies = ["Equal Weight", "Sample Cov MVO", "CAPM Factor MVO", "FF3 Factor MVO"]

    def capm_cov(returns_df):
        dates = returns_df.index.intersection(ff_factors.index)
        excess = returns_df.loc[dates].subtract(ff_factors.loc[dates, "RF"], axis=0)
        mkt = ff_factors.loc[dates, "Mkt-RF"].values
        X = np.column_stack([np.ones(len(dates)), mkt])
        n = excess.shape[1]
        betas, resid_var = np.zeros(n), np.zeros(n)
        for i, col in enumerate(excess.columns):
            coefs = np.linalg.lstsq(X, excess[col].values, rcond=None)[0]
            betas[i] = coefs[1]
            resid_var[i] = np.var(excess[col].values - X @ coefs, ddof=1)
        return np.outer(betas, betas) * np.var(mkt, ddof=1) + np.diag(resid_var)

    ports: dict[str, list] = {k: [] for k in strategies}
    idx_list: list = []
    for start in range(lookback, T - rebal, rebal):
        train = returns.iloc[start - lookback : start]
        test  = returns.iloc[start : start + rebal]
        w_eq   = np.ones(N) / N
        w_sc   = min_var(sample_cov(train))
        w_capm = min_var(capm_cov(train))
        w_ff3  = min_var(fama_french(train))
        for di in range(len(test)):
            r = test.iloc[di].values
            ports["Equal Weight"].append(w_eq @ r)
            ports["Sample Cov MVO"].append(w_sc @ r)
            ports["CAPM Factor MVO"].append(w_capm @ r)
            ports["FF3 Factor MVO"].append(w_ff3 @ r)
            idx_list.append(test.index[di])
    ret_df = pd.DataFrame(ports, index=idx_list)
    pv_df  = (1 + ret_df).cumprod()
    return ret_df, pv_df


def main() -> None:
    print("Loading data...")
    returns    = pd.read_csv(ROOT / "data" / "returns.csv", index_col=0, parse_dates=True)
    ff_factors = pd.read_csv(ROOT / "data" / "ff_factors.csv", index_col=0, parse_dates=True)

    cache: dict = {}

    print("M1: full-sample diagnostics...")
    cache["m1_full_diag"] = m1_full_diag(returns)

    print("M1: rolling diagnostics (this takes ~30s)...")
    cache["m1_rolling_diag"] = m1_rolling_diag(returns)

    print("M1: rolling backtest (LW)...")
    cache["m1_backtest"] = rolling_backtest(returns, "Ledoit-Wolf Shrinkage")

    print("M2: rolling backtest (RMT)...")
    cache["m2_backtest"] = rolling_backtest(returns, "RMT Eigenvalue Cleaning")

    print("M3: rolling backtest (CAPM + FF3)...")
    cache["m3_backtest"] = rolling_backtest_m3(returns, ff_factors)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Saved to {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
