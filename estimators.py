"""
estimators.py — Covariance Estimators for M6 Portfolio Optimizer
================================================================
Each function takes a returns DataFrame (T×N) and returns an N×N
numpy covariance matrix (symmetric, positive semi-definite).

Status:
  ✓ sample_cov      — implemented
  ✓ ledoit_wolf     — implemented (sklearn)
  ✓ fama_french     — implemented (M4)
  ○ rmt_clean       — placeholder (returns sample cov)
  ○ dcc_garch       — placeholder (returns sample cov)
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. Sample Covariance
# ─────────────────────────────────────────────────────────────
def sample_cov(returns_df, **kwargs):
    """Textbook sample covariance. No shrinkage, no structure."""
    return returns_df.cov().values


# ─────────────────────────────────────────────────────────────
# 2. Ledoit-Wolf Shrinkage
# ─────────────────────────────────────────────────────────────
def ledoit_wolf(returns_df, **kwargs):
    """
    Ledoit-Wolf (2004) linear shrinkage toward scaled identity.
    Uses sklearn's analytical formula for optimal shrinkage intensity.
    """
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns_df.values)
    return lw.covariance_


# ─────────────────────────────────────────────────────────────
# 3. Random Matrix Theory — Eigenvalue Cleaning
# ─────────────────────────────────────────────────────────────
def rmt_clean(returns_df, **kwargs):
    """
    TODO: Replace with M2 implementation.

    Should:
      1. Compute eigenvalues/eigenvectors of sample cov.
      2. Identify noise eigenvalues via Marchenko-Pastur bounds.
      3. Replace noise eigenvalues (e.g., shrink to mean).
      4. Reconstruct cleaned covariance matrix.

    Currently falls back to sample covariance.
    """
    print("⚠ RMT estimator is a placeholder — using sample covariance.")
    return returns_df.cov().values


# ─────────────────────────────────────────────────────────────
# 4. Fama-French 3-Factor Model (from M4)
# ─────────────────────────────────────────────────────────────

# FF3 factor data — loaded once on import
_ff_cache = {}


def _load_ff_factors():
    """Download and cache FF3 daily factor data."""
    if "df" in _ff_cache:
        return _ff_cache["df"]

    import io, zipfile, urllib.request

    FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    resp = urllib.request.urlopen(FF_URL)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(csv_name).decode("utf-8")

    lines = raw.split("\n")
    start = next(
        i
        for i, l in enumerate(lines)
        if l.strip()[:1].isdigit() and len(l.split(",")[0].strip()) == 8
    )
    data_lines = []
    for l in lines[start:]:
        s = l.strip()
        if not s or not s[0].isdigit():
            break
        data_lines.append(s)

    ff = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["date", "Mkt-RF", "SMB", "HML", "RF"],
    )
    ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
    ff = ff.set_index("date") / 100
    _ff_cache["df"] = ff
    return ff


def fama_french(returns_df, **kwargs):
    """
    Fama-French 3-factor covariance estimator (M4).
    Σ = B Σ_f B' + D
    """
    ff_df = _load_ff_factors()
    common = returns_df.index.intersection(ff_df.index)

    if len(common) < 60:
        print("⚠ FF3: insufficient overlapping dates — falling back to sample cov.")
        return returns_df.cov().values

    R = returns_df.loc[common].values
    F = ff_df.loc[common][["Mkt-RF", "SMB", "HML"]].values
    rf = ff_df.loc[common]["RF"].values
    Re = R - rf[:, None]

    B = (np.linalg.inv(F.T @ F) @ F.T @ Re).T  # N×3
    E = Re - F @ B.T  # T×N residuals

    return B @ np.cov(F, rowvar=False) @ B.T + np.diag(np.var(E, axis=0, ddof=1))


# ─────────────────────────────────────────────────────────────
# 5. DCC-GARCH
# ─────────────────────────────────────────────────────────────
def dcc_garch(returns_df, **kwargs):
    """
    TODO: Replace with M3 implementation.

    Should:
      1. Fit univariate GARCH(1,1) to each asset.
      2. Standardize residuals.
      3. Estimate dynamic conditional correlations.
      4. Reconstruct time-varying covariance matrix.

    Currently falls back to sample covariance.
    """
    print("⚠ DCC-GARCH estimator is a placeholder — using sample covariance.")
    return returns_df.cov().values


# ─────────────────────────────────────────────────────────────
# Registry — used by M6 GUI
# ─────────────────────────────────────────────────────────────
ESTIMATORS = {
    "Sample Covariance": sample_cov,
    "Ledoit-Wolf Shrinkage": ledoit_wolf,
    "RMT Eigenvalue Cleaning": rmt_clean,
    "Fama-French 3-Factor": fama_french,
    "DCC-GARCH": dcc_garch,
}
