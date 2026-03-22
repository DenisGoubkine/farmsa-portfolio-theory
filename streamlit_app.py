from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm

from estimators import ESTIMATORS


st.set_page_config(
    page_title="FARMSA Portfolio Theory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

MODULES = {
    "Preface": {
        "code": "00",
        "title": "Preface",
        "subtitle": "Project setup, notation, universe construction, and the overall research question.",
        "notebook": ROOT / "00_preface.ipynb",
    },
    "Summary": {
        "code": "00",
        "title": "Comparative Summary",
        "subtitle": "How covariance estimation changes portfolio outcomes under one shared backtest.",
        "notebook": ROOT / "00_summary.ipynb",
    },
    "Ledoit-Wolf Shrinkage": {
        "code": "M1",
        "title": "Ledoit-Wolf Shrinkage",
        "subtitle": "Shrink the noisy sample covariance toward a more stable target.",
        "notebook": ROOT / "M1_ledoit_wolf.ipynb",
    },
    "RMT Eigenvalue Cleaning": {
        "code": "M2",
        "title": "RMT Eigenvalue Cleaning",
        "subtitle": "Use a random-matrix noise boundary to clean weak eigenmodes.",
        "notebook": ROOT / "M2_rmt_cleaning.ipynb",
    },
    "Fama-French 3-Factor Covariance": {
        "code": "M3",
        "title": "Fama-French 3-Factor Covariance",
        "subtitle": "Build covariance structure from market, size, and value factors.",
        "notebook": ROOT / "M3_factor_models.ipynb",
    },
    "DCC-GARCH Dynamic Covariance": {
        "code": "M4",
        "title": "DCC-GARCH Dynamic Covariance",
        "subtitle": "Model covariance as a dynamic object that evolves with volatility regimes.",
        "notebook": ROOT / "M4_dcc_garch.ipynb",
    },
    "MPT vs PMPT": {
        "code": "M5",
        "title": "MPT vs PMPT",
        "subtitle": "Compare variance-based and downside-based portfolio construction.",
        "notebook": ROOT / "M5_mpt_vs_pmpt.ipynb",
    },
    "Portfolio Optimizer": {
        "code": "M6",
        "title": "Portfolio Optimizer",
        "subtitle": "Notebook logic presented as a native base-settings run plus the full source cells.",
        "notebook": ROOT / "M6_portfolio_optimizer.ipynb",
    },
}

NOTEBOOK_TO_MODULE = {
    "00_preface.ipynb": "Preface",
    "00_summary.ipynb": "Summary",
    "M1_ledoit_wolf.ipynb": "Ledoit-Wolf Shrinkage",
    "M2_rmt_cleaning.ipynb": "RMT Eigenvalue Cleaning",
    "M3_factor_models.ipynb": "Fama-French 3-Factor Covariance",
    "M4_dcc_garch.ipynb": "DCC-GARCH Dynamic Covariance",
    "M5_mpt_vs_pmpt.ipynb": "MPT vs PMPT",
    "M6_portfolio_optimizer.ipynb": "Portfolio Optimizer",
}


def inject_styles() -> None:
    st.markdown(
        """
<style>
  :root {
    --ink: #11233f;
    --muted: #62758d;
    --line: #d9e2ec;
    --panel: rgba(255, 255, 255, 0.92);
    --wash: linear-gradient(180deg, #f7fafc 0%, #edf4f9 100%);
    --accent: #1f4b73;
    --accent-soft: #0f766e;
    --warn: #d97706;
  }
  .stApp {
    color: var(--ink);
    font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    background:
      radial-gradient(circle at top left, rgba(31, 75, 115, 0.09), transparent 28%),
      radial-gradient(circle at top right, rgba(15, 118, 110, 0.06), transparent 22%),
      #f4f7fb;
  }
  html, body, [class*="css"]  {
    color: var(--ink);
    font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
  }
  [data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, #172235 0%, #1f2937 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
  }
  [data-testid="stSidebar"] * {
    color: #eef4ff !important;
  }
  [data-testid="stSidebar"] a {
    color: #93c5fd !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #eef4ff !important;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .st-bq,
  [data-testid="stSidebar"] .st-br,
  [data-testid="stSidebar"] .st-bs,
  [data-testid="stSidebar"] .st-bt {
    color: #eef4ff !important;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label {
    color: #eef4ff !important;
  }
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    color: #eef4ff !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
  }
  .block-container {
    max-width: 1480px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
  }
  .hero {
    background: var(--wash);
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 1.35rem 1.45rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 48px rgba(17, 35, 63, 0.06);
  }
  .hero-kicker {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.78rem;
    font-weight: 850;
    margin-bottom: 0.35rem;
  }
  .hero-title {
    color: var(--ink);
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.02;
    font-weight: 850;
    margin: 0;
  }
  .hero-subtitle {
    color: var(--muted);
    margin-top: 0.65rem;
    max-width: 70rem;
    font-size: 1rem;
  }
  .surface {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.05rem;
  }
  .section-title {
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 850;
    margin: 0.35rem 0 0.45rem;
  }
  .note-kicker {
    color: var(--warn);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.73rem;
    font-weight: 850;
    margin-bottom: 0.35rem;
  }
  .code-header {
    color: var(--accent);
    font-size: 0.8rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
  }
  .output-header {
    color: var(--accent-soft);
    font-size: 0.76rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.6rem 0 0.35rem;
  }
  .metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    min-height: 104px;
  }
  .metric-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 850;
  }
  .metric-value {
    color: var(--ink);
    font-size: 1.85rem;
    font-weight: 850;
    margin-top: 0.15rem;
  }
  .metric-note {
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 0.25rem;
  }
  div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border-radius: 18px;
    box-shadow: 0 10px 24px rgba(17, 35, 63, 0.04);
  }
  div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stMarkdownContainer"] li,
  div[data-testid="stMarkdownContainer"] td,
  div[data-testid="stMarkdownContainer"] th,
  div[data-testid="stMarkdownContainer"] blockquote,
  div[data-testid="stMarkdownContainer"] strong,
  div[data-testid="stMarkdownContainer"] em,
  div[data-testid="stMarkdownContainer"] h1,
  div[data-testid="stMarkdownContainer"] h2,
  div[data-testid="stMarkdownContainer"] h3,
  div[data-testid="stMarkdownContainer"] h4,
  div[data-testid="stMarkdownContainer"] h5,
  div[data-testid="stMarkdownContainer"] h6 {
    color: var(--ink) !important;
  }
  div[data-testid="stMarkdownContainer"] table {
    background: rgba(255, 255, 255, 0.72);
    color: var(--ink) !important;
    border-collapse: collapse;
  }
  div[data-testid="stMarkdownContainer"] th,
  div[data-testid="stMarkdownContainer"] td {
    border-bottom: 1px solid var(--line);
    padding: 0.45rem 0.6rem;
  }
  div[data-testid="stMarkdownContainer"] a {
    color: #2563eb !important;
    text-decoration: underline !important;
    font-weight: 700;
  }
  .stApp a {
    color: #2563eb !important;
    text-decoration: underline !important;
  }
  .katex, .katex * {
    color: var(--ink) !important;
    background: transparent !important;
  }
  .katex-display {
    color: var(--ink) !important;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.25rem 0;
  }
  pre {
    white-space: pre-wrap !important;
    word-break: break-word;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero(module_key: str) -> None:
    module = MODULES[module_key]
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-kicker">FARMSA Portfolio Theory | {module["code"]}</div>
  <div class="hero-title">{module["title"]}</div>
  <div class="hero-subtitle">{module["subtitle"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}</div>
  <div class="metric-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def rewrite_notebook_links(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        target = match.group(2).strip()
        module_key = NOTEBOOK_TO_MODULE.get(Path(target).name)
        if not module_key:
            return match.group(0)
        return f"[{label}](?module={quote(module_key)})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+\.ipynb)\)", replace, source)


@st.cache_data(show_spinner=False)
def load_notebook(module_key: str) -> dict:
    return json.loads(MODULES[module_key]["notebook"].read_text())


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(ROOT / "data" / "returns.csv", index_col=0, parse_dates=True)
    metadata = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("ticker")
    return prices, returns, metadata


def estimate_mu(returns_df: pd.DataFrame) -> np.ndarray:
    return returns_df.mean().values * 252


def regularize_covariance(cov: np.ndarray) -> np.ndarray:
    cov = cov.copy()
    minimum = np.min(np.linalg.eigvalsh(cov))
    if minimum < 1e-10:
        cov += np.eye(cov.shape[0]) * (1e-10 - minimum)
    return cov


def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    metadata: pd.DataFrame,
    risk_tolerance: float,
    max_weight: float,
    sector_limit: float,
    selected_tickers: list[str],
) -> dict[str, object] | None:
    if len(selected_tickers) < 2:
        return None

    idx_map = [tickers.index(ticker) for ticker in selected_tickers]
    mu_sub = mu[idx_map]
    cov_sub = regularize_covariance(cov[np.ix_(idx_map, idx_map)])
    n_assets = len(idx_map)
    w0 = np.ones(n_assets) / n_assets
    bounds = [(0.0, min(max_weight, 1.0))] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    sectors = metadata["sector"].fillna("Other")
    if sector_limit < 1.0:
        for sector_name in sorted(sectors.unique()):
            sector_idx = [j for j, ticker in enumerate(selected_tickers) if sectors.get(ticker, "Other") == sector_name]
            if sector_idx:
                constraints.append(
                    {"type": "ineq", "fun": lambda w, idx=sector_idx, cap=sector_limit: cap - np.sum(w[idx])}
                )

    rf_annual = 0.04
    if risk_tolerance < 0.01:
        objective = lambda w: w @ cov_sub @ w
    else:
        delta = 10.0 * (1.0 - risk_tolerance) + 0.5 * risk_tolerance
        objective = lambda w: -(w @ mu_sub - 0.5 * delta * (w @ cov_sub @ w))

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    weights_sub = result.x if result.success else w0
    weights_sub = np.maximum(weights_sub, 0)
    weights_sub /= weights_sub.sum()

    full_weights = np.zeros(len(tickers))
    for j, idx in enumerate(idx_map):
        full_weights[idx] = weights_sub[j]

    expected_return = float(weights_sub @ mu_sub)
    expected_vol = float(np.sqrt(weights_sub @ cov_sub @ weights_sub * 252))
    sharpe = float((expected_return - rf_annual) / expected_vol) if expected_vol > 0 else 0.0

    return {
        "weights": full_weights,
        "tickers": tickers,
        "expected_return": expected_return,
        "expected_vol": expected_vol,
        "sharpe": sharpe,
        "success": bool(result.success),
    }


def compute_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    selected_tickers: list[str],
    all_tickers: list[str],
    metadata: pd.DataFrame,
    max_weight: float,
    sector_limit: float,
    n_points: int = 22,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    idx = [all_tickers.index(ticker) for ticker in selected_tickers]
    mu_sub = mu[idx]
    cov_sub = regularize_covariance(cov[np.ix_(idx, idx)])
    n_assets = len(idx)
    w0 = np.ones(n_assets) / n_assets
    bounds = [(0.0, min(max_weight, 1.0))] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    sectors = metadata["sector"].fillna("Other")
    if sector_limit < 1.0:
        for sector_name in sorted(sectors.unique()):
            sector_idx = [j for j, ticker in enumerate(selected_tickers) if sectors.get(ticker, "Other") == sector_name]
            if sector_idx:
                constraints.append(
                    {"type": "ineq", "fun": lambda w, idx=sector_idx, cap=sector_limit: cap - np.sum(w[idx])}
                )

    minimum = minimize(
        lambda w: w @ cov_sub @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    w_min = minimum.x if minimum.success else w0
    ret_min = float(w_min @ mu_sub)
    ret_max = float(min(np.max(mu_sub), np.percentile(mu_sub, 90)))

    frontier_returns: list[float] = []
    frontier_vols: list[float] = []
    for target in np.linspace(ret_min, ret_max, n_points):
        result = minimize(
            lambda w: w @ cov_sub @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints + [{"type": "eq", "fun": lambda w, t=target: w @ mu_sub - t}],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if result.success:
            frontier_returns.append(float(target))
            frontier_vols.append(float(np.sqrt(result.x @ cov_sub @ result.x * 252)))

    if not frontier_returns:
        return None, None
    return np.array(frontier_returns), np.array(frontier_vols)


def render_m6_dynamic_tool() -> None:
    _, returns, metadata = load_data()
    tickers = list(returns.columns)
    sectors = sorted(metadata["sector"].fillna("Other").unique())

    st.markdown('<div class="section-title">Interactive Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="surface">'
            '<strong>This is the native Streamlit version of Module 6.</strong> '
            'Pick an estimator, adjust the constraints, choose a universe, and run the optimizer live.'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        estimator_name = st.selectbox("Covariance estimator", list(ESTIMATORS.keys()), index=0)
        lookback = st.slider("Lookback window", 60, min(504, len(returns)), 252, step=21)
        risk_tolerance = st.slider("Risk tolerance", 0.0, 1.0, 0.0, step=0.05)
        max_weight = st.slider("Max position weight", 0.02, 1.0, 1.0, step=0.01)
        sector_limit = st.slider("Max sector weight", 0.10, 1.0, 1.0, step=0.05)
        show_frontier = st.checkbox("Show efficient frontier", value=True)

    available_tickers = tickers
    with right:
        selected_sectors = st.multiselect("Sectors", sectors, default=sectors)
        if selected_sectors:
            available_tickers = [ticker for ticker in tickers if metadata.loc[ticker, "sector"] in selected_sectors]
        default_tickers = [ticker for ticker in st.session_state.get("m6_selected_tickers", available_tickers) if ticker in available_tickers]
        if not default_tickers:
            default_tickers = available_tickers

        select_a, select_b = st.columns(2)
        if select_a.button("Select all", use_container_width=True):
            st.session_state["m6_selected_tickers"] = available_tickers
        if select_b.button("Clear all", use_container_width=True):
            st.session_state["m6_selected_tickers"] = []

        selected_tickers = st.multiselect(
            "Stocks",
            available_tickers,
            default=default_tickers,
            key="m6_selected_tickers",
            format_func=lambda ticker: f"{ticker} | {metadata.loc[ticker, 'name']}",
        )

    run = st.button("Run Optimizer", type="primary", use_container_width=True)
    if not run:
        return

    if len(selected_tickers) < 2:
        st.error("Select at least two stocks.")
        return
    if max_weight * len(selected_tickers) < 1.0:
        st.error("The max weight is too low for the number of selected stocks.")
        return

    window = returns.iloc[-lookback:]
    estimator = ESTIMATORS[estimator_name]
    try:
        cov = estimator(window)
    except Exception as exc:
        st.warning(f"{estimator_name} failed, using sample covariance instead: {exc}")
        cov = window.cov().values

    mu = estimate_mu(window)
    result = optimize_portfolio(
        mu=mu,
        cov=cov,
        tickers=tickers,
        metadata=metadata,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
        sector_limit=sector_limit,
        selected_tickers=selected_tickers,
    )
    if result is None:
        st.error("Optimization failed.")
        return

    frontier_ret, frontier_vol = None, None
    if show_frontier:
        frontier_ret, frontier_vol = compute_efficient_frontier(
            mu=mu,
            cov=cov,
            selected_tickers=selected_tickers,
            all_tickers=tickers,
            metadata=metadata,
            max_weight=max_weight,
            sector_limit=sector_limit,
        )

    stats = st.columns(4)
    with stats[0]:
        render_metric("Expected Return", f"{result['expected_return'] * 100:.2f}%", "Annualized mean return")
    with stats[1]:
        render_metric("Expected Volatility", f"{result['expected_vol'] * 100:.2f}%", "Annualized portfolio risk")
    with stats[2]:
        render_metric("Sharpe", f"{result['sharpe']:.2f}", "Excess return per unit of risk")
    with stats[3]:
        render_metric("Positions", f"{int(np.sum(result['weights'] > 0.001))}", "Non-trivial allocations")

    weight_frame = pd.DataFrame(
        {
            "Ticker": tickers,
            "Company": [metadata.loc[ticker, "name"] for ticker in tickers],
            "Sector": [metadata.loc[ticker, "sector"] for ticker in tickers],
            "Weight": result["weights"],
        }
    )
    weight_frame = weight_frame[weight_frame["Weight"] > 0.001].sort_values("Weight", ascending=False)
    weight_frame["Weight (%)"] = (weight_frame["Weight"] * 100).round(2)

    left, right = st.columns([1.02, 0.98], gap="large")
    with left:
        fig, ax = plt.subplots(figsize=(9, 5))
        top = weight_frame.head(20).iloc[::-1]
        ax.barh(top["Ticker"], top["Weight (%)"], color="#1f4b73")
        ax.set_title("Top Holdings", fontweight="bold")
        ax.set_xlabel("Weight (%)")
        ax.grid(True, axis="x", alpha=0.18)
        st.pyplot(fig, clear_figure=True)
    with right:
        sector_weights = weight_frame.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            sector_weights.values,
            labels=sector_weights.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.set_title("Sector Allocation", fontweight="bold")
        st.pyplot(fig, clear_figure=True)

    left, right = st.columns([0.98, 1.02], gap="large")
    with left:
        st.markdown('<div class="section-title">Full Weights Table</div>', unsafe_allow_html=True)
        st.dataframe(weight_frame[["Ticker", "Company", "Sector", "Weight (%)"]], width="stretch", hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(8, 5))
        if frontier_ret is not None and frontier_vol is not None and len(frontier_ret) > 1:
            ax.plot(frontier_vol * 100, frontier_ret * 100, lw=2.2, color="#0f766e", label="Efficient frontier")
        ax.scatter(
            result["expected_vol"] * 100,
            result["expected_return"] * 100,
            s=100,
            color="#b91c1c",
            edgecolors="white",
            lw=1.5,
            label="Selected portfolio",
        )
        ax.set_xlabel("Volatility (%)")
        ax.set_ylabel("Return (%)")
        ax.set_title("Risk-Return Trade-off", fontweight="bold")
        ax.grid(True, alpha=0.18)
        ax.legend()
        st.pyplot(fig, clear_figure=True)


def output_html_height(html_text: str) -> int:
    row_count = html_text.count("<tr")
    if row_count:
        return max(220, min(720, 120 + row_count * 28))
    return 420


def clean_stream_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "FigureCanvasAgg is non-interactive" in line:
            continue
        if "ipykernel_" in line and "UserWarning:" in line:
            continue
        if "ClusterWarning:" in line:
            continue
        if "of 50 completed" in line:
            continue
        if line in {"plt.show()", "[                       0%                       ]"}:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ── M1 / M2 live-render helpers — load from precomputed.pkl ──────────────

@st.cache_resource(show_spinner=False)
def _precomputed() -> dict:
    import pickle
    path = ROOT / "data" / "precomputed.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _precomputed_safe() -> dict | None:
    try:
        return _precomputed()
    except Exception:
        return None


def _m1_full_diag() -> dict | None:
    p = _precomputed_safe()
    return p["m1_full_diag"] if p else None


def _m1_rolling_diag():
    p = _precomputed_safe()
    if not p:
        return None, None, None, None, None
    d = p["m1_rolling_diag"]
    return d["dates"], d["alphas"], d["rho_bars"], d["cond_s"], d["cond_lw"]


def _rolling_backtest(estimator_key: str, **_):
    p = _precomputed_safe()
    if not p:
        return None, None
    key = "m1_backtest" if estimator_key == "Ledoit-Wolf Shrinkage" else "m2_backtest"
    return p[key]


def _drawdown(pv: pd.Series) -> pd.Series:
    return (pv / pv.cummax()) - 1


def _backtest_chart_and_table(
    ret_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    for k in pv_df.columns:
        axes[0].plot(pv_df.index, pv_df[k], label=labels.get(k, k), color=colors[k], lw=1.8)
    axes[0].set_ylabel("Portfolio Value ($1 start)")
    axes[0].set_title("Rolling Out-of-Sample Backtest  (252-day window, monthly rebalance)", fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].grid(True, alpha=0.2)

    for k in pv_df.columns:
        dd = _drawdown(pv_df[k]) * 100
        axes[1].fill_between(pv_df.index, dd, 0, alpha=0.35, color=colors[k])
        axes[1].plot(pv_df.index, dd, color=colors[k], lw=1.2)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    rows = {}
    for k, lbl in labels.items():
        r = ret_df[k]
        ar = r.mean() * 252
        av = r.std() * np.sqrt(252)
        maxdd = _drawdown((1 + r).cumprod()).min()
        rows[lbl] = {
            "Ann. Return (%)": round(ar * 100, 2),
            "Ann. Volatility (%)": round(av * 100, 2),
            "Sharpe Ratio": round(ar / av if av > 0 else 0, 3),
            "Max Drawdown (%)": round(maxdd * 100, 2),
            "Calmar Ratio": round(ar / abs(maxdd) if maxdd < 0 else float("nan"), 3),
        }
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)


_M1_COLORS = {"Equal Weight": "#888888", "Sample Cov MVO": "#2c5282", "Estimator MVO": "#c53030"}
_M1_LABELS = {"Equal Weight": "Equal Weight (1/N)", "Sample Cov MVO": "Sample Covariance MVO", "Estimator MVO": "LW Estimator MVO"}
_M2_COLORS = {"Equal Weight": "#888888", "Sample Cov MVO": "#2c5282", "Estimator MVO": "#c53030"}
_M2_LABELS = {"Equal Weight": "Equal Weight (1/N)", "Sample Cov MVO": "Sample Covariance MVO", "Estimator MVO": "RMT Cleaned MVO"}


def render_m1_live_output(cell_index: int) -> bool:
    if cell_index == 4:
        # Eigenvalue spectrum + heatmaps + shrinkage dial
        d = _m1_full_diag()
        S, F, LW, alpha, rho_bar, N = d["S"], d["F"], d["LW"], d["alpha"], d["rho_bar"], d["N"]

        def to_corr(cov):
            s = np.sqrt(np.diag(cov))
            c = cov / np.outer(s, s)
            np.fill_diagonal(c, 1.0)
            return c

        eig_S = np.sort(np.linalg.eigvalsh(S))[::-1]
        eig_LW = np.sort(np.linalg.eigvalsh(LW))[::-1]

        fig = plt.figure(figsize=(14, 9))
        gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_eig = fig.add_subplot(gs_fig[0, :2])
        k = np.arange(1, N + 1)
        ax_eig.semilogy(k, eig_S,  "o-", color="#2c5282", lw=1.8, ms=5, label="Sample Cov")
        ax_eig.semilogy(k, eig_LW, "s-", color="#c53030", lw=1.8, ms=5, label="LW Shrinkage")
        ax_eig.annotate(
            f"Condition S = {np.linalg.cond(S):.0f}\nCondition LW = {np.linalg.cond(LW):.0f}",
            xy=(N * 0.6, eig_S[-1] * 3), fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff9f0", ec="#c53030", alpha=0.9),
        )
        ax_eig.set_xlabel("Eigenvalue rank")
        ax_eig.set_ylabel("Eigenvalue (log scale)")
        ax_eig.set_title("Eigenvalue Spectrum: Sample vs Ledoit-Wolf", fontweight="bold")
        ax_eig.legend(); ax_eig.grid(True, alpha=0.2)

        ax_dial = fig.add_subplot(gs_fig[0, 2])
        ax_dial.axis("off")
        bar_x = np.linspace(0, 1, 300)
        for x, c in zip(bar_x, plt.cm.RdYlBu_r(bar_x)):
            ax_dial.barh(0.55, 1 / 300, left=x, height=0.12, color=c, linewidth=0)
        ax_dial.annotate("", xy=(alpha, 0.35), xytext=(alpha, 0.53),
                         arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
        ax_dial.text(alpha, 0.28, f"α* = {alpha:.3f}", ha="center", fontsize=13, fontweight="bold", color="#0f1f3d")
        ax_dial.text(0.0, 0.73, "Pure Sample\n(α = 0)", ha="left", fontsize=9, color="#2c5282")
        ax_dial.text(1.0, 0.73, "Pure Target\n(α = 1)", ha="right", fontsize=9, color="#c53030")
        ax_dial.text(0.5, 0.88, "Shrinkage Intensity Dial", ha="center", fontsize=11, fontweight="bold", color="#0f1f3d")
        ax_dial.set_xlim(-0.05, 1.05); ax_dial.set_ylim(0, 1.1)

        for col_idx, (mat, title) in enumerate([
            (to_corr(S),  "Sample Correlation"),
            (to_corr(F),  f"Target F  (ρ̄ = {rho_bar:.3f})"),
            (to_corr(LW), f"LW Shrinkage  (α = {alpha:.3f})"),
        ]):
            ax = fig.add_subplot(gs_fig[1, col_idx])
            im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.1, vmax=1.0, aspect="auto")
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.code(f"Eigenvalue ratio  S: {eig_S[0]/eig_S[-1]:.1f}   LW: {eig_LW[0]/eig_LW[-1]:.1f}", language="text")
        return True

    if cell_index == 5:
        # Rolling diagnostics
        dates, alphas, rho_bars, cond_s, cond_lw = _m1_rolling_diag()
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("M1 — Rolling Diagnostics (252-day window)", fontsize=13, fontweight="bold", color="#0f1f3d")

        axes[0].fill_between(dates, alphas, alpha=0.25, color="#c53030")
        axes[0].plot(dates, alphas, color="#c53030", lw=1.6)
        axes[0].set_ylabel("Shrinkage Intensity α*")
        axes[0].set_title("Optimal Shrinkage Intensity", fontweight="bold")
        axes[0].grid(True, alpha=0.2)

        axes[1].fill_between(dates, rho_bars, alpha=0.25, color="#2c5282")
        axes[1].plot(dates, rho_bars, color="#2c5282", lw=1.6)
        axes[1].set_ylabel("Mean Pairwise Correlation ρ̄")
        axes[1].set_title("Average Pairwise Correlation", fontweight="bold")
        axes[1].grid(True, alpha=0.2)

        axes[2].semilogy(dates, cond_s,  color="#2c5282", lw=1.6, label="Sample Cov")
        axes[2].semilogy(dates, cond_lw, color="#c53030", lw=1.6, label="LW Shrinkage")
        axes[2].set_ylabel("Condition Number (log)")
        axes[2].set_title("Numerical Stability", fontweight="bold")
        axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.code(
            f"α* range: [{alphas.min():.3f}, {alphas.max():.3f}]   "
            f"ρ̄ range: [{rho_bars.min():.3f}, {rho_bars.max():.3f}]",
            language="text",
        )
        return True

    if cell_index == 7:
        ret_df, pv_df = _rolling_backtest("Ledoit-Wolf Shrinkage")
        if ret_df is None:
            st.warning("Precomputed data not found — run `python precompute.py` and redeploy.")
            return True
        _backtest_chart_and_table(ret_df, pv_df, _M1_LABELS, _M1_COLORS)
        return True

    return False


def render_m2_live_output(cell_index: int) -> bool:
    if cell_index == 9:
        ret_df, pv_df = _rolling_backtest("RMT Eigenvalue Cleaning")
        if ret_df is None:
            st.warning("Precomputed data not found — run `python precompute.py` and redeploy.")
            return True
        _backtest_chart_and_table(ret_df, pv_df, _M2_LABELS, _M2_COLORS)
        return True
    return False


_M3_COLORS = {
    "Equal Weight":    "#888888",
    "Sample Cov MVO":  "#2c5282",
    "CAPM Factor MVO": "#d97706",
    "FF3 Factor MVO":  "#c53030",
}
_M3_LABELS = {
    "Equal Weight":    "Equal Weight (1/N)",
    "Sample Cov MVO":  "Sample Covariance MVO",
    "CAPM Factor MVO": "CAPM Factor MVO",
    "FF3 Factor MVO":  "FF3 Factor MVO",
}


def render_m3_live_output(cell_index: int) -> bool:
    if cell_index == 9:
        p = _precomputed_safe()
        if p is None or "m3_backtest" not in p:
            st.warning("Precomputed data not found — run `python precompute.py` and redeploy.")
            return True
        ret_df, pv_df = p["m3_backtest"]
        _backtest_chart_and_table(ret_df, pv_df, _M3_LABELS, _M3_COLORS)
        return True
    return False


def render_preface_live_output(cell_index: int) -> bool:
    prices, returns, _ = load_data()

    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "NVDA": "Tech", "ADBE": "Tech", "CRM": "Tech", "INTC": "Tech", "CSCO": "Tech",
        "JPM": "Fin", "BAC": "Fin", "GS": "Fin", "MS": "Fin", "BLK": "Fin", "AXP": "Fin",
        "JNJ": "Health", "UNH": "Health", "PFE": "Health", "ABBV": "Health", "LLY": "Health", "MRK": "Health",
        "AMZN": "Discr", "HD": "Discr", "MCD": "Discr", "NKE": "Discr", "SBUX": "Discr",
        "PG": "Staples", "KO": "Staples", "PEP": "Staples", "COST": "Staples",
        "CAT": "Indust", "HON": "Indust", "UPS": "Indust", "BA": "Indust",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        "META": "Comm", "DIS": "Comm", "NFLX": "Comm",
        "NEE": "Util", "DUK": "Util", "SO": "Util",
        "AMT": "RE", "PLD": "RE", "CCI": "RE",
        "LIN": "Mat", "APD": "Mat", "SHW": "Mat", "ECL": "Mat",
    }
    sector_colors = {
        "Tech": "#2563eb", "Fin": "#059669", "Health": "#dc2626", "Discr": "#d97706",
        "Staples": "#7c3aed", "Indust": "#6b7280", "Energy": "#a16207", "Comm": "#db2777",
        "Util": "#0891b2", "RE": "#4f46e5", "Mat": "#65a30d",
    }

    if cell_index == 6:
        normalized = prices / prices.iloc[0] * 100
        fig, ax = plt.subplots(figsize=(14, 5.5))
        legend_done: set[str] = set()
        for ticker in normalized.columns:
            sector = sector_map.get(ticker, "Other")
            label = sector if sector not in legend_done else None
            legend_done.add(sector)
            ax.plot(
                normalized.index,
                normalized[ticker],
                color=sector_colors.get(sector, "#999999"),
                alpha=0.55,
                linewidth=0.9,
                label=label,
            )
        ax.axhline(100, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
        ax.set_ylabel("Normalized Price (start = 100)", fontsize=10)
        ax.set_title("Price Trajectories — 50-Stock Universe (colored by GICS sector)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.85, title="Sector", title_fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(normalized.index[0], normalized.index[-1])
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        return True

    if cell_index == 7:
        corr_matrix = returns.corr()
        distance = 1 - corr_matrix.values
        np.fill_diagonal(distance, 0)
        link = linkage(squareform(distance, checks=False), method="ward")
        order = leaves_list(link)
        corr_sorted = corr_matrix.iloc[order, order]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_sorted.values, cmap="RdBu_r", vmin=-0.3, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_sorted.columns)))
        ax.set_xticklabels(corr_sorted.columns, fontsize=6, rotation=90)
        ax.set_yticks(range(len(corr_sorted.index)))
        ax.set_yticklabels(corr_sorted.index, fontsize=6)
        ax.set_title("Pairwise Return Correlations (hierarchically clustered)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        upper = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
        st.code(
            f"Correlation stats  ->  mean: {upper.mean():.3f}   min: {upper.min():.3f}   max: {upper.max():.3f}",
            language="text",
        )
        return True

    if cell_index == 8:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        all_returns = returns.values.flatten()
        axes[0].hist(all_returns, bins=150, density=True, alpha=0.65, color="#2563eb", edgecolor="white", linewidth=0.3)
        x_grid = np.linspace(-0.15, 0.15, 300)
        axes[0].plot(x_grid, norm.pdf(x_grid, all_returns.mean(), all_returns.std()), color="#dc2626", linewidth=1.5, label="Normal fit")
        axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Daily Return")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Pooled Return Distribution vs Normal", fontweight="bold")
        axes[0].set_xlim(-0.12, 0.12)
        axes[0].legend(fontsize=8)

        ann_ret = returns.mean() * 252 * 100
        ann_vol = returns.std() * np.sqrt(252) * 100
        for sector in sorted(set(sector_map.values())):
            tickers = [ticker for ticker in returns.columns if sector_map.get(ticker) == sector]
            axes[1].scatter(
                ann_vol[tickers],
                ann_ret[tickers],
                color=sector_colors[sector],
                s=40,
                alpha=0.8,
                label=sector,
                edgecolors="white",
                linewidth=0.4,
            )
            for ticker in tickers:
                axes[1].annotate(ticker, (ann_vol[ticker], ann_ret[ticker]), fontsize=5.5, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        axes[1].set_xlabel("Annualized Volatility (%)")
        axes[1].set_ylabel("Annualized Return (%)")
        axes[1].set_title("Risk-Return Scatter (by sector)", fontweight="bold")
        axes[1].legend(fontsize=6.5, ncol=3, loc="upper left", framealpha=0.8)
        axes[1].grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        return True

    if cell_index == 11:
        cov_matrix = returns.cov().values
        eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
        n_assets = returns.shape[1]
        t_obs = returns.shape[0]
        q = n_assets / t_obs
        sigma2 = np.mean(np.diag(cov_matrix))
        lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        colors = ["#dc2626" if ev > lambda_plus else "#94a3b8" for ev in eigenvalues]
        axes[0].bar(range(1, n_assets + 1), eigenvalues, color=colors, alpha=0.8, width=0.8)
        axes[0].axhline(lambda_plus, color="#dc2626", linestyle="--", linewidth=1.2, label="Noise ceiling (Marchenko-Pastur)")
        axes[0].set_xlabel("Component (ranked by size)")
        axes[0].set_ylabel("Eigenvalue (importance)")
        axes[0].set_title("Signal vs Noise in the Covariance Matrix", fontweight="bold")
        axes[0].legend(fontsize=8)

        cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100
        axes[1].plot(range(1, n_assets + 1), cum_var, "o-", color="#2563eb", ms=4, linewidth=1.5)
        axes[1].axhline(90, color="#dc2626", linestyle="--", linewidth=1, alpha=0.6, label="90 % threshold")
        axes[1].fill_between(range(1, n_assets + 1), cum_var, alpha=0.08, color="#2563eb")
        axes[1].set_xlabel("Number of components kept")
        axes[1].set_ylabel("Cumulative variance explained (%)")
        axes[1].set_title("How Many Components Matter?", fontweight="bold")
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(0, 105)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        n_signal = int(np.sum(eigenvalues > lambda_plus))
        n_90 = int(np.argmax(cum_var >= 90) + 1)
        st.code(
            f"Only {n_signal} of {n_assets} components carry real signal — the other {n_assets - n_signal} are noise.\n"
            f"Just {n_90} components explain 90% of the variance ({cum_var[n_90-1]:.1f}%).",
            language="text",
        )
        return True

    return False


def render_output(output: dict, module_key: str) -> None:
    output_type = output.get("output_type")

    if output_type == "stream":
        text = clean_stream_text("".join(output.get("text", [])))
        if text.strip():
            st.code(text, language="text")
        return

    if output_type == "error":
        traceback = "\n".join(output.get("traceback", []))
        message = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        st.error(message)
        if traceback:
            st.code(traceback, language="text")
        return

    data = output.get("data", {})

    if "image/png" in data:
        st.image(base64.b64decode(data["image/png"]), width="stretch")

    if "text/html" in data:
        html_text = data["text/html"]
        if isinstance(html_text, list):
            html_text = "".join(html_text)
        components.html(str(html_text), height=output_html_height(str(html_text)), scrolling=True)
        return

    if "application/vnd.jupyter.widget-view+json" in data:
        if module_key != "Portfolio Optimizer":
            st.info("This notebook output is a Jupyter widget and cannot render natively in Streamlit.")
        return

    if "text/plain" in data:
        plain = data["text/plain"]
        if isinstance(plain, list):
            plain = "".join(plain)
        if str(plain).strip():
            st.code(str(plain), language="text")


def render_notebook_cells(module_key: str, show_code: bool) -> None:
    notebook = load_notebook(module_key)
    st.markdown('<div class="section-title">Notebook Cells</div>', unsafe_allow_html=True)

    code_index = 0
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            if cell_index == 1 and (source.lstrip().startswith("<div") or "module-banner" in source):
                continue
            source = rewrite_notebook_links(source)
            with st.container(border=True):
                st.markdown('<div class="note-kicker">Markdown Cell</div>', unsafe_allow_html=True)
                st.markdown(source, unsafe_allow_html=True)
            continue

        if cell_type != "code":
            continue

        code_index += 1
        with st.container(border=True):
            st.markdown(f'<div class="code-header">Code Cell {code_index}</div>', unsafe_allow_html=True)
            if show_code:
                st.code(source, language="python")
            else:
                st.caption("Code hidden. Enable `Show code cells` in the sidebar to display the full source.")

            outputs = cell.get("outputs", [])

            # M1 and M2 live rendering (fires even when no saved outputs)
            if module_key == "Ledoit-Wolf Shrinkage" and cell_index in (4, 5, 7):
                st.markdown('<div class="output-header">Cell Output</div>', unsafe_allow_html=True)
                if render_m1_live_output(cell_index):
                    stream_lines = [
                        clean_stream_text("".join(o.get("text", [])))
                        for o in outputs if o.get("output_type") == "stream"
                    ]
                    stream_lines = [s for s in stream_lines if s]
                    if stream_lines:
                        st.code("\n".join(stream_lines), language="text")
                    continue
            if module_key == "Fama-French 3-Factor Covariance" and cell_index == 9:
                st.markdown('<div class="output-header">Cell Output</div>', unsafe_allow_html=True)
                if render_m3_live_output(cell_index):
                    stream_lines = [
                        clean_stream_text("".join(o.get("text", [])))
                        for o in outputs if o.get("output_type") == "stream"
                    ]
                    stream_lines = [s for s in stream_lines if s]
                    if stream_lines:
                        st.code("\n".join(stream_lines), language="text")
                    continue
            if module_key == "RMT Eigenvalue Cleaning" and cell_index == 9:
                st.markdown('<div class="output-header">Cell Output</div>', unsafe_allow_html=True)
                if render_m2_live_output(cell_index):
                    stream_lines = [
                        clean_stream_text("".join(o.get("text", [])))
                        for o in outputs if o.get("output_type") == "stream"
                    ]
                    stream_lines = [s for s in stream_lines if s]
                    if stream_lines:
                        st.code("\n".join(stream_lines), language="text")
                    continue

            if outputs:
                st.markdown('<div class="output-header">Cell Output</div>', unsafe_allow_html=True)
                if module_key == "Portfolio Optimizer" and cell_index == 7:
                    st.info("The ipywidget control panel from the notebook is replaced by the live Streamlit optimizer above.")
                    continue
                if module_key == "Preface" and render_preface_live_output(cell_index):
                    meaningful_streams = []
                    for output in outputs:
                        if output.get("output_type") == "stream":
                            cleaned = clean_stream_text("".join(output.get("text", [])))
                            if cleaned:
                                meaningful_streams.append(cleaned)
                        elif output.get("output_type") != "stream":
                            render_output(output, module_key)
                    if meaningful_streams:
                        st.code("\n".join(meaningful_streams), language="text")
                else:
                    for output in outputs:
                        render_output(output, module_key)


def main() -> None:
    inject_styles()

    st.sidebar.title("FARMSA Navigation")
    requested_module = st.query_params.get("module")
    module_keys = list(MODULES.keys())
    if requested_module not in module_keys:
        requested_module = module_keys[0]
    module_key = st.sidebar.radio("Open module", module_keys, index=module_keys.index(requested_module))
    st.query_params["module"] = module_key
    show_code = st.sidebar.toggle("Show code cells", value=True)

    render_hero(module_key)

    if module_key == "Portfolio Optimizer":
        render_m6_dynamic_tool()

    render_notebook_cells(module_key, show_code=show_code)


if __name__ == "__main__":
    main()
