"""Shared paths, constants and helper functions for the analysis pipeline.

Every script in ``src/`` imports from here, so that file locations, indicator
definitions, the photoperiod model and the regression helpers are defined once.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root, so the pipeline runs anywhere)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT_DATA = OUT / "data"
OUT_TAB = OUT / "tables"
OUT_FIG = OUT / "figures"
RESULTS_JSON = OUT / "results.json"
for _p in (OUT_DATA, OUT_TAB, OUT_FIG):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
IMD_LOWER = RAW / "File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx"
IMD_UPPER = RAW / "File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx"
CENTROIDS = RAW / "Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BFE.csv"
NCMP_TABLES = RAW / "nati-chil-meas-prog-eng-2019-2020-tab.xlsx"
CENSUS_ETHNICITY = RAW / "census2021_RM033_dependent_children_ethnic_group_ctyua2023.csv"
BOUNDARIES = RAW / "Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BUC_England.geojson"

# Fingertips indicators, ordered left-to-right across the BMI distribution.
# key -> (file, cohort, threshold label, short label)
INDICATORS = {
    "rec_uw":  ("indicator90316.csv", "Reception", "Underweight", "Underweight"),
    "rec_owob": ("indicator20601.csv", "Reception", "Overweight incl. obesity (≥85th)", "OW+OB"),
    "rec_ob":  ("indicator90319.csv", "Reception", "Obesity incl. severe (≥95th)", "Obesity"),
    "rec_sev": ("indicator93194.csv", "Reception", "Severe obesity (≥99.6th)", "Severe obesity"),
    "yr6_uw":  ("indicator90320.csv", "Year 6", "Underweight", "Underweight"),
    "yr6_owob": ("indicator20602.csv", "Year 6", "Overweight incl. obesity (≥85th)", "OW+OB"),
    "yr6_ob":  ("indicator90323.csv", "Year 6", "Obesity incl. severe (≥95th)", "Obesity"),
    "yr6_sev": ("indicator93195.csv", "Year 6", "Severe obesity (≥99.6th)", "Severe obesity"),
}
FINGERTIPS_IDS = {k: v[0].replace("indicator", "").replace(".csv", "") for k, v in INDICATORS.items()}

PRIMARY_YEAR = "2023/24"
REPLICATION_YEAR = "2024/25"
SECOND_REPLICATION_YEAR = "2025/26"   # published November 2026; added as a further out-of-sample test
YEARS = [PRIMARY_YEAR, REPLICATION_YEAR, SECOND_REPLICATION_YEAR]
# 2023/24 and 2024/25 come from the original Fingertips exports in data/raw/; 2025/26 from the
# Fingertips API files in data/raw/fingertips_api/ (identical to the exports for the earlier years).
FINGERTIPS_DIR = {PRIMARY_YEAR: RAW, REPLICATION_YEAR: RAW, SECOND_REPLICATION_YEAR: RAW / "fingertips_api"}

# UK90 population-monitoring cut-points (NCMP definitions) as z-scores
CUTPOINTS = {
    "underweight": stats.norm.ppf(0.02),   # at or below 2nd centile
    "overweight": stats.norm.ppf(0.85),    # at or above 85th centile
    "obese": stats.norm.ppf(0.95),         # at or above 95th centile
    "severe": stats.norm.ppf(0.996),       # at or above 99.6th centile
}

# Academic-month order used throughout (NCMP collection runs September-July)
ACAD_MONTHS = ["September", "October", "November", "December", "January", "February",
               "March", "April", "May", "June", "July"]
MONTH_NUM = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}
ACAD_INDEX = {m: i for i, m in enumerate(ACAD_MONTHS, start=1)}  # September = 1

REFERENCE_LATITUDE = 52.5  # mid-England, used for the national monthly series


# ---------------------------------------------------------------------------
# Photoperiod: CBM model (Forsythe et al. 1995, Ecological Modelling 80:87-95)
# ---------------------------------------------------------------------------
def cbm_declination(day_of_year):
    """Solar declination (radians) from the CBM model.

    theta = 0.2163108 + 2*atan(0.9671396*tan(0.00860*(J - 186)))   [revolution angle]
    phi   = asin(0.39795*cos(theta))                                  [declination]
    """
    j = np.asarray(day_of_year, dtype=float)
    theta = 0.2163108 + 2.0 * np.arctan(0.9671396 * np.tan(0.00860 * (j - 186.0)))
    return np.arcsin(0.39795 * np.cos(theta))


def day_length(latitude_deg, day_of_year, p_deg: float = 0.8333):
    """Day length in hours (CBM model, Forsythe et al. 1995).

    D = 24 - (24/pi) * acos[(sin(p) + sin(L) sin(phi)) / (cos(L) cos(phi))]

    p = 0.8333 deg defines sunrise/sunset as the moment the top of the sun is at
    the apparent horizon (atmospheric refraction plus solar semi-diameter), i.e.
    the conventional sunrise-to-sunset day length.
    """
    lat = np.radians(np.asarray(latitude_deg, dtype=float))
    phi = cbm_declination(day_of_year)
    a = (np.sin(np.radians(p_deg)) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    a = np.clip(a, -1.0, 1.0)
    return 24.0 - (24.0 / np.pi) * np.arccos(a)


def photoperiod_amplitude(latitude_deg):
    """Annual range of day length (summer-solstice minus winter-solstice day length)."""
    doys = np.arange(1, 366)
    lat = np.atleast_1d(np.asarray(latitude_deg, dtype=float))
    dl = day_length(lat[:, None], doys[None, :])
    out = dl.max(axis=1) - dl.min(axis=1)
    return out if out.size > 1 else float(out[0])


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------
def ols(y, X, cov_type: str = "HC1", **cov_kwds):
    """OLS via statsmodels; X is a DataFrame (constant added here)."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(np.asarray(y, dtype=float), Xc).fit(cov_type=cov_type, cov_kwds=cov_kwds or None)


def manual_hc1(y, X):
    """Independent HC1 implementation used to cross-check statsmodels."""
    y = np.asarray(y, dtype=float)
    Xm = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    n, k = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    e = y - Xm @ beta
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    meat = (Xm * e[:, None] ** 2).T @ Xm * n / (n - k)
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    return beta, se


def partial_r(fit_classical, term: str) -> float:
    """Partial correlation of `term` from the classical t statistic."""
    t = fit_classical.tvalues[term]
    return float(t / math.sqrt(t ** 2 + fit_classical.df_resid))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def distance_matrix_km(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def conley_se(y, X, lat, lon, cutoff_km: float):
    """Spatial-HAC (Conley 1999) standard errors with a Bartlett kernel."""
    y = np.asarray(y, dtype=float)
    Xm = sm.add_constant(X, has_constant="add")
    names = list(Xm.columns)
    Xm = np.asarray(Xm, dtype=float)
    n, k = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    e = y - Xm @ beta
    d = distance_matrix_km(lat, lon)
    w = np.clip(1.0 - d / cutoff_km, 0.0, None)  # Bartlett weights, 1 on diagonal
    xe = Xm * e[:, None]
    meat = xe.T @ w @ xe
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    V = xtx_inv @ meat @ xtx_inv * n / (n - k)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return pd.Series(beta, index=names), pd.Series(se, index=names)


def knn_weights(lat, lon, k: int = 8):
    """Row-standardised k-nearest-neighbour spatial weights."""
    d = distance_matrix_km(lat, lon)
    np.fill_diagonal(d, np.inf)
    n = len(d)
    W = np.zeros((n, n))
    idx = np.argsort(d, axis=1)[:, :k]
    for i in range(n):
        W[i, idx[i]] = 1.0 / k
    return W


def morans_i(x, W, permutations: int = 9999, seed: int = 20261124):
    """Moran's I with a one-sided (positive autocorrelation) permutation p-value."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    n = len(z)
    s0 = W.sum()
    I = (n / s0) * (z @ W @ z) / (z @ z)
    rng = np.random.default_rng(seed)
    sims = np.empty(permutations)
    for b in range(permutations):
        zp = rng.permutation(z)
        sims[b] = (n / s0) * (zp @ W @ zp) / (zp @ zp)
    p = (1 + np.sum(sims >= I)) / (permutations + 1)
    return float(I), float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Results registry: every number quoted in the paper is written here
# ---------------------------------------------------------------------------
def save_results(section: str, values: dict):
    data = {}
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
    data[section] = values
    RESULTS_JSON.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default))


def load_results() -> dict:
    return json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
