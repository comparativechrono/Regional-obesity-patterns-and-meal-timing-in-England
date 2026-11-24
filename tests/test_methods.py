"""Unit checks of the statistical helpers in src/common.py against independent references."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from common import CUTPOINTS, conley_se, holm, knn_weights, manual_cr1, manual_hc1, morans_i, ols

rng = np.random.default_rng(1)


def _data(n=120):
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 1 + 0.5 * X.a - 0.3 * X.b + rng.normal(size=n) * (1 + np.abs(X.a))
    return y, X


def test_hc1_matches_statsmodels():
    y, X = _data()
    fit = ols(y, X, "HC1")
    b, se = manual_hc1(y, X)
    assert np.allclose(b, fit.params.values) and np.allclose(se, fit.bse.values)


def test_holm_matches_statsmodels():
    p = rng.uniform(0, 0.2, size=8)
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])


def test_conley_with_tiny_cutoff_equals_hc1():
    y, X = _data()
    lat = 50 + rng.uniform(0, 5, len(y))
    lon = -5 + rng.uniform(0, 6, len(y))
    _, se = conley_se(y.values, X, lat, lon, cutoff_km=1e-6)
    assert np.allclose(se.values, ols(y, X, "HC1").bse.values)


def test_morans_i_detects_spatial_clustering():
    lat = np.repeat(np.linspace(50, 55, 12), 12)
    lon = np.tile(np.linspace(-5, 1, 12), 12)
    W = knn_weights(lat, lon, k=8)
    I_trend, p_trend = morans_i(lat + rng.normal(scale=0.1, size=lat.size), W, permutations=499)
    I_noise, p_noise = morans_i(rng.normal(size=lat.size), W, permutations=499)
    assert I_trend > 0.5 and p_trend < 0.01
    assert abs(I_noise) < 0.15 and p_noise > 0.01


def test_uk90_cutpoints():
    assert CUTPOINTS["underweight"] == pytest.approx(-2.054, abs=1e-3)
    assert CUTPOINTS["overweight"] == pytest.approx(1.036, abs=1e-3)
    assert CUTPOINTS["obese"] == pytest.approx(1.645, abs=1e-3)
    assert CUTPOINTS["severe"] == pytest.approx(2.652, abs=1e-3)


def _panel(n_groups=60, years=3):
    """Stacked panel: each group observed in several years at a fixed location, with a shared
    group effect (so errors are correlated within group)."""
    g = np.repeat(np.arange(n_groups), years)
    lat = np.repeat(50 + rng.uniform(0, 5, n_groups), years)
    lon = np.repeat(-5 + rng.uniform(0, 6, n_groups), years)
    X = pd.DataFrame({"x": np.repeat(rng.normal(size=n_groups), years), "t": np.tile(np.arange(years), n_groups)})
    y = 0.4 * X.x + np.repeat(rng.normal(size=n_groups), years) + rng.normal(size=len(g))
    return y.values, X, g, lat, lon


def test_cluster_robust_matches_statsmodels():
    y, X, g, *_ = _panel()
    Xc = sm.add_constant(X)
    fit = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": g})
    b, se = manual_cr1(y, Xc.values, g)
    assert np.allclose(b, fit.params.values) and np.allclose(se, fit.bse.values)


def test_conley_on_stacked_data_reduces_to_clustering_by_location():
    """With a tiny cutoff, only observations at the same location (the same authority in other
    years) are correlated, which is clustering by authority up to the small-sample factor."""
    y, X, g, lat, lon = _panel()
    _, se_c = conley_se(y, X, lat, lon, cutoff_km=1e-6)
    _, se_cr1 = manual_cr1(y, sm.add_constant(X).values, g)
    n, G = len(y), len(np.unique(g))
    assert np.allclose(se_c.values * np.sqrt(G / (G - 1) * (n - 1) / n), se_cr1)
