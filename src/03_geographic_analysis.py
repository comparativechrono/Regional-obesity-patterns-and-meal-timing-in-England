"""Step 3: local-authority analysis of latitude and NCMP prevalence, 2023/24-2025/26.

One pooled analysis of the three NCMP years (151 upper-tier local authorities x 3 years),
for each of 8 cohort x threshold indicators and two covariate sets:

    Model A  prevalence ~ year + IMD 2019 average score + latitude
    Model B  prevalence ~ year + IMD 2019 average score + child ethnic composition + latitude
             (Census 2021 % of dependent children who are Asian, Black, Mixed and Other;
              White is the reference group)

OLS on the stacked LA-year data with standard errors clustered by local authority (checked
against an independent implementation); Holm correction across the 8 indicators within each
model; spatial-HAC (Conley) standard errors at 100 and 200 km on the stacked data, so that
correlation within an authority across years and between nearby authorities are both allowed
for; Moran's I of authority-mean residuals.

Robustness: IDACI in place of IMD; London excluded; longitude added; weights = children
measured; region fixed effects; values suppressed for disclosure control set to 1, 4 or 7
children. Estimates for each year separately are reported as a consistency check.

Outputs: outputs/tables/geo_*.csv and results.json["geographic"]
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import (INDICATORS, OUT_DATA, OUT_TAB, YEARS, conley_se, holm, knn_weights, manual_cr1, morans_i,
                    partial_r, save_results)

ETHNICITY = ["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"]  # % of dependent children; White = reference
MODELS = {"A": [], "B": ETHNICITY}
MODEL_LABEL = {"A": "IMD + latitude", "B": "IMD + child ethnic composition + latitude"}


def load(year):
    return pd.read_csv(OUT_DATA / f"la_dataset_{year.replace('/', '_')}.csv").assign(year=year)


panel = pd.concat([load(y) for y in YEARS], ignore_index=True)
region_dummies = pd.get_dummies(panel["region"], prefix="reg", drop_first=True, dtype=float)
REGION_COLS = list(region_dummies.columns)
panel = pd.concat([panel, region_dummies], axis=1)
YEAR_COLS = []
for y in YEARS[1:]:
    col = "yr_" + y.replace("/", "_")
    panel[col] = (panel.year == y).astype(float)
    YEAR_COLS.append(col)


def design(d, covs, with_lat=True, years=True):
    cols = (YEAR_COLS if years else []) + covs + (["latitude"] if with_lat else [])
    return sm.add_constant(d[cols].astype(float), has_constant="add")


def fit(data, y, dep="imd_score", extra=None, weights=None, drop_london=False, years=True):
    """Pooled (or single-year, years=False) fit of prevalence on covariates + latitude,
    SEs clustered by local authority. Returns (data, fit, fit without latitude, classical fit)."""
    d = data[data[y].notna()].copy()
    if drop_london:
        d = d[d.london == 0]
    covs = [dep] + (extra or [])
    X, X0 = design(d, covs, years=years), design(d, covs, with_lat=False, years=years)
    w = d[weights].values if weights else np.ones(len(d))
    cluster = {"groups": pd.factorize(d.la_code)[0]}
    robust = sm.WLS(d[y].values, X, weights=w).fit(cov_type="cluster", cov_kwds=cluster)
    base = sm.WLS(d[y].values, X0, weights=w).fit()
    classical = sm.WLS(d[y].values, X, weights=w).fit()
    return d, robust, base, classical


def conley(d, y, X, cutoff):
    b, se = conley_se(d[y].values, X.drop(columns="const"), d.latitude.values, d.longitude.values, cutoff)
    dof = d.la_code.nunique() - 1
    t = b["latitude"] / se["latitude"]
    return se["latitude"], 2 * stats.t.sf(abs(t), dof), dof


# ---------------------------------------------------------------------------
# Main analysis: pooled 2023/24-2025/26
# ---------------------------------------------------------------------------
rows, moran_rows = [], []
for key, (_, cohort, label, short) in INDICATORS.items():
    for model, extra in MODELS.items():
        covs = ["imd_score"] + extra
        d, f, base, classical = fit(panel, key, extra=extra)
        X = design(d, covs)
        b_man, se_man = manual_cr1(d[key].values, X.values, d.la_code.values)
        i = list(X.columns).index("latitude")
        assert np.isclose(b_man[i], f.params["latitude"]) and np.isclose(se_man[i], f.bse["latitude"])
        ci = f.conf_int().loc["latitude"]
        se100, p100, dof = conley(d, key, X, 100)
        se200, p200, _ = conley(d, key, X, 200)
        tcrit = stats.t.ppf(0.975, dof)
        row = {"model": model, "indicator": key, "cohort": cohort, "threshold": label,
               "n_obs": int(f.nobs), "n_las": int(d.la_code.nunique()),
               "mean_prevalence": d[key].mean(), "imd_beta": f.params["imd_score"], "imd_p": f.pvalues["imd_score"],
               "lat_beta": f.params["latitude"], "lat_se": f.bse["latitude"], "lat_ci_low": ci[0], "lat_ci_high": ci[1],
               "lat_p": f.pvalues["latitude"],
               "lat_se_conley100": se100, "lat_p_conley100": p100,
               "conley100_ci_low": f.params["latitude"] - tcrit * se100,
               "conley100_ci_high": f.params["latitude"] + tcrit * se100,
               "lat_se_conley200": se200, "lat_p_conley200": p200,
               "r2": f.rsquared, "base_r2": base.rsquared, "delta_r2": f.rsquared - base.rsquared,
               "partial_r": partial_r(classical, "latitude")}
        for e in extra:
            row[f"{e}_beta"], row[f"{e}_p"] = f.params[e], f.pvalues[e]
        rows.append(row)
        # Moran's I of authority-mean residuals (without and with latitude)
        las = d.assign(r_base=base.resid, r_full=f.resid).groupby("la_code").agg(
            r_base=("r_base", "mean"), r_full=("r_full", "mean"), lat=("latitude", "first"), lon=("longitude", "first"))
        W = knn_weights(las.lat.values, las.lon.values, k=8)
        I0, p0 = morans_i(las.r_base.values, W, permutations=4999)
        I1, p1 = morans_i(las.r_full.values, W, permutations=4999)
        moran_rows.append({"model": model, "indicator": key, "morans_I_without_latitude": I0, "p_without_latitude": p0,
                           "morans_I_with_latitude": I1, "p_with_latitude": p1})
models = pd.DataFrame(rows)
for model in MODELS:
    sel = models.model == model
    models.loc[sel, "lat_p_holm"] = holm(models.loc[sel, "lat_p"].values)
    models.loc[sel, "lat_p_conley100_holm"] = holm(models.loc[sel, "lat_p_conley100"].values)
models.to_csv(OUT_TAB / "geo_models.csv", index=False)
moran = pd.DataFrame(moran_rows)
moran.to_csv(OUT_TAB / "geo_morans_i.csv", index=False)

# ---------------------------------------------------------------------------
# Robustness (pooled)
# ---------------------------------------------------------------------------
rob = []
for key, (_, cohort, label, short) in INDICATORS.items():
    meas = "rec_measured" if cohort == "Reception" else "yr6_measured"
    specs = {
        "Model A: IMD + latitude": {},
        "IDACI instead of IMD": {"dep": "idaci_score"},
        "London excluded": {"drop_london": True},
        "Longitude added": {"extra": ["longitude"]},
        "Weighted by children measured": {"weights": meas},
        "Region fixed effects": {"extra": REGION_COLS},
        "Model B: + child ethnic composition": {"extra": ETHNICITY},
        "Model B, London excluded": {"extra": ETHNICITY, "drop_london": True},
        "Model B, weighted by children measured": {"extra": ETHNICITY, "weights": meas},
        "Model B + region fixed effects": {"extra": ETHNICITY + REGION_COLS},
    }
    for spec, kw in specs.items():
        _, f, _, _ = fit(panel, key, **kw)
        rob.append({"indicator": key, "cohort": cohort, "threshold": short, "spec": spec, "n_obs": int(f.nobs),
                    "lat_beta": f.params["latitude"], "lat_se": f.bse["latitude"], "lat_p": f.pvalues["latitude"]})
rob = pd.DataFrame(rob)
rob.to_csv(OUT_TAB / "geo_robustness.csv", index=False)

# ---------------------------------------------------------------------------
# Disclosure-control bounds: suppressed counts (1-7 children) set to 1, 4 or 7
# ---------------------------------------------------------------------------
bounds = []
for key in [k for k in INDICATORS if panel[k].isna().any()]:
    cohort = INDICATORS[key][1]
    meas = "rec_measured" if cohort == "Reception" else "yr6_measured"
    for model, extra in MODELS.items():
        for label, count in (("excluded (main analysis)", None), ("set to 1 child", 1),
                             ("set to 4 children", 4), ("set to 7 children", 7)):
            d = panel.copy()
            if count is not None:
                d[key] = d[key].fillna(100 * count / d[meas])
            _, f, _, _ = fit(d, key, extra=extra)
            bounds.append({"model": model, "indicator": key, "n_suppressed": int(panel[key].isna().sum()),
                           "treatment": label, "n_obs": int(f.nobs), "lat_beta": f.params["latitude"],
                           "lat_p": f.pvalues["latitude"]})
bounds = pd.DataFrame(bounds)
bounds.to_csv(OUT_TAB / "geo_suppression_bounds.csv", index=False)

# ---------------------------------------------------------------------------
# Consistency: each year separately (HC1-equivalent: one observation per authority)
# ---------------------------------------------------------------------------
by_year = []
for year in YEARS:
    one = panel[panel.year == year]
    for key, (_, cohort, label, short) in INDICATORS.items():
        for model, extra in MODELS.items():
            d, f, base, _ = fit(one, key, extra=extra, years=False)
            ci = f.conf_int().loc["latitude"]
            se100, p100, _ = conley(d, key, design(d, ["imd_score"] + extra, years=False), 100)
            by_year.append({"year": year, "model": model, "indicator": key, "cohort": cohort, "threshold": label,
                            "n": int(f.nobs), "lat_beta": f.params["latitude"], "lat_ci_low": ci[0],
                            "lat_ci_high": ci[1], "lat_p": f.pvalues["latitude"], "lat_p_conley100": p100})
by_year = pd.DataFrame(by_year)
by_year.to_csv(OUT_TAB / "geo_models_by_year.csv", index=False)

# Stability of LA prevalence between consecutive years
stab = {}
for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
    a, b = panel[panel.year == y0], panel[panel.year == y1]
    stab[f"{y0}-{y1}"] = {k: float(a[["la_code", k]].merge(b[["la_code", k]], on="la_code").dropna().iloc[:, 1:]
                                   .corr().iloc[0, 1]) for k in INDICATORS}

# Correlations between covariates and three-year mean prevalence
la_means = panel.groupby("la_code").agg({**{k: "mean" for k in INDICATORS},
                                         **{c: "first" for c in ["latitude", "imd_score"] + ETHNICITY}})
corr_vars = ["latitude", "imd_score"] + ETHNICITY + list(INDICATORS)
corr = la_means[corr_vars].corr()
corr.to_csv(OUT_TAB / "geo_correlations.csv")

save_results("geographic", {
    "models": {f"{r['model']}|{r['indicator']}": r for r in models.to_dict(orient="records")},
    "robustness": rob.to_dict(orient="records"),
    "morans_i": moran.to_dict(orient="records"),
    "suppression_bounds": bounds.to_dict(orient="records"),
    "by_year": by_year.to_dict(orient="records"),
    "between_year_la_correlation": stab,
    "correlations_three_year_means": corr.round(4).to_dict(),
    "n_obs_per_indicator": {k: int(panel[k].notna().sum()) for k in INDICATORS},
})

pd.set_option("display.width", 250)
for model in MODELS:
    s = models[models.model == model]
    print(f"\nPooled 2023/24-2025/26, Model {model}: {MODEL_LABEL[model]}")
    print(s[["indicator", "n_obs", "n_las", "lat_beta", "lat_ci_low", "lat_ci_high", "lat_p", "lat_p_holm",
             "lat_p_conley100", "lat_p_conley100_holm", "delta_r2", "base_r2", "r2"]].round(4).to_string(index=False))
rob["txt"] = rob.apply(lambda x: f"{x.lat_beta:+.3f} ({x.lat_p:.4f})", axis=1)
print("\nRobustness (latitude beta, p):")
print(rob.pivot_table(index="spec", columns="indicator", values="txt", aggfunc="first", sort=False)[list(INDICATORS)].to_string())
print("\nMoran's I:"); print(moran.round(3).to_string(index=False))
print("\nSuppression bounds:"); print(bounds.round(4).to_string(index=False))
print("\nBy year:"); print(by_year.pivot_table(index=["indicator", "model"], columns="year", values="lat_beta", sort=False).round(3).to_string())
