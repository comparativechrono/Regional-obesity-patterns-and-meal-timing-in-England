"""Step 3: cross-sectional (local-authority) analysis of latitude and NCMP prevalence.

Two models for each of 8 cohort x threshold indicators (NCMP 2023/24), each replicated in
NCMP 2024/25 and NCMP 2025/26 (different cohorts of children):

    Model A  prevalence ~ IMD 2019 average score + latitude
    Model B  prevalence ~ IMD 2019 average score + child ethnic composition + latitude
             (Census 2021 % of dependent children who are Asian, Black, Mixed and Other;
              White is the reference group)

OLS with HC1 standard errors (cross-checked against an independent implementation; HC3
also reported); Holm correction across the 8 tests within each model and year; spatial-HAC
(Conley) standard errors at 100 and 200 km; Moran's I of residuals.

Robustness: IDACI in place of IMD; London excluded; longitude added; precision weights
(children measured); region fixed effects (Models A and B); bounds for values suppressed
for disclosure control.

Outputs: outputs/tables/geo_*.csv and results.json["geographic"]
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import (INDICATORS, OUT_DATA, OUT_TAB, PRIMARY_YEAR, REPLICATION_YEAR, YEARS, conley_se,
                    holm, knn_weights, manual_hc1, morans_i, ols, partial_r, save_results)

ETHNICITY = ["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"]  # % of dependent children; White = reference
MODELS = {"A": [], "B": ETHNICITY}
MODEL_LABEL = {"A": "IMD + latitude", "B": "IMD + child ethnic composition + latitude"}


def load(year):
    return pd.read_csv(OUT_DATA / f"la_dataset_{year.replace('/', '_')}.csv")


def fit(df, y, dep="imd_score", extra=None, weights=None, drop_london=False, cov="HC1"):
    """prevalence ~ dep + extra + latitude.
    Returns (data used, robust fit, fit without latitude, classical fit)."""
    d = df[df[y].notna()].copy()
    if drop_london:
        d = d[d.london == 0]
    covs = [dep] + (extra or [])
    X = d[covs + ["latitude"]]
    if weights is not None:
        w = d[weights].values
        robust = sm.WLS(d[y].values, sm.add_constant(X), weights=w).fit(cov_type=cov)
        base = sm.WLS(d[y].values, sm.add_constant(d[covs]), weights=w).fit()
        classical = sm.WLS(d[y].values, sm.add_constant(X), weights=w).fit()
    else:
        robust = ols(d[y], X, cov)
        base = ols(d[y], d[covs], "nonrobust")
        classical = ols(d[y], X, "nonrobust")
    return d, robust, base, classical


def conley(d, y, covs, cutoff):
    """Latitude SE and p from spatial-HAC (Bartlett kernel) standard errors."""
    b, se = conley_se(d[y].values, d[covs + ["latitude"]], d.latitude.values, d.longitude.values, cutoff)
    dof = len(d) - len(covs) - 2
    t = b["latitude"] / se["latitude"]
    return se["latitude"], 2 * stats.t.sf(abs(t), dof), dof


def analyse_year(year):
    rows, robust_rows, moran_rows = [], [], []
    df = load(year)
    region_dummies = pd.get_dummies(df["region"], prefix="reg", drop_first=True, dtype=float)
    df = pd.concat([df, region_dummies], axis=1)
    region_cols = list(region_dummies.columns)
    for key, (_, cohort, label, short) in INDICATORS.items():
        dk = df[df[key].notna()]
        raw = ols(dk[key], dk[["latitude"]], "HC1")
        W = knn_weights(dk.latitude.values, dk.longitude.values, k=8)
        for model, extra in MODELS.items():
            covs = ["imd_score"] + extra
            d, f, base, classical = fit(df, key, extra=extra)
            hc3 = fit(df, key, extra=extra, cov="HC3")[1]
            b_man, se_man = manual_hc1(d[key], d[covs + ["latitude"]])
            assert np.isclose(b_man[-1], f.params["latitude"]) and np.isclose(se_man[-1], f.bse["latitude"])
            ci = f.conf_int().loc["latitude"]
            se100, p100, dof = conley(d, key, covs, 100)
            se200, p200, _ = conley(d, key, covs, 200)
            tcrit = stats.t.ppf(0.975, dof)
            row = {
                "year": year, "model": model, "indicator": key, "cohort": cohort, "threshold": label,
                "n": int(f.nobs), "mean_prevalence": d[key].mean(), "sd_prevalence": d[key].std(),
                "raw_beta": raw.params["latitude"], "raw_p": raw.pvalues["latitude"],
                "raw_r": np.sign(raw.params["latitude"]) * np.sqrt(raw.rsquared),
                "base_r2": base.rsquared, "imd_beta": f.params["imd_score"], "imd_p": f.pvalues["imd_score"],
                "lat_beta": f.params["latitude"], "lat_se": f.bse["latitude"], "lat_ci_low": ci[0],
                "lat_ci_high": ci[1], "lat_p": f.pvalues["latitude"], "lat_p_hc3": hc3.pvalues["latitude"],
                "lat_se_conley100": se100, "lat_p_conley100": p100,
                "conley100_ci_low": f.params["latitude"] - tcrit * se100,
                "conley100_ci_high": f.params["latitude"] + tcrit * se100,
                "lat_se_conley200": se200, "lat_p_conley200": p200,
                "r2": f.rsquared, "delta_r2": f.rsquared - base.rsquared,
                "partial_r": partial_r(classical, "latitude"),
            }
            for e in extra:
                row[f"{e}_beta"], row[f"{e}_p"] = f.params[e], f.pvalues[e]
            rows.append(row)
            I_base, p_base = morans_i(base.resid, W, permutations=4999)
            I_full, p_full = morans_i(f.resid, W, permutations=4999)
            moran_rows.append({"year": year, "model": model, "indicator": key,
                               "morans_I_without_latitude": I_base, "p_without_latitude": p_base,
                               "morans_I_with_latitude": I_full, "p_with_latitude": p_full})

        def add(spec, res):
            robust_rows.append({"year": year, "indicator": key, "cohort": cohort, "threshold": short, "spec": spec,
                                "n": int(res.nobs), "lat_beta": res.params["latitude"],
                                "lat_se": res.bse["latitude"], "lat_p": res.pvalues["latitude"]})
        meas = "rec_measured" if cohort == "Reception" else "yr6_measured"
        add("Model A: IMD + latitude", fit(df, key)[1])
        add("IDACI instead of IMD", fit(df, key, dep="idaci_score")[1])
        add("London excluded", fit(df, key, drop_london=True)[1])
        add("Longitude added", fit(df, key, extra=["longitude"])[1])
        add("Weighted by children measured", fit(df, key, weights=meas)[1])
        add("Region fixed effects", fit(df, key, extra=region_cols)[1])
        add("Model B: + child ethnic composition", fit(df, key, extra=ETHNICITY)[1])
        add("Model B, London excluded", fit(df, key, extra=ETHNICITY, drop_london=True)[1])
        add("Model B, weighted by children measured", fit(df, key, extra=ETHNICITY, weights=meas)[1])
        add("Model B + region fixed effects", fit(df, key, extra=ETHNICITY + region_cols)[1])
    models = pd.DataFrame(rows)
    for model in MODELS:
        sel = models.model == model
        models.loc[sel, "lat_p_holm"] = holm(models.loc[sel, "lat_p"].values)
        models.loc[sel, "lat_p_conley100_holm"] = holm(models.loc[sel, "lat_p_conley100"].values)
    return df, models, pd.DataFrame(robust_rows), pd.DataFrame(moran_rows)


per_year = {year: analyse_year(year) for year in YEARS}
df = per_year[PRIMARY_YEAR][0]
models = pd.concat([v[1] for v in per_year.values()], ignore_index=True)
models.to_csv(OUT_TAB / "geo_models.csv", index=False)
robust = pd.concat([v[2] for v in per_year.values()], ignore_index=True)
robust.to_csv(OUT_TAB / "geo_robustness.csv", index=False)
moran = pd.concat([v[3] for v in per_year.values()], ignore_index=True)
moran.to_csv(OUT_TAB / "geo_morans_i.csv", index=False)

# ---------------------------------------------------------------------------
# Disclosure-control bounds: suppressed counts are 1-7 children.
# Re-fit with suppressed LAs imputed at 1, 4 and 7 children (both models, 2023/24).
# ---------------------------------------------------------------------------
bounds = []
for key in [k for k in INDICATORS if df[k].isna().any()]:
    cohort = INDICATORS[key][1]
    meas = "rec_measured" if cohort == "Reception" else "yr6_measured"
    for model, extra in MODELS.items():
        for label, count in (("excluded (primary)", None), ("imputed at 1 child", 1),
                             ("imputed at 4 children", 4), ("imputed at 7 children", 7)):
            d = df.copy()
            if count is not None:
                d[key] = d[key].fillna(100 * count / d[meas])
            _, f, _, _ = fit(d, key, extra=extra)
            bounds.append({"model": model, "indicator": key, "n_suppressed": int(df[key].isna().sum()),
                           "treatment": label, "n": int(f.nobs), "lat_beta": f.params["latitude"],
                           "lat_p": f.pvalues["latitude"]})
bounds = pd.DataFrame(bounds)
bounds.to_csv(OUT_TAB / "geo_suppression_bounds_2023_24.csv", index=False)

# Stability of LA prevalence between consecutive years
d23 = load(PRIMARY_YEAR)
stab = {}
for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
    a, b = load(y0), load(y1)
    stab[f"{y0}-{y1}"] = {k: float(a[["la_code", k]].merge(b[["la_code", k]], on="la_code").dropna().iloc[:, 1:]
                                   .corr().iloc[0, 1]) for k in INDICATORS}

# Covariate and outcome correlations (2023/24)
corr_vars = ["latitude", "imd_score", "pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"] + list(INDICATORS)
corr = d23[corr_vars].corr()
corr.to_csv(OUT_TAB / "geo_correlations_2023_24.csv")

save_results("geographic", {
    "models": {f"{r['year']}|{r['model']}|{r['indicator']}": r for r in models.to_dict(orient="records")},
    "robustness": robust.to_dict(orient="records"),
    "morans_i": moran.to_dict(orient="records"),
    "suppression_bounds_2023_24": bounds.to_dict(orient="records"),
    "between_year_la_correlation": stab,
    "correlations_2023_24": corr.round(4).to_dict(),
})

pd.set_option("display.width", 250)
for yr in YEARS:
    for model in MODELS:
        s = models[(models.year == yr) & (models.model == model)]
        print(f"\n{yr} Model {model}: {MODEL_LABEL[model]}")
        print(s[["indicator", "n", "lat_beta", "lat_ci_low", "lat_ci_high", "lat_p", "lat_p_holm", "lat_p_conley100",
                 "lat_p_conley100_holm", "partial_r", "delta_r2", "base_r2", "r2"]].round(4).to_string(index=False))
print("\nRobustness (latitude beta, p):")
robust["txt"] = robust.apply(lambda x: f"{x.lat_beta:+.3f} ({x.lat_p:.4f})", axis=1)
for yr in YEARS:
    print(yr)
    print(robust[robust.year == yr].pivot_table(index="spec", columns="indicator", values="txt", aggfunc="first",
                                                sort=False)[list(INDICATORS)].to_string())
print("\nMoran's I:"); print(moran.round(3).to_string(index=False))
print("\nSuppression bounds:"); print(bounds.round(4).to_string(index=False))
print("\nBetween-year LA correlation:", {p: {k: round(v, 3) for k, v in d.items()} for p, d in stab.items()})
