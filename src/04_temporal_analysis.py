"""Step 4: analysis of the national monthly NCMP series (NHS Digital Tables 3-4).

Primary sample: the 12 complete collection years 2007/08-2018/19 (Sept-July, 132 months
per cohort). 2019/20 is excluded from the primary analysis because data stop in March
2020 and March is incomplete (COVID-19 school closures); it is used in a sensitivity check.

Analyses
  1. When children are measured: share of each cohort measured by month and by term.
  2. Within-year trend in each BMI category (academic-year fixed effects, standard errors
     clustered by year), plus Sept-Dec vs Jan-Jul slopes.
  3. Linear cosinor (fixed 12-month period) with zero-amplitude F test.
  4. Discriminating model comparison: does photoperiod (or any sinusoid) explain anything
     once a linear within-school-year trend is allowed for?  Pooled (year FE, clustered SE)
     and collapsed to the 11 monthly means (the most conservative form).
  5. Location/scale of the BMI z-score distribution through the year.
  6. The September reset (July -> following September) as a discriminating test.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import ACAD_MONTHS, CUTPOINTS, OUT_DATA, OUT_TAB, save_results

df_all = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
df = df_all[df_all.complete_year].copy()
CATS = ["underweight", "healthy", "overweight", "obese", "severe", "owob"]
COHORTS = ["Reception", "Year 6"]
results = {}

# ---------------------------------------------------------------------------
# 1. Measurement timing
# ---------------------------------------------------------------------------
timing = []
for c in COHORTS:
    s = df[df.cohort == c]
    tot = s.groupby("month")["n_measured"].sum().reindex(ACAD_MONTHS)
    share = 100 * tot / tot.sum()
    # month blocks approximating the three school terms (Easter falls in late March or April)
    terms = {"Sep-Dec (autumn term)": share[ACAD_MONTHS[0:4]].sum(),
             "Jan-Mar (spring term)": share[ACAD_MONTHS[4:7]].sum(),
             "Apr-Jul (summer term)": share[ACAD_MONTHS[7:11]].sum()}
    per_year = s.pivot_table(index="ncmp_year", columns="month", values="n_measured")[ACAD_MONTHS]
    per_year_share = per_year.div(per_year.sum(axis=1), axis=0) * 100
    for m in ACAD_MONTHS:
        timing.append({"cohort": c, "month": m, "children": int(tot[m]), "share_pct": share[m],
                       "share_min_year": per_year_share[m].min(), "share_max_year": per_year_share[m].max()})
    # mean measurement position (academic month) and its year-to-year range
    pos = (per_year_share * np.arange(1, 12)).sum(axis=1) / 100
    results.setdefault("timing", {})[c] = {
        "terms_pct": terms, "modal_month": share.idxmax(), "share_by_month_pct": share.round(2).to_dict(),
        "mean_academic_month": float((share * np.arange(1, 12)).sum() / 100),
        "mean_academic_month_range_by_year": [float(pos.min()), float(pos.max())],
        "children_measured": int(tot.sum())}
timing = pd.DataFrame(timing)
timing.to_csv(OUT_TAB / "monthly_measurement_timing.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Within-year trends (year fixed effects; SE clustered by collection year)
# ---------------------------------------------------------------------------
def cluster_fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["ncmp_year"]})

trend_rows, monthly_means = [], []
for c in COHORTS:
    s = df[df.cohort == c]
    for cat in CATS + ["z_mean", "z_sd"]:
        y = f"pct_{cat}" if cat in CATS else cat
        f = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", s)
        autumn = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", s[s.acad_month <= 4])
        rest = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", s[s.acad_month >= 5])
        mm = s.groupby("acad_month")[y].agg(["mean", "std", "count"])
        mm["ci95"] = stats.t.ppf(0.975, mm["count"] - 1) * mm["std"] / np.sqrt(mm["count"])
        sep_jul = mm.loc[11, "mean"] - mm.loc[1, "mean"]
        trend_rows.append({"cohort": c, "outcome": cat, "mean": s[y].mean(),
                           "slope_per_month": f.params["acad_month"], "slope_ci_low": f.conf_int().loc["acad_month", 0],
                           "slope_ci_high": f.conf_int().loc["acad_month", 1], "slope_p": f.pvalues["acad_month"],
                           "slope_sep_dec": autumn.params["acad_month"], "p_sep_dec": autumn.pvalues["acad_month"],
                           "slope_jan_jul": rest.params["acad_month"], "p_jan_jul": rest.pvalues["acad_month"],
                           "sep_mean": mm.loc[1, "mean"], "jul_mean": mm.loc[11, "mean"], "jul_minus_sep": sep_jul})
        for m, r in mm.iterrows():
            monthly_means.append({"cohort": c, "outcome": cat, "acad_month": m, "month": ACAD_MONTHS[m - 1],
                                  "mean": r["mean"], "ci95": r["ci95"], "n_years": int(r["count"])})
trends = pd.DataFrame(trend_rows)
trends.to_csv(OUT_TAB / "monthly_within_year_trends.csv", index=False)
monthly_means = pd.DataFrame(monthly_means)
monthly_means.to_csv(OUT_TAB / "monthly_means_by_month.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Linear cosinor, 12-month period (Cornelissen 2014): y = M + b(t - tbar) + a cos + g sin
# ---------------------------------------------------------------------------
def cosinor(s, y):
    d = s.copy()
    d["tc"] = d["t_years"] - d["t_years"].mean()
    d["cosw"] = np.cos(2 * np.pi * d["t_years"])
    d["sinw"] = np.sin(2 * np.pi * d["t_years"])
    full = smf.ols(f"{y} ~ tc + cosw + sinw", data=d).fit()
    red = smf.ols(f"{y} ~ tc", data=d).fit()
    F, p, _ = full.compare_f_test(red)
    a, g = full.params["cosw"], full.params["sinw"]
    amp = float(np.hypot(a, g))
    V = full.cov_params().loc[["cosw", "sinw"], ["cosw", "sinw"]].values
    grad = np.array([a, g]) / amp
    amp_se = float(np.sqrt(grad @ V @ grad))
    acro_years = (np.arctan2(g, a) / (2 * np.pi)) % 1.0          # fraction of calendar year at the peak
    peak_doy = acro_years * 365.25
    peak = (pd.Timestamp("2021-01-01") + pd.Timedelta(days=float(peak_doy))).strftime("%d %b")
    return {"amplitude": amp, "amplitude_ci_low": amp - 1.96 * amp_se, "amplitude_ci_high": amp + 1.96 * amp_se,
            "peak_date": peak, "peak_doy": float(peak_doy), "trend_per_year": full.params["tc"],
            "zero_amplitude_F": float(F), "zero_amplitude_p": float(p), "r2": full.rsquared, "aic": full.aic}

cos_rows = []
for c in COHORTS:
    s = df[df.cohort == c]
    for cat in CATS:
        cos_rows.append({"cohort": c, "outcome": cat, **cosinor(s, f"pct_{cat}")})
cos_tab = pd.DataFrame(cos_rows)
cos_tab.to_csv(OUT_TAB / "monthly_cosinor.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Model comparison: school-year trend vs photoperiod vs sinusoid
# ---------------------------------------------------------------------------
def add_terms(d):
    d = d.copy()
    d["cosw"] = np.cos(2 * np.pi * d["t_years"])
    d["sinw"] = np.sin(2 * np.pi * d["t_years"])
    d["after_dec"] = np.clip(d["acad_month"] - 4, 0, None)  # hinge: separate slope from January
    return d

comp_rows = []
for c in COHORTS:
    s = add_terms(df[df.cohort == c])
    for cat in ["obese", "owob", "underweight", "z_mean"]:
        y = f"pct_{cat}" if cat != "z_mean" else cat
        specs = {
            "year FE only": f"{y} ~ C(ncmp_year)",
            "school-year trend": f"{y} ~ C(ncmp_year) + acad_month",
            "photoperiod": f"{y} ~ C(ncmp_year) + photoperiod_h",
            "sinusoid": f"{y} ~ C(ncmp_year) + cosw + sinw",
            "trend + photoperiod": f"{y} ~ C(ncmp_year) + acad_month + photoperiod_h",
            "trend + sinusoid": f"{y} ~ C(ncmp_year) + acad_month + cosw + sinw",
            "quadratic trend": f"{y} ~ C(ncmp_year) + acad_month + I(acad_month ** 2)",
            "quadratic trend + photoperiod": f"{y} ~ C(ncmp_year) + acad_month + I(acad_month ** 2) + photoperiod_h",
            "two-phase trend": f"{y} ~ C(ncmp_year) + acad_month + after_dec",
            "two-phase trend + photoperiod": f"{y} ~ C(ncmp_year) + acad_month + after_dec + photoperiod_h",
            "month FE (saturated)": f"{y} ~ C(ncmp_year) + C(acad_month)",
        }
        fits = {k: smf.ols(v, data=s).fit() for k, v in specs.items()}
        clustered = cluster_fit(specs["trend + photoperiod"], s)
        F_photo = fits["trend + photoperiod"].compare_f_test(fits["school-year trend"])
        F_sin = fits["trend + sinusoid"].compare_f_test(fits["school-year trend"])
        F_trend_given_photo = fits["trend + photoperiod"].compare_f_test(fits["photoperiod"])
        F_photo_given_quad = fits["quadratic trend + photoperiod"].compare_f_test(fits["quadratic trend"])
        clustered_quad = cluster_fit(specs["quadratic trend + photoperiod"], s)
        F_photo_given_2ph = fits["two-phase trend + photoperiod"].compare_f_test(fits["two-phase trend"])
        clustered_2ph = cluster_fit(specs["two-phase trend + photoperiod"], s)
        # collapsed to the 11 month means (most conservative)
        mm = s.groupby("acad_month").agg(y=(y, "mean"), photo=("photoperiod_h", "first")).reset_index()
        col = smf.ols("y ~ acad_month + photo", data=mm).fit()
        col_photo_only = smf.ols("y ~ photo", data=mm).fit()
        comp_rows.append({
            "cohort": c, "outcome": cat,
            **{f"aic_{k}": f.aic for k, f in fits.items()},
            **{f"adjR2_{k}": f.rsquared_adj for k, f in fits.items()},
            "photo_coef_alone": fits["photoperiod"].params["photoperiod_h"],
            "photo_coef_given_trend": fits["trend + photoperiod"].params["photoperiod_h"],
            "photo_coef_given_trend_clustered_p": clustered.pvalues["photoperiod_h"],
            "F_photo_given_trend_p": F_photo[1], "F_sinusoid_given_trend_p": F_sin[1],
            "F_trend_given_photo_p": F_trend_given_photo[1],
            "F_photo_given_quadratic_trend_p": F_photo_given_quad[1],
            "photo_coef_given_quadratic_trend": fits["quadratic trend + photoperiod"].params["photoperiod_h"],
            "photo_coef_given_quadratic_trend_clustered_p": clustered_quad.pvalues["photoperiod_h"],
            "F_photo_given_two_phase_trend_p": F_photo_given_2ph[1],
            "photo_coef_given_two_phase_trend": fits["two-phase trend + photoperiod"].params["photoperiod_h"],
            "photo_coef_given_two_phase_trend_clustered_p": clustered_2ph.pvalues["photoperiod_h"],
            "trend_coef_given_photo": fits["trend + photoperiod"].params["acad_month"],
            "trend_coef_given_photo_clustered_p": clustered.pvalues["acad_month"],
            "collapsed_photo_coef": col.params["photo"], "collapsed_photo_p": col.pvalues["photo"],
            "collapsed_trend_coef": col.params["acad_month"], "collapsed_trend_p": col.pvalues["acad_month"],
            "collapsed_photo_only_r": float(np.sign(col_photo_only.params["photo"]) * np.sqrt(col_photo_only.rsquared)),
            "collapsed_photo_only_p": col_photo_only.pvalues["photo"],
        })
comp = pd.DataFrame(comp_rows)
comp.to_csv(OUT_TAB / "monthly_model_comparison.csv", index=False)

# Residual seasonal structure after the school-year trend (Reception obesity), by month
resid_rows = []
for c in COHORTS:
    s = df[df.cohort == c].copy()
    f = smf.ols("pct_obese ~ C(ncmp_year) + acad_month", data=s).fit()
    s["resid"] = f.resid
    r = s.groupby("acad_month").agg(resid=("resid", "mean"), photo=("photoperiod_h", "first")).reset_index()
    r["cohort"] = c
    resid_rows.append(r)
resid = pd.concat(resid_rows)
resid.to_csv(OUT_TAB / "monthly_obesity_residuals_after_trend.csv", index=False)
resid_corr = {c: float(np.corrcoef(resid[resid.cohort == c].resid, resid[resid.cohort == c].photo)[0, 1]) for c in COHORTS}

# Collinearity: over September-July, how much of day length is a smooth function of school month?
mm_photo = df.groupby("acad_month")["photoperiod_h"].first()
am = mm_photo.index.values.astype(float)
X_lin = np.column_stack([np.ones(11), am])
X_quad = np.column_stack([np.ones(11), am, am ** 2])
r2_lin = 1 - np.sum((mm_photo.values - X_lin @ np.linalg.lstsq(X_lin, mm_photo.values, rcond=None)[0]) ** 2) / np.sum((mm_photo.values - mm_photo.mean()) ** 2)
r2_quad = 1 - np.sum((mm_photo.values - X_quad @ np.linalg.lstsq(X_quad, mm_photo.values, rcond=None)[0]) ** 2) / np.sum((mm_photo.values - mm_photo.mean()) ** 2)
results["photoperiod_vs_school_month_r2"] = {"linear": float(r2_lin), "quadratic": float(r2_quad)}

# ---------------------------------------------------------------------------
# 4b. The September reset: July of one school year -> September of the next.
#     Day length falls by ~3.6 h between mid-July and mid-September, so a photoperiod-driven
#     process predicts little change (photoperiod alone) or a fall (positive coefficient given
#     the trend); a school-year-anchored process predicts a rise of ~10 x the monthly slope.
# ---------------------------------------------------------------------------
reset_rows = []
years = sorted(df_all.ncmp_year.unique())
for c in COHORTS:
    s_all = df_all[df_all.cohort == c]
    comp_c = comp[comp.cohort == c].set_index("outcome")
    tr_c = trends[trends.cohort == c].set_index("outcome")
    for cat in ["obese", "owob", "underweight"]:
        y = f"pct_{cat}"
        changes = []
        for y0, y1 in zip(years[:-1], years[1:]):
            jul = s_all[(s_all.ncmp_year == y0) & (s_all.acad_month == 11)]
            sep = s_all[(s_all.ncmp_year == y1) & (s_all.acad_month == 1)]
            if len(jul) and len(sep):
                changes.append(float(sep[y].iloc[0] - jul[y].iloc[0]))
        changes = np.array(changes)
        d_photo = float(mm_photo.loc[1] - mm_photo.loc[11])
        t_ci = stats.t.ppf(0.975, len(changes) - 1) * changes.std(ddof=1) / np.sqrt(len(changes))
        reset_rows.append({
            "cohort": c, "outcome": cat, "n_transitions": len(changes),
            "observed_mean_change": changes.mean(), "ci_low": changes.mean() - t_ci, "ci_high": changes.mean() + t_ci,
            "n_positive": int((changes > 0).sum()), "sign_test_p": float(stats.binomtest(int((changes > 0).sum()), len(changes)).pvalue),
            "day_length_change_h": d_photo,
            "pred_school_year_trend": -10 * tr_c.loc[cat, "slope_per_month"],
            "pred_photoperiod_alone": comp_c.loc[cat, "photo_coef_alone"] * d_photo,
            "pred_photoperiod_coef_given_trend": comp_c.loc[cat, "photo_coef_given_trend"] * d_photo,
        })
reset = pd.DataFrame(reset_rows)
reset.to_csv(OUT_TAB / "monthly_september_reset.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Distribution location vs scale: does a pure shift of the mean explain each tail?
#    Predicted change in each category over the year if only the mean z shifted.
# ---------------------------------------------------------------------------
#    Changes are the fitted September->July change (10 x within-year slope, year fixed effects),
#    which is less noisy than comparing the two smallest months directly.
shift_rows = []
for c in COHORTS:
    s = df[df.cohort == c]
    tr = trends[trends.cohort == c].set_index("outcome")
    mu_bar, sd_bar = s["z_mean"].mean(), s["z_sd"].mean()
    d_mu, d_sd = 10 * tr.loc["z_mean", "slope_per_month"], 10 * tr.loc["z_sd", "slope_per_month"]
    def prev(mu, sd):
        return {"underweight": 100 * stats.norm.cdf((CUTPOINTS["underweight"] - mu) / sd),
                "owob": 100 * stats.norm.sf((CUTPOINTS["overweight"] - mu) / sd),
                "obese": 100 * stats.norm.sf((CUTPOINTS["obese"] - mu) / sd),
                "severe": 100 * stats.norm.sf((CUTPOINTS["severe"] - mu) / sd)}
    p_sep_loc, p_jul_loc = prev(mu_bar - d_mu / 2, sd_bar), prev(mu_bar + d_mu / 2, sd_bar)
    p_sep_both, p_jul_both = prev(mu_bar - d_mu / 2, sd_bar - d_sd / 2), prev(mu_bar + d_mu / 2, sd_bar + d_sd / 2)
    for k in p_sep_loc:
        shift_rows.append({"cohort": c, "outcome": k, "observed_sep_to_jul": 10 * tr.loc[k, "slope_per_month"],
                           "predicted_mean_shift_only": p_jul_loc[k] - p_sep_loc[k],
                           "predicted_mean_and_sd": p_jul_both[k] - p_sep_both[k]})
    results.setdefault("distribution", {})[c] = {
        "z_mean_overall": mu_bar, "z_sd_overall": sd_bar, "z_mean_change_sep_to_jul": d_mu,
        "z_mean_change_ci": [10 * tr.loc["z_mean", "slope_ci_low"], 10 * tr.loc["z_mean", "slope_ci_high"]],
        "z_sd_change_sep_to_jul": d_sd, "z_sd_change_p": tr.loc["z_sd", "slope_p"],
        "probit_fit_rmse_median": float(s["probit_fit_rmse"].median())}
shift = pd.DataFrame(shift_rows)
shift.to_csv(OUT_TAB / "monthly_distribution_shift.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Sensitivity: include 2019/20 (Sept-Feb); Year 6 excluding 2007/08-2008/09
# ---------------------------------------------------------------------------
sens = []
extra = df_all[(df_all.ncmp_year == "2019/20") & (df_all.acad_month <= 6)]
for label, data in {"primary (2007/08-2018/19)": df,
                    "plus 2019/20 Sep-Feb": pd.concat([df, extra]),
                    "excluding 2007/08-2008/09": df[~df.ncmp_year.isin(["2007/08", "2008/09"])],
                    "before universal infant free school meals (2007/08-2013/14)": df[df.ncmp_year < "2014/15"],
                    "after universal infant free school meals (2014/15-2018/19)": df[df.ncmp_year >= "2014/15"]}.items():
    for c in COHORTS:
        s = data[data.cohort == c]
        f = cluster_fit("pct_obese ~ C(ncmp_year) + acad_month", s)
        g = cluster_fit("pct_obese ~ C(ncmp_year) + acad_month + photoperiod_h", s)
        sens.append({"sample": label, "cohort": c, "n_months": len(s), "obese_slope": f.params["acad_month"],
                     "obese_slope_p": f.pvalues["acad_month"], "photo_given_trend": g.params["photoperiod_h"],
                     "photo_given_trend_p": g.pvalues["photoperiod_h"]})
sens = pd.DataFrame(sens)
sens.to_csv(OUT_TAB / "monthly_sensitivity.csv", index=False)

# Did the within-year slope change when universal infant free school meals began (September 2014)?
uifsm = []
for c in COHORTS:
    s_ = df[df.cohort == c].assign(post=lambda d: (d.ncmp_year >= "2014/15").astype(int))
    for y in ["pct_obese", "pct_owob", "pct_underweight", "z_mean"]:
        f = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month + acad_month:post", s_)
        uifsm.append({"cohort": c, "outcome": y, "slope_before": f.params["acad_month"],
                      "slope_change_after": f.params["acad_month:post"], "p_change": f.pvalues["acad_month:post"]})
uifsm = pd.DataFrame(uifsm)
uifsm.to_csv(OUT_TAB / "monthly_slope_before_after_uifsm.csv", index=False)


results.update({
    "trends": trends.set_index(["cohort", "outcome"]).to_dict(orient="index"),
    "cosinor": cos_tab.set_index(["cohort", "outcome"]).to_dict(orient="index"),
    "model_comparison": comp.set_index(["cohort", "outcome"]).to_dict(orient="index"),
    "residual_vs_photoperiod_corr": resid_corr,
    "distribution_shift": shift.to_dict(orient="records"),
    "sensitivity": sens.to_dict(orient="records"),
    "september_reset": reset.to_dict(orient="records"),
    "slope_before_after_uifsm": uifsm.to_dict(orient="records"),
})
# JSON keys must be strings
results = {k: ({"|".join(kk) if isinstance(kk, tuple) else kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v)
           for k, v in results.items()}
save_results("temporal", results)

pd.set_option("display.width", 250)
print("TIMING"); print({c: (results['timing'][c]['terms_pct'], results['timing'][c]['modal_month']) for c in COHORTS})
print("\nTRENDS"); print(trends.round(4).to_string(index=False))
print("\nCOSINOR"); print(cos_tab.round(4).to_string(index=False))
print("\nMODEL COMPARISON"); print(comp[["cohort", "outcome", "photo_coef_alone", "photo_coef_given_trend", "photo_coef_given_trend_clustered_p",
                                       "F_photo_given_trend_p", "F_sinusoid_given_trend_p", "trend_coef_given_photo", "trend_coef_given_photo_clustered_p",
                                       "collapsed_photo_coef", "collapsed_photo_p", "collapsed_trend_coef", "collapsed_trend_p",
                                       "aic_school-year trend", "aic_photoperiod", "aic_sinusoid", "aic_trend + photoperiod"]].round(4).to_string(index=False))
print("\nRESIDUALS (Reception obesity after school-year trend)"); print(resid[resid.cohort == "Reception"].round(3).to_string(index=False))
print("residual~photoperiod r:", resid_corr)
print("\nDISTRIBUTION SHIFT"); print(shift.round(3).to_string(index=False)); print(results["distribution"])
print("\nSENSITIVITY"); print(sens.round(4).to_string(index=False))
print("\nSEPTEMBER RESET"); print(reset.round(3).to_string(index=False))
print("\nSLOPE BEFORE/AFTER UIFSM"); print(uifsm.round(4).to_string(index=False))
print("photoperiod vs school month R2:", results["photoperiod_vs_school_month_r2"])
print(comp[["cohort", "outcome", "aic_school-year trend", "aic_quadratic trend", "aic_two-phase trend", "aic_trend + photoperiod",
            "aic_two-phase trend + photoperiod", "aic_month FE (saturated)", "photo_coef_given_trend", "photo_coef_given_trend_clustered_p",
            "photo_coef_given_quadratic_trend", "photo_coef_given_quadratic_trend_clustered_p",
            "photo_coef_given_two_phase_trend", "photo_coef_given_two_phase_trend_clustered_p"]].round(4).to_string(index=False))
