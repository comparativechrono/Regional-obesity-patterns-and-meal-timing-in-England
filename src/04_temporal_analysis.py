"""Step 4: analysis of the national monthly NCMP series (NHS Digital Tables 3-4).

Sample: the 12 complete school years 2007/08-2018/19 (September-July, 132 months per cohort).
2019/20 stops in March 2020 (COVID-19 school closures) and is used only in a sensitivity check.

Every September a new cohort of children is measured, so each school year is a separate group
of children: the analyses use school-year fixed effects and never link July of one year to
September of the next. Within a school year, the month of measurement advances the calendar
(day length, season, school terms and holidays) and the children's age and time in school
together. Reception (age 4-5) and Year 6 (age 10-11) are measured under the same calendar but
at different ages.

Analyses
  1. When children are measured.
  2. The within-year pattern month by month, and its change within periods: the autumn term
     (September-December, when days shorten), the spring term (January-March), and May-July
     (after the spring peak); the spring peak (April against March and May).
  3. The shape of the within-year change: a single linear school-year trend against a decline
     that stops part-way through the year (end of the decline estimated from the data), a
     linear trend plus day length, and unrestricted month effects (AIC, BIC).
  4. Location (mean) and spread (SD) of the BMI z-score distribution during the autumn decline
     and in May-July.
  5. Month-to-month pattern shared by the two cohorts and cohort-specific change:
       y[c, t, m] = a[c, t] + b[c] * m + s[m]
     s is identified only up to a linear function of m, so the shared component is the
     non-linear part of the within-year pattern.
  6. The spring peak and the date of Easter. Easter moves the Easter school holiday by up to
     five weeks between years; day length on a given date and the start of British Summer Time
     (last Sunday in March) do not move with it.
  7. Sensitivity analyses.
"""
import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dateutil.easter import easter
from scipy import stats

from common import ACAD_MONTHS, CUTPOINTS, OUT_DATA, OUT_TAB, REFERENCE_LATITUDE, day_length, save_results

df_all = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
df = df_all[df_all.complete_year].copy().reset_index(drop=True)
CATS = ["underweight", "healthy", "overweight", "obese", "severe", "owob"]
COHORTS = ["Reception", "Year 6"]
MONTHS = np.arange(1, 12)
APRIL, MAY, MARCH = 8, 9, 7
PERIODS = {"sep_dec": (1, 4), "jan_mar": (5, 7), "may_jul": (9, 11), "jan_jul": (5, 11)}
SHAPE_OUTCOMES = {"pct_obese": "Obesity", "pct_owob": "Overweight incl. obesity", "pct_underweight": "Underweight",
                  "z_mean": "Mean BMI z-score"}
years = sorted(df.ncmp_year.unique())
results = {}
rng = np.random.default_rng(20261124)


def ycol(cat):
    return f"pct_{cat}" if cat in CATS else cat


def cluster_fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["ncmp_year"]})


def year_by_month(data, cohort, y):
    """School years x months matrix for one cohort."""
    return data[data.cohort == cohort].pivot_table(index="ncmp_year", columns="acad_month", values=y).reindex(years)


day = df.groupby("acad_month")["mid_month_doy"].first().reindex(MONTHS).values
D_month = day_length(REFERENCE_LATITUDE, day)

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
    pos = (per_year_share * np.arange(1, 12)).sum(axis=1) / 100
    autumn_by_year = per_year_share[ACAD_MONTHS[0:4]].sum(axis=1)
    results.setdefault("timing", {})[c] = {
        "autumn_term_pct_range_by_year": [float(autumn_by_year.min()), float(autumn_by_year.max())],
        "terms_pct": terms, "modal_month": share.idxmax(), "share_by_month_pct": share.round(2).to_dict(),
        "mean_academic_month": float((share * np.arange(1, 12)).sum() / 100),
        "mean_academic_month_range_by_year": [float(pos.min()), float(pos.max())],
        "children_measured": int(tot.sum())}
timing = pd.DataFrame(timing)
timing.to_csv(OUT_TAB / "monthly_measurement_timing.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Within-year pattern: month means, change within periods, the spring peak
# ---------------------------------------------------------------------------
period_rows, monthly_means = [], []
for c in COHORTS:
    s = df[df.cohort == c]
    for cat in CATS + ["z_mean", "z_sd"]:
        y = ycol(cat)
        row = {"cohort": c, "outcome": cat, "mean": s[y].mean()}
        for name, (m0, m1) in PERIODS.items():
            f = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", s[(s.acad_month >= m0) & (s.acad_month <= m1)])
            ci = f.conf_int().loc["acad_month"]
            row.update({f"slope_{name}": f.params["acad_month"], f"ci_low_{name}": ci[0], f"ci_high_{name}": ci[1],
                        f"p_{name}": f.pvalues["acad_month"]})
        # autumn slope vs January-July slope: one model, period-specific year effects and slopes
        s2 = s.assign(autumn=(s.acad_month <= 4).astype(int))
        j = cluster_fit(f"{y} ~ C(ncmp_year):C(autumn) + acad_month:C(autumn)", s2)
        row["p_autumn_vs_jan_jul"] = float(j.t_test("acad_month:C(autumn)[1] - acad_month:C(autumn)[0] = 0").pvalue)
        # a single linear school-year trend, for comparison
        lin = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", s)
        row.update({"slope_linear": lin.params["acad_month"], "p_linear": lin.pvalues["acad_month"]})
        # spring peak: April against the mean of March and May in the same school year
        piv = year_by_month(df, c, y)
        peak = piv[APRIL] - (piv[MARCH] + piv[MAY]) / 2
        tt = stats.ttest_1samp(peak, 0)
        half = stats.t.ppf(0.975, len(peak) - 1) * peak.std(ddof=1) / np.sqrt(len(peak))
        row.update({"april_peak": peak.mean(), "april_peak_ci_low": peak.mean() - half,
                    "april_peak_ci_high": peak.mean() + half, "april_peak_p": tt.pvalue,
                    "april_peak_n_positive": int((peak > 0).sum())})
        # the same contrast after the Christmas holiday: January against December and February
        jan = piv[5] - (piv[4] + piv[6]) / 2
        half_j = stats.t.ppf(0.975, len(jan) - 1) * jan.std(ddof=1) / np.sqrt(len(jan))
        row.update({"january_peak": jan.mean(), "january_peak_ci_low": jan.mean() - half_j,
                    "january_peak_ci_high": jan.mean() + half_j, "january_peak_p": stats.ttest_1samp(jan, 0).pvalue,
                    "january_peak_n_positive": int((jan > 0).sum())})
        # month means: deviation from each school year's mean, and year-adjusted level
        dev = piv.sub(piv.mean(axis=1), axis=0)
        for mth in MONTHS:
            v = dev[mth].dropna()
            monthly_means.append({"cohort": c, "outcome": cat, "acad_month": mth, "month": ACAD_MONTHS[mth - 1],
                                  "deviation": v.mean(),
                                  "ci95": stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)),
                                  "level": v.mean() + piv.values.mean(), "n_years": len(v)})
        period_rows.append(row)
periods = pd.DataFrame(period_rows)
periods.to_csv(OUT_TAB / "monthly_within_year_change.csv", index=False)
monthly_means = pd.DataFrame(monthly_means)
monthly_means.to_csv(OUT_TAB / "monthly_means_by_month.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Shape of the within-year change (per cohort; school-year fixed effects)
# ---------------------------------------------------------------------------
def fit_shape(Y, X):
    """Y: years x months; X: months x k shape regressors (school-year effects removed by
    demeaning within year). Returns residual sum of squares and number of parameters."""
    ok = ~np.isnan(Y)
    Yd = Y - np.nanmean(Y, axis=1, keepdims=True)
    rows_, ys = [], []
    for i in range(Y.shape[0]):
        Xi = X[ok[i]]
        rows_.append(Xi - Xi.mean(axis=0, keepdims=True))
        ys.append(Yd[i, ok[i]] - Yd[i, ok[i]].mean())
    Xs, ys = np.vstack(rows_), np.concatenate(ys)
    e = ys - Xs @ np.linalg.lstsq(Xs, ys, rcond=None)[0] if Xs.shape[1] else ys
    return float(e @ e), Y.shape[0] + X.shape[1], len(ys)


def info(ssr, k, n):
    return n * np.log(ssr / n) + 2 * k, n * np.log(ssr / n) + np.log(n) * k


apr = (MONTHS == APRIL).astype(float)[:, None]
KNOTS = range(2, 11)          # last month of the decline: October ... June


def knot_x(k):
    return np.minimum(MONTHS, k)[:, None].astype(float)


def best_knot(Y, with_april=True):
    fits = [(fit_shape(Y, np.column_stack([knot_x(k), apr]) if with_april else knot_x(k))[0], k) for k in KNOTS]
    return min(fits)[1]


FORMS = {
    "linear school-year trend": lambda Y: MONTHS[:, None].astype(float),
    "linear trend + April": lambda Y: np.column_stack([MONTHS, apr]),
    "decline, then level": lambda Y: knot_x(best_knot(Y, with_april=False)),
    "decline, then level + April": lambda Y: np.column_stack([knot_x(best_knot(Y)), apr]),
    "linear trend + day length": lambda Y: np.column_stack([MONTHS, D_month]),
    "linear trend + day length + April": lambda Y: np.column_stack([MONTHS, D_month, apr]),
    "month effects": lambda Y: np.eye(11)[:, 1:],
}
shape_rows = []
for c in COHORTS:
    for y, label in SHAPE_OUTCOMES.items():
        Y = year_by_month(df, c, y).values
        row = {"cohort": c, "outcome": y, "label": label}
        for form, make in FORMS.items():
            ssr, k, n = fit_shape(Y, make(Y))
            k += 1 if form.startswith("decline") else 0           # the estimated end of the decline
            row[f"aic_{form}"], row[f"bic_{form}"] = info(ssr, k, n)
        row["end_of_decline"] = ACAD_MONTHS[best_knot(Y) - 1]
        boot = [best_knot(Y[rng.integers(0, len(years), len(years))]) for _ in range(1000)]
        counts = pd.Series(boot).value_counts(normalize=True)
        row["end_of_decline_boot_mode"] = ACAD_MONTHS[int(counts.idxmax()) - 1]
        row["end_of_decline_boot_dec_or_jan"] = float(counts.reindex([4, 5]).fillna(0).sum())
        row["end_of_decline_boot_range"] = "-".join(ACAD_MONTHS[int(v) - 1][:3] for v in np.percentile(boot, [2.5, 97.5]))
        # lack of fit of the decline-then-level form against unrestricted month effects
        kk = best_knot(Y)
        s_r, k_r, n_ = fit_shape(Y, np.column_stack([knot_x(kk), apr]))
        s_f, k_f, _ = fit_shape(Y, np.eye(11)[:, 1:])
        F = ((s_r - s_f) / (k_f - k_r)) / (s_f / (n_ - k_f))
        row["lack_of_fit_F"], row["lack_of_fit_p"] = F, stats.f.sf(F, k_f - k_r, n_ - k_f)
        shape_rows.append(row)
shape = pd.DataFrame(shape_rows)
shape.to_csv(OUT_TAB / "monthly_shape_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Location vs spread: does a shift of the mean reproduce each category's change?
# ---------------------------------------------------------------------------
shift_rows = []


def prevalence(mu, sd):
    return {"underweight": 100 * stats.norm.cdf((CUTPOINTS["underweight"] - mu) / sd),
            "owob": 100 * stats.norm.sf((CUTPOINTS["overweight"] - mu) / sd),
            "obese": 100 * stats.norm.sf((CUTPOINTS["obese"] - mu) / sd),
            "severe": 100 * stats.norm.sf((CUTPOINTS["severe"] - mu) / sd)}


for c in COHORTS:
    s = df[df.cohort == c]
    pr = periods[periods.cohort == c].set_index("outcome")
    mu_bar, sd_bar = s["z_mean"].mean(), s["z_sd"].mean()
    for name, months in (("sep_dec", 3), ("may_jul", 2)):          # change from first to last month of the period
        d_mu, d_sd = months * pr.loc["z_mean", f"slope_{name}"], months * pr.loc["z_sd", f"slope_{name}"]
        a_loc, b_loc = prevalence(mu_bar - d_mu / 2, sd_bar), prevalence(mu_bar + d_mu / 2, sd_bar)
        a_both, b_both = prevalence(mu_bar - d_mu / 2, sd_bar - d_sd / 2), prevalence(mu_bar + d_mu / 2, sd_bar + d_sd / 2)
        for k in a_loc:
            shift_rows.append({"cohort": c, "period": name, "outcome": k,
                               "observed_change": months * pr.loc[k, f"slope_{name}"],
                               "predicted_mean_shift_only": b_loc[k] - a_loc[k],
                               "predicted_mean_and_sd": b_both[k] - a_both[k],
                               "z_mean_change": d_mu, "z_sd_change": d_sd})
    results.setdefault("distribution", {})[c] = {
        "z_mean_overall": mu_bar, "z_sd_overall": sd_bar,
        "probit_fit_rmse_median": float(s["probit_fit_rmse"].median())}
shift = pd.DataFrame(shift_rows)
shift.to_csv(OUT_TAB / "monthly_distribution_shift.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Pattern shared by the two cohorts, and cohort-specific change
# ---------------------------------------------------------------------------
df["cy"] = df.cohort + "_" + df.ncmp_year
df["rec"] = (df.cohort == "Reception").astype(float)


def designs(data):
    """Explicit design matrices (avoids ambiguity about rank in nested F tests)."""
    fe = pd.get_dummies(data.cy, dtype=float).values
    m = data.acad_month.values
    rec = data.rec.values
    y6 = 1 - rec
    Dm = np.column_stack([(m == k).astype(float) for k in range(2, 12)])
    X0 = np.column_stack([fe, rec * m, y6 * m])                       # cohort-specific linear trends
    X1 = np.column_stack([X0, Dm])                                    # + month effects shared by cohorts
    X2 = np.column_stack([fe, rec[:, None] * Dm, y6[:, None] * Dm])   # cohort-specific month effects
    return X0, X1, X2


def nested_f(Xr, Xf, y):
    def ssr(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        return e @ e, np.linalg.matrix_rank(X)
    (s0, r0), (s1, r1) = ssr(Xr), ssr(Xf)
    df1, df2 = r1 - r0, len(y) - r1
    F = ((s0 - s1) / df1) / (s1 / df2)
    return float(F), int(df1), int(df2), float(stats.f.sf(F, df1, df2))


def detrend(v):
    return v - np.polyval(np.polyfit(MONTHS, v, 1), MONTHS)


def detrended_profiles(data, y):
    """Per cohort: deviation from each school year's mean, averaged over years, minus its own
    linear fit across school months."""
    return pd.DataFrame({c: pd.Series(detrend(year_by_month(data, c, y).pipe(
        lambda p: p.sub(p.mean(axis=1), axis=0)).mean().values), index=MONTHS) for c in COHORTS})


decomp_rows, shared_rows = [], []
P = np.polyfit(MONTHS, np.eye(11), 1)                                 # linear-fit operator
DETR = np.eye(11) - np.vander(MONTHS, 2) @ P                          # residual-maker for a linear fit
for y, label in SHAPE_OUTCOMES.items():
    X0, X1, X2 = designs(df)
    yy = df[y].values
    F_shared = nested_f(X0, X1, yy)
    F_cohort = nested_f(X1, X2, yy)
    fr = cluster_fit(f"{y} ~ C(cy) + C(acad_month) + rec:acad_month", df)
    prof = detrended_profiles(df, y)
    shared = prof.mean(axis=1)
    r_c, p_c = stats.pearsonr(prof["Reception"], prof["Year 6"])
    r_d, p_d = stats.pearsonr(shared, detrend(D_month))
    dev = {c: year_by_month(df, c, y).pipe(lambda p: p.sub(p.mean(axis=1), axis=0)).values for c in COHORTS}
    boots = np.array([np.mean([DETR @ dev[c][pick].mean(axis=0) for c in COHORTS], axis=0)
                      for pick in (rng.integers(0, len(years), len(years)) for _ in range(2000))])
    decomp_rows.append({
        "outcome": y, "label": label,
        "F_shared": F_shared[0], "df_shared": f"{F_shared[1]},{F_shared[2]}", "p_shared": F_shared[3],
        "F_cohort_diff": F_cohort[0], "df_cohort_diff": f"{F_cohort[1]},{F_cohort[2]}", "p_cohort_diff": F_cohort[3],
        "r_cohort_profiles": r_c, "p_cohort_profiles": p_c,
        "reception_extra_slope": fr.params["rec:acad_month"],
        "reception_extra_slope_ci_low": fr.conf_int().loc["rec:acad_month", 0],
        "reception_extra_slope_ci_high": fr.conf_int().loc["rec:acad_month", 1],
        "reception_extra_slope_p": fr.pvalues["rec:acad_month"],
        "shared_range": float(shared.max() - shared.min()),
        "shared_lowest": ACAD_MONTHS[int(shared.idxmin()) - 1], "shared_highest": ACAD_MONTHS[int(shared.idxmax()) - 1],
        "r_shared_vs_day_length": r_d, "p_shared_vs_day_length": p_d,
    })
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    for i, mth in enumerate(MONTHS):
        shared_rows.append({"outcome": y, "acad_month": mth, "month": ACAD_MONTHS[mth - 1],
                            "reception": prof.loc[mth, "Reception"], "year6": prof.loc[mth, "Year 6"],
                            "shared": shared.loc[mth], "shared_ci_low": lo[i], "shared_ci_high": hi[i],
                            "day_length_h": D_month[i], "day_length_detrended": detrend(D_month)[i]})
decomp = pd.DataFrame(decomp_rows)
decomp.to_csv(OUT_TAB / "monthly_decomposition.csv", index=False)
shared_tab = pd.DataFrame(shared_rows)
shared_tab.to_csv(OUT_TAB / "monthly_shared_component.csv", index=False)

# ---------------------------------------------------------------------------
# 6. The spring peak and the date of Easter
# ---------------------------------------------------------------------------
def bst_start(year):
    d = dt.date(year, 3, 31)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)            # last Sunday in March


EASTER_OUTCOMES = ["pct_obese", "pct_owob", "z_mean", "pct_underweight"]
erows = []
for yr in years:
    cal = int(yr[:4]) + 1
    e = easter(cal)
    row = {"ncmp_year": yr, "easter": e.isoformat(), "easter_doy": e.timetuple().tm_yday,
           "bst_start": bst_start(cal).isoformat(), "bst_start_doy": bst_start(cal).timetuple().tm_yday}
    for c in COHORTS:
        s = df[(df.cohort == c) & (df.ncmp_year == yr)].set_index("acad_month")
        for y in EASTER_OUTCOMES:
            d_ = s[y] - s[y].mean()
            for mth, nm in ((MARCH, "mar"), (APRIL, "apr"), (MAY, "may"), (10, "jun")):
                row[f"{c}|{y}|{nm}"] = d_.loc[mth]
            row[f"{c}|{y}|apr_minus_may"] = d_.loc[APRIL] - d_.loc[MAY]
        n = s["n_measured"] / s["n_measured"].sum() * 100
        row[f"{c}|april_share_pct"], row[f"{c}|march_share_pct"] = n.loc[APRIL], n.loc[MARCH]
    for y in EASTER_OUTCOMES:
        row[f"both|{y}|apr_minus_may"] = (row[f"Reception|{y}|apr_minus_may"] + row[f"Year 6|{y}|apr_minus_may"]) / 2
    erows.append(row)
east = pd.DataFrame(erows)
east.to_csv(OUT_TAB / "monthly_april_easter.csv", index=False)

median_doy = east.easter_doy.median()
early = east.easter_doy < median_doy
perm = np.array([rng.permutation(len(east)) for _ in range(20000)])
easter_tests = []
for c in COHORTS + ["both"]:
    for y in EASTER_OUTCOMES:
        v = east[f"{c}|{y}|apr_minus_may"].values
        x_ = east.easter_doy.values
        r, p = stats.pearsonr(x_, v)
        r_perm = np.array([np.corrcoef(x_, v[pp])[0, 1] for pp in perm])
        loo = [stats.pearsonr(np.delete(x_, i), np.delete(v, i))[0] for i in range(len(v))]
        tt = stats.ttest_ind(v[early], v[~early])
        r_bst = stats.pearsonr(east.bst_start_doy, v)[0]
        rec = {"cohort": c, "outcome": y, "r_easter": r, "p_easter": p,
               "p_easter_permutation": float((np.abs(r_perm) >= abs(r) - 1e-12).mean()),
               "r_leave_one_out_min": min(loo), "r_leave_one_out_max": max(loo),
               "early_mean": v[early].mean(), "late_mean": v[~early].mean(),
               "early_minus_late": v[early].mean() - v[~early].mean(), "p_early_vs_late": tt.pvalue,
               "r_bst_start": r_bst}
        if c != "both":
            for nm in ("mar", "apr", "may", "jun"):
                rec[f"r_easter_{nm}"] = stats.pearsonr(x_, east[f"{c}|{y}|{nm}"])[0]
                rec[f"p_easter_{nm}"] = stats.pearsonr(x_, east[f"{c}|{y}|{nm}"])[1]
        easter_tests.append(rec)
easter_tests = pd.DataFrame(easter_tests)
easter_tests.to_csv(OUT_TAB / "monthly_easter_tests.csv", index=False)

# Spring profiles (deviation from each year's January-July mean) in early- and late-Easter years
sp_rows = []
for c in COHORTS + ["both"]:
    for y in EASTER_OUTCOMES + ["pct_severe"]:
        piv = (year_by_month(df, "Reception", y) + year_by_month(df, "Year 6", y)) / 2 if c == "both" \
            else year_by_month(df, c, y)
        dev = piv.loc[:, 5:11].sub(piv.loc[:, 5:11].mean(axis=1), axis=0)
        for grp, mask in (("early", early.values), ("late", ~early.values)):
            g = dev[mask]
            for mth in range(5, 12):
                v = g[mth]
                sp_rows.append({"cohort": c, "outcome": y, "easter": grp, "acad_month": mth,
                                "month": ACAD_MONTHS[mth - 1], "deviation": v.mean(),
                                "ci95": stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))})
spring = pd.DataFrame(sp_rows)
spring.to_csv(OUT_TAB / "monthly_spring_by_easter.csv", index=False)
results["easter"] = {
    "median_easter_doy": float(median_doy), "n_early": int(early.sum()), "n_late": int((~early).sum()),
    "early_dates": east.loc[early, "easter"].tolist(), "late_dates": east.loc[~early, "easter"].tolist(),
    "bst_start_range": [east.bst_start.min(), east.bst_start.max()],
    "r_april_share_easter": {c: float(stats.pearsonr(east.easter_doy, east[f"{c}|april_share_pct"])[0]) for c in COHORTS},
    "r_march_share_easter": {c: float(stats.pearsonr(east.easter_doy, east[f"{c}|march_share_pct"])[0]) for c in COHORTS},
    "tests": easter_tests.to_dict(orient="records"),
}

# ---------------------------------------------------------------------------
# 7. Sensitivity analyses
# ---------------------------------------------------------------------------
sens = []
extra = df_all[(df_all.ncmp_year == "2019/20") & (df_all.acad_month <= 6)].copy()
extra["cy"] = extra.cohort + "_" + extra.ncmp_year
extra["rec"] = (extra.cohort == "Reception").astype(float)
samples = {"all 12 years (2007/08-2018/19)": df,
           "plus 2019/20 (September-February)": pd.concat([df, extra], ignore_index=True),
           "excluding September and July (fewest children measured)": df[df.acad_month.between(2, 10)],
           "excluding 2007/08 and 2008/09": df[~df.ncmp_year.isin(["2007/08", "2008/09"])],
           "2007/08-2013/14 (before universal infant free school meals)": df[df.ncmp_year < "2014/15"],
           "2014/15-2018/19 (after universal infant free school meals)": df[df.ncmp_year >= "2014/15"]}
for label, data in samples.items():
    data = data.reset_index(drop=True)
    row = {"sample": label, "months_per_cohort": int(len(data) / 2)}
    if data.acad_month.nunique() == 11:
        X0, X1, _ = designs(data)
        Fs = nested_f(X0, X1, data["z_mean"].values)
        row.update({"F_shared_z": Fs[0], "p_shared_z": Fs[3]})
    for c in COHORTS:
        s = data[data.cohort == c]
        for y in ["pct_obese", "z_mean"]:
            for name in ("sep_dec", "jan_jul"):
                m0, m1 = PERIODS[name]
                sub = s[(s.acad_month >= m0) & (s.acad_month <= m1)]
                if sub.acad_month.nunique() >= 2:
                    f = cluster_fit(f"{y} ~ C(ncmp_year) + acad_month", sub)
                    row[f"{c}|{y}|{name}"], row[f"{c}|{y}|{name}|p"] = f.params["acad_month"], f.pvalues["acad_month"]
    sens.append(row)
sens = pd.DataFrame(sens)
sens.to_csv(OUT_TAB / "monthly_sensitivity.csv", index=False)

results.update({
    "within_year": periods.set_index(["cohort", "outcome"]).to_dict(orient="index"),
    "shape": shape.set_index(["cohort", "outcome"]).to_dict(orient="index"),
    "distribution_shift": shift.to_dict(orient="records"),
    "decomposition": decomp.set_index("outcome").to_dict(orient="index"),
    "shared_component": shared_tab.to_dict(orient="records"),
    "sensitivity": sens.to_dict(orient="records"),
})
results = {k: ({"|".join(kk) if isinstance(kk, tuple) else kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v)
           for k, v in results.items()}
save_results("temporal", results)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
print("TIMING", {c: (results["timing"][c]["terms_pct"], results["timing"][c]["modal_month"]) for c in COHORTS})
cols = ["cohort", "outcome", "slope_sep_dec", "p_sep_dec", "slope_jan_mar", "p_jan_mar", "slope_may_jul", "p_may_jul",
        "slope_jan_jul", "p_jan_jul", "p_autumn_vs_jan_jul", "april_peak", "april_peak_p", "april_peak_n_positive",
        "january_peak", "january_peak_p"]
print("\nWITHIN-YEAR CHANGE BY PERIOD"); print(periods[cols].round(4).to_string(index=False))
aic_cols = [c for c in shape.columns if c.startswith("aic_")]
d_aic = shape[aic_cols].sub(shape[aic_cols].min(axis=1), axis=0).round(1)
d_aic.columns = [c[4:] for c in aic_cols]
print("\nSHAPE: delta AIC"); print(pd.concat([shape[["cohort", "outcome"]], d_aic], axis=1).to_string(index=False))
print(shape[["cohort", "outcome", "end_of_decline", "end_of_decline_boot_mode", "end_of_decline_boot_dec_or_jan",
             "end_of_decline_boot_range", "lack_of_fit_F", "lack_of_fit_p"]].round(3).to_string(index=False))
print("\nDISTRIBUTION SHIFT"); print(shift.round(3).to_string(index=False))
print("\nDECOMPOSITION"); print(decomp.round(4).T.to_string())
print("\nEASTER"); print(easter_tests.round(3).to_string(index=False))
print({k: v for k, v in results["easter"].items() if k != "tests"})
print("\nSENSITIVITY"); print(sens.round(4).T.to_string())
