"""Step 7: formatted tables for the manuscript (main Tables 1-4, supplementary Tables S1-S8).

Every number is read from outputs/ (results.json and the analysis tables) -- nothing is typed in.
Writes outputs/tables/formatted/*.csv and one workbook, outputs/tables/manuscript_tables.xlsx.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import (ACAD_MONTHS, INDICATORS, OUT_TAB, PRIMARY_YEAR, YEARS, day_length, load_results)
from photoperiod import compute

FMT = OUT_TAB / "formatted"
FMT.mkdir(exist_ok=True)
R = load_results()
COHORTS = ["Reception", "Year 6"]


def p_fmt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def est_ci(b, lo, hi, d=2):
    return f"{b:+.{d}f} ({lo:+.{d}f} to {hi:+.{d}f})"


tables = {}

# ---------------------------------------------------------------------------
# Table 1. Data analysed
# ---------------------------------------------------------------------------
la = R["la_dataset"]
ms = R["monthly_series"]
tm = R["temporal"]["timing"]
rows = []
rows.append(["National monthly series (NHS Digital Tables 3-4)", "", ""])
rows.append(["  School years analysed", "2007/08 to 2018/19 (12 complete years)", "2007/08 to 2018/19 (12 complete years)"])
rows.append(["  Children measured", f"{ms['children_measured_complete_years']['Reception']:,}",
             f"{ms['children_measured_complete_years']['Year 6']:,}"])
for term in ["Sep-Dec (autumn term)", "Jan-Mar (spring term)", "Apr-Jul (summer term)"]:
    rows.append([f"  Measured {term}, %", f"{tm['Reception']['terms_pct'][term]:.1f}", f"{tm['Year 6']['terms_pct'][term]:.1f}"])
rows.append(["  Modal month of measurement", tm["Reception"]["modal_month"], tm["Year 6"]["modal_month"]])
for c in COHORTS:
    pass
rows.append(["  Mean school month of measurement (Sep = 1), range across years",
             "{:.1f} ({:.1f}-{:.1f})".format(tm["Reception"]["mean_academic_month"], *tm["Reception"]["mean_academic_month_range_by_year"]),
             "{:.1f} ({:.1f}-{:.1f})".format(tm["Year 6"]["mean_academic_month"], *tm["Year 6"]["mean_academic_month_range_by_year"])])
for year in YEARS:
    s = la[year]
    rows.append([f"Local-authority data, {year} (OHID Fingertips)", "", ""])
    rows.append(["  Children measured (England)", f"{s['england']['rec_owob']['england_children_measured']:,}",
                 f"{s['england']['yr6_owob']['england_children_measured']:,}"])
    for suffix, lab in (("uw", "Underweight"), ("owob", "Overweight incl. obesity"), ("ob", "Obesity incl. severe"),
                        ("sev", "Severe obesity")):
        cells = []
        for pre in ("rec", "yr6"):
            k = f"{pre}_{suffix}"
            cells.append(f"{s['england'][k]['england_prevalence']:.1f} ({s['la_ranges'][k][0]:.1f}-{s['la_ranges'][k][1]:.1f}); "
                         f"N = {s['n_by_indicator'][k]}")
        rows.append([f"  {lab}: England % (LA range); LAs analysed"] + cells)
s = la[PRIMARY_YEAR]
rows.append(["Local-authority covariates (151 LAs)", "", ""])
rows.append(["  Latitude of LA centroid, degrees N: mean (range)",
             "{:.1f} ({:.1f}-{:.1f})".format(s["latitude_mean_sd"][0], *s["latitude_range"]), ""])
rows.append(["  Annual range of day length, hours (range across LAs)",
             "{:.1f}-{:.1f}".format(*s["photoperiod_amplitude_range_h"]), ""])
rows.append(["  IMD 2019 average score: mean (range)", "{:.1f} ({:.1f}-{:.1f})".format(s["imd_mean_sd"][0], *s["imd_range"]), ""])
for k, lab in (("pct_asian", "Asian"), ("pct_black", "Black"), ("pct_mixed", "Mixed"), ("pct_other_ethnic", "Other"),
               ("pct_white", "White")):
    v = s["ethnicity_mean_range"][k]
    rows.append([f"  Dependent children who are {lab}, %: mean (range)", f"{v[0]:.1f} ({v[1]:.1f}-{v[2]:.1f})", ""])
t1 = pd.DataFrame(rows, columns=["", "Reception (age 4-5)", "Year 6 (age 10-11)"])
tables["Table_1_data"] = t1

# ---------------------------------------------------------------------------
# Table 2. Within-school-year change and the September reset
# ---------------------------------------------------------------------------
tr = pd.read_csv(OUT_TAB / "monthly_within_year_trends.csv")
reset = pd.read_csv(OUT_TAB / "monthly_september_reset.csv")
labels = {"underweight": "Underweight", "healthy": "Healthy weight", "overweight": "Overweight", "obese": "Obesity",
          "severe": "Severe obesity", "owob": "Overweight incl. obesity", "z_mean": "Mean BMI z-score",
          "z_sd": "SD of BMI z-score"}
rows = []
for c in COHORTS:
    for cat in ["underweight", "healthy", "overweight", "obese", "severe", "owob", "z_mean", "z_sd"]:
        t = tr[(tr.cohort == c) & (tr.outcome == cat)].iloc[0]
        d = 3 if cat.startswith("z_") else 2
        rr = reset[(reset.cohort == c) & (reset.outcome == cat)]
        reset_txt = (f"{rr.observed_mean_change.iloc[0]:+.2f} ({rr.ci_low.iloc[0]:+.2f} to {rr.ci_high.iloc[0]:+.2f}); "
                     f"{rr.n_positive.iloc[0] if rr.observed_mean_change.iloc[0] > 0 else rr.n_transitions.iloc[0] - rr.n_positive.iloc[0]}/12"
                     if len(rr) else "")
        rows.append([c, labels[cat], f"{t['mean']:.{d}f}", est_ci(t.slope_per_month, t.slope_ci_low, t.slope_ci_high, d + 1),
                     p_fmt(t.slope_p), f"{t.slope_sep_dec:+.{d + 1}f} ({p_fmt(t.p_sep_dec)})",
                     f"{t.slope_jan_jul:+.{d + 1}f} ({p_fmt(t.p_jan_jul)})", f"{10 * t.slope_per_month:+.{d}f}", reset_txt])
t2 = pd.DataFrame(rows, columns=["Cohort", "Outcome", "Mean (% or z)", "Within-year slope per month (95% CI)", "p",
                                 "Slope Sep-Dec (p)", "Slope Jan-Jul (p)", "Fitted change Sep to Jul",
                                 "July to next September: mean change (95% CI); years in that direction"])
tables["Table_2_within_year"] = t2

# ---------------------------------------------------------------------------
# Table 3. Day length versus position in the school year
# ---------------------------------------------------------------------------
comp = pd.read_csv(OUT_TAB / "monthly_model_comparison.csv")
rows = []
for c in COHORTS:
    for cat in ["z_mean", "obese", "owob", "underweight"]:
        r = comp[(comp.cohort == c) & (comp.outcome == cat)].iloc[0]
        d = 4 if cat == "z_mean" else 3
        rows.append([
            c, labels[cat],
            f"{r['adjR2_year FE only']:.2f}", f"{r['adjR2_photoperiod']:.2f}", f"{r['adjR2_school-year trend']:.2f}",
            f"{r['adjR2_trend + photoperiod']:.2f}", f"{r['adjR2_month FE (saturated)']:.2f}",
            f"{r.photo_coef_alone:+.{d}f}",
            f"{r.photo_coef_given_trend:+.{d}f} ({p_fmt(r.photo_coef_given_trend_clustered_p)})",
            f"{r.photo_coef_given_quadratic_trend:+.{d}f} ({p_fmt(r.photo_coef_given_quadratic_trend_clustered_p)})",
            f"{r.photo_coef_given_two_phase_trend:+.{d}f} ({p_fmt(r.photo_coef_given_two_phase_trend_clustered_p)})",
        ])
t3 = pd.DataFrame(rows, columns=["Cohort", "Outcome", "Adj. R2: year only", "+ day length", "+ linear school-year trend",
                                 "+ trend and day length", "+ month (saturated)",
                                 "Day-length coefficient, alone", "given linear trend (p)", "given quadratic trend (p)",
                                 "given two-phase trend (p)"])
tables["Table_3_day_length_vs_school_year"] = t3

# ---------------------------------------------------------------------------
# Table 4. Latitude coefficients, Models A and B, both years
# ---------------------------------------------------------------------------
gm = pd.read_csv(OUT_TAB / "geo_models.csv")
rows = []
for key, (_, cohort, label, short) in INDICATORS.items():
    for year in YEARS:
        row = [cohort, short, year]
        for model in ("A", "B"):
            g = gm[(gm.indicator == key) & (gm.year == year) & (gm.model == model)].iloc[0]
            row += [f"{g.lat_beta:+.2f} ({g.lat_ci_low:+.2f} to {g.lat_ci_high:+.2f})",
                    f"{p_fmt(g.lat_p)} / {p_fmt(g.lat_p_holm)} / {p_fmt(g.lat_p_conley100)}"]
        rows.append(row)
t4 = pd.DataFrame(rows, columns=["Cohort", "Outcome", "NCMP year",
                                 "Model A: beta, pp per degree N (95% CI)", "Model A: p / Holm p / spatial-HAC p",
                                 "Model B: beta, pp per degree N (95% CI)", "Model B: p / Holm p / spatial-HAC p"])
tables["Table_4_latitude"] = t4

# ---------------------------------------------------------------------------
# Supplementary tables
# ---------------------------------------------------------------------------
timing = pd.read_csv(OUT_TAB / "monthly_measurement_timing.csv")
ts = timing.assign(share=lambda d: d.apply(lambda r: f"{r.share_pct:.1f} ({r.share_min_year:.1f}-{r.share_max_year:.1f})", axis=1))
ts = ts.pivot_table(index="month", columns="cohort", values=["children", "share"], aggfunc="first").reindex(ACAD_MONTHS)
ts.columns = [f"{c} {'children measured' if v == 'children' else '% of year (range across years)'}" for v, c in ts.columns]
ts = ts[["Reception children measured", "Reception % of year (range across years)", "Year 6 children measured",
         "Year 6 % of year (range across years)"]].reset_index().rename(columns={"month": "Month"})
ts["Day length at 52.5 N on the 15th (h)"] = [f"{R['monthly_series']['photoperiod_52_5N_by_month_h'][m]:.1f}" for m in ACAD_MONTHS]
tables["Table_S1_measurement_timing"] = ts

reorg = pd.read_csv(OUT_TAB / "imd_reorganised_las.csv")
tables["Table_S2_imd_reorganised_LAs"] = reorg.round(3)

sup = pd.read_csv(OUT_TAB / "geo_suppression_bounds_2023_24.csv")
sup["latitude beta (p)"] = sup.apply(lambda r: f"{r.lat_beta:+.3f} ({p_fmt(r.lat_p)})", axis=1)
tables["Table_S3_suppression_bounds"] = sup[["model", "indicator", "n_suppressed", "treatment", "n", "latitude beta (p)"]]

rob = pd.read_csv(OUT_TAB / "geo_robustness.csv")
rob["txt"] = rob.apply(lambda r: f"{r.lat_beta:+.2f} ({p_fmt(r.lat_p)})", axis=1)
rob_w = rob.pivot_table(index=["year", "spec"], columns="indicator", values="txt", aggfunc="first", sort=False)[list(INDICATORS)]
rob_w.columns = [f"{INDICATORS[k][1]} {INDICATORS[k][3]}" for k in rob_w.columns]
tables["Table_S4_robustness"] = rob_w.reset_index()

mor = pd.read_csv(OUT_TAB / "geo_morans_i.csv")
mor["I without latitude (p)"] = mor.apply(lambda r: f"{r.morans_I_without_latitude:.3f} ({p_fmt(r.p_without_latitude)})", axis=1)
mor["I with latitude (p)"] = mor.apply(lambda r: f"{r.morans_I_with_latitude:.3f} ({p_fmt(r.p_with_latitude)})", axis=1)
tables["Table_S5_morans_I"] = mor[["year", "model", "indicator", "I without latitude (p)", "I with latitude (p)"]]

cos = pd.read_csv(OUT_TAB / "monthly_cosinor.csv")
cos_t = pd.DataFrame({
    "Cohort": cos.cohort, "Outcome": cos.outcome.map(labels),
    "Amplitude, pp (95% CI)": cos.apply(lambda r: f"{r.amplitude:.2f} ({max(r.amplitude_ci_low, 0):.2f}-{r.amplitude_ci_high:.2f})", axis=1),
    "Peak (acrophase)": cos.peak_date, "Zero-amplitude F test p": cos.zero_amplitude_p.map(p_fmt),
    "Linear trend, pp per year": cos.trend_per_year.map(lambda v: f"{v:+.3f}"), "R2": cos.r2.map(lambda v: f"{v:.2f}")})
tables["Table_S6_cosinor"] = cos_t

cmp_cols = ["year FE only", "photoperiod", "sinusoid", "school-year trend", "quadratic trend", "two-phase trend",
            "trend + photoperiod", "trend + sinusoid", "quadratic trend + photoperiod", "two-phase trend + photoperiod",
            "month FE (saturated)"]
aic = comp[["cohort", "outcome"] + [f"aic_{c}" for c in cmp_cols]].copy()
best = aic[[f"aic_{c}" for c in cmp_cols]].min(axis=1)
for c in cmp_cols:
    aic[f"aic_{c}"] = (aic[f"aic_{c}"] - best).map(lambda v: f"{v:.1f}")
aic.columns = ["Cohort", "Outcome"] + [f"dAIC: {c}" for c in cmp_cols]
aic["Outcome"] = aic["Outcome"].map(labels)
tables["Table_S7_model_comparison_AIC"] = aic

# Photoperiod model check: CBM (used in the paper) vs the NOAA solar-position algorithm
val = []
for lat in (50.4, 51.5, 52.5, 53.5, 54.5, 55.3):
    diffs = []
    for i in range(366):
        d = date(2024, 1, 1) + timedelta(days=i)
        r = compute(lat, -1.5, d)
        diffs.append(60 * (r["day_length_cbm_h"] - r["day_length_noaa_h"]))
    diffs = np.array(diffs)
    val.append({"Latitude (N)": lat, "Shortest day, CBM (h)": f"{day_length(lat, 356):.2f}",
                "Longest day, CBM (h)": f"{day_length(lat, 172):.2f}",
                "Mean CBM - NOAA (min)": f"{diffs.mean():+.2f}", "Max |CBM - NOAA| (min)": f"{np.abs(diffs).max():.2f}"})
tables["Table_S8_photoperiod_model_check"] = pd.DataFrame(val)

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUT_TAB / "manuscript_tables.xlsx") as xl:
    for name, t in tables.items():
        t.to_csv(FMT / f"{name}.csv", index=False)
        t.to_excel(xl, sheet_name=name[:31], index=False)
print("wrote", len(tables), "tables:", ", ".join(tables))
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
for name in ["Table_1_data", "Table_2_within_year", "Table_3_day_length_vs_school_year", "Table_4_latitude",
             "Table_S8_photoperiod_model_check"]:
    print("\n" + name)
    print(tables[name].to_string(index=False))
