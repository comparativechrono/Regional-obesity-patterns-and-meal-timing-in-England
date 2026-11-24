"""Step 7: formatted tables for the manuscript (main Tables 1-4, supplementary Tables S1-S12).

Every number is read from outputs/ (results.json and the analysis tables) -- nothing is typed in.
Writes outputs/tables/formatted/*.csv and one workbook, outputs/tables/manuscript_tables.xlsx.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common import ACAD_MONTHS, INDICATORS, OUT_DATA, OUT_TAB, YEARS, day_length, load_results
from photoperiod import compute

FMT = OUT_TAB / "formatted"
FMT.mkdir(exist_ok=True)
R = load_results()
COHORTS = ["Reception", "Year 6"]
LABELS = {"underweight": "Underweight", "healthy": "Healthy weight", "overweight": "Overweight", "obese": "Obesity",
          "severe": "Severe obesity", "owob": "Overweight incl. obesity", "z_mean": "Mean BMI z-score",
          "z_sd": "SD of BMI z-score", "pct_obese": "Obesity", "pct_owob": "Overweight incl. obesity",
          "pct_underweight": "Underweight", "pct_severe": "Severe obesity"}


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
rows.append(["  Measured Sep-Dec (autumn term), % (range across years)"] + [
    "{:.1f} ({:.1f}-{:.1f})".format(tm[c]["terms_pct"]["Sep-Dec (autumn term)"], *tm[c]["autumn_term_pct_range_by_year"])
    for c in COHORTS])
for term in ["Jan-Mar (spring term)", "Apr-Jul (summer term)"]:
    rows.append([f"  Measured {term}, %", f"{tm['Reception']['terms_pct'][term]:.1f}", f"{tm['Year 6']['terms_pct'][term]:.1f}"])
rows.append(["  Modal month of measurement", tm["Reception"]["modal_month"], tm["Year 6"]["modal_month"]])
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
s = la[YEARS[0]]
rows.append(["Local-authority characteristics (151 LAs; the same in every year)", "", ""])
rows.append(["  Latitude of LA centroid, degrees N: mean (range)",
             "{:.1f} ({:.1f}-{:.1f})".format(s["latitude_mean_sd"][0], *s["latitude_range"]), ""])
rows.append(["  Annual range of day length, hours (range across LAs)",
             "{:.1f}-{:.1f}".format(*s["photoperiod_amplitude_range_h"]), ""])
rows.append(["  IMD 2019 average score: mean (range)", "{:.1f} ({:.1f}-{:.1f})".format(s["imd_mean_sd"][0], *s["imd_range"]), ""])
for k, lab in (("pct_asian", "Asian"), ("pct_black", "Black"), ("pct_mixed", "Mixed"), ("pct_other_ethnic", "Other"),
               ("pct_white", "White")):
    v = s["ethnicity_mean_range"][k]
    rows.append([f"  Dependent children who are {lab}, %: mean (range)", f"{v[0]:.1f} ({v[1]:.1f}-{v[2]:.1f})", ""])
tables["Table_1_data"] = pd.DataFrame(rows, columns=["", "Reception (age 4-5)", "Year 6 (age 10-11)"])

# ---------------------------------------------------------------------------
# Table 2. Change while days are shortening and while they are lengthening; the April peak
# ---------------------------------------------------------------------------
wy = pd.read_csv(OUT_TAB / "monthly_within_year_change.csv")
rows = []
for c in COHORTS:
    for cat in ["underweight", "owob", "obese", "severe", "z_mean", "z_sd"]:
        t = wy[(wy.cohort == c) & (wy.outcome == cat)].iloc[0]
        d = 3 if cat.startswith("z_") else 2
        rows.append([c, LABELS[cat], f"{t['mean']:.{d}f}",
                     est_ci(t.chg_short, t.chg_short_ci_low, t.chg_short_ci_high, d + 1),
                     est_ci(t.chg_long, t.chg_long_ci_low, t.chg_long_ci_high, d + 1),
                     f"{p_fmt(t.p_short_vs_long)}; {int(t.n_years_same_direction)}/12",
                     f"{est_ci(t.april_peak, t.april_peak_ci_low, t.april_peak_ci_high, d)}; {int(t.april_peak_n_positive)}/12"])
tables["Table_2_within_year"] = pd.DataFrame(rows, columns=[
    "Cohort", "Outcome", "Mean (% or z)", "Days shortening: change per month (95% CI)",
    "Days lengthening: change per month (95% CI)", "Difference: p; years in that direction",
    "April minus mean of March and May (95% CI); years higher"])

# ---------------------------------------------------------------------------
# Table 3. The shortening-days model
# ---------------------------------------------------------------------------
shape = pd.read_csv(OUT_TAB / "monthly_shape_comparison.csv")
coh = pd.read_csv(OUT_TAB / "monthly_shortening_cohort_comparison.csv").set_index("outcome")
OUTC = {"pct_obese": "obese", "pct_owob": "owob", "pct_underweight": "underweight", "z_mean": "z_mean"}
all_forms = [c[4:] for c in shape.columns if c.startswith("aic_")]
rows = []
for c in COHORTS:
    for y, cat in OUTC.items():
        t = wy[(wy.cohort == c) & (wy.outcome == cat)].iloc[0]
        sr = shape[(shape.cohort == c) & (shape.outcome == y)].iloc[0]
        d = 4 if cat == "z_mean" else 2
        aic = {f: sr[f"aic_{f}"] for f in all_forms}
        rows.append([c, LABELS[cat], f"{est_ci(t.per_short, t.per_short_ci_low, t.per_short_ci_high, d)}; {p_fmt(t.per_short_p)}",
                     f"{est_ci(t.steady, t.steady_ci_low, t.steady_ci_high, d)}; {p_fmt(t.steady_p)}",
                     p_fmt(coh.loc[y, "p_cohort_difference"]),
                     f"{aic['single linear trend'] - aic['shortening days + linear trend + April']:.1f}",
                     f"{aic['shortening days + linear trend + April'] - aic['month effects']:+.1f}",
                     p_fmt(sr.lack_of_fit_p)])
tables["Table_3_shortening_days_model"] = pd.DataFrame(rows, columns=[
    "Cohort", "Outcome", "Change per month of shortening days (95% CI); p", "Steady change per school month (95% CI); p",
    "Same rate in both cohorts: p", "AIC improvement over a single linear trend", "AIC relative to month effects",
    "Lack of fit vs month effects: p"])

# ---------------------------------------------------------------------------
# Table 4. Latitude coefficients, 2023/24-2025/26 pooled, Models A and B
# ---------------------------------------------------------------------------
gm = pd.read_csv(OUT_TAB / "geo_models.csv")
rows = []
for key, (_, cohort, label, short) in INDICATORS.items():
    a = gm[(gm.indicator == key) & (gm.model == "A")].iloc[0]
    row = [cohort, short, f"{int(a.n_obs)} ({int(a.n_las)})"]
    for model in ("A", "B"):
        g = gm[(gm.indicator == key) & (gm.model == model)].iloc[0]
        row += [f"{g.lat_beta:+.2f} ({g.lat_ci_low:+.2f} to {g.lat_ci_high:+.2f})",
                f"{p_fmt(g.lat_p)} / {p_fmt(g.lat_p_holm)} / {p_fmt(g.lat_p_conley100)}", f"{100 * g.delta_r2:.1f}"]
    rows.append(row)
tables["Table_4_latitude"] = pd.DataFrame(rows, columns=[
    "Cohort", "Outcome", "Authority-years (authorities)",
    "Model A: beta, pp per degree N (95% CI)", "Model A: p / Holm p / spatial-HAC p", "Model A: R2 added by latitude, %",
    "Model B: beta, pp per degree N (95% CI)", "Model B: p / Holm p / spatial-HAC p", "Model B: R2 added by latitude, %"])

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

LABEL = {k: f"{v[1]} {v[3]}" for k, v in INDICATORS.items()}
LA_NAMES = pd.read_csv(OUT_DATA / "la_dataset_2023_24.csv").set_index("la_code")["la_name"]

reorg = pd.read_csv(OUT_TAB / "imd_reorganised_las.csv")
tables["Table_S2_imd_reorganised_LAs"] = pd.DataFrame({
    "Authority (code)": [f"{LA_NAMES.get(c, c)} ({c})" for c in reorg.new_code],
    "Method": reorg.method, "IMD 2019 average score": reorg.imd_score.round(2),
    "IDACI average score": reorg.idaci_score.round(3)})

sup = pd.read_csv(OUT_TAB / "geo_suppression_bounds.csv")
tables["Table_S3_suppression_bounds"] = pd.DataFrame({
    "Model": sup.model, "Indicator": sup.indicator.map(LABEL), "Authority-years suppressed": sup.n_suppressed,
    "Treatment of suppressed values": sup.treatment, "Authority-years analysed": sup.n_obs,
    "Latitude beta, pp per degree (p)": sup.apply(lambda r: f"{r.lat_beta:+.3f} ({p_fmt(r.lat_p)})", axis=1)})

rob = pd.read_csv(OUT_TAB / "geo_robustness.csv")
rob["txt"] = rob.apply(lambda r: f"{r.lat_beta:+.2f} ({p_fmt(r.lat_p)})", axis=1)
rob_w = rob.pivot_table(index="spec", columns="indicator", values="txt", aggfunc="first", sort=False)[list(INDICATORS)]
rob_w.columns = [LABEL[k] for k in rob_w.columns]
tables["Table_S4_robustness"] = rob_w.reset_index().rename(columns={"spec": "Specification"})

mor = pd.read_csv(OUT_TAB / "geo_morans_i.csv")
tables["Table_S5_morans_I"] = pd.DataFrame({
    "Model": mor.model, "Indicator": mor.indicator.map(LABEL),
    "Moran's I, model without latitude (p)": mor.apply(
        lambda r: f"{r.morans_I_without_latitude:.3f} ({p_fmt(r.p_without_latitude)})", axis=1),
    "Moran's I, model with latitude (p)": mor.apply(
        lambda r: f"{r.morans_I_with_latitude:.3f} ({p_fmt(r.p_with_latitude)})", axis=1)})

gy = pd.read_csv(OUT_TAB / "geo_models_by_year.csv")
rows = []
for key in INDICATORS:
    for model in ("A", "B"):
        row = [LABEL[key], model]
        for year in YEARS:
            g = gy[(gy.indicator == key) & (gy.model == model) & (gy.year == year)].iloc[0]
            row.append(f"{g.lat_beta:+.2f} ({g.lat_ci_low:+.2f} to {g.lat_ci_high:+.2f}); N = {int(g.n)}")
        rows.append(row)
tables["Table_S6_latitude_by_year"] = pd.DataFrame(rows, columns=["Indicator", "Model"] + [
    f"{y}: beta, pp per degree N (95% CI)" for y in YEARS])

dec = pd.read_csv(OUT_TAB / "monthly_decomposition.csv")
tables["Table_S7_shared_and_cohort_specific"] = pd.DataFrame({
    "Outcome": dec.label,
    "Month pattern shared by the cohorts: F (df), p": dec.apply(lambda r: f"{r.F_shared:.1f} ({r.df_shared}), {p_fmt(r.p_shared)}", axis=1),
    "Cohorts' patterns differ: F (df), p": dec.apply(lambda r: f"{r.F_cohort_diff:.2f} ({r.df_cohort_diff}), {p_fmt(r.p_cohort_diff)}", axis=1),
    "r between the cohorts' non-linear patterns": dec.r_cohort_profiles.map(lambda v: f"{v:.2f}"),
    "Shared pattern: lowest / highest month": dec.shared_lowest + " / " + dec.shared_highest,
    "r of shared pattern with day length": dec.apply(lambda r: f"{r.r_shared_vs_day_length:+.2f} ({p_fmt(r.p_shared_vs_day_length)})", axis=1),
    "Reception minus Year 6 linear change per month (95% CI)": dec.apply(
        lambda r: est_ci(r.reception_extra_slope, r.reception_extra_slope_ci_low, r.reception_extra_slope_ci_high,
                         4 if r.outcome == "z_mean" else 3), axis=1)})

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

sens = pd.read_csv(OUT_TAB / "monthly_sensitivity.csv")


def sl(r, col, d=3):
    return f"{r[col]:+.{d}f} ({p_fmt(r[col + '|p'])})" if pd.notna(r.get(col)) else ""


tables["Table_S9_temporal_sensitivity"] = pd.DataFrame({
    "Sample": sens["sample"], "Months per cohort": sens.months_per_cohort,
    "Reception mean z: per month of shortening days (p)": sens.apply(lambda r: sl(r, "Reception|z_mean|per_short", 4), axis=1),
    "Reception mean z: steady change (p)": sens.apply(lambda r: sl(r, "Reception|z_mean|steady", 4), axis=1),
    "Year 6 mean z: per month of shortening days (p)": sens.apply(lambda r: sl(r, "Year 6|z_mean|per_short", 4), axis=1),
    "Reception obesity: per month of shortening days (p)": sens.apply(lambda r: sl(r, "Reception|pct_obese|per_short"), axis=1),
    "Year 6 obesity: per month of shortening days (p)": sens.apply(lambda r: sl(r, "Year 6|pct_obese|per_short"), axis=1),
    "Year 6 underweight: per month of shortening days (p)": sens.apply(lambda r: sl(r, "Year 6|pct_underweight|per_short"), axis=1),
    "Shared month pattern, mean z: F (p)": sens.apply(
        lambda r: f"{r.F_shared_z:.1f} ({p_fmt(r.p_shared_z)})" if pd.notna(r.F_shared_z) else "", axis=1)})

east = pd.read_csv(OUT_TAB / "monthly_april_easter.csv")
fmt_day = lambda iso: date.fromisoformat(iso).strftime("%-d %b %Y")
tables["Table_S10_easter_by_year"] = pd.DataFrame({
    "School year": east.ncmp_year, "Easter Sunday": east.easter.map(fmt_day),
    "British Summer Time began": east.bst_start.map(fmt_day),
    "Reception children measured in April, % of year": east["Reception|april_share_pct"].map(lambda v: f"{v:.1f}"),
    "Obesity, April minus May (pp): Reception": east["Reception|pct_obese|apr_minus_may"].map(lambda v: f"{v:+.2f}"),
    "Year 6": east["Year 6|pct_obese|apr_minus_may"].map(lambda v: f"{v:+.2f}"),
    "Mean of both cohorts": east["both|pct_obese|apr_minus_may"].map(lambda v: f"{v:+.2f}")})

et = pd.read_csv(OUT_TAB / "monthly_easter_tests.csv")
tables["Table_S11_easter_tests"] = pd.DataFrame({
    "Cohort": et.cohort.replace({"both": "Mean of both cohorts"}), "Outcome": et.outcome.map(LABELS),
    "r with date of Easter (permutation p)": et.apply(lambda r: f"{r.r_easter:+.2f} ({p_fmt(r.p_easter_permutation)})", axis=1),
    "r, leaving out one year at a time": et.apply(lambda r: f"{r.r_leave_one_out_min:+.2f} to {r.r_leave_one_out_max:+.2f}", axis=1),
    "Early-Easter years: mean April minus May": et.apply(lambda r: f"{r.early_mean:+.3f}", axis=1),
    "Late-Easter years: mean April minus May": et.apply(lambda r: f"{r.late_mean:+.3f}", axis=1),
    "Difference (p)": et.apply(lambda r: f"{r.early_minus_late:+.3f} ({p_fmt(r.p_early_vs_late)})", axis=1),
    "r with date British Summer Time began": et.r_bst_start.map(lambda v: f"{v:+.2f}")})

rows = []
for _, r in shape.iterrows():
    best = min(r[f"aic_{f}"] for f in all_forms)
    rows.append([r.cohort, LABELS[r.outcome]] + [f"{r[f'aic_{f}'] - best:.1f}" for f in all_forms] +
                [f"{r.end_of_decline} ({100 * r.end_of_decline_boot_dec_or_jan:.0f}%)"])
tables["Table_S12_within_year_forms"] = pd.DataFrame(rows, columns=["Cohort", "Outcome"] + [f"dAIC: {f}" for f in all_forms] +
                                                     ["Decline, then level: end month (bootstrap % December or January)"])

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
for old in FMT.glob("*.csv"):
    old.unlink()
with pd.ExcelWriter(OUT_TAB / "manuscript_tables.xlsx") as xl:
    for name, t in tables.items():
        t.to_csv(FMT / f"{name}.csv", index=False)
        t.to_excel(xl, sheet_name=name[:31], index=False)
    # fixed document timestamps so that re-running the pipeline produces an identical file
    from datetime import datetime
    if hasattr(xl.book, "set_properties"):          # xlsxwriter engine
        xl.book.set_properties({"created": datetime(2026, 11, 24)})
    else:                                           # openpyxl engine
        xl.book.properties.created = xl.book.properties.modified = datetime(2026, 11, 24)
print("wrote", len(tables), "tables:", ", ".join(tables))
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
for name in ["Table_2_within_year", "Table_3_shortening_days_model", "Table_4_latitude", "Table_S9_temporal_sensitivity",
             "Table_S12_within_year_forms"]:
    print("\n" + name)
    print(tables[name].to_string(index=False))
