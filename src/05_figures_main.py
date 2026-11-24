"""Step 5: main-text figures (Figures 1-6)."""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from scipy import stats

from common import OUT_DATA, OUT_TAB, YEARS, day_length
from plotstyle import (AXIS, COHORT_COLOUR, DIVERGING, DOUBLE_COL_IN, GRID, INK, INK2, MONTH_ABBR,
                       MUTED, REC, YR6, save, titled)

m_all = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
m = m_all[m_all.complete_year].copy()
COHORTS = ["Reception", "Year 6"]
x = np.arange(1, 12)


def deviation_from_year_mean(df, col):
    """Per-year deviation from that school year's mean; mean and 95% CI across years."""
    piv = df.pivot_table(index="ncmp_year", columns="acad_month", values=col)
    d = piv.sub(piv.mean(axis=1), axis=0)
    n = d.count()
    mean = d.mean()
    ci = stats.t.ppf(0.975, n - 1) * d.std() / np.sqrt(n)
    return mean, ci.fillna(0)


# ===========================================================================
# Figure 1. When children are measured, and day length through the school year
# ===========================================================================
timing = pd.read_csv(OUT_TAB / "monthly_measurement_timing.csv")
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.6), gridspec_kw={"wspace": 0.32})
ax = axes[0]
for c in COHORTS:
    t = timing[timing.cohort == c]
    ax.plot(x, t.share_pct, "-o", color=COHORT_COLOUR[c], label=c, markersize=4)
    ax.fill_between(x, t.share_min_year, t.share_max_year, color=COHORT_COLOUR[c], alpha=0.12, lw=0)
ax.set_xticks(x, MONTH_ABBR)
ax.set_ylabel("Children measured (% of school-year total)")
ax.set_ylim(0, None)
ax.legend(loc="upper left")
titled(ax, "A", "Share of each cohort measured, by month")

ax = axes[1]
start = dt.date(2017, 9, 1)
days = [start + dt.timedelta(days=i) for i in range((dt.date(2018, 7, 31) - start).days + 1)]
doy = np.array([d.timetuple().tm_yday for d in days])
pos = np.array([(d - start).days for d in days])
lat_cols = {55.3: "#104281", 52.5: "#3987e5", 50.4: "#86b6ef"}
for lat, col in lat_cols.items():
    ax.plot(pos, day_length(lat, doy), color=col, lw=1.6, label=f"{lat}°N")
solstice = (dt.date(2017, 12, 21) - start).days
ax.axvline(solstice, color=MUTED, lw=0.6)
ax.text(solstice + 3, 16.6, "winter solstice", color=INK2, fontsize=6.5, va="top")
month_starts = [(dt.date(2017 + (mm < 9), mm, 1) - start).days for mm in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]]
ax.set_xticks([p + 15 for p in month_starts], MONTH_ABBR, fontsize=6.3)
ax.set_ylabel("Day length (hours)")
ax.set_ylim(6.5, 18)
ax.legend(loc="upper left", title="Latitude", title_fontsize=7)
titled(ax, "B", "Day length, September to July")
save(fig, "Figure_1_measurement_timing_and_day_length")

# ===========================================================================
# Figure 2. The whole BMI distribution shifts through the school year
# ===========================================================================
fig = plt.figure(figsize=(DOUBLE_COL_IN, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
panels = [("pct_underweight", "Underweight"), ("pct_owob", "Overweight + obesity"), ("pct_obese", "Obesity"),
          ("pct_severe", "Severe obesity"), ("z_mean", "Mean BMI z-score")]
for i, (col, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    for c in COHORTS:
        mean, ci = deviation_from_year_mean(m[m.cohort == c], col)
        off = -0.12 if c == "Reception" else 0.12
        ax.errorbar(mean.index + off, mean.values, yerr=ci.values, fmt="-o", color=COHORT_COLOUR[c], ms=3.5,
                    lw=1.4, elinewidth=0.8, capsize=0, label=c)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x, MONTH_ABBR, rotation=90)
    ax.set_ylabel("Deviation from year mean (SD)" if col == "z_mean" else "Deviation from year mean (pp)")
    titled(ax, "ABCDE"[i], title)
    if i == 0:
        ax.legend(loc="upper left")

# Panel F: observed Sep->Jul change vs change predicted by a pure shift of the mean
shift = pd.read_csv(OUT_TAB / "monthly_distribution_shift.csv")
ax = fig.add_subplot(gs[1, 2])
cats = ["underweight", "owob", "obese", "severe"]
labels = ["UW", "OW+OB", "Obese", "Severe"]
for j, c in enumerate(COHORTS):
    s = shift[shift.cohort == c].set_index("outcome").loc[cats]
    xs = np.arange(len(cats)) + (-0.15 if j == 0 else 0.15)
    ax.scatter(xs, s.observed_sep_to_jul, color=COHORT_COLOUR[c], s=22, zorder=3)
    ax.scatter(xs, s.predicted_mean_shift_only, facecolor="white", edgecolor=COHORT_COLOUR[c], s=22, lw=1.1, zorder=3)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xticks(range(len(cats)), labels)
ax.set_ylabel("Fitted change, Sep to Jul (pp)")
titled(ax, "F", "Observed vs mean shift only")
ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=INK2, ms=4.5, label="observed"),
                   Line2D([], [], marker="o", lw=0, mfc="white", mec=INK2, ms=4.5, label="mean shift only")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
save(fig, "Figure_2_distribution_shift")

# ===========================================================================
# Figure 3. Does the within-year pattern follow day length or the school year?
# ===========================================================================
resid = pd.read_csv(OUT_TAB / "monthly_obesity_residuals_after_trend.csv")
reset = pd.read_csv(OUT_TAB / "monthly_september_reset.csv")
fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.36)
for k, c in enumerate(COHORTS):
    ax = fig.add_subplot(gs[0, k])
    s = m[m.cohort == c]
    fit = smf.ols("pct_obese ~ C(ncmp_year) + acad_month", data=s).fit()
    # month means after removing year-to-year level differences (re-centred on the overall mean)
    fe = fit.params.filter(like="C(ncmp_year)").reindex(["C(ncmp_year)[T.%s]" % y for y in s.ncmp_year]).fillna(0).values
    s = s.assign(adj=s.pct_obese - fe + fe.mean())
    g = s.groupby("acad_month")["adj"]
    mean = g.mean(); ci = stats.t.ppf(0.975, g.count() - 1) * g.std() / np.sqrt(g.count())
    ax.axvspan(0.5, 4.5, color="#f0efec", lw=0)
    ax.text(2.5, 0.98, "days shortening", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6, color=INK2)
    ax.text(8, 0.98, "days lengthening", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6, color=INK2)
    ax.errorbar(x, mean, yerr=ci, fmt="o", color=COHORT_COLOUR[c], ms=4, elinewidth=0.8, capsize=0, zorder=3)
    line = fit.params["Intercept"] + fe.mean() + fit.params["acad_month"] * x
    ax.plot(x, line, color=INK2, lw=1.0, zorder=2, label="linear school-year trend")
    ax.set_xticks(x, MONTH_ABBR)
    ax.tick_params(axis="x", labelsize=6.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    ax.set_ylabel("Obesity prevalence (%), year-adjusted")
    titled(ax, "AB"[k], f"{c} obesity, by month measured")
    ax.legend(loc="lower left")

# C: the September reset
ax = fig.add_subplot(gs[1, 0])
rows = [("Reception", "owob", "Reception OW+OB"), ("Reception", "obese", "Reception obesity"),
        ("Year 6", "owob", "Year 6 OW+OB"), ("Year 6", "obese", "Year 6 obesity")]
for i, (c, cat, lab) in enumerate(rows):
    r = reset[(reset.cohort == c) & (reset.outcome == cat)].iloc[0]
    yv = len(rows) - 1 - i
    col = COHORT_COLOUR[c]
    ax.plot([r.ci_low, r.ci_high], [yv, yv], color=col, lw=1.4, solid_capstyle="round")
    ax.scatter(r.observed_mean_change, yv, color=col, s=26, zorder=4)
    ax.scatter(r.pred_school_year_trend, yv + 0.22, marker="s", facecolor="white", edgecolor=INK2, s=18, lw=0.9, zorder=3)
    ax.scatter(r.pred_photoperiod_alone, yv - 0.22, marker="^", facecolor="white", edgecolor=INK2, s=20, lw=0.9, zorder=3)
    ax.scatter(r.pred_photoperiod_coef_given_trend, yv - 0.22, marker="v", facecolor="white", edgecolor=INK2, s=20,
               lw=0.9, zorder=3)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_yticks(range(len(rows)), [r[2] for r in rows][::-1])
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_xlabel("Change from July to the following September (pp)")
titled(ax, "C", "The September reset (12 transitions)")
ax.legend(handles=[Line2D([], [], color=INK2, marker="o", lw=1.4, ms=4, label="observed (95% CI)"),
                   Line2D([], [], marker="s", lw=0, mfc="white", mec=INK2, ms=4, label="school-year trend predicts"),
                   Line2D([], [], marker="^", lw=0, mfc="white", mec=INK2, ms=4.5, label="day length predicts (alone)"),
                   Line2D([], [], marker="v", lw=0, mfc="white", mec=INK2, ms=4.5, label="day length predicts (given trend)")],
          loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6, handletextpad=0.3, columnspacing=1.0)

# D: residual after the linear school-year trend vs day length
ax = fig.add_subplot(gs[1, 1])
offsets = {1: (4, 2), 2: (4, 2), 3: (4, -1), 4: (4, -3), 5: (-14, -6), 6: (4, -6), 7: (4, -3), 8: (4, 2), 9: (-15, 3),
           10: (3, 3), 11: (4, 1)}
for c in COHORTS:
    r = resid[resid.cohort == c]
    sl = stats.linregress(r.photo, r.resid)
    ax.scatter(r.photo, r.resid, color=COHORT_COLOUR[c], s=18, zorder=3, label=f"{c} (r = {sl.rvalue:+.2f})")
    xx = np.linspace(7.5, 17, 10)
    ax.plot(xx, sl.intercept + sl.slope * xx, color=COHORT_COLOUR[c], lw=1.0)
    if c == "Reception":
        for _, row in r.iterrows():
            ax.annotate(MONTH_ABBR[int(row.acad_month) - 1], (row.photo, row.resid), xytext=offsets[int(row.acad_month)],
                        textcoords="offset points", fontsize=5.5, color=INK2)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xlabel("Day length at 52.5°N on the 15th of the month (hours)")
ax.set_ylabel("Residual after linear trend (pp)")
titled(ax, "D", "Residual obesity vs day length")
ax.legend(loc="lower right", fontsize=6)
save(fig, "Figure_3_school_year_vs_day_length")

# ===========================================================================
# Figure 4. Double-plotted actograms (phase locked to the school year)
# ===========================================================================
def actogram_matrix(cohort, col):
    s = m[m.cohort == cohort]
    years = sorted(s.ncmp_year.unique())
    z = {}
    for y in years:
        v = s[s.ncmp_year == y].sort_values("acad_month")[col].values
        z[y] = (v - v.mean()) / v.std(ddof=0)
    mat = np.array([np.concatenate([z[years[i]], z[years[i + 1]]]) for i in range(len(years) - 1)])
    labels = [f"{years[i]} | {years[i + 1]}" for i in range(len(years) - 1)]
    return mat, labels


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 3.4), gridspec_kw={"wspace": 0.08})
specs = [("Reception", "pct_obese", "Reception obesity", 1), ("Reception", "pct_underweight", "Reception underweight†", -1),
         ("Year 6", "pct_obese", "Year 6 obesity", 1)]
for k, (c, col, title, sign) in enumerate(specs):
    ax = axes[k]
    mat, labels = actogram_matrix(c, col)
    im = ax.imshow(sign * mat, aspect="auto", cmap=DIVERGING, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.axvline(10.5, color="white", lw=2.5)
    ax.set_xticks([0, 3, 6, 9, 11, 14, 17, 20], ["Sep", "Dec", "Mar", "Jun"] * 2, rotation=90)
    ax.set_yticks(range(len(labels)), labels if k == 0 else [])
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    titled(ax, "ABC"[k], title)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Within-year z-score (red = higher-BMI\ndirection; † underweight colour inverted)", fontsize=6.5, color=INK2)
cb.outline.set_visible(False)
save(fig, "Figure_4_actograms")

# ===========================================================================
# Figure 5. Latitude coefficients: Model A (IMD) and Model B (IMD + child ethnic composition)
# ===========================================================================
gm = pd.read_csv(OUT_TAB / "geo_models.csv")
fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL_IN, 3.9), gridspec_kw={"wspace": 0.3})
thresholds = [("uw", "Underweight"), ("owob", "Overweight + obesity"), ("ob", "Obesity"), ("sev", "Severe obesity")]
rows = [(coh, yr) for coh in ("rec", "yr6") for yr in YEARS]
for k, (suffix, title) in enumerate(thresholds):
    ax = axes[k]
    for i, (coh, yr) in enumerate(rows):
        key = f"{coh}_{suffix}"
        col = REC if coh == "rec" else YR6
        base = len(rows) - 1 - i + (0.4 if coh == "rec" else 0)
        for model, dy, filled in (("A", 0.17, False), ("B", -0.17, True)):
            p = gm[(gm.indicator == key) & (gm.year == yr) & (gm.model == model)].iloc[0]
            yv = base + dy
            ax.plot([p.conley100_ci_low, p.conley100_ci_high], [yv, yv], color=col, lw=1.2 if filled else 0.9,
                    solid_capstyle="round", alpha=1 if filled else 0.75)
            ax.scatter(p.lat_beta, yv, s=20 if filled else 18, zorder=3, facecolor=col if filled else "white",
                       edgecolor=col, lw=1.0)
    ax.axvline(0, color=AXIS, lw=0.8)
    ticks = [len(rows) - 1 - i + (0.4 if c == "rec" else 0) for i, (c, _) in enumerate(rows)]
    ax.set_yticks(ticks, [f"{'Reception' if c == 'rec' else 'Year 6'} {y}" for c, y in rows] if k == 0 else [])
    ax.set_ylim(-0.6, len(rows) - 0.0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlabel("pp per degree north")
    titled(ax, "ABCD"[k], title)
handles = [Line2D([], [], color=INK2, marker="o", lw=0.9, ms=4, mfc="white", label="Model A: adjusted for IMD"),
           Line2D([], [], color=INK2, marker="o", lw=1.2, ms=4, label="Model B: adjusted for IMD and child ethnic composition")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
save(fig, "Figure_5_latitude_coefficients")

# ===========================================================================
# Figure 6. Maps: what latitude stands in for (2023/24)
# ===========================================================================
import maps  # noqa: E402

ETH = ["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"]
la = pd.read_csv(OUT_DATA / "la_dataset_2023_24.csv")


def residual_map_values(key, covariates):
    d = la[la[key].notna()]
    f = smf.ols(f"{key} ~ " + " + ".join(covariates), data=d).fit()
    return dict(zip(d.la_code, f.resid))


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 3.7), gridspec_kw={"wspace": 0.04})
panels = [("rec_uw", ["imd_score"], "Reception underweight\n      adjusted for IMD"),
          ("rec_uw", ["imd_score"] + ETH, "Reception underweight\n      adjusted for IMD + ethnicity"),
          ("yr6_sev", ["imd_score"] + ETH, "Year 6 severe obesity\n      adjusted for IMD + ethnicity")]
for k, (key, covs, title) in enumerate(panels):
    ax = axes[k]
    vals = residual_map_values(key, covs)
    v = 1.2 if key == "rec_uw" else float(np.nanpercentile(np.abs(list(vals.values())), 97))
    sm = maps.draw(ax, vals, DIVERGING, -v, v)
    ax.set_title(f"{'ABC'[k]}   {title}", loc="left", fontsize=8, fontweight="semibold", color=INK, pad=4)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=22)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Residual prevalence (pp)", fontsize=6.5, color=INK2)
save(fig, "Figure_6_residual_maps")
