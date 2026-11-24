"""Step 5: main-text figures (Figures 1-6)."""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from scipy import stats

from common import OUT_DATA, OUT_TAB, YEARS, day_length, load_results
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


def shortening_spans():
    """x-ranges (school-month axis, mid-month = integer) when days are shortening: September to the
    winter solstice, and the summer solstice to the end of July."""
    doy_mid = m.groupby("acad_month")["mid_month_doy"].first().reindex(x).values.astype(float)
    daily = day_length(52.5, np.arange(1, 366))
    winter, summer = float(np.argmin(daily) + 1), float(np.argmax(daily) + 1)
    pos_w = 4 + (winter - doy_mid[3]) / 31          # December (x = 4) to January (x = 5)
    pos_s = 10 + (summer - doy_mid[9]) / 30         # June (x = 10) to July (x = 11)
    return [(0.5, pos_w), (pos_s, 11.5)]


SHORT_SPANS = shortening_spans()


def shade_shortening(ax, labels=False, fontsize=6):
    for a, b in SHORT_SPANS:
        ax.axvspan(a, b, color="#f0efec", lw=0, zorder=0)
    if labels:
        tr = ax.get_xaxis_transform()
        ax.text(sum(SHORT_SPANS[0]) / 2, 0.98, "days shortening", transform=tr, ha="center", va="top", fontsize=fontsize, color=INK2)
        ax.text((SHORT_SPANS[0][1] + SHORT_SPANS[1][0]) / 2, 0.98, "days lengthening", transform=tr, ha="center", va="top",
                fontsize=fontsize, color=INK2)
        ax.text(sum(SHORT_SPANS[1]) / 2, 0.98, "short-\nening", transform=tr, ha="center", va="top", fontsize=fontsize - 0.5,
                color=INK2, linespacing=0.9)


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
    shade_shortening(ax)
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
        from matplotlib.patches import Patch
        h, l_ = ax.get_legend_handles_labels()
        fig.legend(h + [Patch(facecolor="#f0efec", edgecolor=AXIS, lw=0.5)], l_ + ["days shortening"], loc="lower left",
                   bbox_to_anchor=(0.1, -0.035), ncol=3, fontsize=7)

# Panel F: observed change over the autumn term (Sep->Dec) vs change predicted by a shift of the mean alone
shift = pd.read_csv(OUT_TAB / "monthly_distribution_shift.csv")
shift = shift[shift.period == "sep_dec"]
ax = fig.add_subplot(gs[1, 2])
cats = ["underweight", "owob", "obese", "severe"]
labels = ["UW", "OW+OB", "Obese", "Severe"]
for j, c in enumerate(COHORTS):
    s_ = shift[shift.cohort == c].set_index("outcome").loc[cats]
    xs = np.arange(len(cats)) + (-0.15 if j == 0 else 0.15)
    ax.scatter(xs, s_.observed_change, color=COHORT_COLOUR[c], s=22, zorder=3)
    ax.scatter(xs, s_.predicted_mean_shift_only, facecolor="white", edgecolor=COHORT_COLOUR[c], s=22, lw=1.1, zorder=3)
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xticks(range(len(cats)), labels)
ax.set_ylabel("Change, September to December (pp)")
titled(ax, "F", "Autumn term: observed vs mean shift")
ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=INK2, ms=4.5, label="observed"),
                   Line2D([], [], marker="o", lw=0, mfc="white", mec=INK2, ms=4.5, label="mean shift only")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
save(fig, "Figure_2_distribution_shift")

# ===========================================================================
# Figure 3. Shape of the within-year change, and the spring peak and the date of Easter
# ===========================================================================
R = load_results()
means = pd.read_csv(OUT_TAB / "monthly_means_by_month.csv")
shape = pd.read_csv(OUT_TAB / "monthly_shape_comparison.csv").set_index(["cohort", "outcome"])
spring = pd.read_csv(OUT_TAB / "monthly_spring_by_easter.csv")
east = pd.read_csv(OUT_TAB / "monthly_april_easter.csv")
etests = pd.read_csv(OUT_TAB / "monthly_easter_tests.csv").set_index(["cohort", "outcome"])
MONTH_NUM = {mm: i + 1 for i, mm in enumerate(["September", "October", "November", "December", "January", "February",
                                               "March", "April", "May", "June", "July"])}


def fitted_profile(cohort, col, X):
    """Profile fitted with school-year fixed effects (data demeaned within year); returned on the
    level scale of the year-adjusted monthly means."""
    piv = m[m.cohort == cohort].pivot_table(index="ncmp_year", columns="acad_month", values=col)
    Y = piv.sub(piv.mean(axis=1), axis=0).values
    Xd = X - X.mean(axis=0)
    b, *_ = np.linalg.lstsq(np.tile(Xd, (Y.shape[0], 1)), Y.ravel(), rcond=None)
    return Xd @ b + piv.values.mean()


fig = plt.figure(figsize=(DOUBLE_COL_IN, 7.9))
gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.34)
for r_, c in enumerate(COHORTS):
    for k, (cat, lab) in enumerate((("obese", "obesity"), ("underweight", "underweight"))):
        ax = fig.add_subplot(gs[r_, k])
        mm_ = means[(means.cohort == c) & (means.outcome == cat)].sort_values("acad_month")
        shade_shortening(ax, labels=True)
        ax.errorbar(x, mm_.level, yerr=mm_.ci95, fmt="-o", color=COHORT_COLOUR[c], ms=3.8, lw=1.0, elinewidth=0.8,
                    capsize=0, zorder=3)
        ax.set_xticks(x, MONTH_ABBR)
        ax.tick_params(axis="x", labelsize=6.3)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.2 * (hi - lo))
        ax.set_ylabel(f"{lab.capitalize()} (%), year-adjusted")
        titled(ax, "ABCD"[2 * r_ + k], f"{c} {lab}, by month measured")

# E: spring profiles in years with an early and a late Easter (both cohorts averaged)
ax = fig.add_subplot(gs[2, 0])
sp = spring[(spring.cohort == "both") & (spring.outcome == "pct_obese")]
dates = R["temporal"]["easter"]


def date_range(isodates):
    ds = sorted(dt.date.fromisoformat(d).replace(year=2021) for d in isodates)   # by day of year
    return f"{ds[0].strftime('%-d %b')} to {ds[-1].strftime('%-d %b')}"


early_lab = f"Easter {date_range(dates['early_dates'])} (6 years)"
late_lab = f"Easter {date_range(dates['late_dates'])} (6 years)"
styles = {"early": dict(color=INK, ls="-", mfc=INK, label=early_lab, off=-0.08),
          "late": dict(color=MUTED, ls=(0, (4, 2)), mfc="white", label=late_lab, off=0.08)}
for grp, st in styles.items():
    g = sp[sp.easter == grp].sort_values("acad_month")
    ax.errorbar(g.acad_month + st["off"], g.deviation, yerr=g.ci95, color=st["color"], ls=st["ls"], marker="o",
                mfc=st["mfc"], mec=st["color"], ms=4, lw=1.2, elinewidth=0.7, capsize=0, label=st["label"])
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xticks(range(5, 12), MONTH_ABBR[4:])
ax.set_ylabel("Obesity, deviation from Jan-Jul mean (pp)")
titled(ax, "E", "Obesity, January to July, by date of Easter")
ax.legend(loc="lower left", fontsize=6.3)

# F: April minus May against the date of Easter
ax = fig.add_subplot(gs[2, 1])
v = east["both|pct_obese|apr_minus_may"]
is_early = east.easter_doy < dates["median_easter_doy"]
ax.scatter(east.easter_doy[is_early], v[is_early], s=26, color=INK, zorder=3)
ax.scatter(east.easter_doy[~is_early], v[~is_early], s=26, facecolor="white", edgecolor=MUTED, lw=1.1, zorder=3)
sl = stats.linregress(east.easter_doy, v)
xx = np.linspace(east.easter_doy.min() - 2, east.easter_doy.max() + 2, 10)
ax.plot(xx, sl.intercept + sl.slope * xx, color=INK2, lw=0.9)
ax.axhline(0, color=AXIS, lw=0.8)
tick_dates = [dt.date(2021, 3, 22), dt.date(2021, 4, 1), dt.date(2021, 4, 10), dt.date(2021, 4, 20)]
ax.set_xticks([d.timetuple().tm_yday for d in tick_dates], [d.strftime("%-d %b") for d in tick_dates])
t = etests.loc[("both", "pct_obese")]
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.3 * (hi - lo))
ax.text(0.97, 0.97, f"r = {t.r_easter:.2f}, permutation p = {t.p_easter_permutation:.3f}; 12 school years".replace("-", "−"),
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
ax.set_xlabel("Date of Easter Sunday")
ax.set_ylabel("Obesity, April minus May (pp)")
titled(ax, "F", "April minus May, by date of Easter")
save(fig, "Figure_3_within_year_shape_and_easter")

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
# Figure 5. Latitude coefficients, 2023/24-2025/26 pooled: Model A (IMD) and Model B (+ ethnic composition)
# ===========================================================================
gm = pd.read_csv(OUT_TAB / "geo_models.csv")
gy = pd.read_csv(OUT_TAB / "geo_models_by_year.csv")
fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL_IN, 2.9), gridspec_kw={"wspace": 0.3})
thresholds = [("uw", "Underweight"), ("owob", "Overweight + obesity"), ("ob", "Obesity"), ("sev", "Severe obesity")]
rows = [("rec", "A"), ("rec", "B"), ("yr6", "A"), ("yr6", "B")]
for k, (suffix, title) in enumerate(thresholds):
    ax = axes[k]
    for i, (coh, model) in enumerate(rows):
        key = f"{coh}_{suffix}"
        col = REC if coh == "rec" else YR6
        yv = len(rows) - 1 - i + (0.35 if coh == "rec" else 0)
        p = gm[(gm.indicator == key) & (gm.model == model)].iloc[0]
        filled = model == "B"
        ax.plot([p.conley100_ci_low, p.conley100_ci_high], [yv, yv], color=col, lw=1.3, solid_capstyle="round")
        ax.scatter(p.lat_beta, yv, s=24, zorder=3, facecolor=col if filled else "white", edgecolor=col, lw=1.1)
        yr = gy[(gy.indicator == key) & (gy.model == model)]
        ax.scatter(yr.lat_beta, np.full(len(yr), yv - 0.22), s=7, marker="|", color=INK2, lw=0.8, zorder=2)
    ax.axvline(0, color=AXIS, lw=0.8)
    ticks = [len(rows) - 1 - i + (0.35 if c == "rec" else 0) for i, (c, _) in enumerate(rows)]
    ax.set_yticks(ticks, [f"{'Reception' if c == 'rec' else 'Year 6'}, Model {mdl}" for c, mdl in rows] if k == 0 else [])
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_xlabel("pp per degree north")
    titled(ax, "ABCD"[k], title)
handles = [Line2D([], [], color=INK2, marker="o", lw=1.3, ms=4.5, mfc="white", label="Model A: adjusted for IMD"),
           Line2D([], [], color=INK2, marker="o", lw=1.3, ms=4.5, label="Model B: adjusted for IMD and child ethnic composition"),
           Line2D([], [], color=INK2, marker="|", lw=0, ms=6, label="each NCMP year separately")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=6.5)
save(fig, "Figure_5_latitude_coefficients")

# ===========================================================================
# Figure 6. Maps: what latitude stands in for (authority means of residuals, 2023/24-2025/26)
# ===========================================================================
import maps  # noqa: E402

ETH = ["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"]
panel_la = pd.concat([pd.read_csv(OUT_DATA / f"la_dataset_{y.replace('/', '_')}.csv").assign(year=y) for y in YEARS],
                     ignore_index=True)


def residual_map_values(key, covariates):
    """Residuals from the pooled model without latitude (year effects + covariates), averaged
    over the three years within each authority."""
    d = panel_la[panel_la[key].notna()]
    f = smf.ols(f"{key} ~ C(year) + " + " + ".join(covariates), data=d).fit()
    return d.assign(r=f.resid).groupby("la_code")["r"].mean().to_dict()


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 3.7), gridspec_kw={"wspace": 0.04})
panels = [("rec_uw", ["imd_score"], "Reception underweight\n      adjusted for IMD"),
          ("rec_uw", ["imd_score"] + ETH, "Reception underweight\n      adjusted for IMD + ethnicity"),
          ("yr6_sev", ["imd_score"] + ETH, "Year 6 severe obesity\n      adjusted for IMD + ethnicity")]
for k, (key, covs, title) in enumerate(panels):
    ax = axes[k]
    vals = residual_map_values(key, covs)
    v = 1.0 if key == "rec_uw" else float(np.nanpercentile(np.abs(list(vals.values())), 97))
    sm = maps.draw(ax, vals, DIVERGING, -v, v)
    ax.set_title(f"{'ABC'[k]}   {title}", loc="left", fontsize=8, fontweight="semibold", color=INK, pad=4)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=20, shrink=0.82)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Residual prevalence (pp), 3-year mean", fontsize=6.5, color=INK2)
save(fig, "Figure_6_residual_maps")
