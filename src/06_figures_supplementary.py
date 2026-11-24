"""Step 6: supplementary figures (S1-S10), covering the full BMI spectrum."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import INDICATORS, OUT_DATA, day_length, photoperiod_amplitude
from plotstyle import (AXIS, COHORT_COLOUR, DIVERGING, DOUBLE_COL_IN, GRID, INK2, MONTH_ABBR, MUTED,
                       save, titled)

m_all = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
m = m_all[m_all.complete_year].copy()
la = pd.read_csv(OUT_DATA / "la_dataset_2023_24.csv")
COHORTS = ["Reception", "Year 6"]
CATS = [("underweight", "Underweight"), ("healthy", "Healthy weight"), ("overweight", "Overweight"),
        ("obese", "Obesity"), ("severe", "Severe obesity")]


def row_labels(fig, axes_rows, labels):
    """Write the cohort name to the left of each row of panels."""
    for axrow, lab in zip(axes_rows, labels):
        bb = axrow[0].get_position()
        fig.text(0.0, (bb.y0 + bb.y1) / 2, lab, rotation=90, ha="left", va="center", fontsize=8.5,
                 fontweight="semibold", color="#0b0b0b")


x = np.arange(1, 12)
LETTERS = "ABCDEFGHIJKLMNOP"

# ---------------------------------------------------------------------------
# S1. Day length across England's latitude range
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.6), gridspec_kw={"wspace": 0.3})
doy = np.arange(1, 366)
lo, hi = la.latitude.min(), la.latitude.max()
ax = axes[0]
ax.fill_between(doy, day_length(lo, doy), day_length(hi, doy), color="#b7d3f6", lw=0)
ax.plot(doy, day_length(hi, doy), color="#104281", lw=1.4, label=f"northernmost LA ({hi:.1f}°N)")
ax.plot(doy, day_length(lo, doy), color="#3987e5", lw=1.4, label=f"southernmost LA ({lo:.1f}°N)")
ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
              ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Day length (hours)")
ax.legend(loc="upper left", fontsize=6.5)
titled(ax, "A", "Day length through the calendar year")
ax = axes[1]
ax.scatter(la.latitude, la.photoperiod_amplitude_h, s=10, color="#2a78d6")
ax.set_xlabel("LA centroid latitude (°N)")
ax.set_ylabel("Annual range of day length (hours)")
r = np.corrcoef(la.latitude, la.photoperiod_amplitude_h)[0, 1]
ax.text(0.03, 0.95, f"r = {r:.3f}, N = {len(la)}", transform=ax.transAxes, va="top", fontsize=7, color=INK2)
titled(ax, "B", "Seasonal photoperiod range by local authority")
save(fig, "Figure_S1_day_length_by_latitude")

# ---------------------------------------------------------------------------
# S2. Monthly prevalence of every BMI category (levels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL_IN, 3.9), gridspec_kw={"wspace": 0.45, "hspace": 0.6})
for r_, c in enumerate(COHORTS):
    for k, (cat, lab) in enumerate(CATS):
        ax = axes[r_, k]
        g = m[m.cohort == c].groupby("acad_month")[f"pct_{cat}"]
        mean, ci = g.mean(), stats.t.ppf(0.975, g.count() - 1) * g.std() / np.sqrt(g.count())
        ax.errorbar(x, mean, yerr=ci, fmt="-o", ms=2.5, lw=1.1, elinewidth=0.7, color=COHORT_COLOUR[c])
        ax.set_xticks([1, 4, 7, 10], ["Sep", "Dec", "Mar", "Jun"])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Prevalence (%)")
        titled(ax, LETTERS[r_ * 5 + k], lab if r_ == 0 else lab)
        ax.title.set_fontsize(6.8)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S2_monthly_prevalence_all_categories")

# ---------------------------------------------------------------------------
# S3. Double-plotted actograms, every category and cohort
# ---------------------------------------------------------------------------
def actogram(cohort, col):
    s = m[m.cohort == cohort]
    years = sorted(s.ncmp_year.unique())
    z = {}
    for y in years:
        v = s[s.ncmp_year == y].sort_values("acad_month")[col].values
        z[y] = (v - v.mean()) / v.std(ddof=0)
    return np.array([np.concatenate([z[years[i]], z[years[i + 1]]]) for i in range(len(years) - 1)]), years


fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL_IN, 4.6), gridspec_kw={"wspace": 0.08, "hspace": 0.35})
for r_, c in enumerate(COHORTS):
    for k, (cat, lab) in enumerate(CATS):
        ax = axes[r_, k]
        mat, years = actogram(c, f"pct_{cat}")
        sign = -1 if cat in ("underweight", "healthy") else 1
        im = ax.imshow(sign * mat, aspect="auto", cmap=DIVERGING, vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.axvline(10.5, color="white", lw=2)
        ax.set_xticks([0, 6, 11, 17], ["Sep", "Mar", "Sep", "Mar"], fontsize=5.5)
        ax.set_yticks(range(len(years) - 1), [f"{years[i][2:]} | {years[i + 1][2:]}" for i in range(len(years) - 1)] if k == 0 else [])
        ax.tick_params(axis="y", labelsize=5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        titled(ax, LETTERS[r_ * 5 + k], {"Healthy weight": "Healthy"}.get(lab, lab) + ("†" if sign < 0 else ""))
        ax.title.set_fontsize(6.5)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("Within-year z-score (red = higher-BMI direction; † inverted)", fontsize=6.5, color=INK2)
cb.outline.set_visible(False)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S3_actograms_all_categories")

# ---------------------------------------------------------------------------
# S4. Waveform overlays: each year's within-year profile and the grand mean
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL_IN, 3.9), gridspec_kw={"wspace": 0.45, "hspace": 0.6})
for r_, c in enumerate(COHORTS):
    for k, (cat, lab) in enumerate(CATS):
        ax = axes[r_, k]
        s = m[m.cohort == c]
        piv = s.pivot_table(index="ncmp_year", columns="acad_month", values=f"pct_{cat}")
        dev = piv.sub(piv.mean(axis=1), axis=0)
        for _, row in dev.iterrows():
            ax.plot(x, row.values, color=COHORT_COLOUR[c], alpha=0.25, lw=0.7)
        ax.plot(x, dev.mean().values, color="#0b0b0b", lw=1.6)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_xticks([1, 4, 7, 10], ["Sep", "Dec", "Mar", "Jun"])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Deviation from year mean (pp)")
        titled(ax, LETTERS[r_ * 5 + k], lab)
        ax.title.set_fontsize(6.8)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S4_waveform_overlays")

# ---------------------------------------------------------------------------
# S5. Linear cosinor fits to the full monthly series
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(DOUBLE_COL_IN, 3.9), gridspec_kw={"wspace": 0.45, "hspace": 0.6})
for r_, c in enumerate(COHORTS):
    for k, (cat, lab) in enumerate(CATS):
        ax = axes[r_, k]
        s = m[m.cohort == c].copy()
        s["tc"] = s.t_years - s.t_years.mean()
        s["cosw"], s["sinw"] = np.cos(2 * np.pi * s.t_years), np.sin(2 * np.pi * s.t_years)
        f = smf.ols(f"pct_{cat} ~ tc + cosw + sinw", data=s).fit()
        tt = np.linspace(s.t_years.min(), s.t_years.max(), 800)
        pred = f.params["Intercept"] + f.params["tc"] * (tt - s.t_years.mean()) + f.params["cosw"] * np.cos(2 * np.pi * tt) + f.params["sinw"] * np.sin(2 * np.pi * tt)
        ax.scatter(s.t_years, s[f"pct_{cat}"], s=3, color=COHORT_COLOUR[c], alpha=0.6, lw=0)
        ax.plot(tt, pred, color="#0b0b0b", lw=0.8)
        amp = np.hypot(f.params["cosw"], f.params["sinw"])
        ax.text(0.03, 0.97, f"A = {amp:.2f} pp", transform=ax.transAxes, va="top", fontsize=6, color=INK2)
        ax.set_xticks([2008, 2012, 2016, 2019], ["2008", "2012", "2016", "2019"])
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel("Prevalence (%)")
        titled(ax, LETTERS[r_ * 5 + k], lab)
        ax.title.set_fontsize(6.8)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S5_cosinor_fits")

# ---------------------------------------------------------------------------
# S6-S8. Geographic: raw associations, added-variable plots, residual maps (2023/24)
# ---------------------------------------------------------------------------
order = list(INDICATORS)
titles = {k: v[3] for k, v in INDICATORS.items()}

fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_IN, 4.1), gridspec_kw={"wspace": 0.4, "hspace": 0.55})
for i, key in enumerate(order):
    ax = axes[i // 4, i % 4]
    d = la[la[key].notna()]
    sl = stats.linregress(d.latitude, d[key])
    col = COHORT_COLOUR[INDICATORS[key][1]]
    ax.scatter(d.latitude, d[key], s=6, color=col, alpha=0.7, lw=0)
    xx = np.linspace(d.latitude.min(), d.latitude.max(), 10)
    ax.plot(xx, sl.intercept + sl.slope * xx, color="#0b0b0b", lw=0.9)
    ax.text(0.03, 0.97, f"β = {sl.slope:+.2f}\nr = {sl.rvalue:+.2f}, N = {len(d)}", transform=ax.transAxes,
            va="top", fontsize=5.8, color=INK2)
    ax.set_xlabel("Latitude (°N)", fontsize=6.5)
    if i % 4 == 0:
        ax.set_ylabel("Prevalence (%)")
    ax.tick_params(labelsize=6)
    titled(ax, LETTERS[i], titles[key])
    ax.title.set_fontsize(6.8)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S6_raw_latitude_associations")

ETH = ["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic"]
MODEL_COVS = {"A": ["imd_score"], "B": ["imd_score"] + ETH}

# S7. Added-variable (partial regression) plots for latitude, Model A and Model B
fig, axes = plt.subplots(4, 4, figsize=(DOUBLE_COL_IN, 7.6), gridspec_kw={"wspace": 0.42, "hspace": 0.75})
for ci_, cohort_prefix in enumerate(["rec", "yr6"]):
    keys = [k for k in order if k.startswith(cohort_prefix)]
    for mi, model in enumerate(["A", "B"]):
        r_ = 2 * ci_ + mi
        covs = " + ".join(MODEL_COVS[model])
        for j, key in enumerate(keys):
            ax = axes[r_, j]
            d = la[la[key].notna()]
            ry = smf.ols(f"{key} ~ {covs}", data=d).fit().resid
            rx = smf.ols(f"latitude ~ {covs}", data=d).fit().resid
            full = smf.ols(f"{key} ~ {covs} + latitude", data=d).fit(cov_type="HC1")
            col = COHORT_COLOUR[INDICATORS[key][1]]
            ax.scatter(rx, ry, s=6, color=col, alpha=0.7 if model == "B" else 0.45, lw=0)
            xx = np.linspace(rx.min(), rx.max(), 10)
            ax.plot(xx, full.params["latitude"] * xx, color="#0b0b0b", lw=0.9)
            ax.axhline(0, color=AXIS, lw=0.5)
            p = full.pvalues["latitude"]
            ax.text(0.03, 0.97, f"β = {full.params['latitude']:+.2f}\np {'< 0.001' if p < 0.001 else f'= {p:.3f}'}",
                    transform=ax.transAxes, va="top", fontsize=5.8, color=INK2)
            ax.set_xlabel("Latitude | covariates (°)", fontsize=6)
            if j == 0:
                ax.set_ylabel("Prevalence | covariates (pp)", fontsize=6)
            ax.tick_params(labelsize=6)
            titled(ax, LETTERS[r_ * 4 + j] if r_ * 4 + j < len(LETTERS) else "", titles[key])
            ax.title.set_fontsize(6.8)
row_labels(fig, axes, ["Reception, Model A", "Reception, Model B", "Year 6, Model A", "Year 6, Model B"])
save(fig, "Figure_S7_added_variable_plots")

# S8. Residual maps for every indicator after IMD + child ethnic composition (Model B without latitude)
import maps  # noqa: E402

fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_IN, 5.2), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
for i, key in enumerate(order):
    ax = axes[i // 4, i % 4]
    d = la[la[key].notna()]
    res = smf.ols(f"{key} ~ " + " + ".join(MODEL_COVS["B"]), data=d).fit().resid
    v = float(np.nanpercentile(np.abs(res), 97))
    sm = maps.draw(ax, dict(zip(d.la_code, res)), DIVERGING, -v, v, inset_rect=(0.68, 0.58, 0.34, 0.30))
    titled(ax, LETTERS[i], titles[key])
    ax.title.set_fontsize(6.8)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.01, aspect=18)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_visible(False)
row_labels(fig, axes, COHORTS)
save(fig, "Figure_S8_residual_maps_all")

# S10. Ethnic composition of dependent children (Census 2021) and underweight prevalence
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#b7d3f6", "#3987e5", "#104281"])
fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.3)
for k, (col, lab) in enumerate([("pct_asian", "Asian"), ("pct_black", "Black")]):
    ax = fig.add_subplot(gs[0, k])
    vmax = float(np.ceil(la[col].max() / 10) * 10)
    sm = maps.draw(ax, dict(zip(la.la_code, la[col])), SEQ, 0, vmax)
    titled(ax, "AB"[k], f"Dependent children who are {lab} (%)")
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=22)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
for k, key in enumerate(["rec_uw", "yr6_uw"]):
    ax = fig.add_subplot(gs[1, k])
    d = la[la[key].notna()]
    sl = stats.linregress(d.pct_asian, d[key])
    ax.scatter(d.pct_asian, d[key], s=8, color=COHORT_COLOUR[INDICATORS[key][1]], alpha=0.75, lw=0)
    xx = np.linspace(0, d.pct_asian.max(), 10)
    ax.plot(xx, sl.intercept + sl.slope * xx, color="#0b0b0b", lw=0.9)
    ax.text(0.03, 0.97, f"r = {sl.rvalue:+.2f}, N = {len(d)}", transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("Dependent children who are Asian (%)")
    ax.set_ylabel("Underweight prevalence (%)")
    titled(ax, "CD"[k], f"{INDICATORS[key][1]} underweight, 2023/24")
save(fig, "Figure_S10_ethnic_composition")

# ---------------------------------------------------------------------------
# S9. Location and spread of the BMI z-score distribution through the year
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.5), gridspec_kw={"wspace": 0.3})
for k, (col, lab) in enumerate([("z_mean", "Mean BMI z-score (location)"), ("z_sd", "SD of BMI z-score (spread)")]):
    ax = axes[k]
    for c in COHORTS:
        g = m[m.cohort == c].groupby("acad_month")[col]
        mean, ci = g.mean(), stats.t.ppf(0.975, g.count() - 1) * g.std() / np.sqrt(g.count())
        ax.errorbar(x + (-0.1 if c == "Reception" else 0.1), mean, yerr=ci, fmt="-o", ms=3, lw=1.2, elinewidth=0.7,
                    color=COHORT_COLOUR[c], label=c)
    ax.set_xticks(x, MONTH_ABBR, rotation=90)
    ax.set_ylabel("z-score units")
    titled(ax, "AB"[k], lab)
    if k == 0:
        ax.legend(loc="center right")
save(fig, "Figure_S9_distribution_location_scale")
