"""Step 2: parse the NCMP monthly time series (NHS Digital 2019/20 Tables 3 and 4).

Each row of the tables is one month of measurement in one NCMP collection year, for
all children measured nationally in that month. Output is a tidy long-format file.

Output: outputs/data/ncmp_monthly.csv
"""
import numpy as np
import pandas as pd

from common import (ACAD_INDEX, CUTPOINTS, MONTH_NUM, NCMP_TABLES, OUT_DATA,
                    REFERENCE_LATITUDE, day_length, save_results)
from scipy import stats

CATEGORY_COLUMNS = {  # category -> (number column, prevalence column)
    "underweight": (2, 3),
    "healthy": (7, 8),
    "overweight": (12, 13),     # overweight but not obese (85th to <95th centile)
    "obese": (17, 18),          # obese including severely obese (>=95th centile)
    "severe": (22, 23),         # severely obese (>=99.6th centile)
    "owob": (27, 28),           # overweight and obese combined (>=85th centile)
}
N_MEASURED_COL = 32

rows = []
for sheet, cohort in (("Table 3", "Reception"), ("Table 4", "Year 6")):
    raw = pd.read_excel(NCMP_TABLES, sheet_name=sheet, header=None)
    header = raw.iloc[12].tolist()
    assert header[2] == "Underweight" and header[17].startswith("Obese") and header[27].startswith("Overweight and obese")
    for _, r in raw.iloc[14:].iterrows():
        year, month = r[0], r[1]
        if not (isinstance(year, str) and year[:2] == "20" and isinstance(month, str)):
            continue
        rec = {"cohort": cohort, "ncmp_year": year.strip(), "month": month.strip(),
               "n_measured": int(r[N_MEASURED_COL])}
        for cat, (nc, pc) in CATEGORY_COLUMNS.items():
            rec[f"n_{cat}"] = int(r[nc])
            rec[f"pct_{cat}"] = float(r[pc])
        rows.append(rec)

df = pd.DataFrame(rows)
df["acad_month"] = df["month"].map(ACAD_INDEX)          # September = 1 ... July = 11
df["cal_month"] = df["month"].map(MONTH_NUM)
df["start_year"] = df["ncmp_year"].str[:4].astype(int)
df["cal_year"] = np.where(df["cal_month"] >= 9, df["start_year"], df["start_year"] + 1)
mid = pd.to_datetime(dict(year=df.cal_year, month=df.cal_month, day=15))
df["mid_month_doy"] = mid.dt.dayofyear
df["t_years"] = df["cal_year"] + (df["mid_month_doy"] - 0.5) / 365.25   # calendar time, fractional years
df["photoperiod_h"] = day_length(REFERENCE_LATITUDE, df["mid_month_doy"].values)
df["complete_year"] = df.groupby(["cohort", "ncmp_year"])["month"].transform("count").eq(11)

# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------
parts = df[["n_underweight", "n_healthy", "n_overweight", "n_obese"]].sum(axis=1)
assert (parts == df["n_measured"]).all(), "categories do not sum to number measured"
assert (df["n_overweight"] + df["n_obese"] == df["n_owob"]).all()
assert (df["n_severe"] <= df["n_obese"]).all()
for cat in CATEGORY_COLUMNS:
    assert np.allclose(df[f"pct_{cat}"], 100 * df[f"n_{cat}"] / df["n_measured"], atol=1e-6), cat

# ---------------------------------------------------------------------------
# Location/scale of the BMI z-score distribution in each month (probit fit to the
# four UK90 cut-points). If the population BMI z-score is ~N(mu, sigma), then
# probit(P(z < c)) = (c - mu) / sigma for each cut-point c.
# ---------------------------------------------------------------------------
cuts = np.array([CUTPOINTS["underweight"], CUTPOINTS["overweight"], CUTPOINTS["obese"], CUTPOINTS["severe"]])
mus, sigmas, rmses = [], [], []
for r in df.itertuples():
    p_below = np.array([
        r.n_underweight / r.n_measured,
        (r.n_underweight + r.n_healthy) / r.n_measured,
        (r.n_underweight + r.n_healthy + r.n_overweight) / r.n_measured,
        1 - r.n_severe / r.n_measured,
    ])
    q = stats.norm.ppf(p_below)
    slope, intercept = np.polyfit(cuts, q, 1)
    sigma = 1 / slope
    mus.append(-intercept * sigma)
    sigmas.append(sigma)
    rmses.append(float(np.sqrt(np.mean((q - (intercept + slope * cuts)) ** 2))))
df["z_mean"] = mus
df["z_sd"] = sigmas
df["probit_fit_rmse"] = rmses

df = df.sort_values(["cohort", "t_years"]).reset_index(drop=True)
df.to_csv(OUT_DATA / "ncmp_monthly.csv", index=False)

summary = {
    "n_records": df.groupby("cohort").size().to_dict(),
    "n_complete_years": df[df.complete_year].groupby("cohort")["ncmp_year"].nunique().to_dict(),
    "complete_years": sorted(df[df.complete_year]["ncmp_year"].unique().tolist()),
    "children_measured_total": df.groupby("cohort")["n_measured"].sum().to_dict(),
    "children_measured_complete_years": df[df.complete_year].groupby("cohort")["n_measured"].sum().to_dict(),
    "photoperiod_52_5N_by_month_h": df.drop_duplicates("month").set_index("month")["photoperiod_h"].round(3).to_dict(),
    "probit_fit_rmse_max": float(df["probit_fit_rmse"].max()),
}
save_results("monthly_series", summary)
print(df.groupby("cohort").size(), "\n", summary["n_complete_years"])
print("Probit fit RMSE: median %.3f, max %.3f" % (df.probit_fit_rmse.median(), df.probit_fit_rmse.max()))
