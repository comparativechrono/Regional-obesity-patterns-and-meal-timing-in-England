"""Step 1: build the local-authority analysis dataset.

Inputs  : Fingertips NCMP indicators (8), IMD 2019 Files 10-11, ONS LA centroids,
          NCMP 2019/20 Table 5 (used only as an official LA -> region lookup).
Outputs : outputs/data/la_dataset_<year>.csv for 2023/24, 2024/25 and 2025/26
          outputs/tables/imd_reorganised_las.csv
"""
import numpy as np
import pandas as pd

from common import (CENSUS_ETHNICITY, CENTROIDS, FINGERTIPS_DIR, IMD_LOWER, IMD_UPPER, INDICATORS, NCMP_TABLES,
                    OUT_DATA, OUT_TAB, YEARS, photoperiod_amplitude, save_results)

# ---------------------------------------------------------------------------
# 1. NCMP prevalence (Fingertips)
# ---------------------------------------------------------------------------
def load_indicator(key, year):
    fname = INDICATORS[key][0]
    df = pd.read_csv(FINGERTIPS_DIR[year] / fname, low_memory=False)
    la = df[(df["Area Type"].str.startswith("Counties & UAs")) & (df["Sex"] == "Persons")
            & (df["Category Type"].isna()) & (df["Time period"] == year)]
    assert la["Area Code"].is_unique and len(la) == 153, (key, year, len(la))
    out = la[["Area Code", "Area Name", "Value", "Count", "Denominator", "Value note"]].rename(
        columns={"Area Code": "la_code", "Area Name": "la_name", "Value": key,
                 "Count": f"{key}_n", "Denominator": f"{key}_denom", "Value note": f"{key}_note"})
    eng = df[(df["Area Type"] == "England") & (df["Sex"] == "Persons") & (df["Category Type"].isna())
             & (df["Time period"] == year)].iloc[0]
    return out, float(eng["Value"]), int(eng["Denominator"])


def build_prevalence(year):
    merged, england = None, {}
    for key in INDICATORS:
        part, eng_val, eng_denom = load_indicator(key, year)
        england[key] = {"england_prevalence": eng_val, "england_children_measured": eng_denom}
        merged = part if merged is None else merged.merge(part.drop(columns="la_name"), on="la_code")
    return merged, england


# ---------------------------------------------------------------------------
# 2. Deprivation (IMD 2019) mapped to April-2023 upper-tier LAs
# ---------------------------------------------------------------------------
def load_imd(path, tier):
    code = ("Upper Tier " if tier == "upper" else "") + "Local Authority District code (2019)"
    imd = pd.read_excel(path, sheet_name="IMD")[[code, "IMD - Average score "]]
    inc = pd.read_excel(path, sheet_name="Income")[[code, "Income - Average score ", "Income scale"]]
    idc = pd.read_excel(path, sheet_name="IDACI")[[code, "IDACI - Average score "]]
    d = imd.merge(inc, on=code).merge(idc, on=code)
    d.columns = ["code", "imd_score", "income_rate", "income_scale", "idaci_score"]
    # Population denominator of the income domain = number income-deprived / rate
    d["population_est"] = d["income_scale"] / d["income_rate"]
    return d


upper = load_imd(IMD_UPPER, "upper")
lower = load_imd(IMD_LOWER, "lower")

# Seven LAs were created between April 2019 (IMD geography) and April 2023 (NCMP geography).
SAME_BOUNDARY = {  # new UA code -> predecessor county with an identical boundary
    "E06000060": ("E10000002", "Buckinghamshire County"),
    "E06000065": ("E10000023", "North Yorkshire County"),
    "E06000066": ("E10000027", "Somerset County"),
}
SPLITS = {  # new UA code -> predecessor districts (population-weighted average)
    "E06000061": ["E07000150", "E07000152", "E07000153", "E07000156"],  # North Northamptonshire
    "E06000062": ["E07000151", "E07000154", "E07000155"],                # West Northamptonshire
    "E06000063": ["E07000026", "E07000028", "E07000029"],                # Cumberland
    "E06000064": ["E07000027", "E07000030", "E07000031"],                # Westmorland and Furness
}


def weighted(df, col):
    return float(np.average(df[col], weights=df["population_est"]))


reorg_rows = []
imd_extra = []
for new, (old, label) in SAME_BOUNDARY.items():
    r = upper[upper.code == old].iloc[0]
    imd_extra.append({"code": new, "imd_score": r.imd_score, "idaci_score": r.idaci_score})
    reorg_rows.append({"new_code": new, "method": f"identical boundary: {label} ({old}) from File 11",
                       "imd_score": r.imd_score, "idaci_score": r.idaci_score})
for new, districts in SPLITS.items():
    s = lower[lower.code.isin(districts)]
    assert len(s) == len(districts), new
    imd_extra.append({"code": new, "imd_score": weighted(s, "imd_score"),
                      "idaci_score": weighted(s, "idaci_score")})
    reorg_rows.append({"new_code": new, "method": "population-weighted mean of 2019 districts "
                       + ", ".join(districts) + " from File 10",
                       "imd_score": weighted(s, "imd_score"), "idaci_score": weighted(s, "idaci_score")})

# Validation: the same weighting reconstructs the File-11 county values from File-10 districts
validation = []
for county, districts in {"E10000021": SPLITS["E06000061"] + SPLITS["E06000062"],
                          "E10000006": SPLITS["E06000063"] + SPLITS["E06000064"]}.items():
    s = lower[lower.code.isin(districts)]
    u = upper[upper.code == county].iloc[0]
    validation.append({"county": county, "file11_imd": u.imd_score, "reconstructed_imd": weighted(s, "imd_score"),
                       "unweighted_imd": s.imd_score.mean()})
validation = pd.DataFrame(validation)
assert (validation.file11_imd - validation.reconstructed_imd).abs().max() < 0.05

imd_all = pd.concat([upper[["code", "imd_score", "idaci_score"]], pd.DataFrame(imd_extra)], ignore_index=True)
imd_all = imd_all.rename(columns={"code": "la_code"})
pd.DataFrame(reorg_rows).to_csv(OUT_TAB / "imd_reorganised_las.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Region lookup (NCMP 2019/20 Table 5 lists every LA under its region)
# ---------------------------------------------------------------------------
t5 = pd.read_excel(NCMP_TABLES, sheet_name="Table 5", header=None)
region_of, current = {}, None
for _, row in t5.iterrows():
    code = row[1]
    if isinstance(code, str) and code.startswith("E12"):
        current = str(row[0]).strip().title().replace("Of", "of").replace("And The", "and the")
    elif isinstance(code, str) and code[:3] in ("E06", "E08", "E09", "E10") and current:
        region_of[code] = current
for new, old in {"E06000060": "E10000002", "E06000061": "E10000021", "E06000062": "E10000021",
                 "E06000063": "E10000006", "E06000064": "E10000006", "E06000065": "E10000023",
                 "E06000066": "E10000027"}.items():
    region_of.setdefault(new, region_of.get(old))

# ---------------------------------------------------------------------------
# 4. Geography (ONS centroids, December 2023 boundaries)
# ---------------------------------------------------------------------------
geo = pd.read_csv(CENTROIDS)
geo = geo[geo["CTYUA23CD"].str.startswith("E")][["CTYUA23CD", "LAT", "LONG"]]
geo.columns = ["la_code", "latitude", "longitude"]

# ---------------------------------------------------------------------------
# 4b. Ethnic composition of dependent children (Census 2021 RM033, April-2023 UTLAs)
#     Used only as a covariate to test whether latitude associations reflect population
#     composition. Isles of Scilly and City of London are added to Cornwall and Hackney,
#     matching how NCMP publishes those areas.
# ---------------------------------------------------------------------------
eth = pd.read_csv(CENSUS_ETHNICITY)
eth = eth[eth["GEOGRAPHY_CODE"].str.startswith("E")]
eth["GEOGRAPHY_CODE"] = eth["GEOGRAPHY_CODE"].replace({"E06000053": "E06000052", "E09000001": "E09000012"})
eth = eth.pivot_table(index="GEOGRAPHY_CODE", columns="C2021_ETH_20_NAME", values="OBS_VALUE", aggfunc="sum")
ETH_GROUPS = {
    "pct_asian": ["Asian, Asian British or Asian Welsh"],
    "pct_south_asian": ["Asian, Asian British or Asian Welsh: Bangladeshi", "Asian, Asian British or Asian Welsh: Indian",
                        "Asian, Asian British or Asian Welsh: Pakistani"],
    "pct_black": ["Black, Black British, Black Welsh, Caribbean or African"],
    "pct_mixed": ["Mixed or Multiple ethnic groups"],
    "pct_other_ethnic": ["Other ethnic group"],
    "pct_white": ["White"],
}
eth_df = pd.DataFrame({k: 100 * eth[cols].sum(axis=1) / eth["Total"] for k, cols in ETH_GROUPS.items()})
assert np.allclose(eth_df[["pct_asian", "pct_black", "pct_mixed", "pct_other_ethnic", "pct_white"]].sum(axis=1), 100)
eth_df["dependent_children_2021"] = eth["Total"]
eth_df = eth_df.reset_index().rename(columns={"GEOGRAPHY_CODE": "la_code"})

# ---------------------------------------------------------------------------
# 5. Assemble one dataset per NCMP year
# ---------------------------------------------------------------------------
summary = {}
for year in YEARS:
    prev, england = build_prevalence(year)
    df = (prev.merge(geo, on="la_code", how="left").merge(imd_all, on="la_code", how="left")
          .merge(eth_df, on="la_code", how="left"))
    df["region"] = df["la_code"].map(region_of)
    df["london"] = df["la_code"].str.startswith("E09").astype(int)
    df["photoperiod_amplitude_h"] = photoperiod_amplitude(df["latitude"].values)

    # City of London and Isles of Scilly are published combined with Hackney and Cornwall
    combined = df[df["rec_ob"].isna()]
    df = df[df["rec_ob"].notna()].copy()
    assert df[["latitude", "imd_score", "idaci_score", "region", "pct_asian"]].notna().all().all()
    # Children measured (same for every indicator within a cohort; taken from OW+OB rows,
    # which are never suppressed) -- needed to bound values suppressed for disclosure control
    df["rec_measured"] = df["rec_owob_denom"]
    df["yr6_measured"] = df["yr6_owob_denom"]
    suppressed = {k: df.loc[df[k].isna(), "la_name"].tolist() for k in INDICATORS}
    n_by_indicator = {k: int(df[k].notna().sum()) for k in INDICATORS}
    tag = year.replace("/", "_")
    df.to_csv(OUT_DATA / f"la_dataset_{tag}.csv", index=False)
    summary[year] = {
        "n_las_published": int(len(prev)),
        "n_las_analysed": int(len(df)),
        "excluded": {r.la_code: f"{r.la_name}: {r.rec_ob_note}" for r in combined.itertuples()},
        "n_by_indicator": n_by_indicator,
        "suppressed_disclosure_control_1_to_7": suppressed,
        "england": england,
        "unweighted_la_means": {k: float(df[k].mean()) for k in INDICATORS},
        "la_ranges": {k: [float(df[k].min()), float(df[k].max())] for k in INDICATORS},
        "la_sd": {k: float(df[k].std()) for k in INDICATORS},
        "latitude_range": [float(df.latitude.min()), float(df.latitude.max())],
        "latitude_mean_sd": [float(df.latitude.mean()), float(df.latitude.std())],
        "imd_range": [float(df.imd_score.min()), float(df.imd_score.max())],
        "imd_mean_sd": [float(df.imd_score.mean()), float(df.imd_score.std())],
        "photoperiod_amplitude_range_h": [float(df.photoperiod_amplitude_h.min()),
                                          float(df.photoperiod_amplitude_h.max())],
        "corr_latitude_amplitude": float(np.corrcoef(df.latitude, df.photoperiod_amplitude_h)[0, 1]),
        "ethnicity_mean_range": {k: [float(df[k].mean()), float(df[k].min()), float(df[k].max())]
                                 for k in ETH_GROUPS},
        "corr_latitude_pct_asian": float(np.corrcoef(df.latitude, df.pct_asian)[0, 1]),
        "corr_latitude_pct_black": float(np.corrcoef(df.latitude, df.pct_black)[0, 1]),
        "n_london": int(df.london.sum()),
        "region_counts": df.region.value_counts().to_dict(),
    }
    print(f"{year}: {len(prev)} LAs published, {len(df)} analysed; "
          f"excluded {list(summary[year]['excluded'].values())}")
    print("   N per indicator:", n_by_indicator)

summary["imd_validation"] = validation.to_dict(orient="records")
summary["imd_reorganised"] = reorg_rows
save_results("la_dataset", summary)
print(validation.round(3).to_string(index=False))
print(pd.DataFrame(reorg_rows)[["new_code", "imd_score", "idaci_score"]].round(3).to_string(index=False))
