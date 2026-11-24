"""Integrity checks on the inputs and on the outputs written by run_all.py."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from common import INDICATORS, OUT_DATA, OUT_TAB, RAW, RESULTS_JSON, ROOT

built = pytest.mark.skipif(not RESULTS_JSON.exists(), reason="run `python run_all.py` first")


def test_raw_data_checksums():
    for line in (ROOT / "data" / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((RAW / name.strip()).read_bytes()).hexdigest() == digest, name


def test_api_files_reproduce_original_exports():
    """The Fingertips API files (used for 2025/26) hold identical 2023/24 and 2024/25 values."""
    def sel(d):
        d = d[d["Area Type"].str.startswith("Counties & UAs") & (d.Sex == "Persons") & d["Category Type"].isna()
              & d["Time period"].isin(["2023/24", "2024/25"])]
        return d.set_index(["Area Code", "Time period"])[["Value", "Count", "Denominator"]].sort_index()
    for fname, *_ in INDICATORS.values():
        a = sel(pd.read_csv(RAW / "fingertips_api" / fname, low_memory=False))
        b = sel(pd.read_csv(RAW / fname, low_memory=False))
        assert a.index.equals(b.index) and len(a) == 306
        pd.testing.assert_frame_equal(a, b)


@built
@pytest.mark.parametrize("year", ["2023_24", "2024_25", "2025_26"])
def test_la_dataset(year):
    d = pd.read_csv(OUT_DATA / f"la_dataset_{year}.csv")
    assert len(d) == 151 and d.la_code.is_unique
    covs = ["latitude", "longitude", "imd_score", "idaci_score", "region", "pct_asian", "pct_black", "pct_mixed",
            "pct_other_ethnic"]
    assert d[covs].notna().all().all()
    assert d.latitude.between(50.0, 55.9).all()
    n_by = json.loads(RESULTS_JSON.read_text())["la_dataset"][year.replace("_", "/")]["n_by_indicator"]
    for k in INDICATORS:
        assert d[k].notna().sum() == n_by[k]
        assert d[k].dropna().between(0, 100).all()
    # nested thresholds: severe <= obesity <= overweight incl. obesity
    ok = d[["rec_sev", "rec_ob", "rec_owob"]].dropna()
    assert (ok.rec_sev <= ok.rec_ob).all() and (ok.rec_ob <= ok.rec_owob).all()


@built
def test_monthly_series():
    m = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
    assert (m.groupby("cohort").size() == 139).all()
    total = m[["pct_underweight", "pct_healthy", "pct_overweight", "pct_obese"]].sum(axis=1)
    assert np.allclose(total, 100, atol=0.05)
    assert m[m.complete_year].ncmp_year.nunique() == 12
    assert (m.pct_severe <= m.pct_obese).all()


@built
def test_tables_agree_with_results_registry():
    r = json.loads(RESULTS_JSON.read_text())
    gm = pd.read_csv(OUT_TAB / "geo_models.csv")
    assert len(gm) == 2 * len(INDICATORS)
    for _, row in gm.iterrows():
        stored = r["geographic"]["models"][f"{row.model}|{row.indicator}"]
        assert stored["lat_beta"] == pytest.approx(row.lat_beta)
        assert row.n_las >= 149 and row.n_obs <= 3 * 151
    wy = pd.read_csv(OUT_TAB / "monthly_within_year_change.csv")
    for _, row in wy.iterrows():
        stored = r["temporal"]["within_year"][f"{row.cohort}|{row.outcome}"]
        assert stored["slope_sep_dec"] == pytest.approx(row.slope_sep_dec)
        assert stored["per_short"] == pytest.approx(row.per_short)
        assert stored["chg_short"] == pytest.approx(row.chg_short)
    # days shorten between mid-September and mid-December, and between mid-June and mid-July
    assert r["temporal"]["shortening_intervals"] == ["Sep-Oct", "Oct-Nov", "Nov-Dec", "Jun-Jul"]


@built
def test_pooled_estimates_lie_among_single_year_estimates():
    """The pooled latitude coefficient is a weighted average of the three years' information, so it
    should fall within (or very near) the range of the single-year estimates."""
    gm = pd.read_csv(OUT_TAB / "geo_models.csv")
    gy = pd.read_csv(OUT_TAB / "geo_models_by_year.csv")
    for _, row in gm.iterrows():
        yr = gy[(gy.indicator == row.indicator) & (gy.model == row.model)].lat_beta
        assert yr.min() - 0.05 <= row.lat_beta <= yr.max() + 0.05


@built
def test_easter_table_covers_complete_school_years():
    e = pd.read_csv(OUT_TAB / "monthly_april_easter.csv")
    assert len(e) == 12 and e.ncmp_year.is_unique
    assert e.easter_doy.between(81, 116).all()                 # Easter falls 22 March to 25 April
    assert pd.to_datetime(e.bst_start).dt.dayofweek.eq(6).all()  # British Summer Time starts on a Sunday


@built
def test_month_to_month_changes_add_up():
    """Each interval's mean change is the mean over school years of a first difference, so the changes
    for consecutive intervals must add up to the change between the end months, and the day-length
    changes must be negative exactly in the shortening intervals."""
    r = json.loads(RESULTS_JSON.read_text())["temporal"]
    ic = pd.read_csv(OUT_TAB / "monthly_interval_changes.csv")
    assert len(ic) == 2 * 4 * 10
    shortening = sorted(k for k, v in r["day_length_change_by_interval"].items() if v < 0)
    assert shortening == sorted(r["shortening_intervals"])
    assert (ic.days.eq("shortening") == ic.day_length_change_h.lt(0)).all()
    m = pd.read_csv(OUT_DATA / "ncmp_monthly.csv")
    m = m[m.complete_year & (m.cohort == "Reception")].pivot(index="ncmp_year", columns="acad_month", values="z_mean")
    autumn = ic[(ic.cohort == "Reception") & (ic.outcome == "z_mean") & ic.interval.isin(["Sep-Oct", "Oct-Nov", "Nov-Dec"])]
    assert autumn.change.sum() == pytest.approx((m[4] - m[1]).mean())
    c = r["interval_contrasts"]["Reception|z_mean|Nov-Dec vs Oct-Nov"]
    assert c["difference"] == pytest.approx(c["later"] - c["earlier"])
