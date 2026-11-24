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
    for _, row in gm.iterrows():
        stored = r["geographic"]["models"][f"{row.year}|{row.model}|{row.indicator}"]
        assert stored["lat_beta"] == pytest.approx(row.lat_beta)
    reset = pd.read_csv(OUT_TAB / "monthly_september_reset.csv")
    assert (reset.n_transitions == 12).all()
