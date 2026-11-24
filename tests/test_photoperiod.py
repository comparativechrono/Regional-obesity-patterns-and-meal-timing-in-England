"""Checks of the photoperiod model used in the paper (CBM, Forsythe et al. 1995) and of the
stand-alone calculator src/photoperiod.py."""
import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import pytest

from common import day_length, photoperiod_amplitude
from photoperiod import PLACES, compute, day_length_cbm

ENGLISH_LATITUDES = np.arange(50.0, 55.81, 0.5)
DAYS_2024 = [date(2024, 1, 1) + timedelta(days=i) for i in range(366)]


def test_calculator_matches_pipeline_model():
    for lat in ENGLISH_LATITUDES:
        for doy in range(1, 366):
            assert day_length_cbm(lat, doy) == pytest.approx(float(day_length(lat, doy)), abs=1e-9)


def test_known_values():
    # equator: ~12 h plus ~7 min for refraction and the solar semi-diameter (p = 0.8333 deg)
    assert all(12.0 < day_length_cbm(0.0, d) < 12.25 for d in range(1, 366))
    assert day_length_cbm(52.5, 356) == pytest.approx(7.65, abs=0.02)   # winter solstice
    assert day_length_cbm(52.5, 172) == pytest.approx(16.83, abs=0.02)  # summer solstice
    # longer summer days and shorter winter days further north
    assert day_length_cbm(55.0, 172) > day_length_cbm(50.0, 172)
    assert day_length_cbm(55.0, 356) < day_length_cbm(50.0, 356)


def test_cbm_agrees_with_noaa_algorithm_at_english_latitudes():
    worst = 0.0
    for lat in ENGLISH_LATITUDES:
        for d in DAYS_2024:
            r = compute(lat, -1.5, d)
            worst = max(worst, abs(r["day_length_cbm_h"] - r["day_length_noaa_h"]) * 60)
    assert worst < 5.0, worst  # minutes


def test_sun_times_against_astral():
    astral = pytest.importorskip("astral")
    from zoneinfo import ZoneInfo

    from astral import LocationInfo
    from astral.sun import sun
    lat, lon, tz = PLACES["london"]
    loc = LocationInfo("London", "England", tz, lat, lon)
    for d in DAYS_2024[::7]:
        s = sun(loc.observer, date=d, tzinfo=ZoneInfo(tz))
        r = compute(lat, lon, d, tz)
        for key in ("sunrise", "sunset"):
            h, m = map(int, r[key].split(":"))
            ref = s[key].hour * 60 + s[key].minute + s[key].second / 60
            assert abs(h * 60 + m - ref) <= 1.5, (d, key, r[key], s[key])


def test_polar_night_and_midnight_sun():
    winter = compute(69.65, 18.96, date(2024, 12, 21), "Europe/Oslo")
    summer = compute(69.65, 18.96, date(2024, 6, 21), "Europe/Oslo")
    assert winter["day_length_cbm_h"] == 0 and winter["sunrise"] == "none"
    assert summer["day_length_cbm_h"] == 24 and summer["sunset"] == "none"


def test_amplitude_increases_with_latitude():
    amp = photoperiod_amplitude(np.array([50.0, 52.0, 54.0, 56.0]))
    assert np.all(np.diff(amp) > 0)


def test_command_line(tmp_path):
    from common import ROOT
    out = subprocess.run([sys.executable, str(ROOT / "src" / "photoperiod.py"), "--place", "cambridge",
                          "--date", "2024-12-21"], capture_output=True, text=True, check=True).stdout
    assert "Sunrise" in out and "7h 42m" in out
    csv = tmp_path / "series.csv"
    subprocess.run([sys.executable, str(ROOT / "src" / "photoperiod.py"), "--lat", "52.2", "--lon", "0.12",
                    "--start", "2024-01-01", "--end", "2024-01-31", "-o", str(csv)], check=True, capture_output=True)
    assert len(csv.read_text().strip().splitlines()) == 32
