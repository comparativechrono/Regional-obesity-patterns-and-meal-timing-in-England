#!/usr/bin/env python3
"""Photoperiod calculator: day length, sunrise, solar noon and sunset for any place and date.

Two models are provided:

* Day length (photoperiod) from the CBM model evaluated by Forsythe et al. (1995),
  Ecological Modelling 80:87-95 -- the model used in the paper. Sunrise and sunset are
  defined as the moment the top of the sun is at the apparent horizon (p = 0.8333 deg).
* Sunrise, solar noon and sunset clock times from the NOAA solar-position algorithm
  (after Meeus, Astronomical Algorithms), which also gives an independent day length.
  The two day lengths agree to within about 5 minutes at English latitudes.

No third-party packages are needed (Python 3.9+). Local clock times need the IANA time-zone
database, which Python's `zoneinfo` provides on most systems.

Examples
--------
    python photoperiod.py --lat 52.2053 --lon 0.1218 --date 2024-12-21 --tz Europe/London
    python photoperiod.py --place cambridge --date 2024-06-21
    python photoperiod.py --place cambridge --start 2024-01-01 --end 2024-12-31 -o cambridge_2024.csv

From Python
-----------
    >>> from datetime import date
    >>> from photoperiod import compute
    >>> r = compute(52.2053, 0.1218, date(2024, 12, 21), tz="Europe/London")
    >>> round(r["day_length_cbm_h"], 2)
    7.71
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

PLACES = {
    "london": (51.5074, -0.1278, "Europe/London"),
    "cambridge": (52.2053, 0.1218, "Europe/London"),
    "oxford": (51.7520, -1.2577, "Europe/London"),
    "birmingham": (52.4862, -1.8904, "Europe/London"),
    "manchester": (53.4808, -2.2426, "Europe/London"),
    "newcastle": (54.9783, -1.6178, "Europe/London"),
    "plymouth": (50.3755, -4.1427, "Europe/London"),
    "edinburgh": (55.9533, -3.1883, "Europe/London"),
    "tromso": (69.6492, 18.9553, "Europe/Oslo"),
    "new york": (40.7128, -74.0060, "America/New_York"),
    "tokyo": (35.6762, 139.6503, "Asia/Tokyo"),
    "sydney": (-33.8688, 151.2093, "Australia/Sydney"),
}


# ---------------------------------------------------------------------------
# CBM day-length model (Forsythe et al. 1995) -- identical to src/common.py
# ---------------------------------------------------------------------------
def day_length_cbm(latitude: float, day_of_year: int, p_deg: float = 0.8333) -> float:
    """Day length in hours. p_deg = 0.8333 -> top of sun at the apparent horizon."""
    theta = 0.2163108 + 2.0 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    lat = math.radians(latitude)
    a = (math.sin(math.radians(p_deg)) + math.sin(lat) * math.sin(phi)) / (math.cos(lat) * math.cos(phi))
    a = max(-1.0, min(1.0, a))
    return 24.0 - (24.0 / math.pi) * math.acos(a)


# ---------------------------------------------------------------------------
# NOAA solar-position algorithm (sunrise, solar noon, sunset)
# ---------------------------------------------------------------------------
def _julian_day(dt_utc: datetime) -> float:
    return dt_utc.toordinal() + 1721424.5 + (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24


def _sun(jd: float):
    """Solar declination (deg) and equation of time (minutes) for a Julian day."""
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * t)
    app_long = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(app_long))
    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eot = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                           - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    return math.degrees(decl), eot


def _event_minutes(latitude, longitude, d: date, which: str):
    """Minutes after 00:00 UTC on date d for 'sunrise', 'noon' or 'sunset' (None if it does not occur)."""
    minutes = 720.0
    for _ in range(3):  # iterate so declination and equation of time are evaluated at the event time
        jd = _julian_day(datetime(d.year, d.month, d.day, tzinfo=timezone.utc)) + minutes / 1440
        decl, eot = _sun(jd)
        noon = 720 - 4 * longitude - eot
        if which == "noon":
            minutes = noon
            continue
        lat, dec = math.radians(latitude), math.radians(decl)
        cos_h = math.cos(math.radians(90.833)) / (math.cos(lat) * math.cos(dec)) - math.tan(lat) * math.tan(dec)
        if cos_h > 1 or cos_h < -1:
            return None
        ha = math.degrees(math.acos(cos_h))
        minutes = noon - 4 * ha if which == "sunrise" else noon + 4 * ha
    return minutes


def _to_clock(d: date, minutes, tz):
    if minutes is None:
        return None
    t = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(minutes=minutes)
    t = (t + timedelta(seconds=30)).replace(second=0, microsecond=0)  # round to the nearest minute
    if tz is not None:
        t = t.astimezone(tz)
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute(latitude: float, longitude: float, d: date, tz: str | None = None) -> dict:
    """All photoperiod quantities for one place and date.

    latitude  : decimal degrees, north positive
    longitude : decimal degrees, east positive (west negative)
    tz        : optional IANA time-zone name (e.g. "Europe/London") for local clock times
    """
    zone = ZoneInfo(tz) if (tz and ZoneInfo) else None
    doy = d.timetuple().tm_yday
    rise = _event_minutes(latitude, longitude, d, "sunrise")
    noon = _event_minutes(latitude, longitude, d, "noon")
    sset = _event_minutes(latitude, longitude, d, "sunset")
    decl, eot = _sun(_julian_day(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)))
    if rise is None or sset is None:
        noaa_dl = 24.0 if decl * latitude > 0 else 0.0
    else:
        noaa_dl = (sset - rise) / 60
    fmt = "%H:%M"
    out = {
        "date": d.isoformat(), "latitude": latitude, "longitude": longitude, "day_of_year": doy,
        "time_zone": tz or "UTC",
        "day_length_cbm_h": day_length_cbm(latitude, doy),
        "day_length_noaa_h": noaa_dl,
        "solar_declination_deg": decl,
        "equation_of_time_min": eot,
        "sunrise": _to_clock(d, rise, zone).strftime(fmt) if rise is not None else "none",
        "solar_noon": _to_clock(d, noon, zone).strftime(fmt),
        "sunset": _to_clock(d, sset, zone).strftime(fmt) if sset is not None else "none",
        "annual_range_cbm_h": max(day_length_cbm(latitude, j) for j in range(1, 366))
                              - min(day_length_cbm(latitude, j) for j in range(1, 366)),
    }
    return out


def hhmm(hours: float) -> str:
    h = int(hours)
    return f"{h}h {round((hours - h) * 60):02d}m"


def _report(r: dict) -> str:
    return "\n".join([
        f"Date            {r['date']} (day {r['day_of_year']})",
        f"Location        {r['latitude']:.4f}, {r['longitude']:.4f}",
        f"Day length      {hhmm(r['day_length_cbm_h'])}  (CBM, Forsythe et al. 1995)",
        f"                {hhmm(r['day_length_noaa_h'])}  (NOAA algorithm)",
        f"Sunrise         {r['sunrise']}  ({r['time_zone']})",
        f"Solar noon      {r['solar_noon']}",
        f"Sunset          {r['sunset']}",
        f"Declination     {r['solar_declination_deg']:+.3f} deg; equation of time {r['equation_of_time_min']:+.2f} min",
        f"Annual range    {r['annual_range_cbm_h']:.2f} h of day length at this latitude",
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lat", type=float, help="latitude, decimal degrees (north positive)")
    ap.add_argument("--lon", type=float, help="longitude, decimal degrees (east positive)")
    ap.add_argument("--place", help="named place: " + ", ".join(PLACES))
    ap.add_argument("--tz", help="IANA time zone for local times, e.g. Europe/London (default UTC)")
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    ap.add_argument("--start", help="first date of a daily series, YYYY-MM-DD")
    ap.add_argument("--end", help="last date of a daily series, YYYY-MM-DD")
    ap.add_argument("-o", "--output", help="CSV file for a daily series")
    a = ap.parse_args(argv)

    tz = a.tz
    if a.place:
        key = a.place.lower()
        if key not in PLACES:
            sys.exit(f"unknown place {a.place!r}; choose from: {', '.join(PLACES)}")
        lat, lon, place_tz = PLACES[key]
        tz = tz or place_tz
    elif a.lat is not None and a.lon is not None:
        lat, lon = a.lat, a.lon
    else:
        ap.error("give --lat and --lon, or --place")

    if a.start:
        d0 = date.fromisoformat(a.start)
        d1 = date.fromisoformat(a.end) if a.end else date(d0.year, 12, 31)
        rows = [compute(lat, lon, d0 + timedelta(days=i), tz) for i in range((d1 - d0).days + 1)]
        if a.output:
            with open(a.output, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} days to {a.output}")
        else:
            for r in rows:
                print(f"{r['date']}  {hhmm(r['day_length_cbm_h'])}  rise {r['sunrise']}  set {r['sunset']}")
        return
    d = date.fromisoformat(a.date) if a.date else date.today()
    print(_report(compute(lat, lon, d, tz)))


if __name__ == "__main__":
    main()
