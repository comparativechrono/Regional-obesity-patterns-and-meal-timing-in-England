# Day length, the school year and children's BMI in England

Data, code and outputs for the paper **"Day length, the school year and children's BMI in England:
testing photoperiod hypotheses with National Child Measurement Programme data"**.

The repository name reflects the project's original scope. The analyses here are about photoperiod
(day length), the school year and latitude.

Everything in `outputs/` is produced from the files in `data/raw/` by the scripts in `src/`.
One command rebuilds all of it in about 30 seconds.

## What the analysis asks

England's National Child Measurement Programme (NCMP) weighs and measures children in Reception
(age 4–5) and Year 6 (age 10–11) at school, at any time from September to July. Two questions are
tested with public data only:

1. **Season.** Does the BMI distribution of measured children vary with the day length at the time
   of measurement? Or does it follow the child's position in the school year? Data: NHS Digital's national
   monthly series, 2007/08–2018/19 (12 complete school years, 13.1 million measurements).
2. **Latitude.** Is latitude, a proxy for the size of the annual photoperiod cycle, associated with
   local-authority prevalence once deprivation and ethnic composition are taken into account? Data:
   151 upper-tier local authorities. NCMP 2023/24 is the primary year, with 2024/25 and 2025/26 as
   replications.

## Main results

All numbers are read from `outputs/results.json` and `outputs/tables/`.

* **When children are measured.** 22% of Reception children are measured in September–December,
  46% in January–March and 31% in April–July (modal month March). Year 6 is measured earlier in the year:
  41% / 35% / 24% (modal month November).
* **A school-year shift of the whole distribution.** Reception overweight-including-obesity
  prevalence falls by 0.34 percentage points (pp) per month of the school year (95% CI 0.28–0.40).
  That is a fitted fall of 3.4 pp from September to July, while underweight rises by 0.27 pp. The
  whole BMI z-score distribution moves down by 0.10 SD (0.08–0.13). The shift in Year 6 is smaller
  (0.04 SD).
* **The September reset.** From July to the following September, day length shortens by 3.6 h,
  yet Reception obesity prevalence rises by 1.8 pp (1.5–2.2). It rose in all 12 year-to-year
  transitions, returning to about the previous September's level. A process driven by day length cannot
  produce this pattern; one anchored to the school year (age at measurement or time in school) can.
* **Day length alone explains almost nothing.** With year effects only, adding day length raises the
  adjusted R² for monthly Reception obesity from −0.03 to 0.02. Adding the school-year trend raises it
  to 0.47. Once that trend is allowed for, a smaller annual component in phase with day length remains.
  Its size depends on how the trend is modelled, so the monthly aggregates cannot attribute it to
  photoperiod.
* **Latitude and underweight reflect ethnic composition.** Adjusted for deprivation only (Model A),
  underweight is lower further north: −0.20 pp per degree in Reception and −0.13 in Year 6
  (2023/24). Adding the Census 2021 ethnic composition of dependent children (Model B) removes the
  association: Reception −0.03 (p = 0.20), Year 6 −0.03 (p = 0.16). The same happens in both
  replication years.
* **Year 6 severe obesity is higher further north once deprivation and ethnicity are adjusted for.**
  The estimate is +0.27 pp per degree in 2023/24, +0.23 in 2024/25 and +0.17 in 2025/26 (spatial-HAC
  p = 0.003, 0.003 and 0.02). There is no comparable association in Reception. This is an ecological
  association. Latitude in England carries many correlated exposures, so it is not evidence for
  photoperiod specifically.

## Repository layout

```
data/
  raw/                         source files exactly as downloaded (see data/README.md)
    fingertips_api/            Fingertips API files used for 2025/26
  SHA256SUMS.txt               checksums of every raw file
src/
  common.py                    paths, indicator definitions, CBM day-length model, regression helpers
  plotstyle.py, maps.py        figure style and choropleth helper
  00_download_supplementary_sources.py   (optional) re-download / verify the API-sourced inputs
  01_build_la_dataset.py       local-authority dataset: prevalence, IMD, ethnicity, latitude, region
  02_build_monthly_series.py   national monthly series from NHS Digital Tables 3-4
  03_geographic_analysis.py    latitude models A and B, robustness, spatial checks, replications
  04_temporal_analysis.py      timing, within-year trends, cosinor, day length vs school year, reset
  05_figures_main.py           Figures 1-6
  06_figures_supplementary.py  Figures S1-S10
  07_tables.py                 formatted Tables 1-4 and S1-S9
  photoperiod.py               stand-alone day length / sunrise / sunset calculator
outputs/
  data/                        analysis datasets (LA x year; national monthly series)
  tables/                      analysis tables (CSV); formatted/ and manuscript_tables.xlsx
  figures/                     PNG (300 dpi) and PDF
  results.json                 every number quoted in the paper
tests/                         pytest checks of methods, photoperiod model and outputs
run_all.py                     runs steps 1-7
```

## Reproducing the analysis

Python 3.10 or later.

```bash
pip install -r requirements.txt
python run_all.py --clean      # rebuild outputs/ from data/raw/
python -m pytest tests         # 19 checks: methods, photoperiod model, data integrity
```

`python run_all.py --download` first re-downloads the API-sourced inputs (step 0). By default the
download script writes `*.downloaded.*` copies and reports whether they match the snapshots.
Figures use the Inter typeface if it is installed and fall back to Arial or DejaVu Sans if not.

## Figures and tables

| Paper | File |
|---|---|
| Figure 1 | `outputs/figures/Figure_1_measurement_timing_and_day_length` |
| Figure 2 | `outputs/figures/Figure_2_distribution_shift` |
| Figure 3 | `outputs/figures/Figure_3_school_year_vs_day_length` |
| Figure 4 | `outputs/figures/Figure_4_actograms` |
| Figure 5 | `outputs/figures/Figure_5_latitude_coefficients` |
| Figure 6 | `outputs/figures/Figure_6_residual_maps` |
| Figures S1–S10 | `outputs/figures/Figure_S*` |
| Tables 1–4, S1–S9 | `outputs/tables/formatted/*.csv` and `outputs/tables/manuscript_tables.xlsx` |

## Photoperiod calculator

`src/photoperiod.py` takes a place and date and returns:

* day length from the CBM model (Forsythe et al. 1995) used in the paper;
* sunrise, solar noon and sunset in local clock time (NOAA solar-position algorithm);
* solar declination, the equation of time and the annual range of day length at that latitude.

```bash
python src/photoperiod.py --place cambridge --date 2024-12-21
python src/photoperiod.py --lat 54.97 --lon -1.61 --tz Europe/London --date 2025-03-20
python src/photoperiod.py --lat 52.2 --lon 0.12 --start 2024-01-01 --end 2024-12-31 -o daylength.csv
```

At English latitudes the CBM and NOAA day lengths agree to within 3.1 minutes (Table S8). Sunrise
and sunset agree with the `astral` package to within one minute.

## Methods in brief

* **Day length.** CBM model (Forsythe et al. 1995, Ecological Modelling 80:87–95), with sunrise and
  sunset defined by the upper limb at the apparent horizon (p = 0.8333°). The monthly series uses
  52.5°N on the 15th of each month. For local authorities, the annual range of day length is almost
  perfectly collinear with latitude (r = 0.999), so latitude is used.
* **Monthly series.** Academic-year fixed effects with standard errors clustered by year. Separate
  September–December and January–July slopes. Linear cosinor with a 12-month period. Day length is
  compared with linear, quadratic and two-phase school-year trends. Location (mean) and scale (SD)
  of the BMI z-score distribution are recovered from the four UK90 cut-points by probit regression.
* **Local authorities.** OLS with HC1 standard errors (checked against an independent
  implementation) and a Holm correction across the 8 indicators. Spatial-HAC (Conley) standard errors
  at 100 and 200 km and Moran's I of residuals. Robustness checks: IDACI instead of IMD, London
  excluded, longitude added, weighting by children measured, region fixed effects, and bounds for
  values suppressed for disclosure control (1–7 children).
* **IMD 2019 for April-2023 authorities.** Buckinghamshire, North Yorkshire and Somerset take their
  predecessor county values (identical boundaries). North and West Northamptonshire, Cumberland, and
  Westmorland and Furness take population-weighted means of their 2019 districts, and the same
  weighting reproduces the published county values (Table S2).
* **Ethnic composition.** Census 2021 table RM033: the percentage of dependent children who are
  Asian, Black, Mixed or Other, with White as the reference group.

## Analysis notes

The analysis was audited and finalised in November 2026. At that stage two things were added: the
ethnic-composition covariate (Model B), prompted by the sign reversal of the underweight–latitude
association within regions, and NCMP 2025/26 as a further replication year. Model B was therefore
specified after the 2023/24 and 2024/25 results had been seen. The predictions for 2025/26 were
written down before any 2025/26 data were analysed:

| Prediction (Model B unless stated) | 2025/26 result |
|---|---|
| Year 6 severe obesity: β > 0, p < 0.05 | +0.17 pp/°N (0.04 to 0.31), p = 0.010 (Holm 0.078) |
| Year 6 obesity, OW+OB: β > 0, weaker | +0.29 (p = 0.074); +0.22 (p = 0.18) |
| Underweight, both cohorts: β ≈ 0 | −0.03 (p = 0.32); −0.02 (p = 0.37) |
| Reception OW+OB, obesity, severe obesity: no robust association | all p > 0.6 |
| Model A underweight, both cohorts: β < 0, p < 0.05 | −0.18 and −0.17, both p < 0.001 |

## Data sources and licences

All inputs are public and used under the Open Government Licence v3.0. Sources, URLs, renamed files,
suppression rules and required attributions are listed in [`data/README.md`](data/README.md).
Contains OS data © Crown copyright and database right 2023. Source: Office for National Statistics
licensed under the Open Government Licence v3.0.
