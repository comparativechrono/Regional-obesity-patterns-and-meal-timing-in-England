# Day length, the school year and children's BMI in England

Data, code and outputs for the paper **"Day length, the school year and children's BMI in England:
testing photoperiod hypotheses with National Child Measurement Programme data"**.

The repository name reflects the project's original scope. The analyses here are about photoperiod
(day length), the school calendar and latitude.

Everything in `outputs/` is produced from the files in `data/raw/` by the scripts in `src/`.
One command rebuilds all of it in under a minute.

## What the analysis asks

England's National Child Measurement Programme (NCMP) weighs and measures children in Reception
(age 4–5) and Year 6 (age 10–11) at school, at any time from September to July. Two questions are
addressed with public data only:

1. **Season.** How does the BMI distribution of measured children change from month to month
   through the school year, and does the pattern follow the calendar of day length or the school
   calendar? Data: NHS Digital's national monthly series, 2007/08–2018/19 (12 complete school years,
   13.1 million measurements). Each September a new cohort is measured, so each school year is
   analysed as a separate group of children.
2. **Latitude.** Is latitude, a proxy for the size of the annual photoperiod cycle, associated with
   local-authority prevalence once deprivation and ethnic composition are taken into account? Data:
   151 upper-tier local authorities, NCMP 2023/24, 2024/25 and 2025/26 analysed together.

## Main results

All numbers are read from `outputs/results.json` and `outputs/tables/`.

* **When children are measured.** 22% of Reception children are measured in September–December,
  46% in January–March and 31% in April–July (modal month March). Year 6 is measured earlier in the
  year: 41% / 35% / 24% (modal month November).
* **The BMI distribution falls in the autumn, most steeply just before the winter solstice.** Mean
  BMI z-score falls by 0.042 SD from November to December in Reception and 0.030 in Year 6, against
  0.018 and 0.003 from October to November (paired across the 12 school years, p = 0.002 in both).
  Days shorten by 1.1 h over that month, against about 2 h in each of the two months before it, so
  the fall is steepest when days are shortest, not when they shorten fastest. In Year 6, overweight
  and obesity also fall from September to October. The mean z-score then changes little over the
  winter (Table S13).
* **It falls again in the summer term, from May, while days are still lengthening.** The pace is
  the same before and after the summer solstice (Reception −0.022 and −0.031 SD from May to June and
  June to July; Year 6 −0.013 and −0.016), so the June–July fall does not by itself point to day
  length. Averaged over the year, the mean z-score changes by −0.024 SD per month while days shorten
  and −0.005 while they lengthen in Reception (p = 0.001), and by −0.015 and −0.001 in Year 6
  (p = 0.002).
* **Underweight is the mirror image.** It rises while days shorten (Year 6: +0.10 pp per month,
  against +0.01 while they lengthen, in all 12 years), most steeply from November to December. It is
  highest in midwinter (January in Year 6) and again in July, when overweight and obesity are lowest.
  Its month-to-month pattern correlates r = −0.98 (Year 6) and −0.87 (Reception) with mean BMI z-score.
* **The same rate of change per month of shortening days in both cohorts.** In a model with a term
  for months of shortening days elapsed since mid-September, mean BMI z-score falls 0.016 SD per
  month of shortening in Reception and 0.014 in Year 6 (difference p = 0.60). The shortening term is
  informed mainly by the autumn, because the summer-term fall starts before the summer solstice.
  * Year 6 shows no other change (steady trend −0.0005 SD per month, p = 0.70).
  * Reception also declines steadily through the year (−0.006 SD per month).
  * This form fits far better than a single linear school-year trend: AIC lower by 27 and 33 units
    for the mean z-score and 51 for Reception obesity.
  * For Year 6 obesity and overweight it fits better than unrestricted month effects.
* **The spring peak follows Easter, not the sun.** Reception obesity is 0.37 pp (0.13–0.61) higher
  in April than the mean of March and May, in 10 of 12 years. In years when Easter fell between
  23 March and 5 April, the peak was in April. When Easter fell between 8 and 24 April, it moved
  into May. Across both cohorts, the April-minus-May difference correlates with the date of Easter
  (r = −0.63, permutation p = 0.034). It does not correlate with the start of British Summer Time
  (r = −0.18). Day length on a given date does not move with Easter, so this feature is tied to
  the school calendar. No comparable peak followed the Christmas holiday.
* **What the monthly data cannot show.** The timing of the autumn fall is more consistent with a
  response to short, shortening days than with recovery from the summer holiday. But each month
  compares different children in different schools, and day length changes together with
  temperature, weather and time spent outdoors, so these data cannot attribute the fall to day
  length. The autumn term ends within days of the winter solstice, so timing alone cannot separate
  the two; a larger autumn fall at higher latitudes, in individual-level data, would.
* **Latitude and underweight reflect ethnic composition.** In the pooled 2023/24–2025/26 analysis
  adjusted for deprivation only (Model A), underweight is lower further north. The estimates are
  −0.19 pp per degree in Reception and −0.16 in Year 6 (spatial-HAC p = 0.011 and 0.013). Adding the
  Census 2021 ethnic composition of dependent children (Model B) removes the association: −0.03
  (p = 0.17) and −0.02 (p = 0.16). The same applies to the positive Model A associations for Reception
  overweight and obesity.
* **Year 6 severe obesity is higher further north once deprivation and ethnicity are adjusted for.**
  The estimate is +0.23 pp per degree (0.12–0.33), with Holm-adjusted p < 0.001 and spatial-HAC
  p = 0.003. Estimated year by year, it was +0.27, +0.23 and +0.17. There is no comparable
  association in Reception. This is an ecological association. Latitude in England carries many
  correlated exposures, so it is not evidence for photoperiod specifically.

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
  03_geographic_analysis.py    pooled latitude models A and B, robustness, spatial checks, single years
  04_temporal_analysis.py      timing, month-to-month change, shortening-days model, Easter
  05_figures_main.py           Figures 1-6
  06_figures_supplementary.py  Figures S1-S10
  07_tables.py                 formatted Tables 1-4 and S1-S13
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
python -m pytest tests         # 24 checks: methods, photoperiod model, data integrity
```

`python run_all.py --download` first re-downloads the API-sourced inputs (step 0). By default the
download script writes `*.downloaded.*` copies and reports whether they match the snapshots.
Figures use the Inter typeface if it is installed and fall back to Arial or DejaVu Sans if not.

## Figures and tables

| Paper | File |
|---|---|
| Figure 1 | `outputs/figures/Figure_1_measurement_timing_and_day_length` |
| Figure 2 | `outputs/figures/Figure_2_distribution_shift` |
| Figure 3 | `outputs/figures/Figure_3_within_year_shape_and_easter` |
| Figure 4 | `outputs/figures/Figure_4_actograms` |
| Figure 5 | `outputs/figures/Figure_5_latitude_coefficients` |
| Figure 6 | `outputs/figures/Figure_6_residual_maps` |
| Figures S1–S10 | `outputs/figures/Figure_S*` |
| Tables 1–4, S1–S13 | `outputs/tables/formatted/*.csv` and `outputs/tables/manuscript_tables.xlsx` |

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
* **Monthly series.** School-year fixed effects throughout, so July of one year is never linked
  to September of the next (a new cohort). Standard errors are clustered by school year.
  * Change between every pair of consecutive months, within each school year, averaged over the 12
    years (Table S13). Paired comparisons across years ask whether the autumn fall is steepest from
    November to December and whether the summer-term fall changes pace at the summer solstice.
  * Change while days shorten (mid-September to mid-December, mid-June to mid-July) against change
    while they lengthen.
  * The shortening-days model: months of shortening days elapsed since mid-September (counted day
    by day at 52.5°N), a steady linear change and an April term. Its rate is compared between
    cohorts.
  * The April peak, compared with March and May.
  * The shape of the within-year change, comparing six forms by AIC: a single linear trend; a
    decline that ends at an estimated month and is then level; the shortening-days forms; a linear
    trend plus day length; and unrestricted month effects.
  * The month pattern shared by the two cohorts, after each cohort's own linear change.
  * Location (mean) and spread (SD) of the BMI z-score distribution, recovered from the four UK90
    cut-points by probit regression.
  * The spring peak and the date of Easter: correlation with permutation p-values, leave-one-out
    ranges, and early-Easter against late-Easter years.
* **Local authorities.** The three NCMP years are stacked (up to 453 authority-years) with year
  fixed effects. Standard errors are clustered by authority and checked against an independent
  implementation, with a Holm correction across the 8 indicators.
  * Spatial-HAC (Conley) standard errors at 100 and 200 km on the stacked data. These allow for
    correlation within an authority across years and between neighbouring authorities.
  * Moran's I of authority-mean residuals.
  * Robustness checks: IDACI instead of IMD, London excluded, longitude added, weighting by children
    measured, region fixed effects, and bounds for values suppressed for disclosure control
    (1–7 children). Each year is also estimated separately.
* **IMD 2019 for April-2023 authorities.** Buckinghamshire, North Yorkshire and Somerset take their
  predecessor county values (identical boundaries). North and West Northamptonshire, Cumberland, and
  Westmorland and Furness take population-weighted means of their 2019 districts, and the same
  weighting reproduces the published county values (Table S2).
* **Ethnic composition.** Census 2021 table RM033: the percentage of dependent children who are
  Asian, Black, Mixed or Other, with White as the reference group. The BMI distribution differs by
  ethnic group, and ethnic composition varies with latitude in England (Figure S10).

## Data sources and licences

All inputs are public and used under the Open Government Licence v3.0. Sources, URLs, renamed files,
suppression rules and required attributions are listed in [`data/README.md`](data/README.md).
Contains OS data © Crown copyright and database right 2023. Source: Office for National Statistics
licensed under the Open Government Licence v3.0.
