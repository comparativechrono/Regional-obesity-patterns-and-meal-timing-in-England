# Data

Every file in `raw/` is a public dataset, stored unchanged apart from the renames noted below.
`SHA256SUMS.txt` lists the checksum of each file, and `tests/test_outputs.py` checks them.
Nothing in this folder is personal data. All sources publish only aggregate statistics, after their
own disclosure control.

## Files

| File(s) | Content | Source | Used for |
|---|---|---|---|
| `indicator90316.csv`, `indicator20601.csv`, `indicator90319.csv`, `indicator93194.csv` | NCMP Reception prevalence of underweight, overweight including obesity, obesity including severe obesity, and severe obesity | OHID Fingertips, [NCMP profile](https://fingertips.phe.org.uk/profile/national-child-measurement-programme), CSV export (that address now redirects to the [Obesity, physical activity and nutrition profile](https://fingertips.phe.org.uk/profile/obesity-physical-activity-nutrition), which carries the same indicators) | 2023/24 and 2024/25 local-authority prevalence |
| `indicator90320.csv`, `indicator20602.csv`, `indicator90323.csv`, `indicator93195.csv` | The same four indicators for Year 6 | as above | as above |
| `fingertips_api/indicator*.csv` | The same eight indicators from the Fingertips API, upper-tier local authorities from April 2023 (area type 502), all years to 2025/26 | `https://fingertips.phe.org.uk/api/all_data/csv/by_indicator_id?indicator_ids=<id>&child_area_type_id=502&parent_area_type_id=15`, downloaded 24 November 2026 | 2025/26 local-authority prevalence |
| `nati-chil-meas-prog-eng-2019-2020-tab.xlsx` | NCMP England 2019/20 tables. Tables 3 and 4 give national monthly numbers and prevalence by BMI category, September 2007 to March 2020. Table 5 is used only as an official local authority → region lookup | NHS Digital, [National Child Measurement Programme, England 2019/20 School Year](https://digital.nhs.uk/data-and-information/publications/statistical/national-child-measurement-programme/2019-20-school-year) | National monthly series; regions |
| `File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx`, `File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx` | English Indices of Deprivation 2019, local authority summaries (IMD, income and IDACI average scores) | MHCLG, [English indices of deprivation 2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | Deprivation covariate |
| `Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BFE.csv` | Attribute table of the December 2023 county and unitary authority boundaries, including centroid latitude and longitude | ONS Open Geography Portal, [Counties and Unitary Authorities (December 2023) Boundaries UK BFE](https://geoportal.statistics.gov.uk/datasets/ons::counties-and-unitary-authorities-december-2023-boundaries-uk-bfe-2) | Latitude and longitude of each local authority |
| `Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BUC_England.geojson` | Ultra-generalised (BUC) boundaries for England's 153 upper-tier authorities, WGS84 | ONS Open Geography Portal feature service (URL in `src/00_download_supplementary_sources.py`), downloaded 24 November 2026 | Drawing maps only |
| `census2021_RM033_dependent_children_ethnic_group_ctyua2023.csv` | Census 2021 RM033, ethnic group of dependent children, local authorities (county / unitary, April 2023) | ONS via [Nomis](https://www.nomisweb.co.uk/datasets/c2021rm033) (dataset NM_2133_1; API URL in `src/00_download_supplementary_sources.py`), downloaded 24 November 2026 | Ethnic-composition covariate (Model B) |

Renamed files: `Indicator20601.csv` became `indicator20601.csv` for consistent capitalisation. The
boundary attribute table's download name ended in a random numeric suffix, which was removed.

## Notes on the NCMP data

* **Indicators.** Fingertips IDs: Reception 90316 (underweight), 20601 (overweight including
  obesity), 90319 (obesity including severe obesity), 93194 (severe obesity). Year 6 uses 90320,
  20602, 90323 and 93195 for the same thresholds. BMI categories use the UK90 reference with
  population-monitoring cut-points: underweight at or below the 2nd centile, overweight at or above
  the 85th, obesity at or above the 95th, and severe obesity at or above the 99.6th.
* **Geography.** 153 upper-tier local authorities (April 2023 boundaries). Fingertips publishes the
  Isles of Scilly combined with Cornwall and the City of London combined with Hackney, so the
  analysis uses 151 authorities. The census covariate is summed over the same pairs.
* **Disclosure control.** Counts of 1–7 children are suppressed. The number of authorities analysed
  per indicator is therefore below 151 for underweight (both cohorts) and Reception severe obesity.
  For example, Reception underweight in 2023/24 uses 143 authorities. `src/03_geographic_analysis.py` refits each affected
  model with the suppressed values set to 1, 4 and 7 children.
* **Consistency of the two Fingertips sources.** For 2023/24 and 2024/25, values, counts and
  denominators in `fingertips_api/` are identical to the original exports. This is checked by
  `tests/test_outputs.py::test_api_files_reproduce_original_exports`. Fingertips data can be revised
  after publication, so a fresh download may not match the snapshot exactly.
* **Monthly series.** Tables 3 and 4 run from September 2007 to March 2020. Measurement stopped
  after March 2020 because of COVID-19 school closures. The analysis uses the 12 complete school
  years, 2007/08 to 2018/19, and adds September 2019 to February 2020 in a sensitivity analysis.

## Licences and attribution

All sources are Crown copyright and are used under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

* NCMP local-authority data: Office for Health Improvement and Disparities, Public health profiles (Fingertips).
* NCMP 2019/20 tables: NHS Digital (now part of NHS England).
* English Indices of Deprivation 2019: Ministry of Housing, Communities and Local Government.
* Census 2021 and geography: Source: Office for National Statistics licensed under the Open
  Government Licence v3.0. Contains OS data © Crown copyright and database right 2023.
