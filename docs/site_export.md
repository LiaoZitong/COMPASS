# Static site-data export

`code/export_site_data.py` is the sole analysis-to-site export path. It reads frozen result tables directly and writes versioned JSON plus flattened CSV downloads. It does not read Word, PDF, screenshots, or figure pixels.

The exporter reproduces the frozen lower-5% Top-20 sequence and then continues the deterministic lower-5% greedy objective through all 2,144 candidate species. Every fixed-sequence prefix is evaluated at the lower-5%, lower-10%, and lower-20% targets with the target-specific frozen probability matrix, national chemical weights, and Gaussian-copula working dependence parameter. The 60 published rank-1--20 values are required to match before export; ranks 21--2,144 are explicitly labeled as post hoc extended-prefix evaluations. Regional outputs retain the 32 frozen lower-5% localized panels and add documented national-default records for all other states and the District of Columbia. The map geometry is generated from the separately obtained U.S. Census 2024 1:500,000 state cartographic boundary archive. Expected capture remains a probability in exported files; the browser formats it as a percentage. Absolute gains are stored and displayed in percentage points.

The export therefore requires all three frozen probability matrices, `results/weights/national_priority_chemicals.csv`, `results/panels/panel_probability_manifest.json`, and `data/external/cb_2024_us_state_500k.zip` in addition to the previously documented result tables.

Example:

```powershell
.\.venv\Scripts\python.exe .\code\export_site_data.py --analysis-root . --out .\outputs\site_data
```

Copy the complete generated directory to the independent `COMPASS_Site_Package/site_data/`, build the site, then run:

```powershell
.\.venv\Scripts\python.exe .\code\validate_results.py --root . --profile release --site-data ..\COMPASS_Site_Package\site_data
```

The export records source-file checksums and a combined source-analysis identifier. Repository commit, release, and DOI fields remain null until those objects exist and must be updated in one release-metadata step.
