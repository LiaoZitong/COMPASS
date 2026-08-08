# Static site-data export

`code/export_site_data.py` is the sole analysis-to-site export path. It reads frozen result tables directly and writes versioned JSON plus flattened CSV downloads. It does not read Word, PDF, screenshots, or figure pixels.

The exporter reproduces the frozen lower-5% Top-20 sequence and attaches lower-10% and lower-20% evaluations to those same validated prefixes. It then continues the deterministic lower-5% greedy objective through all 2,144 candidate species for complete catalog exploration. The continuation is explicitly scope-labeled, and dependence-adjusted cumulative capture remains null beyond rank 20. Regional outputs use the frozen lower-5% state sequences. Expected capture remains a probability in exported files; the browser formats it as a percentage. Absolute gains are stored and displayed in percentage points.

Example:

```powershell
.\.venv\Scripts\python.exe .\code\export_site_data.py --analysis-root . --out .\outputs\site_data
```

Copy the complete generated directory to the independent `COMPASS_Site_Package/site_data/`, build the site, then run:

```powershell
.\.venv\Scripts\python.exe .\code\validate_results.py --root . --profile release --site-data ..\COMPASS_Site_Package\site_data
```

The export records source-file checksums and a combined source-analysis identifier. Repository commit, release, and DOI fields remain null until those objects exist and must be updated in one release-metadata step.
