# Static R1 site-data export

`code/revision_r1/export_r1_site_data.py` is the release path from the approved
R1 analysis to the browser-readable Explorer files. It reads the strict-LOO R1
probability matrices, the AP05 dependence audit, the AP04 localization ledger,
the Figure 2/5 source tables, and unchanged base metadata inputs. It does not
read Word, PDF, screenshots, or figure pixels.

The exporter requires exact reproduction of all 60 frozen R1 Figure 2 prefix
values before it extends the lower-5% greedy sequence through all 2,144
candidates. Every fixed-sequence prefix is evaluated at the lower-5%,
lower-10%, and lower-20% targets with the corresponding strict-LOO matrix,
national chemical weights, and target-specific AP05 dependence value. Ranks
21–2,144 are explicitly labeled as extended post hoc evaluations. Regional
outputs retain the 32 approved R1 localized panels, the PA/TN/WV national-panel
fallbacks, and documented defaults for states outside the 35-state occurrence
weight universe. Census 2024 state geometry is obtained separately.

Example:

```powershell
.\.venv\Scripts\python.exe .\code\revision_r1\export_r1_site_data.py `
  --analysis-root . `
  --revision-root .\results\revision_r1 `
  --out .\outputs\site_data `
  --boundary-zip .\data\external\cb_2024_us_state_500k.zip `
  --git-commit <full-commit-sha> `
  --github-release https://github.com/LiaoZitong/COMPASS/releases/tag/v1.0.1

.\.venv\Scripts\python.exe .\code\revision_r1\validate_revision_results.py `
  --revision-root .\results\revision_r1 `
  --expected .\config\r1_expected_results.json `
  --site-data .\outputs\site_data
```

Copy the complete generated directory to `site/explorer/site_data/`, validate
the static application locally, and commit the matching `site/` snapshot. The
GitHub Pages workflow publishes `site/` at
https://liaozitong.github.io/COMPASS/explorer/. The export records source
hashes, the Git commit, the GitHub release URL, and a combined R1
source-analysis identifier. The DOI field remains null unless a real archival
DOI has been returned and verified.
