# COMPASS core reproducibility package

COMPASS is a research workflow for selecting compact aquatic sentinel panels from sparse and imbalanced toxicity evidence. It estimates measured-tail probabilities, selects a fixed national sequence at the lower-5% target, evaluates the same sequence at broader lower-tail targets, localizes panels with state-priority chemical and species-relevance weights, and identifies evidence-acquisition priorities.

This repository is a compact, code-only reproducibility package for the formal R1 release. It contains the base v16.5 analysis, the ordered reviewer-requested R1 analysis layer, and the figure/site export code needed to reproduce the approved strict focal-species leave-one-out results after the required inputs are obtained separately. Large public and provider-hosted source datasets, including EPA ECOTOX, are not mirrored in the repository; users can obtain them from the official sources documented below.

> The repository does not redistribute the full ECOTOX-derived or other third-party raw datasets. Full-data execution requires users to obtain these data from the cited providers and place them in the expected local directories.

## Scientific scope

The R1 national sequence was constructed at the lower-5% measured-tail target (`x = 0.95`) from strict focal-species leave-one-out probabilities. The same membership order is evaluated at the lower-10% and lower-20% targets without reoptimization. Expected capture is stored as a probability from 0 to 1 and may be displayed as a percentage. Regional localization is fixed at the lower-5% target.

The outputs support protection-oriented screening and follow-up planning within the analyzed evidence domain. Confirmatory apical testing and complete, framework-appropriate measured evidence remain necessary for formal SSD, HC5, and water-quality-criterion work. Chemical follow-up scores rank evidence-acquisition priority; their interpretation requires the stated chemical and evidence context.

## What is included

- `code/pipeline/`: frozen semantic stages 01–22;
- `code/revision_r1/`: ordered AP01 → AP02 → AP05 → AP03A → AP03B → AP04 → G3 analyses, R1 validator, and Explorer exporter;
- `code/run_analysis.py` and `run_analysis.ps1`: sequential public runner;
- `code/figures/`: the five main Matplotlib figure scripts;
- `code/validate_inputs.py`: lightweight schema and value checks;
- `code/validate_results.py`: release-blocking scientific checkpoints;
- `code/export_site_data.py`: one-way export of frozen results for the static explorer;
- `config/`: dependency locks, seeds, targets, and expected checkpoints;
- `data/`: source and placement documentation only;
- `tests/`: base and R1 data-free tests using temporary synthetic fixtures;
- `docs/` and `release/`: audit, provenance, validation, and release metadata.

Local verification results are summarized in [`docs/release_validation_report.md`](docs/release_validation_report.md), with the scientific crosswalk in [`docs/result_consistency_report.md`](docs/result_consistency_report.md).

The package excludes third-party acquisition/ETL utilities, manuscript and SI assembly, Word templates, confidential review material, historical runs, caches, logs, and all real data/results.

## Reproduction levels

### A. Full-data reconstruction

Obtain the cited third-party sources and prepare the canonical ECOTOX partitions and auxiliary processed tables described in [`data/README.md`](data/README.md). Raw acquisition code is outside this core-code release, so this level also requires an independently audited preparation of the documented input contracts.

### B. Canonical/processed-input reproduction — primary route

Place the separately obtained inputs at the paths documented in [`data/README.md`](data/README.md), then run stages 01–22. No input table is bundled here.

### C. Demo/test run

The test suite creates temporary synthetic fixtures to test validators and package behavior. These fixtures test software contracts only and never represent manuscript results.

## Environment

Python 3.11 is required. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\config\requirements-lock.txt
```

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Prepare and validate inputs

The expected layout is:

```text
data/
├── intermediate/ecotox_canonical_partitions/part_*.csv.gz
├── processed/chemical_master.csv.gz
├── processed/state_chemical_exposure_2024.csv.gz
├── processed/state_species_occurrence_with_usgs_nas.csv.gz
└── reference/
    ├── AOP_MOA_data.zip
    ├── lifestage_codes.txt
    ├── regulatory_baseline_catalog.csv
    └── species_traits_curated.csv
```

Run the lightweight checks before analysis:

```powershell
.\.venv\Scripts\python.exe .\code\validate_inputs.py --root . --level upstream
```

## Run order

The public runner retains the frozen semantic order:

```powershell
# Show the planned commands; missing data are reported without execution
.\run_analysis.ps1 -Mode Full -DryRun

# Production-strength stages 01–22
.\run_analysis.ps1 -Mode Full -Seed 20260622 -Bootstrap 500 -ProfileDraws 2000 -TraditionalRandomReplicates 500

# Resume from existing stage 01–05 outputs
.\run_analysis.ps1 -Mode Core -Seed 20260622 -Bootstrap 500 -TraditionalRandomReplicates 500

# Profile-likelihood and MNAR sensitivity only
.\run_analysis.ps1 -Mode Robustness -Seed 20260622 -ProfileDraws 2000
```

Stages 01–05 prepare censored toxicity cells, evidence layers, chemistry/trait annotations, and national/state chemical weights. Stages 06–20 estimate and calibrate measured-tail probabilities, select national and localized panels, compare framework implementations, and build testing priorities. Stages 21–22 propagate profile-likelihood uncertainty and assess MNAR scenarios. See [`docs/methods_to_code.md`](docs/methods_to_code.md).

After the base stages have completed, run the R1 analysis layer in its frozen dependency order:

```powershell
.\.venv\Scripts\python.exe .\code\revision_r1\run_revision_analysis.py --analysis-root . --out results\revision_r1
```

Do not reorder these packages: AP05 depends on AP02, AP03A depends on AP02 and AP05, AP03B depends on AP02 and AP03A, AP04 consumes all earlier gates, and G3 is the final integration gate. See [`docs/r1_revision_workflow.md`](docs/r1_revision_workflow.md).

## Outputs

Analysis products are written beneath `results/`; logs are written beneath `logs/`. Both are ignored by Git. The primary R1 checkpoints are:

- national Top-5 membership, frozen R1 Top-20 ordering, and a deterministic full 2,144-species continuation for the explorer;
- fixed Top-5 expected capture of 33.8%, 62.4%, and 88.8% at the lower-5%, lower-10%, and lower-20% targets;
- frozen panel-size coverage checkpoints at `k = 1, 5, 10, 20`, plus post hoc cumulative expected capture for every prefix through rank 2,144;
- comparator, Top5-apical, warning, and regional summaries;
- 32 supported localized state panels, explicit national-default records for the remaining states and the District of Columbia, and Census-derived interactive map geometry.

Validate computed results with:

```powershell
.\.venv\Scripts\python.exe .\code\validate_results.py --root . --profile analysis
.\.venv\Scripts\python.exe .\code\revision_r1\validate_revision_results.py --revision-root .\results\revision_r1 --expected .\config\r1_expected_results.json
```

The command exits nonzero on any mismatch and never edits a result.

## Figures 1–5

After analysis outputs exist:

```powershell
.\.venv\Scripts\python.exe .\code\figures\render_all.py
```

Figure 5 also requires the U.S. Census cartographic boundary archive at `data/external/cb_2024_us_state_500k.zip`; the archive is not redistributed. Generated SVG, PDF, TIFF, PNG, source tables, and figure manifests are written to `outputs/figures/`. See [`docs/figure_reproduction.md`](docs/figure_reproduction.md).

## Static explorer export

The browser never executes Equations 1–5 or reoptimizes a panel. Export the frozen R1 precomputed results with:

```powershell
.\.venv\Scripts\python.exe .\code\revision_r1\export_r1_site_data.py --analysis-root . --revision-root .\results\revision_r1 --out .\outputs\site_data --git-commit <full-commit-sha> --github-release https://github.com/LiaoZitong/COMPASS/releases/tag/v1.0.0
```

The export contains the frozen R1 Top-20 sequence, a scope-labeled continuation through all 2,144 candidates, cumulative expected capture for all three targets at every prefix, all 50 states plus the District of Columbia, and Census-derived map geometry. The independent site package consumes that directory. See [`docs/site_export.md`](docs/site_export.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

CI compiles the public Python sources, runs data-free tests, and scans the tracked candidate files for accidental secrets, absolute local paths, and prohibited large-data extensions.

## Data availability

The GitHub code package does not mirror source or analysis datasets. The principal source datasets, including EPA ECOTOX, are available from their official providers; access links, provenance dates, and local input contracts are documented in [`data/README.md`](data/README.md). The versioned source is released at [github.com/LiaoZitong/COMPASS](https://github.com/LiaoZitong/COMPASS), with the R1 snapshot identified by tag `v1.0.0` and its full commit SHA. No archival DOI is claimed unless and until a separate archive returns a verified identifier. The synchronized statement is in [`docs/data_availability_statement.md`](docs/data_availability_statement.md).

## Known limitations

The main inputs are large public or provider-hosted datasets, including EPA ECOTOX, which are distributed through their official services and are impractical to mirror in a compact GitHub code repository. External reproduction therefore begins by obtaining those sources and preparing inputs that satisfy the frozen contracts. Chemical-held-out assessment is an internal generalization check within the analyzed evidence corpus. Profile-likelihood intervals address the stated record-censoring and working-SSD uncertainty, while MNAR scenarios provide sensitivity bounds. See [`docs/known_limitations.md`](docs/known_limitations.md).

## License, citation, and contact

The source is publicly viewable under the included all-rights-reserved source-availability notice; no open-source reuse licence is implied. Citation metadata are provided in `CITATION.cff`; no archival DOI is claimed. Repository coordination: [LiaoZitong](https://github.com/LiaoZitong).
