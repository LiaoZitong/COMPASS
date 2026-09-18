# COMPASS core code 1.0.1

Version 1.0.1 aligns the uncertainty and testing-selection sensitivity analyses
with the strict focal-species leave-one-out probability layer used throughout
the revised manuscript.

## Robustness update

- The profile-likelihood analysis propagates censored-record likelihood
  uncertainty for 24,254 identified protective model cells across 413 measured
  protective contexts.
- The MNAR analysis models direct-testing propensity without using the tail
  outcome as a predictor, applies inverse-probability weighting, and evaluates
  prespecified tail-positive testing odds ratios of 0.25, 0.5, 1, 2, and 4.
- The five national panel members remain unchanged throughout the tested MNAR
  range; rank-order changes are retained in the machine-readable output.
- SI Figure S6 is rebuilt from these strict-LOO outputs with editable SVG/PDF,
  300-dpi PNG, 600-dpi TIFF, source-data tables, and a figure contract.

## Scope

These analyses quantify stated record-censoring uncertainty and sensitivity to
unidentifiable testing-selection assumptions. They leave the primary strict-LOO
probability matrix and national Top-5 unchanged and do not estimate unmeasured
absolute toxicity or replace formal SSD/HC5 assessment.

## Data boundary

The repository contains source code, configuration, numerical contracts, tests,
and release metadata. It does not redistribute EPA ECOTOX or other third-party
raw, processed, or derived analysis datasets. Those sources must be obtained
from their official providers and prepared to the documented input contracts.

No archival DOI is claimed for this release.
