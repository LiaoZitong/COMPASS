# COMPASS core code 1.0.0

Version 1.0.0 is the reproducibility release supporting the R1 revision of
“Developing Complementary Aquatic Sentinel Panels for Cross-Chemical Monitoring
and Regional Localization.”

## R1 analysis layer

- AP01 freezes the censoring, model-cell, SSD-context, and probability-eligibility denominators.
- AP02 applies strict focal-species leave-one-out estimation to the 1,613 originally n ≥ 6 contexts; 413 originally n = 5 contexts remain separate diagnostics.
- AP05 re-estimates the target-specific working dependence values.
- AP03A audits available-member Top-5/HC5 concordance and shared-input sensitivity.
- AP03B evaluates maximum, balanced effect-family, mortality-excluded, and common-support chemical follow-up rankings.
- AP04 evaluates localization weights, occurrence policies, and priority-chemical support reduction.
- G3 integrates the upstream gates and freezes the manuscript-facing R1 values.

The R1 lower-5% Top-5 is *Gastrophryne carolinensis*, *Neocloeon triangulifer*,
*Hyalella azteca*, *Daphnia ambigua*, and *Daphnia magna*. The same fixed
sequence has expected capture 0.337764, 0.624097, and 0.887755 at the lower-5%,
lower-10%, and lower-20% evaluation targets, respectively.

## Data boundary

The repository contains source code, configuration, numerical contracts, tests,
and release metadata. It does not redistribute EPA ECOTOX or other third-party
raw, processed, or derived analysis datasets. Those sources must be obtained
from their official providers and prepared to the documented input contracts.

## Release verification

The release is blocked unless the R1 gates, numerical checkpoints, data-free
tests, source audit, site-data cross-checks, and tracked-file checksums pass.
The COMPASS Results Explorer is a separate static deployment of precomputed R1
outputs and does not execute the model in the browser.
