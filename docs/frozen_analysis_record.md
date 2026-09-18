# Base v16.5 frozen analysis record (historical)

This file records the analysis baseline from which the R1 modules were run. It
is not the current release result ledger. See `docs/r1_revision_workflow.md`,
`config/r1_expected_results.json`, and the two R1 validation reports for the
current strict-LOO release.

## Identity

- Manuscript title: *Selecting Sensitive Local Aquatic Sentinel Panels for Protection-Oriented Monitoring under Sparse and Imbalanced Evidence*.
- Analysis version: v16.5.
- Data freeze date: 2026-08-02, defined by the latest frozen national-sequence/cross-target outputs used by the synchronized manuscript.
- Author-confirmed manuscript source version: 2026-07-22.
- Author-confirmed Supporting Information source version: 2026-07-21.
- Current generated manuscript output synchronized: 2026-08-02.
- Git version identifier: unavailable because the source workspace has no valid Git history; SHA-256 manifest used instead.

## Environment and stochastic settings

- CPython 3.11.9, Windows native.
- Default pipeline seed: 20260622.
- Production bootstrap replicates: 500.
- Profile-likelihood draws: 2000.
- Traditional random-panel replicates: 500.
- Figure 2 feasible-panel seed: 20260701; 2000 draws per design and target.
- MNAR testing odds ratios: 0.25, 0.5, 1, 2, and 4.

Exact package versions are in `config/requirements-lock.txt`. The scientific settings are in `config/reproduction_config.json`.

## Frozen sequence and key checkpoints

National Top-5:

1. *Gastrophryne carolinensis*
2. *Daphnia ambigua*
3. *Neocloeon triangulifer*
4. *Daphnia magna*
5. *Hyalella azteca*

National ranks 6–10 are *Ceriodaphnia dubia*, *Daphnia pulex*, *Daphnia carinata*, *Gammarus lacustris*, and *Gammarus fasciatus*.

The sequence was constructed at `x = 0.95` (lower-5%). Expected capture of the same Top-5 is 0.3055373086, 0.5802658237, and 0.8765105591 at the lower-5%, lower-10%, and lower-20% evaluations, respectively.

Regional localization is fixed at `x = 0.95`. Thirty-two states meet the minimum support threshold. Mean localized expected capture is 0.1380271077 versus 0.0681537919 for the fixed national Top-5 under the same state-specific weights; mean gain is 6.9873 percentage points. Connecticut has the maximum frozen gain, 9.8748 percentage points. Pennsylvania, Tennessee, and West Virginia each have one eligible priority chemical and were not localized.

## Output sources

- National sequence, fixed cross-target curves, and Top-5 audit: `results/panels/`.
- Regional sequences and comparisons: `results/state_panels/`.
- Top5-apical and warning outputs: `results/warning_hc5_bridge/` and `results/warning_lead_potential/`.
- Comparator outputs: `results/hc5_framework_comparison/`, `results/panels/`, and Figure 2 source tables.
- Testing-priority outputs: `results/testing/`.
- Profile/MNAR outputs: `results/validation/`.
- Main Figures 1–5: source `manuscript_plot/figure_01` through `figure_05`.
- SI Figures S1–S7: source `src/build_si_figures.py` and the synchronized SI manifests; code not copied into this core-only package.

The public equivalent version record is completed by `outputs/source_manifest_sha256.csv`, `outputs/key_output_manifest_sha256.csv`, and `outputs/run_metadata.json`.
