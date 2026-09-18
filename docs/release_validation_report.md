# Release validation report

Release candidate: `1.0.1`
Primary revision analysis: `R1-strict-LOO-2026-09-18`
Base analysis: `v16.5`
Data freeze: `2026-08-02`
Validation date: `2026-09-18`

## Outcome

The compact package passes the locally executable release checks against the
author-approved R1 analysis outputs. Third-party inputs and derived matrices are
not redistributed; their contracts and hashes remain documented.

| Check | Outcome |
|---|---:|
| Python compilation | pass |
| Base data-free unit tests | 8/8 pass |
| R1 data-free/integration test assertions | 16/16 pass |
| Ordered R1 runner dry run | 8/8 steps in the frozen dependency order |
| R1 analysis validation | 56/56 pass |
| R1 analysis plus Explorer cross-check | 62/62 pass |
| Public-source path, secret, marker, extension, size, and licence-notice audit | pass; zero failures |

The numerical validation reads the completed R1 gates rather than rerunning the
approved computationally intensive analysis. This is intentional: the R1
analysis had already been executed in dependency order and approved by the
authors, and the release task verifies that exact frozen state.

## R1 checkpoints

- Eligibility: 426,588 records; 2,026 eligible protective SSD contexts; 1,613
  originally n ≥ 6 strict-LOO contexts; 413 originally n = 5 diagnostics; 2,144
  candidate species.
- National Top-5: *Gastrophryne carolinensis*, *Neocloeon triangulifer*,
  *Hyalella azteca*, *Daphnia ambigua*, and *Daphnia magna*.
- Fixed x = 0.95 Top-5 expected capture: 33.776% at lower-5%, 62.410% at
  lower-10%, and 88.775% at lower-20%.
- Working dependence: 0.132467, 0.119452, and 0.117446 at x = 0.95, 0.90, and
  0.80, respectively.
- Available-member validation: n = 115; Spearman ρ = 0.847290; no chemical has
  all five panel members measured.
- Follow-up ranking: 639 primary-score chemicals; 145 mortality-excluded
  scoreable and 494 not scoreable.
- Localization: 32 supported states; mean model-expected gain 6.7506 percentage
  points; PA, TN, and WV retain the national fallback.
- Explorer export: 2,144 unique species, 6,432 prefix-target values, 51 state/DC
  records, 51 map geometries, and the same R1 Top-20/capture checkpoints.
- Strict-LOO robustness: 24,254 profiled cells across 413 contexts; 17,037
  directly observed species-target pairs in the MNAR propensity layer; unchanged
  Top-5 membership across odds ratios 0.25–4.

Machine-readable evidence is stored in `docs/r1_result_validation_report.json`,
`docs/r1_site_validation_report.json`, `docs/release_audit_report.json`,
`config/r1_expected_results.json`, and `release/checksums.sha256`.

## External release record

The release target is https://github.com/LiaoZitong/COMPASS with tag `v1.0.1`.
The repository uses the included all-rights-reserved source-availability notice;
no open-source reuse licence or archival DOI is claimed. The release tag and
full commit SHA are verified after push and are then synchronized into the
manuscript, response letter, cover letter, and Explorer metadata.
