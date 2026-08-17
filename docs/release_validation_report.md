# Release validation report

Local release candidate: `0.2.0`
Frozen analysis: `v16.5`
Data freeze: `2026-08-02`
Validation date: `2026-08-17`

## Outcome

The independent GitHub package passes all locally executable release checks. It contains the compact core code and documentation needed to prepare inputs and reproduce the workflow. Large public and provider-hosted datasets, including EPA ECOTOX, remain obtainable from their official services and are not mirrored in the repository.

| Check | Outcome |
|---|---:|
| Python compilation | pass |
| Data-free unit tests | 8/8 pass |
| Full 22-stage dry run | pass |
| Frozen analysis/site consistency | 51/51 pass |
| Static-site automated validation | 34/34 pass |
| Private-release path, secret, marker, extension, and size audit | 942/942 pass |
| Chromium Edge desktop, 500-pixel mobile, interaction, and private-context QA | 10 screenshots; pass |

The dry run reports eight externally acquired inputs in the compact checkout. This is expected because the large source datasets are distributed through their official services. Required filenames, provider links, access dates, and field contracts are documented under `data/`.

## Frozen checkpoints

- National Top-5: *Gastrophryne carolinensis*, *Daphnia ambigua*, *Neocloeon triangulifer*, *Daphnia magna*, and *Hyalella azteca*.
- The site export contains a complete, unique ordering of all 2,144 formal-binomial candidates. Ranks 1-20 reproduce the frozen panel sequence; ranks 21-2,144 continue the same deterministic greedy selection objective and are explicitly scope-labeled.
- Cumulative expected capture is present for all 2,144 prefixes at all three targets (6,432 values). The exporter reproduces the 60 frozen rank-1--20 curve values before accepting the post hoc extension.
- Fixed Top-5 expected capture: 30.6% at lower-5%, 58.0% at lower-10%, and 87.7% at lower-20%.
- Regional export: all 50 states plus the District of Columbia. Thirty-two have frozen localized panels; the other 19 show the fixed national Top-5 as a documented default without fabricating localized estimates. PA, TN, and WV each have one eligible priority chemical.
- The interactive map contains 51 Census-derived geometries, synchronized map/menu selection, selected-state emphasis, and five-species hover/focus details.
- Complete machine-readable comparisons are recorded in `release/result_crosswalk.csv` and `config/expected_results.json`.

## Pending external release checks

Public-license approval, release tagging, Zenodo archiving, DOI insertion, and a Google Chrome-specific pass remain for the eventual public archival code release. The GitHub repository remains private while the license decision is pending; the companion results site is deployed separately with public access.
