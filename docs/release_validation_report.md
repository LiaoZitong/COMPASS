# Release validation report

Local release candidate: `0.1.0-draft`  
Frozen analysis: `v16.5`  
Data freeze: `2026-08-02`  
Validation date: `2026-08-08`

## Outcome

The independent GitHub package passes all locally executable release checks. It contains the compact core code and documentation needed to prepare inputs and reproduce the workflow. Large public and provider-hosted datasets, including EPA ECOTOX, remain obtainable from their official services and are not mirrored in the repository.

| Check | Outcome |
|---|---:|
| Python compilation | pass |
| Data-free unit tests | 7/7 pass |
| Full 22-stage dry run | pass |
| Frozen analysis/site consistency | 41/41 pass |
| Static-site automated validation | 27/27 pass |
| Private-release path, secret, marker, extension, and size audit | 942/942 pass |
| Chromium desktop, mobile-width, interaction, and private-browsing QA | pass |

The dry run reports eight externally acquired inputs in the compact checkout. This is expected because the large source datasets are distributed through their official services. Required filenames, provider links, access dates, and field contracts are documented under `data/`.

## Frozen checkpoints

- National Top-5: *Gastrophryne carolinensis*, *Daphnia ambigua*, *Neocloeon triangulifer*, *Daphnia magna*, and *Hyalella azteca*.
- The site export contains a complete, unique ordering of all 2,144 formal-binomial candidates. Ranks 1-20 reproduce the frozen panel sequence; ranks 21-2,144 continue the same deterministic greedy selection objective and are explicitly scope-labeled.
- Fixed Top-5 expected capture: 30.6% at lower-5%, 58.0% at lower-10%, and 87.7% at lower-20%.
- Localized panels: 32 supported states; PA, TN, and WV have insufficient chemical support under the frozen minimum-support rule.
- Complete machine-readable comparisons are recorded in `release/result_crosswalk.csv` and `config/expected_results.json`.

## Pending external release checks

Release tagging, public-license approval, Zenodo archiving, DOI insertion, and a Google Chrome-specific pass remain for the eventual public archival release. The private repository and owner-access Sites deployment are recorded in the task handoff after remote verification.
