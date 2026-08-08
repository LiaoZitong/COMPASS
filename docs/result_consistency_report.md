# Frozen-result consistency report

The frozen source analysis and the independent static-site export agree on all 41 programmed scientific and cross-package checks at an absolute tolerance of `1e-9`.

| Checkpoint | Frozen analysis | Static site | Status |
|---|---:|---:|---:|
| Top-5 lower-5% expected capture | 0.3055373085768699 | 0.3055373085768699 | pass |
| Same Top-5 lower-10% expected capture | 0.5802658236765963 | 0.5802658236765963 | pass |
| Same Top-5 lower-20% expected capture | 0.8765105591257395 | 0.8765105591257395 | pass |
| National sequence basis | 0.95 | 0.95 | pass |
| Supported localized states | 32 | 32 | pass |
| Analysis version | v16.5 | v16.5 | pass |

The programmed checks also cover the frozen Top-20 order; the complete, unique 2,144-species site ordering; panel-size checkpoints at `k = 1, 5, 10, 20`; Top5-apical and warning metrics; regional means, maximum gain, and insufficient-support states; and the EPA WET/random comparator checkpoints. The validator reads frozen outputs without modifying them. Hashes of all source tables used for the site export are stored in the site metadata and `release/key_output_manifest_sha256.csv`; the large source and result tables remain outside this compact code package.
