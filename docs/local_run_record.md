# Local run record

Validation host: native Windows, Python 3.11.9 from the controlled analysis
environment. All task-created caches and temporary outputs stayed below the
COMPASS package root.

| Command class | Result |
|---|---|
| Python compilation for public code and tests | pass |
| Base `unittest` suite | 8/8 pass |
| Revision data-free/integration assertions, including strict-LOO robustness | 16/16 pass |
| `run_revision_analysis.py --dry-run` | 8/8 packages resolved in the frozen order |
| `validate_revision_results.py` against approved gates | 56/56 pass |
| Explorer export plus cross-package validation | 62/62 pass |
| `audit_release.py --profile public` | pass, zero failures |

The author-approved primary analyses were not rerun during packaging. The
strict-LOO profile-likelihood and MNAR robustness stage was rerun with 2,000
profile draws after its probability inputs were aligned with the primary
strict-LOO layer. Release validation reads the completed gates, source tables,
matrices, and integrated numerical freeze, while the data-free tests exercise
the key mathematical and routing contracts.

The package is versioned in Git; base and R1 provenance remain recorded through
the source/output manifests, machine-readable validation reports, and
`release/checksums.sha256`. The Explorer export is regenerated from the approved
R1 strict-LOO state and is validated before site deployment.
