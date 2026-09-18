# Local run record

Validation host: native Windows, Python 3.11.9 from the controlled analysis
environment. All task-created caches and temporary outputs stayed below the
COMPASS package root.

| Command class | Result |
|---|---|
| Python compilation for public code and tests | pass |
| Base `unittest` suite | 8/8 pass |
| R1 AP01/AP02/AP03A/AP03B/AP04/AP05/G3 assertion entry points | 13/13 pass |
| `run_revision_analysis.py --dry-run` | 7/7 packages resolved in the frozen order |
| `validate_revision_results.py` against approved R1 gates | 41/41 pass |
| R1 Explorer export plus cross-package validation | 47/47 pass |
| `audit_release.py --profile public` | pass, zero failures |

The computationally intensive R1 analyses were not rerun during packaging: the
author-approved outputs had already been generated in the required dependency
order. Release validation reads the completed gates, source tables, matrices,
and integrated numerical freeze, while the data-free tests exercise the key
mathematical and routing contracts.

The package is versioned in Git; base and R1 provenance remain recorded through
the source/output manifests, machine-readable validation reports, and
`release/checksums.sha256`. The Explorer export is regenerated from the approved
R1 strict-LOO state and is validated before site deployment.
