# Local run record

Validation host: native Windows, Python 3.11.9 from the source project's controlled `.venv`. The package itself remains environment-neutral and records exact dependencies under `config/`.

| Command class | Result |
|---|---|
| `python -m compileall` for code, tests, and site scripts | pass |
| `python -m unittest discover -s tests -v` | 6 tests passed |
| `python code/run_analysis.py --mode full --dry-run --seed 20260622` | all stages 01–22 resolved in semantic order |
| `python code/validate_inputs.py --level all` | eight external inputs absent as expected in the data-free package |
| `python code/validate_results.py --profile analysis` | 38/38 checks passed against frozen v16.5 outputs and site export |
| `python code/audit_release.py --profile private` | pass, zero failures |
| `python scripts/validate_site.py --profile local` | pass |

The source workspace had no usable Git history, so equivalent versioning is supplied through modification times and SHA-256 records in `release/source_manifest_sha256.csv`, `release/key_output_manifest_sha256.csv`, and `release/checksums.sha256`. No production analysis stage was rerun because the requested GitHub deliverable excludes all inputs and result data.
