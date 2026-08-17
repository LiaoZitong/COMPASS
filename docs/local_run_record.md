# Local run record

Validation host: native Windows, Python 3.11.9 from the source project's controlled `.venv`. The package itself remains environment-neutral and records exact dependencies under `config/`.

| Command class | Result |
|---|---|
| `python -m compileall` for code, tests, and site scripts | pass |
| `python -m unittest discover -s tests -v` | 8 tests passed |
| `python code/run_analysis.py --mode full --dry-run --seed 20260622` | all stages 01–22 resolved in semantic order |
| `python code/validate_inputs.py --level all` | eight external inputs absent as expected in the data-free package |
| `python code/validate_results.py --profile release --site-data ...` | 51/51 checks passed against frozen v16.5 outputs and the complete site export |
| `python code/audit_release.py --profile private` | pass, zero failures |
| `python scripts/validate_site.py --profile local` | 34/34 checks passed |
| `node scripts/browser_qa.js` with bundled Playwright and Edge | 10 screenshots; zero browser errors |

The independent package is versioned in Git, while source/output provenance remains recorded through `release/source_manifest_sha256.csv`, `release/key_output_manifest_sha256.csv`, and `release/checksums.sha256`. No production analysis stage was rerun because the requested GitHub deliverable excludes all inputs and result data; the site export was regenerated from the existing frozen v16.5 results.
