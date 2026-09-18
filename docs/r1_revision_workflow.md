# R1 revision workflow

## Purpose and frozen order

The R1 layer answers the reviewer requests that cannot be represented by the
base v16.5 pipeline alone. Execute the packages in this order:

1. **AP01 — eligibility ledger:** freezes censoring and denominator flow.
2. **AP02 — strict focal-species LOO:** excludes the focal species from each
   context reference distribution; originally n ≥ 6 contexts form the primary
   layer and originally n = 5 contexts remain diagnostics.
3. **AP05 — dependence audit:** re-estimates one working Gaussian-copula
   dependence value per target from the AP02 event layer.
4. **AP03A — available-member validation:** evaluates measured Top-5 member
   minima against full-data HC5 within the eligible 115-chemical subset.
5. **AP03B — follow-up robustness:** compares primary maximum, balanced
   effect-family, mortality-excluded, and matched-support rankings.
6. **AP04 — localization sensitivity:** evaluates alternative chemical weights,
   occurrence policies, and priority-chemical support reduction.
7. **G3 — integration gate:** reconciles the six upstream packages and freezes
   manuscript-facing values.
8. **Strict-LOO robustness:** propagates record-level profile likelihoods and
   evaluates IPW/MNAR testing-selection scenarios using the approved strict-LOO
   direct and full probability matrices.

AP05 must follow AP02; AP03A requires AP02 and AP05; AP03B requires AP02 and
AP03A; AP04 checks every earlier gate; G3 integrates the primary packages; and
the robustness stage runs only after G3. The runner enforces this order and
refuses a resume when an earlier PASS gate is missing.

## Command

After running the base semantic stages 01–22 and placing all documented inputs:

```powershell
.\.venv\Scripts\python.exe .\code\revision_r1\run_revision_analysis.py `
  --analysis-root . `
  --out results\revision_r1
```

To inspect the commands without execution, add `--dry-run`. To resume, pass
`--from-step` with one of the directory names printed by the runner. Existing
earlier gates must be present and report `PASS`.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\revision_r1 -q
.\.venv\Scripts\python.exe .\code\revision_r1\validate_revision_results.py `
  --revision-root .\results\revision_r1 `
  --expected .\config\r1_expected_results.json
```

The validator checks all eight gates, eligibility denominators, the R1 Top-20,
12 capture checkpoints, three dependence values, available-member validation,
follow-up counts, regional gains/fallbacks, strict-LOO profile/MNAR checkpoints,
and—when supplied—the site export.

## Data and claim boundaries

The release contains code and contracts, not third-party data. Strict LOO
measured-tail probabilities and expected capture are model-based screening
quantities. The available-member HC5 comparison applies only to the 115
eligible chemicals, and no chemical has all five panel members measured.
Follow-up rankings are evidence-conditioned heuristics. Regional gains are
model-expected values under the stated weights and require prospective,
matrix-specific validation before deployment.
