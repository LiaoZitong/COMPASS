## COMPASS R1 reproducibility release

This is the versioned code release supporting the R1 revision of “Developing
Complementary Aquatic Sentinel Panels for Cross-Chemical Monitoring and Regional
Localization.”

## Scope

- Base COMPASS analysis code, semantic stages 01–22
- Ordered R1 AP01/AP02/AP05/AP03A/AP03B/AP04/G3 revision analyses
- Frozen R1 strict-LOO values and numerical validation contracts
- Main-figure source and the publication-ready SI Figure S8/Table S7 builders
- Input/data-source contracts and an R1-specific static Explorer exporter

## Data boundary

Primary inputs include large public and provider-hosted resources such as EPA ECOTOX. They are available from their official services and are not mirrored in this compact code release because of their size and provider distribution arrangements. Follow the official links and preparation contracts in `data/README.md`.

## Reproducibility status

The tagged source state passed the data-free unit/assertion suite, Python compile
check, R1 numerical validation, public-source audit, site-data cross-check, and
tracked-file checksum verification. See `docs/release_validation_report.md` and
`release/checksums.sha256` for the machine-readable evidence.

The Git tag and commit identify this release. No archival DOI is claimed unless
and until a separate archive returns a verified identifier.
