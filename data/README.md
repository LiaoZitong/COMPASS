# Data acquisition and placement

The primary inputs are large public or provider-hosted resources, including EPA ECOTOX. They are not mirrored in this compact code repository because of their size and provider distribution arrangements. Obtain them from the official links below, retain their provenance and terms, and place locally prepared inputs at the documented paths. Do not commit files beneath `data/intermediate/`, `data/processed/`, or `data/external/`, and do not commit the reference archives listed below.

The dates and local compressed sizes below describe the frozen v16.5 source workspace. They are provenance aids, not redistribution permissions.

| Provider | Source used | Access date | Frozen local input | Approx. compressed size | Redistribution in this repository | Main stage |
|---|---|---:|---|---:|---|---:|
| U.S. EPA | ECOTOX Knowledgebase | 2025-12-07 | `data/intermediate/ecotox_canonical_partitions/part_*.csv.gz` (13 parts) | 211 MB | No; obtain source data from EPA and prepare the canonical partitions | 01 |
| National Water Quality Monitoring Council / USGS / U.S. EPA | Water Quality Portal, 2024 records | 2026-02-26 | `data/processed/state_chemical_exposure_2024.csv.gz` | 0.12 MB | No | 05, 20 |
| U.S. EPA | Tools for Automated Data Analysis (TADA) workflow | 2026-02-26 | incorporated into the state exposure table | — | Workflow attribution only; no downloaded bundle is redistributed | 05 |
| GBIF | State-species occurrence downloads (DOIs below) | 2026-04-05 | `data/processed/state_species_occurrence_with_usgs_nas.csv.gz` | 1.20 MB | No | 06, 20 |
| USGS | Nonindigenous Aquatic Species Database | 2026-03-26 | incorporated into the state occurrence table | source archive ~80 MB | No | 20 |
| U.S. EPA | ToxCast / invitroDB | 2026-02-25 | `data/reference/AOP_MOA_data.zip` | 47.3 MB | No | 03–04 |
| AOP-Wiki | AOP event and relationship tables | 2026-01-01 | incorporated into `AOP_MOA_data.zip` | included above | No | 03–04 |
| U.S. EPA | CompTox Dashboard and CTX APIs | 2026-02-25 / 2026-02-27 | `data/processed/chemical_master.csv.gz` and mechanism bundle mappings | 2.91 MB for chemical master | No | 03–05 |
| Author-curated references | Species taxonomy/traits and life-stage mapping | frozen 2026-06-22 | `species_traits_curated.csv`, `lifestage_codes.txt` | 0.85 MB | No in this release | 01, 03 |
| EPA/OECD/CCME/ANZG/EU source documents | Operational comparator catalog | frozen 2026-06-22 | `regulatory_baseline_catalog.csv` | 0.004 MB | No in this release | 13, 19 |
| U.S. Census Bureau | 2024 1:500,000 state cartographic boundaries | current figure source | `data/external/cb_2024_us_state_500k.zip` | external | No; download from Census | Figure 5 |

GBIF occurrence download DOIs recorded by the frozen manuscript are `10.15468/dl.jw3vuc`, `10.15468/dl.jcupcd`, `10.15468/dl.7jdzjh`, `10.15468/dl.y8ptv8`, and `10.15468/dl.5bdt74`.

## Provider links

- EPA ECOTOX: <https://cfpub.epa.gov/ecotox/>
- Water Quality Portal: <https://www.waterqualitydata.us/>
- EPA TADA: <https://www.epa.gov/waterdata/TADA>
- GBIF: <https://www.gbif.org/>
- USGS NAS: <https://nas.er.usgs.gov/>
- EPA ToxCast: <https://www.epa.gov/comptox-tools/exploring-toxcast-data>
- EPA CTX APIs: <https://www.epa.gov/comptox-tools/computational-toxicology-and-exposure-apis>
- AOP-Wiki: <https://aopwiki.org/>
- EPA CompTox Dashboard: <https://comptox.epa.gov/dashboard/>
- Census state boundaries: <https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip>

## Minimum input contracts

The machine checks in `code/validate_inputs.py` enforce the principal fields. Detailed definitions are in [`dictionaries/input_contracts.md`](dictionaries/input_contracts.md).

- Canonical ECOTOX partitions: record/test/reference identifiers, DTXSID, formal scientific name, evidence/effect/endpoint fields, exposure duration, life stage, medium, censoring class, molar point/lower/upper concentration, and waterborne eligibility.
- Chemical master: `DTXSID`, preferred name, and structure/form fields used by the chemical annotation stage.
- State exposure: state code, DTXSID, mapping/exposure domain, monitoring support, detection frequency, and nonnegative pre-HC5 weight.
- State occurrence: state code, scientific name, exact-occurrence flag, and state-candidate flag.
- Trait table: one row per scientific name with taxonomy and available trait fields.
- Mechanism bundle: the chemical, feature, active-score, and confidence matrices for MIE, route, and chain levels.

Concentrations entering stage 01 are positive and expressed in `µmol/L`. Expected-capture and probability fields are always stored on a 0–1 scale. State codes use two-letter U.S. abbreviations. The allowed measured-tail targets are `x = 0.95`, `0.90`, and `0.80`, corresponding to lower-5%, lower-10%, and lower-20% evaluation.

## Redistribution boundary

Provider terms and any database-specific citation requirements must be checked at the time of a public deposit. This release takes the conservative route: no complete source snapshot, canonical partition, processed table, reference archive, or result matrix is included. The package publishes input contracts, core code, and a minimal set of numerical validation checkpoints only.
