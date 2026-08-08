# Input contracts

## Canonical ECOTOX partitions

Path: `data/intermediate/ecotox_canonical_partitions/part_*.csv.gz`

Primary record key: `result_id`; study linkage: `test_id` plus `reference_number`. DTXSID values use the `DTXSID` prefix followed by digits. `latin_name` is the scientific name used for candidate-universe construction. Concentrations are positive `µmol/L` values represented by `concentration_point_umol_L`, `concentration_lower_umol_L`, and `concentration_upper_umol_L` according to `censoring_class`.

Allowed censoring classes are `exact`, `approximate_exact`, `left_censored`, `right_censored`, and `interval_censored`. Exact, left-, right-, and interval-censored observations enter the likelihood; the public code does not substitute half limits or interval midpoints for final inference.

Required classification fields are `effect_family`, `effect_evidence_layer`, `endpoint_family`, `endpoint_level_band`, `exposure_duration_h`, `organism_lifestage`, `medium_family`, and `waterborne_main`.

## Chemical master

Path: `data/processed/chemical_master.csv.gz`

Key: `DTXSID`. At minimum, `PREFERRED_NAME` is required. Structure/form classification uses available fields such as `SMILES`, `QSAR_READY_SMILES`, and `MOLECULAR_FORMULA`; unavailable structure information is retained as unresolved and is not silently imputed.

## State chemical exposure

Path: `data/processed/state_chemical_exposure_2024.csv.gz`

Key: `state_code` + `DTXSID`. State codes are two-letter abbreviations. Required fields include `mapping_status`, `exposure_domain`, `pre_hc5_exposure_weight`, `detection_frequency`, `n_sites`, and `n_result_records`. Counts and weights are nonnegative; detection frequency lies in `[0, 1]`. Concentration fields, where present, retain their stated `µg/L` unit and are not used as toxicity-effect concentrations.

## State species occurrence

Path: `data/processed/state_species_occurrence_with_usgs_nas.csv.gz`

Key: `state_code` + `latin_name`. Required fields include `exact_species_occurrence` and `eligible_state_candidate`. Occurrence and NAS evidence contribute relevance signals. A NAS non-hit is not treated as proof of native status.

## Species traits

Path: `data/reference/species_traits_curated.csv`

Key: `latin_name`. Taxonomy and traits may be missing. Missingness is retained and contributes to support/reliability reporting; sparse species are not deleted solely for having few records.

## Mechanism reference bundle

Path: `data/reference/AOP_MOA_data.zip`

The archive contains matched chemical and feature indices plus sparse active-score and confidence matrices for MIE, route, and chain levels. Shapes and row/column indices must agree. MOA/form layers can alter the core probability model only after the frozen whole-chemical out-of-fold gate passes.

## Life-stage and comparator references

`lifestage_codes.txt` is pipe-delimited and requires a `code` column. `regulatory_baseline_catalog.csv` requires scientific-name and comparator metadata used by the documented operational proxies.

## Missing values and uniqueness

Blank strings and standard CSV missing values are treated as missing. Keys must be unique at their stated level. The validators report duplicates and invalid ranges and exit nonzero on errors. They report non-binomial scientific names as warnings at the broad input-audit layer because raw provider tables can retain unresolved labels; the formal candidate universe applies the stricter name rule downstream.

