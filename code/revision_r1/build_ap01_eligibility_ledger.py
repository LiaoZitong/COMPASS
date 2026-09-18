#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTECTIVE = {
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def ledger_row(
    stage: str,
    total_in: int,
    retained: int,
    reason: str,
    source: str,
    scope: str,
) -> dict[str, object]:
    excluded = int(total_in - retained)
    return {
        "stage": stage,
        "scope": scope,
        "total_in": int(total_in),
        "retained": int(retained),
        "excluded": excluded,
        "retention_percent": 100.0 * retained / total_in if total_in else np.nan,
        "exclusion_or_status_reason": reason,
        "source": source,
    }


def load_npz_counts(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=True) as data:
        direct = np.asarray(data["direct_n"])
        return {
            "rows": int(direct.shape[0]),
            "columns": int(direct.shape[1]),
            "cells": int(direct.size),
            "direct_cells": int((direct > 0).sum()),
            "borrowed_or_no_direct_cells": int((direct == 0).sum()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhbt-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.bhbt_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "toxicity_manifest": root / "results/toxicity/censored_toxicity_manifest.json",
        "censoring_summary": root / "results/toxicity/censoring_summary_eligible.csv",
        "study_cells": root / "results/toxicity/censored_study_level_cells.csv.gz",
        "model_cells": root / "results/toxicity/censored_model_cells.csv.gz",
        "candidate_universe": root / "results/probability/candidate_universe_locked.csv",
        "targets": root / "results/probability/protective_chemical_effect_targets.csv",
        "panel_manifest": root / "results/panels/panel_probability_manifest.json",
    }
    for target in (80, 90, 95):
        paths[f"target_npz_x{target}"] = root / f"results/probability/species_protective_target_tail_probability_x{target}.npz"
        paths[f"chemical_npz_x{target}"] = root / f"results/probability/species_chemical_tail_probability_x{target}.npz"

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AP01 input(s):\n" + "\n".join(missing))

    manifest = json.loads(paths["toxicity_manifest"].read_text(encoding="utf-8"))
    panel_manifest = json.loads(paths["panel_manifest"].read_text(encoding="utf-8"))
    censoring = pd.read_csv(paths["censoring_summary"])
    study = pd.read_csv(paths["study_cells"], low_memory=False)
    model = pd.read_csv(paths["model_cells"], low_memory=False)
    candidates = pd.read_csv(paths["candidate_universe"])
    targets = pd.read_csv(paths["targets"])

    identified = model["mle_status"].astype(str).str.startswith("identified") & np.isfinite(
        pd.to_numeric(model["mu_log10_umol_L"], errors="coerce")
    )
    protective = model[identified & model["effect_family"].isin(PROTECTIVE)].copy()
    protective["context_id"] = protective[
        ["dtxsid", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]
    ].astype(str).agg("|".join, axis=1)
    context = (
        protective.groupby("context_id", as_index=False)
        .agg(
            n_species=("latin_name", "nunique"),
            ssd_sd=("mu_log10_umol_L", "std"),
        )
    )
    eligible_context = context[
        context["n_species"].ge(5)
        & context["ssd_sd"].notna()
        & context["ssd_sd"].gt(0.05)
    ].copy()
    eligible_ids = set(eligible_context["context_id"].astype(str))
    eligible_species_context_rows = protective[protective["context_id"].isin(eligible_ids)]

    raw_records = int(manifest["raw_results_rows"])
    eligible_records = int(manifest["eligible_waterborne_molar_records"])
    model_count = len(model)
    identified_count = int(identified.sum())
    protective_identified_count = len(protective)
    formal_name = candidates["latin_name"].astype(str).str.match(
        r"^[A-Z][a-z-]+ [a-z][A-Za-z.-]+$", na=False
    )

    rows = [
        ledger_row(
            "raw_to_eligible_records",
            raw_records,
            eligible_records,
            "Excluded records fail the controlled waterborne, molar-concentration, chemical-ID, species-name, or positive-bound/value eligibility rules.",
            "results/toxicity/censored_toxicity_manifest.json",
            "record",
        ),
        ledger_row(
            "eligible_records_to_study_cells",
            eligible_records,
            len(study),
            "Aggregation to unique chemical-species-effect-band-duration-life-stage-medium-reference-test study cells; this is aggregation, not record exclusion.",
            "results/toxicity/censored_study_level_cells.csv.gz",
            "study_cell",
        ),
        ledger_row(
            "model_cells_identified",
            model_count,
            identified_count,
            "Nonidentified cells are one-sided bounds or have no numeric center; they remain in the evidence audit but do not supply a fitted center.",
            "results/toxicity/censored_model_cells.csv.gz",
            "model_cell",
        ),
        ledger_row(
            "identified_to_protective_identified",
            identified_count,
            protective_identified_count,
            "Parallel warning and mechanism-support families are excluded from the protective probability layer.",
            "results/toxicity/censored_model_cells.csv.gz",
            "protective_model_cell",
        ),
        ledger_row(
            "protective_context_eligibility",
            len(context),
            len(eligible_context),
            "Retain contexts with at least five measured species, finite between-species SD, and SD > 0.05 log10 units.",
            "recomputed from results/toxicity/censored_model_cells.csv.gz",
            "ssd_context",
        ),
        ledger_row(
            "protective_species_context_rows",
            len(protective),
            len(eligible_species_context_rows),
            "Species-context rows outside eligible SSD contexts do not enter direct measured-tail probabilities.",
            "recomputed from results/toxicity/censored_model_cells.csv.gz",
            "species_context",
        ),
        ledger_row(
            "formal_candidate_species",
            len(candidates),
            int(formal_name.sum()),
            "Candidate table is already filtered to formal binomial species with at least one valid protective context; this row rechecks the naming contract.",
            "results/probability/candidate_universe_locked.csv",
            "species",
        ),
        ledger_row(
            "chemical_effect_targets",
            int(eligible_context[["context_id"]].shape[0]),
            len(targets),
            "Eligible contexts are collapsed to unique chemical-effect-family probability targets.",
            "results/probability/protective_chemical_effect_targets.csv",
            "chemical_effect_target",
        ),
    ]

    matrix_contract: dict[str, dict[str, int]] = {}
    for target in (80, 90, 95):
        target_counts = load_npz_counts(paths[f"target_npz_x{target}"])
        chemical_counts = load_npz_counts(paths[f"chemical_npz_x{target}"])
        matrix_contract[f"x{target}"] = {
            "target_probability_rows_species": target_counts["rows"],
            "target_probability_columns_chemical_effect": target_counts["columns"],
            "target_probability_cells": target_counts["cells"],
            "target_probability_direct_cells": target_counts["direct_cells"],
            "target_probability_borrowed_or_no_direct_cells": target_counts["borrowed_or_no_direct_cells"],
            "chemical_probability_rows_species": chemical_counts["rows"],
            "chemical_probability_columns_chemicals": chemical_counts["columns"],
            "chemical_probability_cells": chemical_counts["cells"],
            "chemical_probability_direct_cells": chemical_counts["direct_cells"],
            "chemical_probability_borrowed_or_no_direct_cells": chemical_counts["borrowed_or_no_direct_cells"],
        }
        rows.append(
            ledger_row(
                f"x{target}_target_probability_direct_support",
                target_counts["cells"],
                target_counts["direct_cells"],
                "Cells without direct species-chemical-effect support use the controlled probability-only borrowing path or remain at its calibrated/shrunk prior; no absolute toxicity threshold is predicted.",
                f"results/probability/species_protective_target_tail_probability_x{target}.npz",
                "species_chemical_effect_probability_cell",
            )
        )
        rows.append(
            ledger_row(
                f"x{target}_chemical_probability_any_direct_support",
                chemical_counts["cells"],
                chemical_counts["direct_cells"],
                "Direct_n records whether at least one observed effect-family target contributes to the completed species-chemical probability; zero is not interpreted as low risk.",
                f"results/probability/species_chemical_tail_probability_x{target}.npz",
                "species_chemical_probability_cell",
            )
        )

    ledger = pd.DataFrame(rows)
    ledger.to_csv(out / "ap01_censoring_and_probability_eligibility_ledger.csv", index=False)

    censoring["n_records"] = pd.to_numeric(censoring["n_records"], errors="raise").astype(int)
    if int(censoring["n_records"].sum()) != eligible_records:
        raise AssertionError("Censoring summary does not reconcile to eligible record count")
    censoring["percent_of_eligible"] = 100.0 * censoring["n_records"] / eligible_records
    censoring.to_csv(out / "ap01_censoring_profile.csv", index=False)

    small_context = context["n_species"].lt(5)
    eligible_size_sd_missing = context["n_species"].ge(5) & context["ssd_sd"].isna()
    eligible_size_sd_too_small = (
        context["n_species"].ge(5)
        & context["ssd_sd"].notna()
        & context["ssd_sd"].le(0.05)
    )
    context_exclusions = {
        "candidate_contexts": int(len(context)),
        "n_species_lt_5": int(small_context.sum()),
        "n_species_ge_5_but_ssd_sd_missing": int(eligible_size_sd_missing.sum()),
        "n_species_ge_5_but_ssd_sd_le_0_05": int(eligible_size_sd_too_small.sum()),
        "eligible_contexts": int(len(eligible_context)),
        "eligible_original_n_eq_5": int(eligible_context["n_species"].eq(5).sum()),
        "eligible_original_n_ge_6": int(eligible_context["n_species"].ge(6).sum()),
    }
    if (
        context_exclusions["eligible_contexts"]
        + context_exclusions["n_species_lt_5"]
        + context_exclusions["n_species_ge_5_but_ssd_sd_missing"]
        + context_exclusions["n_species_ge_5_but_ssd_sd_le_0_05"]
        != context_exclusions["candidate_contexts"]
    ):
        raise AssertionError("Context eligibility categories do not reconcile")

    checks = {
        "manifest_eligible_records_match_censoring_summary": True,
        "manifest_study_cells_match_file": int(manifest["study_cells"]) == len(study),
        "manifest_model_cells_match_file": int(manifest["model_cells"]) == len(model),
        "manifest_identified_model_cells_match": int(manifest["identified_model_cells"]) == identified_count,
        "manifest_protective_identified_cells_match": int(manifest["protective_model_cells_identified"]) == protective_identified_count,
        "candidate_species_match_panel_manifest": int(panel_manifest["candidate_species"]) == len(candidates),
        "target_count_matches_each_target_npz": all(
            matrix_contract[f"x{target}"]["target_probability_columns_chemical_effect"] == len(targets)
            for target in (80, 90, 95)
        ),
        "all_context_exclusions_reconcile": True,
    }
    overall_pass = all(checks.values())

    input_manifest = {
        "schema_version": 1,
        "bhbt_root": str(root),
        "inputs": [
            {
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        ],
    }
    (out / "ap01_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    contract = {
        "schema_version": 1,
        "method": "R0 denominator reconstruction from frozen controlled products; no model refit",
        "context_eligibility": context_exclusions,
        "matrix_contract": matrix_contract,
        "checks": checks,
        "overall_pass": overall_pass,
    }
    (out / "ap01_denominator_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    censor_map = dict(zip(censoring["censoring_class"].astype(str), censoring["n_records"].astype(int)))
    censored_total = sum(
        censor_map.get(label, 0)
        for label in ("left_censored", "right_censored", "interval_censored")
    )
    report = f"""# AP01 eligibility and censoring audit

## Gate result

`{'PASS' if overall_pass else 'FAIL'}` — all frozen product counts and probability-matrix dimensions {'reconcile' if overall_pass else 'do not reconcile'}.

## Key denominators

- Eligible waterborne molar records: {eligible_records:,}
- Censored records retained in the likelihood: {censored_total:,} ({100*censored_total/eligible_records:.2f}%)
- Study-level cells: {len(study):,}
- Model cells: {len(model):,}; identified centers: {identified_count:,}; one-sided/no-center cells: {len(model)-identified_count:,}
- Identified protective model cells: {protective_identified_count:,}
- Candidate protective SSD contexts: {len(context):,}; eligible contexts: {len(eligible_context):,}
- Eligible contexts with original n = 5: {context_exclusions['eligible_original_n_eq_5']:,}; strict-LOO-eligible original n >= 6: {context_exclusions['eligible_original_n_ge_6']:,}
- Candidate species: {len(candidates):,}; chemical-effect targets: {len(targets):,}

## Interpretation

Exact, approximate-exact, left-censored, right-censored, and interval-censored observations remain distinct inputs to the fixed-scale censored-normal likelihood. One-sided-only model cells remain auditable bounds and do not receive fabricated centers. The strict LOO analysis must use only original n >= 6 contexts for its threshold-compliant main layer and must report original n = 5 contexts separately.
"""
    (out / "AP01_ELIGIBILITY_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
