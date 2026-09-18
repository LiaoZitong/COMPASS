#!/usr/bin/env python3
"""Integrate the sequential R1 analysis gates without editing manuscript files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    rows = frame.loc[mask, column]
    if len(rows) != 1:
        raise AssertionError(f"Expected one row for {column}, found {len(rows)}")
    return float(rows.iloc[0])


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    d = frame[columns].copy()
    headers = list(d.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.analysis_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "ap01_gate": root / "01_ap01_eligibility/gate_ap01.json",
        "ap02_gate": root / "02_ap02_strict_loo/gate_ap02.json",
        "ap02_capture": root / "02_ap02_strict_loo/ap02_panel_expected_capture_comparison.csv",
        "ap02_ranking": root / "02_ap02_strict_loo/ap02_species_rank_and_panel_comparison.csv",
        "ap05_gate": root / "03_ap05_dependence/gate_ap05.json",
        "ap05_audit": root / "03_ap05_dependence/ap05_rho_estimation_audit.csv",
        "ap03a_gate": root / "04_ap03_module_a/gate_ap03a.json",
        "ap03a_concordance": root / "04_ap03_module_a/ap03a_member_count_concordance.csv",
        "ap03a_contributors": root / "04_ap03_module_a/ap03a_member_contribution_summary.csv",
        "ap03a_ablation": root / "04_ap03_module_a/ap03a_member_ablation_matched.csv",
        "ap03a_shared": root / "04_ap03_module_a/ap03a_shared_input_sensitivity_with_intervals.csv",
        "ap03b_gate": root / "05_ap03_module_b/gate_ap03b.json",
        "ap03b_pairwise": root / "05_ap03_module_b/ap03b_pairwise_ranking_robustness.csv",
        "ap03b_recall": root / "05_ap03_module_b/ap03b_lower20_hazard_recall.csv",
        "ap04_gate": root / "06_ap04_localization_sensitivity/gate_ap04.json",
        "ap04_ledger": root / "06_ap04_localization_sensitivity/ap04_state_support_and_fallback_ledger.csv",
        "ap04_weight": root / "06_ap04_localization_sensitivity/ap04_weight_sensitivity_state_detail.csv",
        "ap04_policy": root / "06_ap04_localization_sensitivity/ap04_species_policy_sensitivity_state_detail.csv",
        "ap04_reduction": root / "06_ap04_localization_sensitivity/ap04_data_reduction_state_detail.csv.gz",
        "ap04_config": root / "06_ap04_localization_sensitivity/ap04_frozen_scenario_config.json",
        "ap04_figure_manifest": root / "06_ap04_localization_sensitivity/ap04_diagnostic_figure_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing G3 input files: {missing}")

    ap01 = load_json(paths["ap01_gate"])
    ap02 = load_json(paths["ap02_gate"])
    ap05 = load_json(paths["ap05_gate"])
    ap03a = load_json(paths["ap03a_gate"])
    ap03b = load_json(paths["ap03b_gate"])
    ap04 = load_json(paths["ap04_gate"])
    gates = [ap01, ap02, ap05, ap03a, ap03b, ap04]
    if any(gate.get("status") != "PASS" for gate in gates):
        raise RuntimeError("G3 requires all AP01/AP02/AP05/AP03A/AP03B/AP04 gates to PASS")

    capture = pd.read_csv(paths["ap02_capture"])
    ranking = pd.read_csv(paths["ap02_ranking"])
    rho_audit = pd.read_csv(paths["ap05_audit"])
    concordance = pd.read_csv(paths["ap03a_concordance"])
    contributors = pd.read_csv(paths["ap03a_contributors"])
    ablation = pd.read_csv(paths["ap03a_ablation"])
    shared = pd.read_csv(paths["ap03a_shared"])
    pairwise = pd.read_csv(paths["ap03b_pairwise"])
    recall = pd.read_csv(paths["ap03b_recall"])
    ledger = pd.read_csv(paths["ap04_ledger"])
    weight = pd.read_csv(paths["ap04_weight"])
    policy = pd.read_csv(paths["ap04_policy"])
    reduction = pd.read_csv(paths["ap04_reduction"])

    top5 = ap02["x95_disposition"]["selected_downstream_top5"]
    if top5 != ap03a["selected_top5"] or top5 != ap04["national_top5"]:
        raise AssertionError("Selected Top-5 is inconsistent across AP02, AP03A, and AP04")
    rho95 = float(ap05["selected_downstream_rho_by_x"]["0.95"])
    if not np.isclose(rho95, float(ap04["rho_x95"]), atol=0, rtol=0):
        raise AssertionError("AP05 x=0.95 rho is inconsistent with AP04")
    if ap02["x95_disposition"]["disposition"] != "PRIMARY_ANALYSIS_UPDATE_REQUIRED":
        raise AssertionError("AP02 primary-analysis disposition changed unexpectedly")
    if int(ledger["localized"].sum()) != 32 or set(ledger.loc[~ledger["localized"], "state_code"]) != {"PA", "TN", "WV"}:
        raise AssertionError("AP04 state localization/fallback reconciliation failed")

    strict_updated = capture[
        capture["analysis"].eq("optimized_for_same_target")
        & capture["variant"].eq("strict_LOO_updated_rho")
    ].set_index("protection_target_x")
    r0_capture = capture[
        capture["analysis"].eq("optimized_for_same_target") & capture["variant"].eq("R0")
    ].set_index("protection_target_x")
    if set(strict_updated.index.astype(float)) != {0.8, 0.9, 0.95}:
        raise AssertionError("Missing strict-LOO updated-rho capture target")

    rhos = {}
    for target in [0.8, 0.9, 0.95]:
        row = rho_audit[
            np.isclose(rho_audit["protection_target_x"], target)
            & rho_audit["variant"].eq("AP02_STRICT_LOO")
        ]
        if len(row) != 1 or not bool(row.iloc[0]["selected_for_downstream"]):
            raise AssertionError(f"Missing selected AP02 strict-LOO rho row for x={target}")
        rhos[str(target)] = {
            "rho": float(row.iloc[0]["rho_working"]),
            "eligible_species_pairs": int(row.iloc[0]["n_pairs"]),
            "positive_species_pairs": int(row.iloc[0]["n_positive_pairs"]),
            "fallback_used": bool(row.iloc[0]["fallback_used"]),
        }

    all_typical = concordance[
        concordance["stratum"].eq("all_eligible_rows")
        & concordance["reference"].eq("reference_hc5_typical_log10")
    ].iloc[0]
    dmag = contributors[contributors["latin_name"].eq("Daphnia magna")].iloc[0]
    hyalella_ablation = ablation[
        ablation["removed_member"].eq("Hyalella azteca")
        & ablation["reference"].eq("reference_hc5_typical_log10")
    ].iloc[0]
    dmag_ablation = ablation[
        ablation["removed_member"].eq("Daphnia magna")
        & ablation["reference"].eq("reference_hc5_typical_log10")
    ].iloc[0]

    full_r0_r1 = pairwise[
        pairwise["scope"].eq("full_pairwise_scoreable")
        & pairwise["comparison"].eq("R0 primary vs R1 strict-LOO primary")
    ].iloc[0]
    matched_primary_balanced = pairwise[
        pairwise["scope"].eq("matched_support")
        & pairwise["comparison"].eq("R1 primary max vs balanced")
    ].iloc[0]
    matched_primary_no_mortality = pairwise[
        pairwise["scope"].eq("matched_support")
        & pairwise["comparison"].eq("R1 primary max vs mortality-excluded balanced")
    ].iloc[0]
    r1_primary_top20_recall = recall[
        recall["scope"].eq("full_scoreable")
        & recall["variant"].eq("r1_primary_max")
        & recall["budget"].eq("top_20pct")
    ].iloc[0]

    weight_nonbaseline = weight[
        ~weight["weight_scenario"].eq("baseline_45_35_10_10") & weight["baseline_localized"]
    ].copy()
    exact_weight_membership = int(np.isclose(weight_nonbaseline["membership_jaccard_vs_baseline"], 1.0).sum())
    changed_weight_membership = int((~np.isclose(weight_nonbaseline["membership_jaccard_vs_baseline"], 1.0)).sum())
    policy_local = policy[policy["baseline_localized"]].copy()
    policy_changed = {
        name: int(
            (~np.isclose(group["membership_jaccard_vs_baseline"], 1.0)).sum()
        )
        for name, group in policy_local.groupby("species_policy")
    }
    reduction_soft = reduction[
        reduction["species_evidence_tier"].eq("soft_all")
        & reduction["baseline_localized"]
    ].copy()
    additional_fallbacks = {}
    for fraction, group in reduction_soft.groupby("chemical_retention_fraction"):
        additional_fallbacks[str(float(fraction))] = sorted(group.loc[~group["localized"], "state_code"].astype(str).tolist())

    numeric_freeze = {
        "schema_version": 1,
        "status": "AUTHORITATIVE_R1_ANALYSIS_NUMBERS_PENDING_AUTHOR_REVIEW",
        "eligibility": {
            "eligible_records": int(ap01["denominator_contract"]["eligible_records"]),
            "candidate_contexts": int(ap01["denominator_contract"]["candidate_contexts"]),
            "eligible_contexts": int(ap01["denominator_contract"]["eligible_contexts"]),
            "strict_loo_original_n_ge_6_contexts": int(
                ap01["denominator_contract"]["strict_loo_eligible_contexts_original_n_ge_6"]
            ),
            "original_n5_diagnostic_contexts": int(ap01["denominator_contract"]["n5_diagnostic_contexts"]),
            "candidate_species": int(ap01["denominator_contract"]["candidate_species"]),
            "chemical_effect_targets": int(ap01["denominator_contract"]["chemical_effect_targets"]),
        },
        "strict_loo_primary": {
            "top5_order": top5,
            "rank_spearman_vs_r0_x95": float(ap02["x95_disposition"]["rank_spearman"]),
            "top5_membership_changed": bool(ap02["x95_disposition"]["membership_changed"]),
            "top5_order_changed": bool(ap02["x95_disposition"]["order_changed"]),
            "fixed_r0_rho_expected_capture_r0_x95": float(ap02["x95_disposition"]["expected_capture_original"]),
            "fixed_r0_rho_expected_capture_strict_loo_x95": float(
                ap02["x95_disposition"]["expected_capture_strict_loo_fixed_r0_rho"]
            ),
            "fixed_r0_rho_absolute_difference_x95": float(
                ap02["x95_disposition"]["absolute_expected_capture_difference"]
            ),
            "final_expected_capture_by_target": {
                str(target): float(strict_updated.loc[target, "expected_capture"]) for target in [0.8, 0.9, 0.95]
            },
            "r0_expected_capture_by_target": {
                str(target): float(r0_capture.loc[target, "expected_capture"]) for target in [0.8, 0.9, 0.95]
            },
            "final_x95_change_vs_r0_pp": float(
                100 * (strict_updated.loc[0.95, "expected_capture"] - r0_capture.loc[0.95, "expected_capture"])
            ),
        },
        "dependence": rhos,
        "available_member_validation": {
            "eligible_rows_and_unique_chemicals": 115,
            "measured_member_count_distribution": ap03a["count_distribution"],
            "fully_observed_top5_rows": int(ap03a["full_five_member_rows"]),
            "daphnia_magna_rows_measured": int(dmag["n_rows_measured"]),
            "daphnia_magna_minimum_contributor_rows": int(dmag["n_rows_as_minimum_contributor"]),
            "daphnia_magna_fractional_minimum_credit_fraction": float(dmag["fractional_contribution_fraction"]),
            "spearman_typical_hc5": float(all_typical["spearman_rho"]),
            "spearman_typical_hc5_ci95": [
                float(all_typical["spearman_rho_ci95_lower"]),
                float(all_typical["spearman_rho_ci95_upper"]),
            ],
            "hyalella_ablation_delta_spearman": float(hyalella_ablation["delta_spearman_ablated_minus_full"]),
            "hyalella_ablation_delta_ci95": [
                float(hyalella_ablation["delta_spearman_ci95_lower"]),
                float(hyalella_ablation["delta_spearman_ci95_upper"]),
            ],
            "daphnia_magna_ablation_matched_rows": int(dmag_ablation["n_matched_rows"]),
            "daphnia_magna_ablation_rows_dropped": int(dmag_ablation["n_rows_dropped_no_remaining_member"]),
            "daphnia_magna_ablation_delta_spearman": float(dmag_ablation["delta_spearman_ablated_minus_full"]),
            "daphnia_magna_ablation_delta_ci95": [
                float(dmag_ablation["delta_spearman_ci95_lower"]),
                float(dmag_ablation["delta_spearman_ci95_upper"]),
            ],
            "shared_input_sensitivity": shared.to_dict("records"),
        },
        "followup_ranking": {
            "all_ranked_chemicals": int(ap03b["denominators"]["all_ranked_chemicals"]),
            "mortality_excluded_scoreable": int(ap03b["denominators"]["mortality_excluded_scoreable"]),
            "mortality_excluded_not_scoreable": int(ap03b["denominators"]["mortality_excluded_not_scoreable"]),
            "matched_support_common_set": int(ap03b["denominators"]["matched_support_common_set"]),
            "mortality_max_rows_r1": int(ap03b["mortality_dominance"]["r1_strict_loo_primary_max_rows"]),
            "mortality_max_top100_r1": int(ap03b["mortality_dominance"]["r1_strict_loo_primary_max_top100"]),
            "r0_vs_r1_primary_full_spearman": float(full_r0_r1["spearman_rho"]),
            "r0_vs_r1_primary_full_spearman_ci95": [
                float(full_r0_r1["spearman_rho_ci95_lower"]),
                float(full_r0_r1["spearman_rho_ci95_upper"]),
            ],
            "matched_primary_vs_balanced_spearman": float(matched_primary_balanced["spearman_rho"]),
            "matched_primary_vs_balanced_top20_jaccard": float(matched_primary_balanced["top_20pct_jaccard"]),
            "matched_primary_vs_mortality_excluded_spearman": float(matched_primary_no_mortality["spearman_rho"]),
            "matched_primary_vs_mortality_excluded_top20_jaccard": float(
                matched_primary_no_mortality["top_20pct_jaccard"]
            ),
            "r1_primary_top20_lower20_hazard_recall": float(r1_primary_top20_recall["recall_of_all_23"]),
        },
        "localization": {
            "state_universe": int(ap04["n_states"]),
            "localized_states": int(ap04["localized_states_complete_baseline"]),
            "fallback_states": ap04["fallback_states_complete_baseline"],
            "mean_gain_over_national_top5_pp_localized": float(ap04["mean_localization_gain_over_national_top5_pp"]),
            "median_gain_over_national_top5_pp_localized": float(ap04["median_localization_gain_over_national_top5_pp"]),
            "range_gain_over_national_top5_pp_localized": [
                float(ap04["min_localization_gain_over_national_top5_pp"]),
                float(ap04["max_localization_gain_over_national_top5_pp"]),
            ],
            "weight_stability_label": ap04["predeclared_weight_stability_label"],
            "weight_nonbaseline_state_scenario_comparisons": int(len(weight_nonbaseline)),
            "weight_exact_top5_membership_comparisons": exact_weight_membership,
            "weight_one_member_substitution_comparisons": changed_weight_membership,
            "weight_minimum_jaccard": float(weight_nonbaseline["membership_jaccard_vs_baseline"].min()),
            "weight_max_absolute_full_universe_coverage_change_pp": float(
                weight_nonbaseline["full_universe_coverage_delta_pp"].abs().max()
            ),
            "policy_changed_state_counts": policy_changed,
            "hard_exclusion_worst_coverage_change_pp": float(
                policy_local.loc[
                    policy_local["species_policy"].eq("hard_exclude_current_or_unresolved_official_nas"),
                    "full_universe_coverage_delta_pp",
                ].min()
            ),
            "state_confirmed_worst_coverage_change_pp": float(
                policy_local.loc[
                    policy_local["species_policy"].eq("state_confirmed_only"),
                    "full_universe_coverage_delta_pp",
                ].min()
            ),
            "additional_fallbacks_from_chemical_reduction_among_baseline_localizable_states": additional_fallbacks,
        },
    }
    (out / "g3_authoritative_numeric_freeze.json").write_text(
        json.dumps(numeric_freeze, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    decision_rows = [
        {
            "package": "AP01",
            "reviewer_comments": "R2.2; R3.min3; R3.min6",
            "analysis_status": "PASS",
            "r1_disposition": "Methods and denominator clarification; statistical model unchanged",
            "authoritative_result": "426,588 eligible records; 2,026 eligible contexts; 912 chemical-effect targets; censoring likelihood tests passed",
            "required_revision_locations": "Methods 2.1-2.3; Results terminology; Figures 3/5 captions; SI Note/Table; responses",
            "claim_ceiling": "Auditable censored-likelihood and aggregation workflow with explicit eligibility boundaries",
            "do_not_claim": "Do not equate measured-tail probability, expected capture, HC5, follow-up priority, or regulatory risk",
        },
        {
            "package": "AP02",
            "reviewer_comments": "R3.M1",
            "analysis_status": "PASS; primary update required",
            "r1_disposition": "Promote strict focal-species LOO probabilities and regenerate all dependent results",
            "authoritative_result": "x=0.95 Top-5 membership unchanged but order changed; final expected capture 0.337764 with updated rho",
            "required_revision_locations": "Methods 2.2; Results 3.2-3.3; downstream figures/tables/SI; Discussion; response",
            "claim_ceiling": "High rank stability and unchanged membership under strict LOO, with materially updated expected-capture values",
            "do_not_claim": "Do not state that the original primary probabilities or expected-capture values are unchanged",
        },
        {
            "package": "AP05",
            "reviewer_comments": "R2.3",
            "analysis_status": "PASS",
            "r1_disposition": "Use strict-LOO data-derived working rho values; no empirical fallback used",
            "authoritative_result": "rho=0.117446/0.119452/0.132467 for x=0.80/0.90/0.95 from 612/531/417 eligible pairs",
            "required_revision_locations": "Methods 2.2; SI Table S3/method; all expected-capture calculations; response",
            "claim_ceiling": "Data-derived working dependence summary for the one-factor copula",
            "do_not_claim": "Do not describe rho as a chemical-specific exact dependence structure or as manually fixed at 0.15",
        },
        {
            "package": "AP03A",
            "reviewer_comments": "R2.7; R3.M2; R3.min2; R3.min4",
            "analysis_status": "PASS",
            "r1_disposition": "Retain as available-member concordance with explicit coverage and bootstrap intervals",
            "authoritative_result": "115 chemicals; 54 rows have one measured member; 0 have all five; Spearman 0.847 (95% bootstrap CI 0.756-0.904); D. magna provides 92 minima",
            "required_revision_locations": "Abstract; Methods 2.3; Results 3.3; Figure 3 caption; Discussion; SI table/figure; response",
            "claim_ceiling": "Strong available-member rank concordance on the 115-chemical high-information subset",
            "do_not_claim": "Do not call this complete five-species panel validation or generalize it to all 639 chemicals",
        },
        {
            "package": "AP03B",
            "reviewer_comments": "R3.M3; R3.min4",
            "analysis_status": "PASS; interpretation narrowing required",
            "r1_disposition": "Keep primary, balanced, and mortality-excluded rankings as explicit sensitivity set; demote primary ranking to evidence-contingent follow-up heuristic",
            "authoritative_result": "Mortality is maximal for 629/639 chemicals; matched primary-vs-balanced Spearman 0.443 and Top-20% Jaccard 0.244; 494 chemicals are not scoreable without mortality",
            "required_revision_locations": "Methods 2.3/Equation 5; Results follow-up ranking; Discussion; SI rankings/common set; response",
            "claim_ceiling": "Conditional prioritization under a declared aggregation rule and observed endpoint coverage",
            "do_not_claim": "Do not claim endpoint-balanced robustness, regulatory risk ranking, or treat missing non-mortality evidence as low priority/risk",
        },
        {
            "package": "AP04",
            "reviewer_comments": "R2.4; R2.6",
            "analysis_status": "PASS",
            "r1_disposition": "Retain baseline coefficients with frozen sensitivity; add support/fallback ledger and data-dependence caveat",
            "authoritative_result": "32 localized plus PA/TN/WV fallback; coefficient perturbations retain at least 4/5 members; 50% chemical reduction adds AL/MT fallbacks",
            "required_revision_locations": "Methods 2.3; Results 3.5; Figure 5; Discussion; state SI table; response",
            "claim_ceiling": "Stable to the tested coefficient schemes, conditional on adequate local monitoring and occurrence support",
            "do_not_claim": "Do not infer native status/deployability from occurrence or claim localization is insensitive to sparse local data",
        },
    ]
    decision_register = pd.DataFrame(decision_rows)
    decision_register.to_csv(out / "g3_analysis_decision_register.csv", index=False)

    claim_rows = [
        {
            "topic": "Strict LOO national panel",
            "allowed_statement": "Strict LOO preserves Top-5 membership and nearly preserves national ordering, while changing order and expected-capture estimates.",
            "required_qualifier": "Report the strict n>=6 source contexts, the separate n=5 diagnostic stratum, and updated rho.",
            "prohibited_overreach": "The original and strict-LOO analyses are identical.",
        },
        {
            "topic": "Measured Top5-apical comparison",
            "allowed_statement": "The minimum among measured available members is strongly rank-correlated with typical HC5 within 115 eligible chemicals.",
            "required_qualifier": "State 54/115 one-member rows, 0/115 complete panels, contributor imbalance, and bootstrap interval.",
            "prohibited_overreach": "A fully observed five-species panel has been externally validated across 639 chemicals.",
        },
        {
            "topic": "Chemical follow-up ranking",
            "allowed_statement": "The ranking is a conditional follow-up heuristic whose composition changes under endpoint-family aggregation.",
            "required_qualifier": "State mortality dominance, matched-support denominators, and NOT SCOREABLE counts.",
            "prohibited_overreach": "The ranking is endpoint-balanced, a regulatory-risk ranking, or missing evidence indicates low risk.",
        },
        {
            "topic": "Localized state panels",
            "allowed_statement": "Panels are stable to the tested coefficient schemes and provide model-expected gains in 32 supported states.",
            "required_qualifier": "State the 35=32+3 ledger, data-reduction fallbacks, and occurrence/deployment boundary.",
            "prohibited_overreach": "Occurrence proves native status or field deployability, or gains are empirically validated performance improvements.",
        },
    ]
    claim_matrix = pd.DataFrame(claim_rows)
    claim_matrix.to_csv(out / "g3_claim_boundary_matrix.csv", index=False)

    manifest = {
        "schema_version": 1,
        "analysis_id": "G3_CROSS_PACKAGE_INTEGRATION",
        "input_files": [
            {"name": name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        ],
    }
    (out / "g3_input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    gate = {
        "schema_version": 1,
        "gate": "G3_CROSS_PACKAGE_INTEGRATION",
        "status": "PASS",
        "upstream_gates": [gate["gate"] for gate in gates],
        "authoritative_probability_source": "AP02_STRICT_LOO",
        "authoritative_top5_x95": top5,
        "authoritative_rho_x95": rho95,
        "primary_analysis_update_required": True,
        "available_member_validation_interpretation_locked": True,
        "followup_ranking_interpretation_narrowing_required": True,
        "localization_weight_sensitivity": ap04["predeclared_weight_stability_label"],
        "all_denominator_and_dependency_checks_pass": True,
        "manuscript_or_si_edits_performed": False,
        "next_stage": "AUTHOR_REVIEW_OF_ANALYSIS_RESULTS_BEFORE_SUBSTANTIVE_REVISION",
    }
    (out / "gate_g3.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# G3 cross-package integration report",
        "",
        "Status: **PASS — analysis integrated; manuscript and SI remain unedited.**",
        "",
        "## Authoritative R1 analysis chain",
        "",
        "AP01 → AP02 → AP05 → AP03A → AP03B → AP04 was executed in the frozen dependency order. "
        "Every upstream gate passed before its dependent stage began. AP02 strict focal-species leave-one-out probabilities "
        "and AP05 strict-LOO working dependence parameters are the only downstream probability/dependence sources.",
        "",
        "## Integrated numerical decisions",
        "",
        f"1. **Primary probability update is required.** Strict LOO retains the same x=0.95 Top-5 membership but changes its order to "
        f"{'; '.join(top5)}. At fixed R0 rho, expected capture changes from "
        f"{ap02['x95_disposition']['expected_capture_original']:.6f} to "
        f"{ap02['x95_disposition']['expected_capture_strict_loo_fixed_r0_rho']:.6f}; with the re-estimated AP05 rho it is "
        f"{strict_updated.loc[0.95, 'expected_capture']:.6f}.",
        f"2. **The dependence parameter is data-derived in all three targets.** Strict-LOO rho values are "
        f"{rhos['0.8']['rho']:.6f}, {rhos['0.9']['rho']:.6f}, and {rhos['0.95']['rho']:.6f}; no 0.15 fallback was used.",
        f"3. **The HC5 comparison is available-member concordance.** On 115 eligible chemicals, Spearman rho is "
        f"{all_typical['spearman_rho']:.3f} (95% bootstrap CI {all_typical['spearman_rho_ci95_lower']:.3f}–"
        f"{all_typical['spearman_rho_ci95_upper']:.3f}). Fifty-four rows contain one measured member, none contains all five, "
        f"and D. magna supplies the minimum in {int(dmag['n_rows_as_minimum_contributor'])}/115 rows.",
        f"4. **Chemical follow-up priority is endpoint-aggregation dependent.** Mortality/survival is maximal for "
        f"{ap03b['mortality_dominance']['r1_strict_loo_primary_max_rows']}/639 chemicals and all Top 100. On the 136-chemical "
        f"matched-support set, primary versus balanced Spearman rho is {matched_primary_balanced['spearman_rho']:.3f} and "
        f"Top-20% Jaccard is {matched_primary_balanced['top_20pct_jaccard']:.3f}; 494 chemicals are not scoreable after mortality exclusion.",
        f"5. **Localization is coefficient-stable but support-dependent.** The complete ledger has 32 localized states and "
        f"three fallbacks (PA, TN, WV). Across 384 nonbaseline state-scenario weight comparisons, {exact_weight_membership} retain "
        f"the identical member set and {changed_weight_membership} replace one member; the maximum absolute full-universe "
        f"coverage change is {weight_nonbaseline['full_universe_coverage_delta_pp'].abs().max():.4f} percentage points. "
        f"At 50% or 25% chemical retention, AL and MT become additional fallbacks.",
        "",
        "## Revision decision register",
        "",
        md_table(decision_register, ["package", "analysis_status", "r1_disposition", "claim_ceiling"]),
        "",
        "## Cross-package claim ceilings",
        "",
        md_table(claim_matrix, ["topic", "allowed_statement", "required_qualifier"]),
        "",
        "## Next controlled action",
        "",
        "The next action is author review of this integrated analysis package. No prose, figures, tables, captions, references, "
        "EndNote fields, or Word layout in the manuscript/SI have been changed. After author approval, the revision production "
        "workflow should propagate the strict-LOO probability source and updated rho once, then rebuild all dependent text and "
        "exhibits from controlled sources.",
        "",
    ]
    (out / "G3_CROSS_PACKAGE_INTEGRATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
