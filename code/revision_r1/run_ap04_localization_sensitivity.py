#!/usr/bin/env python3
"""Run the frozen AP04 state-localization sensitivity analysis.

This script reads the frozen BHBT release and R1 upstream gates without
modifying them.  All outputs are written to the requested R1 directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINE_SCENARIO = "baseline_45_35_10_10"
SOFT_POLICY = "soft_baseline"
WEIGHT_COMPONENTS = [
    "detection_rank_state",
    "concentration_rank_state",
    "site_rank_state",
    "analytical_invisibility_proxy",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_stage20(path: Path):
    spec = importlib.util.spec_from_file_location("bhbt_stage20_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stage 20 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_top5_footrule(a: list[str], b: list[str], k: int = 5) -> float:
    """Normalized top-k Spearman-footrule similarity with absent rank k+1."""

    ra = {name: rank for rank, name in enumerate(a[:k], 1)}
    rb = {name: rank for rank, name in enumerate(b[:k], 1)}
    union = set(ra) | set(rb)
    distance = sum(abs(ra.get(name, k + 1) - rb.get(name, k + 1)) for name in union)
    max_distance = k * (k + 1)
    return float(np.clip(1.0 - distance / max_distance, 0.0, 1.0))


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / len(sa | sb)) if sa or sb else 1.0


def component_weights(group: pd.DataFrame, coefficients: list[float]) -> np.ndarray:
    x = group[WEIGHT_COMPONENTS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    raw = np.clip(x @ np.asarray(coefficients, dtype=float), 0.0, None)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        return np.repeat(1.0 / len(group), len(group))
    return raw / raw.sum()


def monitoring_support(group: pd.DataFrame) -> pd.Series:
    records = np.log1p(pd.to_numeric(group["n_result_records"], errors="coerce").fillna(0.0).clip(lower=0.0))
    sites = np.log1p(pd.to_numeric(group["n_sites"], errors="coerce").fillna(0.0).clip(lower=0.0))
    records = records / records.max() if records.max() > 0 else records * 0.0
    sites = sites / sites.max() if sites.max() > 0 else sites * 0.0
    return 0.5 * records + 0.5 * sites


def retained_group(group: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid retention fraction: {fraction}")
    n_keep = max(1, int(math.ceil(len(group) * fraction)))
    ranked = group.assign(_monitoring_support=monitoring_support(group)).sort_values(
        ["_monitoring_support", "DTXSID"], ascending=[False, True]
    )
    keep = set(ranked.head(n_keep)["DTXSID"].astype(str))
    return group[group["DTXSID"].astype(str).isin(keep)].copy()


def aggregate_occurrence_flags(occurrence: pd.DataFrame) -> pd.DataFrame:
    d = occurrence.copy()
    for col in [
        "official_state_nonindigenous_recorded",
        "current_or_unresolved_record",
        "exact_species_occurrence",
        "eligible_state_candidate",
    ]:
        d[col] = d[col].fillna(False).astype(bool)
    d["current_or_unresolved_official_nas"] = (
        d["official_state_nonindigenous_recorded"] & d["current_or_unresolved_record"]
    )
    return (
        d.groupby(["state_code", "latin_name"], as_index=False)
        .agg(
            current_or_unresolved_official_nas=("current_or_unresolved_official_nas", "max"),
            exact_species_occurrence=("exact_species_occurrence", "max"),
            eligible_state_candidate=("eligible_state_candidate", "max"),
        )
    )


def state_species_frame(
    state_code: str,
    species: list[str],
    soft_weights: pd.DataFrame,
    occurrence_flags: pd.DataFrame,
) -> pd.DataFrame:
    soft = (
        soft_weights[soft_weights["state_code"].astype(str).eq(state_code)]
        .set_index("latin_name")
        .reindex(species)
    )
    fill = {
        "soft_local_species_weight": 0.03,
        "state_occurrence_score": 0.0,
        "neighbor_occurrence_score": 0.0,
        "occurrence_breadth_score": 0.0,
        "method_support_representativeness_score": 0.35,
        "soft_local_evidence_class": "model_transfer_low_weight",
        "eligible_state_candidate": False,
        "exact_state_occurrence": False,
    }
    soft = soft.fillna(fill)
    flags = (
        occurrence_flags[occurrence_flags["state_code"].astype(str).eq(state_code)]
        .set_index("latin_name")
        .reindex(species)
        .fillna(False)
    )
    for col in flags.columns:
        soft[col if col not in soft.columns else f"source_{col}"] = flags[col].to_numpy()
    if "source_eligible_state_candidate" in soft:
        soft["eligible_state_candidate"] = soft["source_eligible_state_candidate"].astype(bool)
    if "source_exact_species_occurrence" in soft:
        soft["exact_state_occurrence"] = soft["source_exact_species_occurrence"].astype(bool)
    if "current_or_unresolved_official_nas" not in soft:
        soft["current_or_unresolved_official_nas"] = False
    soft["current_or_unresolved_official_nas"] = soft["current_or_unresolved_official_nas"].astype(bool)
    return soft


def candidate_mask(frame: pd.DataFrame, policy: str) -> np.ndarray:
    soft = pd.to_numeric(frame["soft_local_species_weight"], errors="coerce").fillna(0.03).to_numpy(float) >= 0.05
    evidence = frame["soft_local_evidence_class"].fillna("model_transfer_low_weight").astype(str)
    eligible = frame["eligible_state_candidate"].fillna(False).astype(bool).to_numpy()
    exact = frame["exact_state_occurrence"].fillna(False).astype(bool).to_numpy()
    nas = frame["current_or_unresolved_official_nas"].fillna(False).astype(bool).to_numpy()
    if policy in {"soft_baseline", "soft_all"}:
        return soft
    if policy == "hard_exclude_current_or_unresolved_official_nas":
        return soft & ~nas
    if policy in {"state_confirmed_only", "state_confirmed"}:
        return eligible & exact
    if policy == "regional_breadth_or_better":
        return soft & ~evidence.eq("model_transfer_low_weight").to_numpy()
    if policy == "neighbor_or_state":
        return soft & evidence.isin(["neighbor_occurrence", "state_occurrence"]).to_numpy()
    raise KeyError(f"Unknown species policy/tier: {policy}")


def select_panel(
    stage20: Any,
    P: np.ndarray,
    species: list[str],
    chemical_idx: np.ndarray,
    chemical_weights: np.ndarray,
    species_weights: np.ndarray,
    candidates: np.ndarray,
    national_top5_idx: list[int],
    min_chemicals: int,
    min_species: int,
) -> tuple[list[int], bool, str, int]:
    candidate_idx = np.where(candidates)[0]
    if len(chemical_idx) < min_chemicals:
        return list(national_top5_idx), False, "fallback_fewer_than_3_priority_chemicals", len(candidate_idx)
    if len(candidate_idx) < min_species:
        return list(national_top5_idx), False, "fallback_fewer_than_5_policy_eligible_species", len(candidate_idx)
    objective = P[np.ix_(candidate_idx, chemical_idx)] * species_weights[candidate_idx, None]
    local = stage20.greedy(
        objective,
        chemical_weights,
        5,
        names=[species[i] for i in candidate_idx],
    )
    chosen = [int(candidate_idx[i]) for i in local]
    if len(chosen) < 5:
        return list(national_top5_idx), False, "fallback_selection_returned_fewer_than_5_species", len(candidate_idx)
    return chosen, True, "localized", len(candidate_idx)


def evaluate_panel(
    stage20: Any,
    P: np.ndarray,
    chemical_idx: np.ndarray,
    chemical_weights: np.ndarray,
    species_weights: np.ndarray,
    panel_idx: list[int],
    rho: float,
) -> float:
    q = stage20.evaluate_sequence(P[:, chemical_idx], species_weights, chemical_weights, panel_idx, rho)
    return float(chemical_weights @ q)


def comparison_fields(panel: list[str], baseline: list[str]) -> dict[str, Any]:
    return {
        "panel_species": "; ".join(panel),
        "baseline_panel_species": "; ".join(baseline),
        "membership_jaccard_vs_baseline": jaccard(panel, baseline),
        "ordered_footrule_similarity_vs_baseline": normalized_top5_footrule(panel, baseline),
        "members_in_common": len(set(panel) & set(baseline)),
        "exact_rank_matches": sum(a == b for a, b in zip(panel, baseline)),
    }


def summarize_scenarios(df: pd.DataFrame, scenario_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, group in df.groupby(scenario_col, sort=False):
        local_scope = group[group["baseline_localized"]].copy()
        if local_scope.empty:
            local_scope = group.copy()
        rows.append(
            {
                scenario_col: scenario,
                "n_states": int(group["state_code"].nunique()),
                "localized_states": int(group["localized"].sum()),
                "fallback_states": int((~group["localized"]).sum()),
                "baseline_localizable_states": int(local_scope["state_code"].nunique()),
                "median_jaccard_baseline_localizable": float(local_scope["membership_jaccard_vs_baseline"].median()),
                "mean_jaccard_baseline_localizable": float(local_scope["membership_jaccard_vs_baseline"].mean()),
                "fraction_jaccard_ge_2_over_3": float((local_scope["membership_jaccard_vs_baseline"] >= (2 / 3 - 1e-12)).mean()),
                "fraction_jaccard_ge_3_over_7": float((local_scope["membership_jaccard_vs_baseline"] >= (3 / 7 - 1e-12)).mean()),
                "median_ordered_similarity_baseline_localizable": float(local_scope["ordered_footrule_similarity_vs_baseline"].median()),
                "mean_full_universe_coverage_delta_pp": float(local_scope["full_universe_coverage_delta_pp"].mean()),
                "mean_absolute_full_universe_coverage_delta_pp": float(local_scope["full_universe_coverage_delta_pp"].abs().mean()),
                "min_full_universe_coverage_delta_pp": float(local_scope["full_universe_coverage_delta_pp"].min()),
                "max_full_universe_coverage_delta_pp": float(local_scope["full_universe_coverage_delta_pp"].max()),
            }
        )
    return pd.DataFrame(rows)


def classify_weight_stability(weight_rows: pd.DataFrame) -> tuple[str, dict[str, float]]:
    d = weight_rows[
        ~weight_rows["weight_scenario"].eq(BASELINE_SCENARIO) & weight_rows["baseline_localized"]
    ].copy()
    metrics = {
        "median_jaccard": float(d["membership_jaccard_vs_baseline"].median()),
        "fraction_jaccard_ge_2_over_3": float((d["membership_jaccard_vs_baseline"] >= (2 / 3 - 1e-12)).mean()),
        "fraction_jaccard_ge_3_over_7": float((d["membership_jaccard_vs_baseline"] >= (3 / 7 - 1e-12)).mean()),
        "mean_absolute_full_universe_coverage_delta_pp": float(d["full_universe_coverage_delta_pp"].abs().mean()),
    }
    if (
        metrics["median_jaccard"] >= 2 / 3 - 1e-12
        and metrics["fraction_jaccard_ge_2_over_3"] >= 0.80
        and metrics["mean_absolute_full_universe_coverage_delta_pp"] <= 2.0
    ):
        return "high", metrics
    if (
        metrics["median_jaccard"] >= 3 / 7 - 1e-12
        and metrics["fraction_jaccard_ge_3_over_7"] >= 0.60
        and metrics["mean_absolute_full_universe_coverage_delta_pp"] <= 5.0
    ):
        return "moderate", metrics
    return "sensitive", metrics


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    if frame.empty:
        return "(no rows)"
    d = frame[columns].copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    headers = [str(x) for x in d.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    analysis_root = Path(args.analysis_root).resolve()
    out = Path(args.out).resolve()
    config_path = Path(args.config).resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    scenarios: dict[str, list[float]] = config["chemical_weight_scenarios"]
    for name, coefficients in scenarios.items():
        if len(coefficients) != 4 or min(coefficients) < 0 or not np.isclose(sum(coefficients), 1.0):
            raise AssertionError(f"Invalid coefficient vector for {name}: {coefficients}")

    gate_paths = [
        analysis_root / "01_ap01_eligibility/gate_ap01.json",
        analysis_root / "02_ap02_strict_loo/gate_ap02.json",
        analysis_root / "03_ap05_dependence/gate_ap05.json",
        analysis_root / "04_ap03_module_a/gate_ap03a.json",
        analysis_root / "05_ap03_module_b/gate_ap03b.json",
    ]
    gates = [json.loads(path.read_text(encoding="utf-8")) for path in gate_paths]
    if any(gate.get("status") != "PASS" for gate in gates):
        raise RuntimeError("AP04 dependency gate failure: all upstream gates must be PASS")

    probability_path = analysis_root / "02_ap02_strict_loo/ap02_strict_loo_species_chemical_tail_probability_x95.npz"
    ap02_gate = gates[1]
    ap05_gate = gates[2]
    state_weight_path = project_root / "results/weights/state_priority_chemical_weights.csv.gz"
    occurrence_path = project_root / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz"
    candidate_path = project_root / "results/probability/candidate_universe_locked.csv"
    stage20_path = project_root / "src/pipeline/20_build_state_panels_and_testing.py"

    stage20 = load_stage20(stage20_path)
    z = np.load(probability_path, allow_pickle=True)
    P = np.asarray(z["p"], dtype=float)
    species = z["species"].astype(str).tolist()
    chemicals = z["chemicals"].astype(str).tolist()
    chemical_index = {name: i for i, name in enumerate(chemicals)}
    species_index = {name: i for i, name in enumerate(species)}
    if P.shape != (len(species), len(chemicals)) or not np.isfinite(P).all():
        raise AssertionError("Invalid strict-LOO probability matrix")

    rho = float(ap05_gate["selected_downstream_rho_by_x"]["0.95"])
    national_top5 = ap02_gate["x95_disposition"]["selected_downstream_top5"]
    if len(national_top5) != 5 or any(name not in species_index for name in national_top5):
        raise AssertionError("AP02 national Top-5 is missing or not aligned to AP04 species")
    national_top5_idx = [species_index[name] for name in national_top5]

    state_weights = pd.read_csv(state_weight_path, low_memory=False)
    occurrence = pd.read_csv(occurrence_path, low_memory=False)
    candidates = pd.read_csv(candidate_path, low_memory=False)
    occurrence_flags = aggregate_occurrence_flags(occurrence)
    soft_weights = stage20.build_soft_local_weights(occurrence, candidates, species)

    state_weights["DTXSID"] = state_weights["DTXSID"].astype(str)
    state_weights = state_weights[state_weights["DTXSID"].isin(chemical_index)].copy()
    if state_weights.duplicated(["state_code", "DTXSID"]).any():
        raise AssertionError("Duplicate state-chemical rows in the frozen state-weight input")
    states = sorted(state_weights["state_code"].astype(str).unique())
    if len(states) != 35:
        raise AssertionError(f"Expected 35 states in the frozen state universe, found {len(states)}")

    max_weight_reproduction_error = 0.0
    for _, group in state_weights.groupby("state_code", sort=False):
        reproduced = component_weights(group, scenarios[BASELINE_SCENARIO])
        recorded = pd.to_numeric(group["state_weight"], errors="coerce").to_numpy(float)
        max_weight_reproduction_error = max(max_weight_reproduction_error, float(np.max(np.abs(reproduced - recorded))))
    if max_weight_reproduction_error > 1e-12:
        raise AssertionError(f"Baseline state-weight reconstruction failed: {max_weight_reproduction_error}")

    min_chemicals = int(config["minimum_priority_chemicals_for_localization"])
    min_species = int(config["minimum_candidate_species_for_localization"])

    baseline_by_state: dict[str, dict[str, Any]] = {}
    ledger_rows: list[dict[str, Any]] = []
    species_frames: dict[str, pd.DataFrame] = {}

    for state_code in states:
        group = state_weights[state_weights["state_code"].astype(str).eq(state_code)].copy()
        chemical_ids = group["DTXSID"].tolist()
        chemical_idx = np.asarray([chemical_index[name] for name in chemical_ids], dtype=int)
        weights = component_weights(group, scenarios[BASELINE_SCENARIO])
        sp_frame = state_species_frame(state_code, species, soft_weights, occurrence_flags)
        species_frames[state_code] = sp_frame
        sp_weights = pd.to_numeric(sp_frame["soft_local_species_weight"], errors="coerce").fillna(0.03).to_numpy(float)
        candidates_mask = candidate_mask(sp_frame, SOFT_POLICY)
        panel_idx, localized, reason, n_candidate = select_panel(
            stage20,
            P,
            species,
            chemical_idx,
            weights,
            sp_weights,
            candidates_mask,
            national_top5_idx,
            min_chemicals,
            min_species,
        )
        panel_names = [species[i] for i in panel_idx]
        panel_coverage = evaluate_panel(stage20, P, chemical_idx, weights, sp_weights, panel_idx, rho)
        national_coverage = evaluate_panel(stage20, P, chemical_idx, weights, sp_weights, national_top5_idx, rho)
        baseline_by_state[state_code] = {
            "group": group,
            "chemical_idx": chemical_idx,
            "weights": weights,
            "species_weights": sp_weights,
            "panel_idx": panel_idx,
            "panel_names": panel_names,
            "localized": localized,
            "reason": reason,
            "coverage": panel_coverage,
            "national_coverage": national_coverage,
        }
        class_counts = sp_frame["soft_local_evidence_class"].value_counts()
        ledger_rows.append(
            {
                "state_code": state_code,
                "n_priority_chemicals": len(group),
                "n_soft_candidate_species": n_candidate,
                "n_state_occurrence_species": int(class_counts.get("state_occurrence", 0)),
                "n_neighbor_occurrence_species": int(class_counts.get("neighbor_occurrence", 0)),
                "n_regional_breadth_species": int(class_counts.get("regional_breadth", 0)),
                "n_model_transfer_low_weight_species": int(class_counts.get("model_transfer_low_weight", 0)),
                "n_state_confirmed_eligible_species": int(
                    (sp_frame["eligible_state_candidate"].astype(bool) & sp_frame["exact_state_occurrence"].astype(bool)).sum()
                ),
                "n_current_or_unresolved_official_nas_pairs": int(sp_frame["current_or_unresolved_official_nas"].sum()),
                "localized": localized,
                "fallback_reason": reason,
                "baseline_panel_species": "; ".join(panel_names),
                "baseline_expected_weighted_joint_coverage_x95": panel_coverage,
                "national_top5_expected_weighted_joint_coverage_x95": national_coverage,
                "localization_gain_over_national_top5_pp": 100.0 * (panel_coverage - national_coverage),
            }
        )

    ledger = pd.DataFrame(ledger_rows).sort_values("state_code")
    expected_fallback = {"PA", "TN", "WV"}
    actual_fallback = set(ledger.loc[~ledger["localized"], "state_code"])
    if actual_fallback != expected_fallback:
        raise AssertionError(f"Unexpected complete-data fallback set: {sorted(actual_fallback)}")

    weight_rows: list[dict[str, Any]] = []
    for scenario_name, coefficients in scenarios.items():
        for state_code in states:
            baseline = baseline_by_state[state_code]
            group = baseline["group"]
            chemical_idx = baseline["chemical_idx"]
            scenario_weights = component_weights(group, coefficients)
            sp_frame = species_frames[state_code]
            sp_weights = baseline["species_weights"]
            panel_idx, localized, reason, n_candidate = select_panel(
                stage20,
                P,
                species,
                chemical_idx,
                scenario_weights,
                sp_weights,
                candidate_mask(sp_frame, SOFT_POLICY),
                national_top5_idx,
                min_chemicals,
                min_species,
            )
            panel_names = [species[i] for i in panel_idx]
            self_coverage = evaluate_panel(stage20, P, chemical_idx, scenario_weights, sp_weights, panel_idx, rho)
            baseline_under_scenario = evaluate_panel(
                stage20, P, chemical_idx, scenario_weights, sp_weights, baseline["panel_idx"], rho
            )
            full_baseline_eval = evaluate_panel(
                stage20, P, chemical_idx, baseline["weights"], sp_weights, panel_idx, rho
            )
            row = {
                "state_code": state_code,
                "weight_scenario": scenario_name,
                "coeff_detection": coefficients[0],
                "coeff_concentration": coefficients[1],
                "coeff_sites": coefficients[2],
                "coeff_censoring": coefficients[3],
                "n_priority_chemicals": len(group),
                "n_candidate_species": n_candidate,
                "localized": localized,
                "fallback_reason": reason,
                "baseline_localized": bool(baseline["localized"]),
                "scenario_expected_coverage": self_coverage,
                "baseline_panel_under_scenario_expected_coverage": baseline_under_scenario,
                "reselection_gain_under_scenario_pp": 100.0 * (self_coverage - baseline_under_scenario),
                "full_universe_baseline_weight_expected_coverage": full_baseline_eval,
                "baseline_full_universe_expected_coverage": baseline["coverage"],
                "full_universe_coverage_delta_pp": 100.0 * (full_baseline_eval - baseline["coverage"]),
                **comparison_fields(panel_names, baseline["panel_names"]),
            }
            weight_rows.append(row)
    weight_detail = pd.DataFrame(weight_rows)
    weight_summary = summarize_scenarios(weight_detail, "weight_scenario")
    weight_label, weight_label_metrics = classify_weight_stability(weight_detail)

    occurrence_rows: list[dict[str, Any]] = []
    for policy in config["species_policy_scenarios"]:
        for state_code in states:
            baseline = baseline_by_state[state_code]
            group = baseline["group"]
            chemical_idx = baseline["chemical_idx"]
            weights = baseline["weights"]
            sp_frame = species_frames[state_code]
            sp_weights = baseline["species_weights"]
            panel_idx, localized, reason, n_candidate = select_panel(
                stage20,
                P,
                species,
                chemical_idx,
                weights,
                sp_weights,
                candidate_mask(sp_frame, policy),
                national_top5_idx,
                min_chemicals,
                min_species,
            )
            panel_names = [species[i] for i in panel_idx]
            full_eval = evaluate_panel(stage20, P, chemical_idx, weights, sp_weights, panel_idx, rho)
            occurrence_rows.append(
                {
                    "state_code": state_code,
                    "species_policy": policy,
                    "n_priority_chemicals": len(group),
                    "n_candidate_species": n_candidate,
                    "localized": localized,
                    "fallback_reason": reason,
                    "baseline_localized": bool(baseline["localized"]),
                    "full_universe_expected_coverage": full_eval,
                    "baseline_full_universe_expected_coverage": baseline["coverage"],
                    "full_universe_coverage_delta_pp": 100.0 * (full_eval - baseline["coverage"]),
                    **comparison_fields(panel_names, baseline["panel_names"]),
                }
            )
    occurrence_detail = pd.DataFrame(occurrence_rows)
    occurrence_summary = summarize_scenarios(occurrence_detail, "species_policy")

    reduction_rows: list[dict[str, Any]] = []
    fractions = [float(x) for x in config["data_reduction"]["chemical_retention_fractions"]]
    tiers = list(config["data_reduction"]["species_evidence_tiers"])
    for fraction in fractions:
        for tier in tiers:
            for state_code in states:
                baseline = baseline_by_state[state_code]
                reduced = retained_group(baseline["group"], fraction)
                reduced_ids = reduced["DTXSID"].astype(str).tolist()
                reduced_idx = np.asarray([chemical_index[name] for name in reduced_ids], dtype=int)
                reduced_weights = component_weights(reduced, scenarios[BASELINE_SCENARIO])
                sp_frame = species_frames[state_code]
                sp_weights = baseline["species_weights"]
                panel_idx, localized, reason, n_candidate = select_panel(
                    stage20,
                    P,
                    species,
                    reduced_idx,
                    reduced_weights,
                    sp_weights,
                    candidate_mask(sp_frame, tier),
                    national_top5_idx,
                    min_chemicals,
                    min_species,
                )
                panel_names = [species[i] for i in panel_idx]
                reduced_eval = evaluate_panel(stage20, P, reduced_idx, reduced_weights, sp_weights, panel_idx, rho)
                full_eval = evaluate_panel(
                    stage20,
                    P,
                    baseline["chemical_idx"],
                    baseline["weights"],
                    sp_weights,
                    panel_idx,
                    rho,
                )
                reduction_rows.append(
                    {
                        "state_code": state_code,
                        "chemical_retention_fraction": fraction,
                        "species_evidence_tier": tier,
                        "n_complete_priority_chemicals": len(baseline["group"]),
                        "n_retained_priority_chemicals": len(reduced),
                        "n_candidate_species": n_candidate,
                        "localized": localized,
                        "fallback_reason": reason,
                        "baseline_localized": bool(baseline["localized"]),
                        "reduced_set_expected_coverage": reduced_eval,
                        "full_universe_expected_coverage": full_eval,
                        "baseline_full_universe_expected_coverage": baseline["coverage"],
                        "full_universe_coverage_delta_pp": 100.0 * (full_eval - baseline["coverage"]),
                        **comparison_fields(panel_names, baseline["panel_names"]),
                    }
                )
    reduction_detail = pd.DataFrame(reduction_rows)
    reduction_summary = summarize_scenarios(
        reduction_detail.assign(
            reduction_scenario=reduction_detail["chemical_retention_fraction"].map(lambda x: f"retain_{int(100*x)}pct")
            + "__"
            + reduction_detail["species_evidence_tier"]
        ),
        "reduction_scenario",
    )

    identity_weight = weight_detail[weight_detail["weight_scenario"].eq(BASELINE_SCENARIO)]
    identity_occurrence = occurrence_detail[occurrence_detail["species_policy"].eq(SOFT_POLICY)]
    identity_reduction = reduction_detail[
        np.isclose(reduction_detail["chemical_retention_fraction"], 1.0)
        & reduction_detail["species_evidence_tier"].eq("soft_all")
    ]
    if not (
        np.allclose(identity_weight["membership_jaccard_vs_baseline"], 1.0)
        and np.allclose(identity_occurrence["membership_jaccard_vs_baseline"], 1.0)
        and np.allclose(identity_reduction["membership_jaccard_vs_baseline"], 1.0)
        and np.allclose(identity_weight["full_universe_coverage_delta_pp"], 0.0, atol=1e-10)
        and np.allclose(identity_occurrence["full_universe_coverage_delta_pp"], 0.0, atol=1e-10)
        and np.allclose(identity_reduction["full_universe_coverage_delta_pp"], 0.0, atol=1e-10)
    ):
        raise AssertionError("AP04 identity scenarios do not reproduce the complete-data baseline")

    ledger.to_csv(out / "ap04_state_support_and_fallback_ledger.csv", index=False)
    soft_weights[soft_weights["state_code"].astype(str).isin(states)].to_csv(
        out / "ap04_reconstructed_soft_local_species_weights.csv.gz", index=False, compression="gzip"
    )
    weight_detail.to_csv(out / "ap04_weight_sensitivity_state_detail.csv", index=False)
    weight_summary.to_csv(out / "ap04_weight_sensitivity_summary.csv", index=False)
    occurrence_detail.to_csv(out / "ap04_species_policy_sensitivity_state_detail.csv", index=False)
    occurrence_summary.to_csv(out / "ap04_species_policy_sensitivity_summary.csv", index=False)
    reduction_detail.to_csv(out / "ap04_data_reduction_state_detail.csv.gz", index=False, compression="gzip")
    reduction_summary.to_csv(out / "ap04_data_reduction_summary.csv", index=False)

    input_paths = [probability_path, state_weight_path, occurrence_path, candidate_path, stage20_path, config_path, *gate_paths]
    manifest = {
        "schema_version": 1,
        "analysis_id": "AP04_LOCALIZATION_SENSITIVITY",
        "probability_source": "AP02_STRICT_LOO",
        "protection_target_x": 0.95,
        "rho_source": "AP05_STRICT_LOO",
        "rho": rho,
        "national_top5_source": "AP02_STRICT_LOO",
        "national_top5": national_top5,
        "states": states,
        "n_states": len(states),
        "n_weight_scenarios": len(scenarios),
        "n_species_policy_scenarios": len(config["species_policy_scenarios"]),
        "n_data_reduction_scenarios": len(fractions) * len(tiers),
        "max_baseline_state_weight_reproduction_error": max_weight_reproduction_error,
        "input_files": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in input_paths
        ],
    }
    (out / "ap04_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    localized = ledger[ledger["localized"]]
    gate = {
        "schema_version": 1,
        "gate": "A6_AP04_LOCALIZATION_SENSITIVITY",
        "status": "PASS",
        "dependency_gates": [path.name for path in gate_paths],
        "probability_source": "AP02_STRICT_LOO",
        "rho_x95": rho,
        "national_top5": national_top5,
        "n_states": len(states),
        "localized_states_complete_baseline": int(ledger["localized"].sum()),
        "fallback_states_complete_baseline": ledger.loc[~ledger["localized"], "state_code"].tolist(),
        "mean_localization_gain_over_national_top5_pp": float(localized["localization_gain_over_national_top5_pp"].mean()),
        "median_localization_gain_over_national_top5_pp": float(localized["localization_gain_over_national_top5_pp"].median()),
        "min_localization_gain_over_national_top5_pp": float(localized["localization_gain_over_national_top5_pp"].min()),
        "max_localization_gain_over_national_top5_pp": float(localized["localization_gain_over_national_top5_pp"].max()),
        "predeclared_weight_stability_label": weight_label,
        "predeclared_weight_stability_metrics": weight_label_metrics,
        "max_baseline_state_weight_reproduction_error": max_weight_reproduction_error,
        "identity_checks_pass": True,
        "claim_boundary": (
            "State occurrence is used as graded relevance evidence; it is not proof of native status, "
            "culturability, or field deployability. Robustness wording must follow the scenario results."
        ),
        "next_stage": "G3_CROSS_PACKAGE_INTEGRATION",
    }
    (out / "gate_ap04.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# AP04 localization sensitivity results",
        "",
        "## Dependency and provenance",
        "",
        f"- Upstream gates: all PASS ({', '.join(path.name for path in gate_paths)}).",
        "- Probability source: AP02 strict focal-species leave-one-out x = 0.95 matrix.",
        f"- Dependence source: AP05 strict-LOO rho = {rho:.9f}.",
        f"- Fixed national Top-5: {'; '.join(national_top5)}.",
        f"- Baseline state-weight reconstruction maximum absolute error: {max_weight_reproduction_error:.3e}.",
        "",
        "## Complete-data state ledger summary",
        "",
        f"The frozen universe contains {len(states)} states. {int(ledger['localized'].sum())} meet the localization support rule; "
        f"{int((~ledger['localized']).sum())} use the fixed national Top-5 fallback ({', '.join(ledger.loc[~ledger['localized'], 'state_code'])}).",
        f"Among localized states, the mean gain over the fixed national Top-5 is "
        f"{localized['localization_gain_over_national_top5_pp'].mean():.3f} percentage points "
        f"(median {localized['localization_gain_over_national_top5_pp'].median():.3f}; "
        f"range {localized['localization_gain_over_national_top5_pp'].min():.3f} to "
        f"{localized['localization_gain_over_national_top5_pp'].max():.3f}).",
        "",
        "## Chemical-weight sensitivity",
        "",
        f"Predeclared classification: **{weight_label}**.",
        markdown_table(
            weight_summary,
            [
                "weight_scenario",
                "localized_states",
                "fallback_states",
                "median_jaccard_baseline_localizable",
                "fraction_jaccard_ge_2_over_3",
                "mean_absolute_full_universe_coverage_delta_pp",
            ],
        ),
        "",
        "## Species-evidence policy sensitivity",
        "",
        markdown_table(
            occurrence_summary,
            [
                "species_policy",
                "localized_states",
                "fallback_states",
                "median_jaccard_baseline_localizable",
                "mean_absolute_full_universe_coverage_delta_pp",
                "min_full_universe_coverage_delta_pp",
            ],
        ),
        "",
        "## Data-reduction sensitivity",
        "",
        markdown_table(
            reduction_summary,
            [
                "reduction_scenario",
                "localized_states",
                "fallback_states",
                "median_jaccard_baseline_localizable",
                "mean_absolute_full_universe_coverage_delta_pp",
            ],
        ),
        "",
        "## Interpretation boundary",
        "",
        "Occurrence and NAS fields are treated as graded relevance evidence. They do not establish native status, "
        "culturability, or field deployability. The weight-stability label follows the prespecified numerical rule; "
        "species-policy and data-reduction results are reported directly without converting them into a significance claim.",
        "",
    ]
    (out / "AP04_LOCALIZATION_SENSITIVITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
