#!/usr/bin/env python3
"""Build the compact SI Table S7 sections from frozen AP04 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHT_LABELS = {
    "baseline_45_35_10_10": "Baseline",
    "equal_25_25_25_25": "Equal weights",
    "detection_heavy_60_20_10_10": "Detection-heavy",
    "concentration_heavy_20_60_10_10": "Concentration-heavy",
    "no_censoring_renormalized": "No censoring term",
    "detection_plus_0p10_renormalized": "Detection +0.10",
    "detection_minus_0p10_renormalized": "Detection -0.10",
    "concentration_plus_0p10_renormalized": "Concentration +0.10",
    "concentration_minus_0p10_renormalized": "Concentration -0.10",
    "site_plus_0p10_renormalized": "Sites +0.10",
    "site_minus_0p10_renormalized": "Sites -0.10",
    "censoring_plus_0p10_renormalized": "Censoring +0.10",
    "censoring_minus_0p10_renormalized": "Censoring -0.10",
}

POLICY_LABELS = {
    "soft_baseline": "Soft relevance (baseline)",
    "hard_exclude_current_or_unresolved_official_nas": "Exclude current or unresolved official NAS",
    "state_confirmed_only": "State-confirmed occurrence only",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def membership_counts(detail: pd.DataFrame, group_col: str) -> pd.DataFrame:
    data = detail.loc[detail["baseline_localized"].astype(bool)].copy()
    return (
        data.groupby(group_col, as_index=False)
        .agg(
            exact_membership_states=("membership_jaccard_vs_baseline", lambda s: int(np.isclose(s, 1.0).sum())),
            minimum_jaccard=("membership_jaccard_vs_baseline", "min"),
            maximum_absolute_coverage_change_pp=(
                "full_universe_coverage_delta_pp",
                lambda s: float(np.abs(s).max()),
            ),
            worst_coverage_change_pp=("full_universe_coverage_delta_pp", "min"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "config": source / "ap04_frozen_scenario_config.json",
        "weight_summary": source / "ap04_weight_sensitivity_summary.csv",
        "weight_detail": source / "ap04_weight_sensitivity_state_detail.csv",
        "policy_summary": source / "ap04_species_policy_sensitivity_summary.csv",
        "policy_detail": source / "ap04_species_policy_sensitivity_state_detail.csv",
        "reduction_summary": source / "ap04_data_reduction_summary.csv",
        "reduction_detail": source / "ap04_data_reduction_state_detail.csv.gz",
        "state_ledger": source / "ap04_state_support_and_fallback_ledger.csv",
        "gate": source / "gate_ap04.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("AP04 must pass before Table S7 is generated")

    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    weight_summary = pd.read_csv(paths["weight_summary"])
    weight_detail = pd.read_csv(paths["weight_detail"])
    policy_summary = pd.read_csv(paths["policy_summary"])
    policy_detail = pd.read_csv(paths["policy_detail"])
    reduction_summary = pd.read_csv(paths["reduction_summary"])
    reduction_detail = pd.read_csv(paths["reduction_detail"])

    weight_counts = membership_counts(weight_detail, "weight_scenario")
    table_a = weight_summary.merge(weight_counts, on="weight_scenario", how="left", suffixes=("", "_detail"))
    coefficients = config["chemical_weight_scenarios"]
    table_a.insert(0, "scenario", table_a["weight_scenario"].map(WEIGHT_LABELS))
    table_a.insert(
        1,
        "coefficients_detection_concentration_sites_censoring",
        table_a["weight_scenario"].map(
            lambda name: "/".join(f"{value:.3f}".rstrip("0").rstrip(".") for value in coefficients[name])
        ),
    )
    table_a["exact_membership_states_of_32"] = table_a["exact_membership_states"].astype(int).astype(str) + "/32"
    table_a = table_a[
        [
            "scenario",
            "coefficients_detection_concentration_sites_censoring",
            "localized_states",
            "fallback_states",
            "exact_membership_states_of_32",
            "mean_jaccard_baseline_localizable",
            "minimum_jaccard",
            "maximum_absolute_coverage_change_pp",
        ]
    ]

    policy_counts = membership_counts(policy_detail, "species_policy")
    table_b = policy_summary.merge(policy_counts, on="species_policy", how="left", suffixes=("", "_detail"))
    table_b.insert(0, "policy", table_b["species_policy"].map(POLICY_LABELS))
    table_b["exact_membership_states_of_32"] = table_b["exact_membership_states"].astype(int).astype(str) + "/32"
    table_b = table_b[
        [
            "policy",
            "localized_states",
            "fallback_states",
            "exact_membership_states_of_32",
            "mean_jaccard_baseline_localizable",
            "worst_coverage_change_pp",
        ]
    ]

    soft = reduction_summary.loc[reduction_summary["reduction_scenario"].str.endswith("__soft_all")].copy()
    detail_soft = reduction_detail.loc[
        reduction_detail["baseline_localized"].astype(bool)
        & reduction_detail["species_evidence_tier"].eq("soft_all")
    ].copy()
    fallback_map = (
        detail_soft.loc[~detail_soft["localized"].astype(bool)]
        .groupby("chemical_retention_fraction")["state_code"]
        .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
        .to_dict()
    )
    soft["chemical_records_retained_percent"] = (
        soft["reduction_scenario"].str.extract(r"retain_(\d+)pct", expand=False).astype(int)
    )
    soft["additional_fallback_states"] = soft["chemical_records_retained_percent"].map(
        lambda pct: fallback_map.get(pct / 100.0, "None") or "None"
    )
    table_c = soft[
        [
            "chemical_records_retained_percent",
            "localized_states",
            "fallback_states",
            "additional_fallback_states",
            "mean_jaccard_baseline_localizable",
            "mean_absolute_full_universe_coverage_delta_pp",
            "min_full_universe_coverage_delta_pp",
        ]
    ].sort_values("chemical_records_retained_percent", ascending=False)

    outputs = {
        "A_weight_scenarios": out / "table_s7a_weight_scenarios.csv",
        "B_species_policies": out / "table_s7b_species_policies.csv",
        "C_monitoring_reduction": out / "table_s7c_monitoring_reduction_soft_baseline.csv",
    }
    table_a.to_csv(outputs["A_weight_scenarios"], index=False)
    table_b.to_csv(outputs["B_species_policies"], index=False)
    table_c.to_csv(outputs["C_monitoring_reduction"], index=False)

    manifest = {
        "schema_version": 1,
        "table_id": "Table S7",
        "status": "PASS",
        "title": "Sensitivity of localized Top-5 panels to weighting, species-evidence policy and monitoring support",
        "section_titles": {
            "A": "Chemical-weight scenarios",
            "B": "Species-evidence policies",
            "C": "Chemical-record reduction under the baseline soft species-relevance policy",
        },
        "note": (
            "All summaries use the 32 states localizable under complete baseline support. Panels selected from reduced "
            "inputs were evaluated on the complete baseline state chemical universe. Fallback denotes use of the fixed "
            "national Top-5 when fewer than three priority chemicals or fewer than five policy-eligible candidate species "
            "remained. Jaccard values compare Top-5 membership with the complete-data state baseline. Occurrence/NAS "
            "fields are relevance evidence and do not establish native status or deployability."
        ),
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path), "rows": int(pd.read_csv(path).shape[0])}
            for key, path in outputs.items()
        },
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
    }
    manifest_path = out / "table_s7_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
