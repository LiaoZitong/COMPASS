#!/usr/bin/env python3
"""Validate the frozen R1 analysis gates, exhibit tables, and optional site export."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    observed: Any
    expected: Any
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=Path("config/r1_expected_results.json"))
    parser.add_argument("--site-data", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def close(observed: float, expected: float, tolerance: float) -> bool:
    return bool(np.isclose(float(observed), float(expected), rtol=0.0, atol=tolerance))


def add(checks: list[Check], name: str, observed: Any, expected: Any, source: Path, passed: bool | None = None) -> None:
    source_label = source.name if source.is_absolute() else source.as_posix()
    checks.append(
        Check(
            name=name,
            passed=bool(observed == expected if passed is None else passed),
            observed=observed,
            expected=expected,
            source=source_label,
        )
    )


def main() -> int:
    args = parse_args()
    root = args.revision_root.resolve()
    expected_path = args.expected.resolve()
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    tolerance = float(expected["tolerance"])
    checks: list[Check] = []

    gates = {
        "AP01": root / "01_ap01_eligibility/gate_ap01.json",
        "AP02": root / "02_ap02_strict_loo/gate_ap02.json",
        "AP05": root / "03_ap05_dependence/gate_ap05.json",
        "AP03A": root / "04_ap03_module_a/gate_ap03a.json",
        "AP03B": root / "05_ap03_module_b/gate_ap03b.json",
        "AP04": root / "06_ap04_localization_sensitivity/gate_ap04.json",
        "G3": root / "07_g3_cross_package_integration/gate_g3.json",
    }
    robustness_dir = root / "08_strict_loo_robustness"
    if not robustness_dir.is_dir():
        robustness_dir = root / "09_strict_loo_robustness"
    gates["strict_LOO_robustness"] = robustness_dir / "gate_strict_loo_robustness.json"
    for name, path in gates.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        add(checks, f"gate_{name}", payload.get("status"), "PASS", path)

    freeze_path = root / "07_g3_cross_package_integration/g3_authoritative_numeric_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for key, value in expected["eligibility"].items():
        mapping = {
            "strict_loo_contexts": "strict_loo_original_n_ge_6_contexts",
            "n5_diagnostic_contexts": "original_n5_diagnostic_contexts",
        }
        observed = freeze["eligibility"][mapping.get(key, key)]
        add(checks, f"eligibility_{key}", observed, value, freeze_path)

    curves_path = root / "08_r1_exhibits/main_figure_02/figure_02a_top_k_curves.csv"
    curves = pd.read_csv(curves_path)
    x95 = curves[curves["protection_target_x"].eq(0.95)].sort_values("rank")
    top20 = x95["latin_name"].astype(str).tolist()
    add(checks, "national_top20", top20, expected["national_top20"], curves_path)
    for row in expected["panel_size_checkpoints"]:
        subset = curves[
            curves["protection_target_x"].eq(float(row["evaluation_target"]))
            & curves["rank"].eq(int(row["panel_size"]))
        ]
        observed = float(subset["expected_weighted_joint_coverage"].iloc[0])
        target = row["evaluation_target"]
        panel_size = row["panel_size"]
        add(
            checks,
            f"capture_x{target}_k{panel_size}",
            observed,
            row["expected_capture"],
            curves_path,
            close(observed, row["expected_capture"], tolerance),
        )

    rho_path = root / "03_ap05_dependence/ap05_rho_estimation_audit.csv"
    rhos = pd.read_csv(rho_path)
    rhos = rhos[rhos["selected_for_downstream"].astype(bool)]
    for target, value in expected["dependence"].items():
        observed = float(rhos.loc[rhos["protection_target_x"].eq(float(target)), "rho_working"].iloc[0])
        add(checks, f"rho_x{target}", observed, value, rho_path, close(observed, value, tolerance))

    regional_path = root / "06_ap04_localization_sensitivity/ap04_state_support_and_fallback_ledger.csv"
    regional = pd.read_csv(regional_path)
    localized = regional[regional["localized"].astype(bool)]
    observed_regional = {
        "supported_states": int(len(localized)),
        "mean_localized_expected_capture": float(localized["baseline_expected_weighted_joint_coverage_x95"].mean()),
        "mean_national_expected_capture": float(localized["national_top5_expected_weighted_joint_coverage_x95"].mean()),
        "mean_gain_percentage_points": float(localized["localization_gain_over_national_top5_pp"].mean()),
        "max_gain_state": str(localized.loc[localized["localization_gain_over_national_top5_pp"].idxmax(), "state_code"]),
        "max_gain_percentage_points": float(localized["localization_gain_over_national_top5_pp"].max()),
        "fallback_states": regional.loc[~regional["localized"].astype(bool), ["state_code", "n_priority_chemicals"]].set_index("state_code")["n_priority_chemicals"].astype(int).to_dict(),
    }
    for key, value in expected["regional"].items():
        observed = observed_regional[key]
        passed = close(observed, value, tolerance) if isinstance(value, float) else observed == value
        add(checks, f"regional_{key}", observed, value, regional_path, passed)

    validation = expected["available_member_validation"]
    observed_validation = freeze["available_member_validation"]
    add(checks, "available_member_n", observed_validation["eligible_rows_and_unique_chemicals"], validation["n"], freeze_path)
    add(
        checks,
        "available_member_spearman",
        observed_validation["spearman_typical_hc5"],
        validation["spearman_rho"],
        freeze_path,
        close(observed_validation["spearman_typical_hc5"], validation["spearman_rho"], tolerance),
    )
    add(checks, "available_member_full_top5", observed_validation["fully_observed_top5_rows"], validation["fully_observed_top5_rows"], freeze_path)
    for key, value in expected["followup_ranking"].items():
        add(checks, f"followup_{key}", freeze["followup_ranking"][key], value, freeze_path)

    robustness_expected = expected["strict_loo_robustness"]
    robustness_manifest_path = robustness_dir / "strict_loo_robustness_manifest.json"
    robustness_manifest = json.loads(robustness_manifest_path.read_text(encoding="utf-8"))
    add(
        checks,
        "robustness_profiled_cells",
        robustness_manifest["profile"]["profiled_cells"],
        robustness_expected["profiled_cells"],
        robustness_manifest_path,
    )
    add(
        checks,
        "robustness_profile_contexts",
        robustness_manifest["profile"]["profile_contexts"],
        robustness_expected["profile_contexts"],
        robustness_manifest_path,
    )
    add(
        checks,
        "robustness_mnar_observed_pairs",
        robustness_manifest["mnar"]["observed_pairs"],
        robustness_expected["mnar_observed_pairs"],
        robustness_manifest_path,
    )
    add(
        checks,
        "robustness_core_coverage",
        robustness_manifest["mnar"]["strict_loo_core_coverage"],
        robustness_expected["strict_loo_core_coverage"],
        robustness_manifest_path,
        close(
            robustness_manifest["mnar"]["strict_loo_core_coverage"],
            robustness_expected["strict_loo_core_coverage"],
            tolerance,
        ),
    )
    add(
        checks,
        "robustness_top5_member_flip_threshold",
        robustness_manifest["mnar"]["top5_member_flip_threshold_within_tested_or_range"],
        robustness_expected["top5_member_flip_threshold_within_tested_or_range"],
        robustness_manifest_path,
    )
    scenarios_path = robustness_dir / "strict_loo_mnar_sensitivity_scenarios.csv"
    scenarios = pd.read_csv(scenarios_path)
    for row in robustness_expected["mnar_scenarios"]:
        observed = float(
            scenarios.loc[
                np.isclose(scenarios.odds_ratio_tail_positive_testing, float(row["odds_ratio"])),
                "coverage_copula",
            ].iloc[0]
        )
        add(
            checks,
            f"robustness_mnar_or_{row['odds_ratio']}",
            observed,
            row["expected_capture"],
            scenarios_path,
            close(observed, row["expected_capture"], tolerance),
        )
    profile_path = robustness_dir / "strict_loo_framework_profile_metrics.csv"
    profile = pd.read_csv(profile_path).iloc[0]
    for key, value in robustness_expected["profile_medians"].items():
        observed = float(profile[f"{key}_q50"])
        add(
            checks,
            f"robustness_profile_{key}",
            observed,
            value,
            profile_path,
            close(observed, value, tolerance),
        )

    if args.site_data:
        site = args.site_data.resolve()
        national_path = site / "national_sequence.json"
        coverage_path = site / "coverage.json"
        regional_site_path = site / "regional.json"
        metadata_path = site / "metadata.json"
        national = json.loads(national_path.read_text(encoding="utf-8"))
        site_top20 = [row["scientific_name"] for row in national[:20]]
        add(checks, "site_national_top20", site_top20, expected["national_top20"], national_path)
        coverage = pd.DataFrame(json.loads(coverage_path.read_text(encoding="utf-8")))
        for target, value in expected["fixed_x95_sequence_expected_capture"].items():
            observed = float(
                coverage.loc[
                    coverage["evaluation_target"].eq(float(target)) & coverage["panel_size"].eq(5),
                    "expected_capture",
                ].iloc[0]
            )
            add(checks, f"site_capture_x{target}_k5", observed, value, coverage_path, close(observed, value, tolerance))
        site_regional = json.loads(regional_site_path.read_text(encoding="utf-8"))
        supported = [row for row in site_regional if row["status"] == "supported"]
        add(checks, "site_supported_states", len(supported), expected["regional"]["supported_states"], regional_site_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        add(checks, "site_analysis_version", metadata.get("analysis_version"), expected["analysis_version"], metadata_path)

    failed = [check for check in checks if not check.passed]
    report = {
        "schema_version": 1,
        "profile": "R1 strict-LOO release",
        "checks": len(checks),
        "failed": len(failed),
        "all_passed": not failed,
        "results": [asdict(check) for check in checks],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8", newline="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
