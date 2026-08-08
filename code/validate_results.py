#!/usr/bin/env python3
"""Validate frozen COMPASS result checkpoints and optional static-site exports."""

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
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--profile", choices=["analysis", "release"], default="analysis")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--site-data", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def close(observed: float, expected: float, tolerance: float) -> bool:
    return bool(np.isclose(float(observed), float(expected), rtol=0.0, atol=tolerance))


def add(checks: list[Check], name: str, passed: bool, observed: Any, expected: Any, source: str) -> None:
    checks.append(Check(name, bool(passed), observed, expected, source))


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_site_json(site_data: Path, name: str) -> Any:
    path = site_data / name
    require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    expected_path = args.expected or root / "config/expected_results.json"
    require(expected_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    tolerance = float(expected.get("tolerance", 1e-9))
    checks: list[Check] = []

    sequence_path = root / "results/panels/national_panel_sequences.csv"
    curves_path = root / "results/panels/fixed_x95_sequence_cross_target_curves.csv"
    audit_path = root / "results/panels/fixed_national_top5_cross_target_audit.csv"
    state_path = root / "results/state_panels/state_x95_k5_soft_local_comparison.csv"
    state_weights_path = root / "results/weights/state_priority_chemical_weights.csv.gz"
    apical_path = root / "results/warning_hc5_bridge/top5_panel_apical_hc5_positive_control_metrics.csv"
    warning_path = root / "results/warning_hc5_bridge/top5_panel_warning_hc5_metrics.csv"
    for path in [sequence_path, curves_path, audit_path, state_path, state_weights_path, apical_path, warning_path]:
        require(path)

    sequence = pd.read_csv(sequence_path)
    fixed = sequence[
        sequence["universe"].eq("priority")
        & sequence["protection_target_x"].eq(0.95)
        & sequence["method"].eq("data_driven")
    ].sort_values("rank")
    observed_top10 = fixed.head(10)["latin_name"].tolist()
    add(checks, "national_top10", observed_top10 == expected["national_top10"], observed_top10, expected["national_top10"], sequence_path.as_posix())
    add(checks, "national_top5", observed_top10[:5] == expected["national_top10"][:5], observed_top10[:5], expected["national_top10"][:5], sequence_path.as_posix())
    observed_top20 = fixed.head(20)["latin_name"].tolist()
    add(checks, "national_top20", observed_top20 == expected["national_top20"], observed_top20, expected["national_top20"], sequence_path.as_posix())

    audit = pd.read_csv(audit_path)
    observed_capture = {
        str(float(row.evaluation_target)).rstrip("0").rstrip("."): float(row.expected_capture)
        for row in audit.itertuples(index=False)
    }
    for target, value in expected["fixed_top5_expected_capture"].items():
        observed = observed_capture.get(target)
        add(
            checks,
            f"fixed_top5_capture_{target}",
            observed is not None and close(observed, value, tolerance),
            observed,
            value,
            audit_path.as_posix(),
        )

    curves = pd.read_csv(curves_path)
    add(
        checks,
        "sequence_basis_locked",
        set(curves["sequence_basis_target"].astype(float)) == {0.95},
        sorted(set(curves["sequence_basis_target"].astype(float))),
        [0.95],
        curves_path.as_posix(),
    )
    for checkpoint in expected["panel_size_checkpoints"]:
        rows = curves[
            curves["evaluation_target"].eq(checkpoint["evaluation_target"])
            & curves["panel_size"].eq(checkpoint["panel_size"])
        ]
        observed = None if rows.empty else float(rows.iloc[0]["expected_capture"])
        name = f"coverage_x{checkpoint['evaluation_target']}_k{checkpoint['panel_size']}"
        add(checks, name, observed is not None and close(observed, checkpoint["expected_capture"], tolerance), observed, checkpoint["expected_capture"], curves_path.as_posix())

    apical = pd.read_csv(apical_path)
    apical_expected = expected["top5_apical"]
    apical_row = apical[apical["reference"].eq(apical_expected["reference"])].iloc[0]
    add(checks, "top5_apical_n", int(apical_row["n"]) == apical_expected["n"], int(apical_row["n"]), apical_expected["n"], apical_path.as_posix())
    add(checks, "top5_apical_spearman", close(apical_row["spearman_r"], apical_expected["spearman_r"], tolerance), float(apical_row["spearman_r"]), apical_expected["spearman_r"], apical_path.as_posix())

    warning = pd.read_csv(warning_path)
    warning_expected = expected["warning"]
    warning_row = warning[warning["reference"].eq(warning_expected["reference"])].iloc[0]
    for field in ["n", "spearman_r", "median_ratio"]:
        observed = int(warning_row[field]) if field == "n" else float(warning_row[field])
        passed = observed == warning_expected[field] if field == "n" else close(observed, warning_expected[field], tolerance)
        add(checks, f"warning_{field}", passed, observed, warning_expected[field], warning_path.as_posix())

    state = pd.read_csv(state_path)
    regional = expected["regional"]
    add(checks, "regional_supported_states", len(state) == regional["supported_states"], len(state), regional["supported_states"], state_path.as_posix())
    state_metrics = {
        "mean_localized_expected_capture": state["state_soft_local_expected_weighted_joint_coverage_x95"].mean(),
        "mean_national_expected_capture": state["national_top5_soft_local_expected_weighted_joint_coverage_x95"].mean(),
        "mean_gain_percentage_points": state["state_gain_over_national_top5_percent_points"].mean(),
    }
    for field, observed in state_metrics.items():
        add(checks, f"regional_{field}", close(observed, regional[field], tolerance), float(observed), regional[field], state_path.as_posix())
    max_row = state.loc[state["state_gain_over_national_top5_percent_points"].idxmax()]
    add(checks, "regional_max_gain_state", max_row["state_code"] == regional["max_gain_state"], max_row["state_code"], regional["max_gain_state"], state_path.as_posix())
    add(checks, "regional_max_gain_value", close(max_row["state_gain_over_national_top5_percent_points"], regional["max_gain_percentage_points"], tolerance), float(max_row["state_gain_over_national_top5_percent_points"]), regional["max_gain_percentage_points"], state_path.as_posix())

    state_weights = pd.read_csv(state_weights_path, usecols=["state_code"])
    counts = state_weights.groupby("state_code").size().to_dict()
    observed_unsupported = {state_code: int(counts.get(state_code, 0)) for state_code in regional["unsupported_states"]}
    add(checks, "regional_unsupported_states", observed_unsupported == regional["unsupported_states"], observed_unsupported, regional["unsupported_states"], state_weights_path.as_posix())

    comparator_candidates = [
        root / "outputs/figures/figure_02/figure_02b_feasible_combination_summary.csv",
        root / "manuscript_plot/figure_02/figure_02b_feasible_combination_summary.csv",
    ]
    comparator_path = next((path for path in comparator_candidates if path.is_file()), comparator_candidates[0])
    if comparator_path.is_file():
        comparator = pd.read_csv(comparator_path)
        wet = comparator[
            comparator["method"].eq("US_EPA_WET_species")
            & comparator["protection_target_x"].eq(0.95)
        ].iloc[0]
        random = comparator[
            comparator["method"].eq("random_all_species_probability")
            & comparator["protection_target_x"].eq(0.95)
        ].iloc[0]
        comp_expected = expected["comparators"]
        add(checks, "comparator_epa_wet_x95", close(wet["deterministic_proxy_coverage"], comp_expected["epa_wet_fixed_k8_expected_capture_0.95"], tolerance), float(wet["deterministic_proxy_coverage"]), comp_expected["epa_wet_fixed_k8_expected_capture_0.95"], comparator_path.as_posix())
        add(checks, "comparator_random_x95_median", close(random["sampled_median"], comp_expected["random_k5_median_expected_capture_0.95"], tolerance), float(random["sampled_median"]), comp_expected["random_k5_median_expected_capture_0.95"], comparator_path.as_posix())
    elif args.profile == "release":
        add(checks, "comparator_output_exists", False, None, comparator_path.as_posix(), comparator_path.as_posix())

    if args.site_data:
        site_data = args.site_data.resolve()
        metadata = load_site_json(site_data, "metadata.json")
        national = load_site_json(site_data, "national_sequence.json")
        coverage = load_site_json(site_data, "coverage.json")
        regional_site = load_site_json(site_data, "regional.json")
        site_top10 = [row["scientific_name"] for row in national[:10]]
        add(checks, "site_top10", site_top10 == expected["national_top10"], site_top10, expected["national_top10"], "site_data/national_sequence.json")
        site_top20 = [row["scientific_name"] for row in national[:20]]
        add(checks, "site_top20", site_top20 == expected["national_top20"], site_top20, expected["national_top20"], "site_data/national_sequence.json")
        site_ranks = [int(row["national_rank"]) for row in national]
        add(checks, "site_complete_national_order", site_ranks == list(range(1, 2145)), {"rows": len(site_ranks), "unique": len(set(site_ranks))}, {"rows": 2144, "unique": 2144}, "site_data/national_sequence.json")
        site_k5 = {
            str(float(row["evaluation_target"])).rstrip("0").rstrip("."): row["expected_capture"]
            for row in coverage
            if int(row["panel_size"]) == 5
        }
        for target, value in expected["fixed_top5_expected_capture"].items():
            add(checks, f"site_capture_{target}", target in site_k5 and close(site_k5[target], value, tolerance), site_k5.get(target), value, "site_data/coverage.json")
        supported = [row for row in regional_site if row["status"] == "supported"]
        add(checks, "site_regional_count", len(supported) == regional["supported_states"], len(supported), regional["supported_states"], "site_data/regional.json")
        add(checks, "site_analysis_version", metadata.get("analysis_version") == expected["analysis_version"], metadata.get("analysis_version"), expected["analysis_version"], "site_data/metadata.json")

    failed = [check for check in checks if not check.passed]
    report = {
        "schema_version": 1,
        "analysis_version": expected["analysis_version"],
        "profile": args.profile,
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "all_passed": not failed,
        "details": [asdict(check) for check in checks],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
