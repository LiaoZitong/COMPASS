#!/usr/bin/env python3
"""Build hash-only COMPASS release records without copying analysis data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


KEY_OUTPUTS = [
    "results/panels/national_panel_sequences.csv",
    "results/panels/fixed_x95_sequence_cross_target_curves.csv",
    "results/panels/fixed_national_top5_cross_target_audit.csv",
    "results/probability/candidate_universe_locked.csv",
    "results/probability/species_chemical_tail_probability_x95.npz",
    "results/probability/rapid_warning_species_evidence_observed_only.csv",
    "results/complementarity/top5_top10_species_incremental_and_taxonomic_contributions.csv",
    "results/toxicity/censored_model_cells.csv.gz",
    "results/state_panels/state_x95_panel_sequences.csv",
    "results/state_panels/state_x95_k5_soft_local_comparison.csv",
    "results/weights/state_priority_chemical_weights.csv.gz",
    "results/weights/national_priority_chemicals.csv",
    "results/testing/national_species_testing_priorities.csv",
    "results/warning_hc5_bridge/top5_panel_apical_hc5_positive_control_metrics.csv",
    "results/warning_hc5_bridge/top5_panel_warning_hc5_metrics.csv",
    "manuscript_plot/figure_02/figure_02b_feasible_combination_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site-root", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_pairs(package_root: Path, analysis_root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for public in sorted((package_root / "code/pipeline").glob("*.py")):
        pairs.append((public.relative_to(package_root).as_posix(), f"src/pipeline/{public.name}"))
    for public in sorted((package_root / "code/figures").rglob("*.py")):
        relative = public.relative_to(package_root).as_posix()
        origin = relative.removeprefix("code/figures/")
        pairs.append((relative, f"manuscript_plot/{origin}"))
    pairs.extend(
        [
            ("code/run_analysis.py", "run_all.ps1"),
            ("run_analysis.ps1", "run_all.ps1"),
            ("config/requirements-lock.txt", "manifests/python_requirements_lock.txt"),
            (
                "config/submission_environment_specification.json",
                "manifests/submission_environment_specification.json",
            ),
        ]
    )
    missing = [origin for _, origin in pairs if not (analysis_root / origin).is_file()]
    if missing:
        raise FileNotFoundError("Missing copied-source origin(s): " + ", ".join(missing))
    return pairs


def build_source_manifest(package_root: Path, analysis_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for public_relative, origin_relative in source_pairs(package_root, analysis_root):
        public = package_root / public_relative
        origin = analysis_root / origin_relative
        if not public.is_file():
            raise FileNotFoundError(public_relative)
        public_hash = sha256(public)
        origin_hash = sha256(origin)
        rows.append(
            {
                "public_path": public_relative,
                "source_path": origin_relative,
                "public_sha256": public_hash,
                "source_sha256": origin_hash,
                "identical_copy": str(public_hash == origin_hash).lower(),
                "public_bytes": public.stat().st_size,
                "source_bytes": origin.stat().st_size,
                "source_modified_utc": datetime.fromtimestamp(
                    origin.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    write_csv(
        package_root / "release/source_manifest_sha256.csv",
        [
            "public_path",
            "source_path",
            "public_sha256",
            "source_sha256",
            "identical_copy",
            "public_bytes",
            "source_bytes",
            "source_modified_utc",
        ],
        rows,
    )


def build_output_manifest(package_root: Path, analysis_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for relative in KEY_OUTPUTS:
        path = analysis_root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append(
            {
                "source_path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "distributed_in_code_package": "false",
            }
        )
    write_csv(
        package_root / "release/key_output_manifest_sha256.csv",
        ["source_path", "sha256", "bytes", "modified_utc", "distributed_in_code_package"],
        rows,
    )


def build_crosswalk(package_root: Path) -> None:
    expected = json.loads((package_root / "config/r1_expected_results.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "claim": "national_top5",
            "value": " | ".join(expected["national_top20"][:5]),
            "analysis_source": "R1 AP02 figure_02a_top_k_curves.csv",
            "site_source": "site_data/national_sequence.json",
        }
    )
    for target, value in expected["fixed_x95_sequence_expected_capture"].items():
        rows.append(
            {
                "claim": f"fixed_x95_sequence_expected_capture_{target}",
                "value": format(float(value), ".16g"),
                "analysis_source": "R1 AP02 figure_02a_top_k_curves.csv",
                "site_source": "site_data/coverage.json",
            }
        )
    regional = expected["regional"]
    rows.extend(
        [
            {
                "claim": "regional_supported_states",
                "value": str(regional["supported_states"]),
                "analysis_source": "R1 AP04 ap04_state_support_and_fallback_ledger.csv",
                "site_source": "site_data/regional.json",
            },
            {
                "claim": "regional_mean_gain_percentage_points",
                "value": format(float(regional["mean_gain_percentage_points"]), ".16g"),
                "analysis_source": "R1 AP04 ap04_state_support_and_fallback_ledger.csv",
                "site_source": "site_data/regional.json",
            },
            {
                "claim": "available_member_spearman",
                "value": format(float(expected["available_member_validation"]["spearman_rho"]), ".16g"),
                "analysis_source": "R1 G3 g3_authoritative_numeric_freeze.json",
                "site_source": "not displayed",
            },
            {
                "claim": "mortality_excluded_scoreable_chemicals",
                "value": str(expected["followup_ranking"]["mortality_excluded_scoreable"]),
                "analysis_source": "R1 G3 g3_authoritative_numeric_freeze.json",
                "site_source": "not displayed",
            },
        ]
    )
    write_csv(
        package_root / "release/result_crosswalk.csv",
        ["claim", "value", "analysis_source", "site_source"],
        rows,
    )


def build_run_metadata(package_root: Path, site_root: Path | None) -> None:
    config = json.loads((package_root / "config/reproduction_config.json").read_text(encoding="utf-8"))
    site_metadata: dict[str, Any] = {}
    if site_root:
        site_path = site_root / "site_data/metadata.json"
        if site_path.is_file():
            site_metadata = json.loads(site_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "analysis_version": config["analysis_version"],
        "data_freeze_date": config["data_freeze_date"],
        "seed": config["random_settings"]["pipeline_seed"],
        "python_runtime_used_for_local_validation": platform.python_version(),
        "source_version_record": "release/source_manifest_sha256.csv",
        "key_output_record": "release/key_output_manifest_sha256.csv",
        "source_workspace_git_history_available": False,
        "site_export_identifier": site_metadata.get("source_analysis_identifier"),
        "site_version": site_metadata.get("site_version"),
        "data_distribution": "Large public and provider-hosted inputs, including EPA ECOTOX, are obtained from their official services and are not mirrored in the compact GitHub code package.",
        "local_validation_commands": [
            "python -m unittest discover -s tests -v",
            "python code/run_analysis.py --mode full --dry-run --seed 20260622",
            "python code/validate_results.py --profile analysis",
            "python code/audit_release.py --profile private",
        ],
    }
    write_json(package_root / "release/run_metadata.json", payload)


def should_checksum(relative: Path) -> bool:
    parts = set(relative.parts)
    if parts.intersection({".git", ".tmp", "__pycache__", "node_modules", ".next", "dist", ".wrangler"}):
        return False
    if len(relative.parts) >= 2 and relative.parts[:2] == ("public", "explorer"):
        return False
    if relative.suffix.lower() in {".pyc", ".pyo"}:
        return False
    if relative.suffix.lower() == ".log":
        return False
    if relative.as_posix() == "release/checksums.sha256":
        return False
    if relative.parts and relative.parts[0] == "outputs":
        return False
    return True


def build_checksums(root: Path) -> None:
    destination = root / "release/checksums.sha256"
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and should_checksum(path.relative_to(root))
    ]
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    analysis_root = args.analysis_root.resolve()
    site_root = args.site_root.resolve() if args.site_root else None
    build_source_manifest(package_root, analysis_root)
    build_output_manifest(package_root, analysis_root)
    build_crosswalk(package_root)
    build_run_metadata(package_root, site_root)
    build_checksums(package_root)
    if site_root:
        build_checksums(site_root)
    print("Release metadata and checksums generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
