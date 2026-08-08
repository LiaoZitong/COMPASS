#!/usr/bin/env python3
"""Validate the public COMPASS input layout without changing any input file."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DTXSID_PATTERN = re.compile(r"^DTXSID\d{7,}$")
FORMAL_BINOMIAL_PATTERN = re.compile(r"^[A-Z][A-Za-z.-]+\s+[a-z][A-Za-z.-]+(?:\s+.*)?$")
US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}


@dataclass
class Finding:
    check: str
    passed: bool
    severity: str
    path: str
    detail: str


TABLE_SPECS = {
    "chemical_master": {
        "path": "data/processed/chemical_master.csv.gz",
        "required": ["DTXSID", "PREFERRED_NAME"],
        "dtxsid": "DTXSID",
    },
    "state_exposure": {
        "path": "data/processed/state_chemical_exposure_2024.csv.gz",
        "required": [
            "state_code", "DTXSID", "mapping_status", "exposure_domain",
            "pre_hc5_exposure_weight", "detection_frequency", "n_sites",
            "n_result_records",
        ],
        "dtxsid": "DTXSID",
        "state": "state_code",
        "nonnegative": ["pre_hc5_exposure_weight", "n_sites", "n_result_records"],
        "probability": ["detection_frequency"],
        "key": ["state_code", "DTXSID"],
    },
    "species_occurrence": {
        "path": "data/processed/state_species_occurrence_with_usgs_nas.csv.gz",
        "required": [
            "state_code", "latin_name", "exact_species_occurrence",
            "eligible_state_candidate",
        ],
        "state": "state_code",
        "species": "latin_name",
        "key": ["state_code", "latin_name"],
    },
    "species_traits": {
        "path": "data/reference/species_traits_curated.csv",
        "required": ["latin_name"],
        "species": "latin_name",
        "key": ["latin_name"],
    },
    "regulatory_catalog": {
        "path": "data/reference/regulatory_baseline_catalog.csv",
        "required": ["latin_name"],
        "species": "latin_name",
    },
}

CANONICAL_REQUIRED = [
    "result_id", "test_id", "reference_number", "dtxsid", "latin_name",
    "effect_family", "effect_evidence_layer", "endpoint_family",
    "endpoint_level_band", "exposure_duration_h", "organism_lifestage",
    "medium_family", "censoring_class", "concentration_point_umol_L",
    "concentration_lower_umol_L", "concentration_upper_umol_L",
    "waterborne_main",
]

AOP_REQUIRED_MEMBERS = [
    "mech_v6_out/chemical_mechanism_chemical_index.csv",
    "mech_v6_out/chemical_mechanism_feature_index.csv",
    "mech_v6_out/chemical_mechanism_active_score.npz",
    "mech_v6_out/chemical_mechanism_confidence.npz",
    "mech_v6_out/chemical_route_chemical_index.csv",
    "mech_v6_out/chemical_route_feature_index.csv",
    "mech_v6_out/chemical_route_active_score.npz",
    "mech_v6_out/chemical_route_confidence.npz",
    "mech_v6_out/chemical_chain_chemical_index.csv",
    "mech_v6_out/chemical_chain_feature_index.csv",
    "mech_v6_out/chemical_chain_active_score.npz",
    "mech_v6_out/chemical_chain_confidence.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--level", choices=["upstream", "core", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def add(
    findings: list[Finding],
    check: str,
    passed: bool,
    path: Path,
    detail: str,
    severity: str = "error",
) -> None:
    findings.append(Finding(check, bool(passed), severity, path.as_posix(), detail))


def read_table(path: Path, max_rows: int, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False, **kwargs)


def check_columns(
    findings: list[Finding], path: Path, frame: pd.DataFrame, required: Iterable[str]
) -> bool:
    missing = sorted(set(required) - set(frame.columns))
    add(
        findings,
        "required_columns",
        not missing,
        path,
        "all required columns present" if not missing else f"missing: {', '.join(missing)}",
    )
    return not missing


def check_dtxsid(findings: list[Finding], path: Path, values: pd.Series) -> None:
    normalized = values.dropna().astype(str).str.strip().str.upper()
    invalid = normalized[~normalized.str.match(DTXSID_PATTERN)]
    add(
        findings,
        "dtxsid_format",
        invalid.empty,
        path,
        f"invalid sampled values: {invalid.head(5).tolist()}" if not invalid.empty else "valid DTXSID format",
    )


def check_states(findings: list[Finding], path: Path, values: pd.Series) -> None:
    normalized = values.dropna().astype(str).str.strip().str.upper()
    invalid = sorted(set(normalized) - US_STATE_CODES)
    add(
        findings,
        "state_codes",
        not invalid,
        path,
        f"invalid values: {invalid[:10]}" if invalid else "valid two-letter U.S. state codes",
    )


def check_species(findings: list[Finding], path: Path, values: pd.Series) -> None:
    normalized = values.dropna().astype(str).str.strip()
    invalid = normalized[~normalized.str.match(FORMAL_BINOMIAL_PATTERN)]
    add(
        findings,
        "species_names",
        invalid.empty,
        path,
        f"non-binomial sampled values: {invalid.head(5).tolist()}" if not invalid.empty else "formal-binomial names",
        severity="warning",
    )


def check_range(
    findings: list[Finding], path: Path, frame: pd.DataFrame, columns: Iterable[str], low: float, high: float
) -> None:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & ((values < low) | (values > high))
        add(
            findings,
            f"range:{column}",
            not invalid.any(),
            path,
            f"expected [{low}, {high}]; invalid rows: {int(invalid.sum())}",
        )


def validate_table(root: Path, name: str, spec: dict, max_rows: int, findings: list[Finding]) -> None:
    path = root / spec["path"]
    if not path.is_file():
        add(findings, "file_exists", False, path, f"missing {name}")
        return
    add(findings, "file_exists", True, path, f"found {name}")
    try:
        frame = read_table(path, max_rows)
    except Exception as exc:  # pragma: no cover - exercised by real malformed inputs
        add(findings, "readable_table", False, path, repr(exc))
        return
    add(findings, "readable_table", True, path, f"sampled {len(frame):,} rows")
    if not check_columns(findings, path, frame, spec["required"]):
        return
    if "dtxsid" in spec:
        check_dtxsid(findings, path, frame[spec["dtxsid"]])
    if "state" in spec:
        check_states(findings, path, frame[spec["state"]])
    if "species" in spec:
        check_species(findings, path, frame[spec["species"]])
    key = spec.get("key", [])
    if key:
        duplicated = frame.duplicated(key, keep=False)
        add(
            findings,
            "duplicate_key",
            not duplicated.any(),
            path,
            f"duplicate sampled rows for {key}: {int(duplicated.sum())}",
        )
    check_range(findings, path, frame, spec.get("probability", []), 0.0, 1.0)
    check_range(findings, path, frame, spec.get("nonnegative", []), 0.0, float("inf"))


def validate_canonical(root: Path, max_rows: int, findings: list[Finding]) -> None:
    directory = root / "data/intermediate/ecotox_canonical_partitions"
    files = sorted(directory.glob("part_*.csv.gz")) if directory.is_dir() else []
    add(
        findings,
        "canonical_partitions",
        bool(files),
        directory,
        f"found {len(files)} partition(s)" if files else "no part_*.csv.gz files",
    )
    if not files:
        return
    per_file = max(1, max_rows // len(files))
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = read_table(path, per_file)
        except Exception as exc:
            add(findings, "readable_partition", False, path, repr(exc))
            continue
        check_columns(findings, path, frame, CANONICAL_REQUIRED)
        frames.append(frame)
    if not frames:
        return
    sample = pd.concat(frames, ignore_index=True)
    check_dtxsid(findings, directory, sample["dtxsid"])
    check_species(findings, directory, sample["latin_name"])
    allowed_censoring = {
        "exact", "approximate_exact", "left_censored", "right_censored", "interval_censored"
    }
    invalid_censoring = sorted(set(sample["censoring_class"].dropna().astype(str)) - allowed_censoring)
    add(
        findings,
        "censoring_values",
        not invalid_censoring,
        directory,
        f"invalid values: {invalid_censoring}" if invalid_censoring else "recognized censoring classes",
    )
    for column in [
        "concentration_point_umol_L", "concentration_lower_umol_L", "concentration_upper_umol_L"
    ]:
        values = pd.to_numeric(sample[column], errors="coerce")
        invalid = values.notna() & (values <= 0)
        add(
            findings,
            f"positive_units:{column}",
            not invalid.any(),
            directory,
            f"unit is µmol/L; nonpositive sampled rows: {int(invalid.sum())}",
        )


def validate_reference_files(root: Path, findings: list[Finding]) -> None:
    life = root / "data/reference/lifestage_codes.txt"
    if not life.is_file():
        add(findings, "file_exists", False, life, "missing life-stage code table")
    else:
        try:
            frame = pd.read_csv(life, sep="|", dtype=str)
            add(findings, "lifestage_code_column", "code" in frame, life, f"columns: {list(frame.columns)}")
        except Exception as exc:
            add(findings, "readable_lifestage_table", False, life, repr(exc))

    aop = root / "data/reference/AOP_MOA_data.zip"
    if not aop.is_file():
        add(findings, "file_exists", False, aop, "missing mechanism reference archive")
    else:
        try:
            with zipfile.ZipFile(aop) as archive:
                names = set(archive.namelist())
            missing = sorted(set(AOP_REQUIRED_MEMBERS) - names)
            add(
                findings,
                "aop_archive_members",
                not missing,
                aop,
                "all required members present" if not missing else f"missing: {missing}",
            )
        except Exception as exc:
            add(findings, "readable_zip", False, aop, repr(exc))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    findings: list[Finding] = []
    if args.level in {"upstream", "all"}:
        validate_canonical(root, args.max_rows, findings)
        for name, spec in TABLE_SPECS.items():
            validate_table(root, name, spec, args.max_rows, findings)
        validate_reference_files(root, findings)
    if args.level == "core":
        required = [
            "results/toxicity/censored_model_cells.csv.gz",
            "results/chemistry/chemical_moa_form_master.csv.gz",
            "results/chemistry/species_traits_taxonomy.csv.gz",
            "results/weights/national_priority_chemicals.csv",
            "results/weights/state_priority_chemical_weights.csv.gz",
            "data/processed/state_species_occurrence_with_usgs_nas.csv.gz",
            "data/reference/regulatory_baseline_catalog.csv",
        ]
        for relative in required:
            path = root / relative
            add(findings, "core_input_exists", path.is_file(), path, "present" if path.is_file() else "missing")

    summary = {
        "root": ".",
        "level": args.level,
        "checks": len(findings),
        "errors": sum(not item.passed and item.severity == "error" for item in findings),
        "warnings": sum(not item.passed and item.severity == "warning" for item in findings),
        "findings": [asdict(item) for item in findings],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

