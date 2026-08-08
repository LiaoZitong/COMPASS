#!/usr/bin/env python3
"""Export frozen COMPASS results into static, browser-readable files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib


MANUSCRIPT_TITLE = (
    "Selecting Sensitive Local Aquatic Sentinel Panels for Protection-Oriented "
    "Monitoring under Sparse and Imbalanced Evidence"
)
AUTHORS = [
    "Zitong Liao", "Zhuo Chen", "Yun Lu", "Dongbin Wei", "Yinhu Wu",
    "Yarong Qi", "Yuming Wang", "Ren Ding", "Hong-Ying Hu",
]
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

SOURCE_FILES = [
    "results/panels/national_panel_sequences.csv",
    "results/panels/fixed_x95_sequence_cross_target_curves.csv",
    "results/panels/fixed_national_top5_cross_target_audit.csv",
    "results/probability/candidate_universe_locked.csv",
    "results/probability/rapid_warning_species_evidence_observed_only.csv",
    "results/complementarity/top5_top10_species_incremental_and_taxonomic_contributions.csv",
    "results/toxicity/censored_model_cells.csv.gz",
    "results/state_panels/state_x95_panel_sequences.csv",
    "results/state_panels/state_x95_k5_soft_local_comparison.csv",
    "results/weights/state_priority_chemical_weights.csv.gz",
    "results/testing/national_species_testing_priorities.csv",
    "results/warning_hc5_bridge/top5_panel_apical_hc5_positive_control_metrics.csv",
    "results/warning_hc5_bridge/top5_panel_warning_hc5_metrics.csv",
    "results/probability/species_chemical_tail_probability_x95.npz",
    "results/weights/national_priority_chemicals.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
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
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def int_or_zero(value: Any) -> int:
    return 0 if pd.isna(value) else int(value)


def support_note(tier: str) -> str:
    if str(tier).startswith("very_sparse"):
        return "Very limited direct protective support; confirmatory testing is a high priority."
    if str(tier).startswith("sparse"):
        return "Limited direct protective support; interpret the rank with the reported uncertainty."
    return ""


def support_label(tier: str) -> str:
    labels = {
        "very_sparse_1_4": "Very limited (1-4 protective contexts)",
        "sparse_5_19": "Limited (5-19 protective contexts)",
        "moderate_20_49": "Moderate (20-49 protective contexts)",
        "strong_50_plus": "Strong (50+ protective contexts)",
    }
    return labels.get(str(tier), str(tier).replace("_", " "))


def require_sources(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        rows.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    if missing:
        raise FileNotFoundError("Missing site-export input(s): " + ", ".join(missing))
    return rows


def selected_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame["universe"].eq("priority")
        & frame["protection_target_x"].eq(0.95)
        & frame["method"].eq("data_driven")
    ].copy()
    selected = selected.sort_values("rank").drop_duplicates("rank")
    if selected.empty or selected["rank"].min() != 1:
        raise ValueError("Frozen lower-5% national sequence is missing or malformed")
    return selected


def build_complete_national_order(
    root: Path,
    frozen_sequence: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Continue the frozen lower-5% greedy objective through every candidate species."""
    probability = np.load(
        root / "results/probability/species_chemical_tail_probability_x95.npz",
        allow_pickle=True,
    )
    matrix = probability["p"].astype(float)
    species = probability["species"].astype(str)
    chemicals = probability["chemicals"].astype(str)
    chemical_index = {name: index for index, name in enumerate(chemicals)}

    candidate_names = set(candidate["latin_name"].astype(str))
    matrix_names = set(species)
    if candidate_names != matrix_names:
        raise ValueError(
            "Candidate universe and x=0.95 probability species differ: "
            f"candidate={len(candidate_names)}, probability={len(matrix_names)}"
        )

    priority = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    priority["DTXSID"] = priority["DTXSID"].astype(str)
    priority = priority[
        priority["DTXSID"].isin(chemical_index)
        & priority["national_weight"].astype(float).gt(0)
    ].copy()
    if len(priority) != 362:
        raise ValueError(f"Expected 362 national priority chemicals, found {len(priority)}")
    weights = priority["national_weight"].to_numpy(float)
    weights /= weights.sum()
    columns = np.asarray([chemical_index[name] for name in priority["DTXSID"]], dtype=int)
    probability_matrix = np.ascontiguousarray(matrix[:, columns])

    no_event = np.ones(probability_matrix.shape[1], dtype=float)
    selected_mask = np.zeros(probability_matrix.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    for rank in range(1, probability_matrix.shape[0] + 1):
        gains = probability_matrix @ (weights * no_event)
        gains[selected_mask] = -np.inf
        index = int(np.argmax(gains))
        selected_mask[index] = True
        incremental_gain = float(gains[index])
        no_event *= 1.0 - probability_matrix[index]
        rows.append(
            {
                "rank": rank,
                "latin_name": str(species[index]),
                "selection_incremental_gain_lower5": incremental_gain,
                "selection_objective_cumulative_lower5": float(weights @ (1.0 - no_event)),
                "rank_scope": "frozen_top20" if rank <= 20 else "extended_greedy_order",
            }
        )

    complete = pd.DataFrame(rows)
    frozen_names = frozen_sequence.sort_values("rank")["latin_name"].astype(str).tolist()
    if complete.head(len(frozen_names))["latin_name"].tolist() != frozen_names:
        raise ValueError("Complete national order does not reproduce the frozen Top-20 sequence")
    if len(complete) != len(candidate) or not complete["latin_name"].is_unique:
        raise ValueError("Complete national order must contain each candidate species exactly once")
    return complete


def build_national(
    sequence: pd.DataFrame,
    candidate: pd.DataFrame,
    warning: pd.DataFrame,
    contributions: pd.DataFrame,
    life: pd.DataFrame,
    curves: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidate_by_name = candidate.set_index("latin_name")
    warning_by_name = warning.set_index("latin_name")
    life_by_name = life.set_index("latin_name")
    top5_contrib = contributions[
        contributions["method"].eq("data_driven") & contributions["k"].eq(5)
    ].drop_duplicates("latin_name").set_index("latin_name")
    curve_map = {
        (round(float(row.evaluation_target), 2), int(row.panel_size)): float(row.expected_capture)
        for row in curves.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for row in sequence.itertuples(index=False):
        rank = int(row.rank)
        name = str(row.latin_name)
        meta = candidate_by_name.loc[name]
        warning_row = warning_by_name.loc[name] if name in warning_by_name.index else None
        life_row = life_by_name.loc[name] if name in life_by_name.index else None
        contribution = top5_contrib.loc[name] if name in top5_contrib.index else None
        lower5 = curve_map.get((0.95, rank))
        previous_lower5 = 0.0 if rank == 1 else curve_map.get((0.95, rank - 1))
        validated_incremental = (
            None
            if lower5 is None or previous_lower5 is None
            else float(lower5 - previous_lower5)
        )
        records.append(
            {
                "national_rank": rank,
                "scientific_name": name,
                "common_name": None,
                "class": str(meta.get("class", "")),
                "order": str(meta.get("tax_order", "")),
                "family": str(meta.get("family", "")),
                "phylum": str(meta.get("phylum_division", "")),
                "top5": rank <= 5,
                "top10": rank <= 10,
                "evidence_support": str(meta.get("support_tier", "")),
                "evidence_support_label": support_label(str(meta.get("support_tier", ""))),
                "direct_protective_context_count": int(meta.get("n_contexts", 0)),
                "direct_protective_chemical_count": int(meta.get("n_chemicals", 0)),
                "warning_context_count": 0 if warning_row is None else int(warning_row.get("warning_contexts", 0)),
                "eligible_state_count": int(meta.get("eligible_states", 0)),
                "broad_life_stage_category_count": 0 if life_row is None else int(life_row.get("broad_life_stage_category_count", 0)),
                "official_or_common_method_species": bool_value(meta.get("official_or_common_method_species", False)),
                "selection_incremental_gain_lower5": float(row.selection_incremental_gain_lower5),
                "selection_objective_cumulative_lower5": float(row.selection_objective_cumulative_lower5),
                "validated_incremental_expected_capture_lower5": validated_incremental,
                "rank_scope": str(row.rank_scope),
                "leave_one_out_loss_top5": None if contribution is None else optional_float(contribution.get("leave_one_out_loss")),
                "cumulative_expected_capture": {
                    "lower-5%": curve_map.get((0.95, rank)),
                    "lower-10%": curve_map.get((0.90, rank)),
                    "lower-20%": curve_map.get((0.80, rank)),
                },
                "support_note": support_note(str(meta.get("support_tier", ""))),
            }
        )
    return records


def build_catalog(
    candidate: pd.DataFrame,
    warning: pd.DataFrame,
    life: pd.DataFrame,
    sequence: pd.DataFrame,
    testing: pd.DataFrame,
) -> list[dict[str, Any]]:
    frame = candidate.copy()
    ranks = sequence[["latin_name", "rank"]].rename(columns={"rank": "national_rank"})
    frame = frame.merge(ranks, on="latin_name", how="left")
    frame = frame.merge(
        warning[["latin_name", "warning_contexts", "warning_chemicals"]],
        on="latin_name",
        how="left",
    )
    frame = frame.merge(life, on="latin_name", how="left")
    frame = frame.merge(
        testing[["latin_name", "species_testing_priority"]],
        on="latin_name",
        how="left",
    )
    frame = frame.sort_values(["national_rank", "latin_name"], na_position="last")
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        rank = optional_int(row.get("national_rank"))
        records.append(
            {
                "scientific_name": str(row["latin_name"]),
                "common_name": None,
                "class": str(row.get("class", "")),
                "order": str(row.get("tax_order", "")),
                "family": str(row.get("family", "")),
                "phylum": str(row.get("phylum_division", "")),
                "national_rank": rank,
                "top5": rank is not None and rank <= 5,
                "top10": rank is not None and rank <= 10,
                "protective_context_count": int_or_zero(row.get("n_contexts", 0)),
                "protective_chemical_count": int_or_zero(row.get("n_chemicals", 0)),
                "warning_context_count": int_or_zero(row.get("warning_contexts", 0)),
                "warning_chemical_count": int_or_zero(row.get("warning_chemicals", 0)),
                "broad_life_stage_category_count": int_or_zero(row.get("broad_life_stage_category_count", 0)),
                "eligible_state_count": int_or_zero(row.get("eligible_states", 0)),
                "state_occurrence_available": bool_value(row.get("state_occurrence_available", False)),
                "evidence_support": str(row.get("support_tier", "")),
                "evidence_support_label": support_label(str(row.get("support_tier", ""))),
                "official_or_common_method_species": bool_value(row.get("official_or_common_method_species", False)),
                "testing_priority": optional_float(row.get("species_testing_priority")),
                "rank_scope": "frozen_top20" if rank is not None and rank <= 20 else "extended_greedy_order",
                "note": support_note(str(row.get("support_tier", ""))),
            }
        )
    return records


def build_regional(
    state_sequences: pd.DataFrame,
    comparisons: pd.DataFrame,
    state_weight_counts: dict[str, int],
    national_top5: list[str],
) -> list[dict[str, Any]]:
    comparison_by_state = comparisons.set_index("state_code")
    sequence_groups = {
        state: group.sort_values("rank")
        for state, group in state_sequences.groupby("state_code", sort=True)
    }
    records: list[dict[str, Any]] = []
    for state_code, count in sorted(state_weight_counts.items()):
        base = {
            "state_code": state_code,
            "state_name": STATE_NAMES.get(state_code, state_code),
            "target": "lower-5%",
            "eligible_priority_chemical_count": int(count),
            "national_top5": national_top5,
        }
        if state_code not in comparison_by_state.index:
            records.append(
                {
                    **base,
                    "status": "insufficient_support",
                    "reason": "Fewer than three eligible priority chemicals; localized optimization was not run.",
                    "localized_top5": [],
                    "localized_sequence": [],
                    "overlap_species": [],
                    "overlap_count": 0,
                    "localized_expected_capture": None,
                    "national_expected_capture": None,
                    "absolute_gain_percentage_points": None,
                }
            )
            continue
        comparison = comparison_by_state.loc[state_code]
        sequence = sequence_groups[state_code]
        localized = [
            {
                "rank": int(row.rank),
                "scientific_name": str(row.latin_name),
                "candidate_status": str(row.candidate_status),
                "direct_support_chemical_count": int(row.marginal_direct_support_chemicals),
                "mean_tail_probability": float(row.mean_tail_probability_state),
                "mean_reliability": float(row.mean_reliability_state),
                "species_relevance_weight": float(row.soft_local_species_weight),
            }
            for row in sequence.itertuples(index=False)
        ]
        localized_top5 = [row["scientific_name"] for row in localized[:5]]
        overlap = [name for name in localized_top5 if name in national_top5]
        records.append(
            {
                **base,
                "status": "supported",
                "reason": "",
                "localized_top5": localized_top5,
                "localized_sequence": localized,
                "overlap_species": overlap,
                "overlap_count": len(overlap),
                "localized_expected_capture": float(comparison["state_soft_local_expected_weighted_joint_coverage_x95"]),
                "national_expected_capture": float(comparison["national_top5_soft_local_expected_weighted_joint_coverage_x95"]),
                "absolute_gain_percentage_points": float(comparison["state_gain_over_national_top5_percent_points"]),
                "weighting_summary": "State-priority chemical weights multiplied by soft species relevance and method-support weights.",
            }
        )
    return records


def write_downloads(
    out: Path,
    national: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    regional: list[dict[str, Any]],
) -> None:
    downloads = out / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    national_rows = []
    for row in national:
        flat = {key: value for key, value in row.items() if key != "cumulative_expected_capture"}
        for label, value in row["cumulative_expected_capture"].items():
            flat[f"cumulative_expected_capture_{label}"] = value
        national_rows.append(flat)
    pd.DataFrame(national_rows).to_csv(downloads / "national_sequence.csv", index=False, lineterminator="\n")
    pd.DataFrame(coverage).to_csv(downloads / "panel_size_coverage.csv", index=False, lineterminator="\n")
    pd.DataFrame(catalog).to_csv(downloads / "species_catalog.csv", index=False, lineterminator="\n")
    regional_summary = []
    regional_sequence_rows = []
    for row in regional:
        regional_summary.append({key: value for key, value in row.items() if key != "localized_sequence"})
        for member in row["localized_sequence"]:
            regional_sequence_rows.append({"state_code": row["state_code"], **member})
    pd.DataFrame(regional_summary).to_csv(downloads / "regional_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(regional_sequence_rows).to_csv(downloads / "regional_sequences.csv", index=False, lineterminator="\n")


def main() -> int:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    package_root = args.package_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    package = tomllib.loads((package_root / "COMPASS_PACKAGE.toml").read_text(encoding="utf-8"))
    sources = require_sources(analysis_root)

    sequence_all = pd.read_csv(analysis_root / SOURCE_FILES[0])
    frozen_sequence = selected_sequence(sequence_all)
    curves = pd.read_csv(analysis_root / SOURCE_FILES[1])
    candidate = pd.read_csv(analysis_root / SOURCE_FILES[3])
    warning = pd.read_csv(analysis_root / SOURCE_FILES[4])
    contributions = pd.read_csv(analysis_root / SOURCE_FILES[5])
    model = pd.read_csv(
        analysis_root / SOURCE_FILES[6],
        usecols=["latin_name", "n_broad_life_stages"],
        low_memory=False,
    )
    life = (
        model.groupby("latin_name", as_index=False)["n_broad_life_stages"]
        .max()
        .rename(columns={"n_broad_life_stages": "broad_life_stage_category_count"})
    )
    state_sequences = pd.read_csv(analysis_root / SOURCE_FILES[7])
    comparisons = pd.read_csv(analysis_root / SOURCE_FILES[8])
    state_weights = pd.read_csv(analysis_root / SOURCE_FILES[9], usecols=["state_code"])
    state_weight_counts = state_weights.groupby("state_code").size().astype(int).to_dict()
    testing = pd.read_csv(analysis_root / SOURCE_FILES[10])

    sequence = build_complete_national_order(analysis_root, frozen_sequence, candidate)
    national = build_national(sequence, candidate, warning, contributions, life, curves)
    catalog = build_catalog(candidate, warning, life, sequence, testing)
    coverage = [
        {
            "sequence_basis_target": float(row.sequence_basis_target),
            "evaluation_target": float(row.evaluation_target),
            "measured_tail": str(row.measured_tail),
            "panel_size": int(row.panel_size),
            "added_species": national[int(row.panel_size) - 1]["scientific_name"],
            "expected_capture": float(row.expected_capture),
            "unit": "probability",
        }
        for row in curves.sort_values(["evaluation_target", "panel_size"], ascending=[False, True]).itertuples(index=False)
    ]
    national_top5 = [row["scientific_name"] for row in national[:5]]
    regional = build_regional(state_sequences, comparisons, state_weight_counts, national_top5)

    write_json(out / "national_sequence.json", national)
    write_json(out / "coverage.json", coverage)
    write_json(out / "species_catalog.json", catalog)
    write_json(out / "regional.json", regional)
    write_downloads(out, national, coverage, catalog, regional)

    source_digest = hashlib.sha256(
        "\n".join(f"{row['path']}:{row['sha256']}" for row in sources).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "site_version": package["site_version"],
        "analysis_version": package["analysis_version"],
        "manuscript_version": package["manuscript_source_version"],
        "supporting_information_version": package["supporting_information_source_version"],
        "data_freeze_date": package["data_freeze_date"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis_identifier": source_digest,
        "git_commit": None,
        "github_release": None,
        "zenodo_doi": None,
        "manuscript_title": MANUSCRIPT_TITLE,
        "authors": AUTHORS,
        "units": {
            "expected_capture_storage": "probability (0-1)",
            "expected_capture_display": "percent",
            "absolute_gain": "percentage points",
        },
        "definitions": {
            "sequence_basis": "The national sequence was constructed at the lower-5% target. Its frozen Top-20 is extended through all 2,144 candidates with the same deterministic greedy complementarity objective for catalog exploration.",
            "cross_target_evaluation": "Broader targets evaluate prefixes of the same fixed sequence without reoptimization.",
            "regional_target": "Regional panels were optimized at the lower-5% target.",
            "support_tiers": "Protective evidence volume is the number of distinct valid protective analysis contexts: very limited 1–4, limited 5–19, moderate 20–49, and strong 50 or more. It describes evidence volume, not sensitivity.",
            "extended_rank_scope": "Dependence-adjusted expected-capture curves are frozen for ranks 1–20. Ranks 21–2,144 continue the selection objective; very small late-stage gains require follow-up evidence for practical discrimination.",
        },
        "limitations": (
            "Precomputed research outputs only; the site does not execute the full model, "
            "generate regulatory SSDs, or provide real-time risk assessment."
        ),
        "source_files": sources,
    }
    write_json(out / "metadata.json", metadata)

    files = sorted(path for path in out.rglob("*") if path.is_file())
    manifest = [
        {
            "path": path.relative_to(out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    write_json(out / "downloads_manifest.json", manifest)
    print(
        json.dumps(
            {
                "analysis_version": package["analysis_version"],
                "national_sequence_rows": len(national),
                "species_catalog_rows": len(catalog),
                "coverage_rows": len(coverage),
                "regional_rows": len(regional),
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
