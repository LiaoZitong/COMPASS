#!/usr/bin/env python3
"""Export the frozen R1 strict-LOO results for the COMPASS Results Explorer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


MANUSCRIPT_TITLE = (
    "Developing Complementary Aquatic Sentinel Panels for Cross-Chemical "
    "Monitoring and Regional Localization"
)
AUTHORS = [
    "Zitong Liao", "Zhuo Chen", "Yun Lu", "Dongbin Wei", "Yin-Hu Wu",
    "Yarong Qi", "Yuming Wang", "Xuan Zhou", "Ren Ding", "Hong-Ying Hu",
]
TARGETS = ((0.95, "lower-5%"), (0.90, "lower-10%"), (0.80, "lower-20%"))


def load_base_exporter(package_root: Path):
    path = package_root / "code/export_site_data.py"
    spec = importlib.util.spec_from_file_location("compass_base_site_export", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--boundary-zip",
        type=Path,
        help="U.S. Census state boundary archive; defaults to data/external below --analysis-root.",
    )
    parser.add_argument("--git-commit")
    parser.add_argument("--github-release")
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


def target_suffix(target: float) -> str:
    return f"{int(round(target * 100)):02d}"


def load_probability(revision_root: Path, target: float) -> dict[str, np.ndarray]:
    path = (
        revision_root
        / "02_ap02_strict_loo"
        / f"ap02_strict_loo_species_chemical_tail_probability_x{target_suffix(target)}.npz"
    )
    with np.load(path, allow_pickle=True) as payload:
        return {name: payload[name] for name in payload.files}


def national_weights(analysis_root: Path, chemicals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = pd.read_csv(analysis_root / "results/weights/national_priority_chemicals.csv")
    index = {name: position for position, name in enumerate(chemicals.astype(str))}
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    weights = weights[
        weights["DTXSID"].isin(index) & weights["national_weight"].astype(float).gt(0)
    ].copy()
    if len(weights) != 362:
        raise ValueError(f"Expected 362 national priority chemicals, found {len(weights)}")
    values = weights["national_weight"].to_numpy(float)
    values /= values.sum()
    columns = np.asarray([index[name] for name in weights["DTXSID"]], dtype=int)
    return columns, values


def frozen_top20(revision_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = revision_root / "08_r1_exhibits/main_figure_02/figure_02a_top_k_curves.csv"
    curves = pd.read_csv(path)
    sequence = (
        curves[curves["protection_target_x"].eq(0.95)][["rank", "latin_name"]]
        .sort_values("rank")
        .drop_duplicates("rank")
        .reset_index(drop=True)
    )
    if len(sequence) != 20 or sequence["rank"].tolist() != list(range(1, 21)):
        raise ValueError("R1 Figure 2 must contain the complete frozen x=0.95 Top-20")
    return sequence, curves


def build_complete_order(
    analysis_root: Path,
    revision_root: Path,
    frozen: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    probability = load_probability(revision_root, 0.95)
    matrix = probability["p"].astype(float)
    species = probability["species"].astype(str)
    chemicals = probability["chemicals"].astype(str)
    if set(candidate["latin_name"].astype(str)) != set(species):
        raise ValueError("Candidate universe and R1 strict-LOO probability species differ")
    columns, weights = national_weights(analysis_root, chemicals)
    matrix = np.ascontiguousarray(matrix[:, columns])
    no_event = np.ones(matrix.shape[1], dtype=float)
    selected = np.zeros(matrix.shape[0], dtype=bool)
    rows: list[dict[str, Any]] = []
    for rank in range(1, matrix.shape[0] + 1):
        gains = matrix @ (weights * no_event)
        gains[selected] = -np.inf
        index = int(np.argmax(gains))
        selected[index] = True
        incremental = float(gains[index])
        no_event *= 1.0 - matrix[index]
        rows.append(
            {
                "rank": rank,
                "latin_name": str(species[index]),
                "selection_incremental_gain_lower5": incremental,
                "selection_objective_cumulative_lower5": float(weights @ (1.0 - no_event)),
                "rank_scope": "frozen_r1_top20" if rank <= 20 else "extended_r1_greedy_order",
            }
        )
    complete = pd.DataFrame(rows)
    if complete.head(20)["latin_name"].tolist() != frozen["latin_name"].tolist():
        raise ValueError("Extended R1 sequence does not reproduce the frozen Top-20")
    if len(complete) != 2144 or not complete["latin_name"].is_unique:
        raise ValueError("Complete R1 sequence must contain 2,144 unique candidates")
    return complete


def selected_rhos(revision_root: Path) -> dict[float, float]:
    path = revision_root / "03_ap05_dependence/ap05_rho_estimation_audit.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["selected_for_downstream"].astype(bool)].copy()
    result = {
        float(row.protection_target_x): float(row.rho_working)
        for row in frame.itertuples(index=False)
    }
    if set(result) != {0.8, 0.9, 0.95}:
        raise ValueError("Expected one selected AP05 dependence value for each target")
    return result


def build_full_capture(
    analysis_root: Path,
    revision_root: Path,
    sequence: pd.DataFrame,
    rhos: dict[float, float],
) -> dict[str, np.ndarray]:
    ordered = sequence.sort_values("rank")["latin_name"].astype(str).tolist()
    # The R1 figure/source-data contract uses 12-point Gauss-Hermite quadrature.
    nodes, weights_h = hermgauss(12)
    nodes = np.sqrt(2.0) * nodes
    weights_h = weights_h / np.sqrt(np.pi)
    curves: dict[str, np.ndarray] = {}
    for target, label in TARGETS:
        probability = load_probability(revision_root, target)
        matrix = probability["p"].astype(float)
        species = probability["species"].astype(str)
        chemicals = probability["chemicals"].astype(str)
        species_index = {name: position for position, name in enumerate(species)}
        columns, chemical_weights = national_weights(analysis_root, chemicals)
        rows = np.asarray([species_index[name] for name in ordered], dtype=int)
        ordered_matrix = np.clip(matrix[np.ix_(rows, columns)], 1e-8, 1 - 1e-8)
        thresholds = norm.ppf(1.0 - ordered_matrix)
        rho = float(rhos[target])
        shared_sd = math.sqrt(max(rho, 0.0))
        residual_sd = math.sqrt(max(1.0 - rho, 1e-8))
        no_event = np.ones((len(nodes), len(columns)), dtype=float)
        captures = np.empty(len(ordered), dtype=float)
        for index, threshold in enumerate(thresholds):
            no_event *= norm.cdf(
                (threshold[None, :] - shared_sd * nodes[:, None]) / residual_sd
            )
            marginal_no_event = weights_h @ no_event
            captures[index] = float(chemical_weights @ np.clip(1.0 - marginal_no_event, 0.0, 1.0))
        if np.any(np.diff(captures) < -1e-12):
            raise ValueError(f"R1 cumulative capture is not monotone at x={target:g}")
        curves[label] = captures
    return curves


def validate_top20(full_capture: dict[str, np.ndarray], frozen_curves: pd.DataFrame) -> None:
    labels = {0.95: "lower-5%", 0.90: "lower-10%", 0.80: "lower-20%"}
    differences = []
    for row in frozen_curves.itertuples(index=False):
        observed = full_capture[labels[float(row.protection_target_x)]][int(row.rank) - 1]
        differences.append(abs(float(observed) - float(row.expected_weighted_joint_coverage)))
    if len(differences) != 60 or max(differences, default=float("inf")) > 1e-9:
        raise ValueError(
            "Full R1 capture curves do not reproduce all 60 frozen Figure 2 prefixes: "
            f"rows={len(differences)}, max_abs_difference={max(differences):.3e}"
        )


def r1_contributions(revision_root: Path) -> pd.DataFrame:
    path = revision_root / "08_r1_exhibits/main_figure_02/figure_02c_gain_loss.csv"
    frame = pd.read_csv(path)
    frame["method"] = "data_driven"
    frame["k"] = 5
    return frame


def regional_inputs(
    analysis_root: Path,
    revision_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ledger = pd.read_csv(
        revision_root / "06_ap04_localization_sensitivity/ap04_state_support_and_fallback_ledger.csv"
    )
    comparisons = ledger[ledger["localized"].astype(bool)].copy()
    comparisons = comparisons.rename(
        columns={
            "baseline_expected_weighted_joint_coverage_x95": "state_soft_local_expected_weighted_joint_coverage_x95",
            "national_top5_expected_weighted_joint_coverage_x95": "national_top5_soft_local_expected_weighted_joint_coverage_x95",
            "localization_gain_over_national_top5_pp": "state_gain_over_national_top5_percent_points",
        }
    )
    selected = pd.read_csv(
        revision_root / "08_r1_exhibits/main_figure_05/figure_05c_localized_top5_sequences.csv"
    )
    soft = pd.read_csv(
        revision_root
        / "06_ap04_localization_sensitivity/ap04_reconstructed_soft_local_species_weights.csv.gz"
    )
    state_weights = pd.read_csv(
        analysis_root / "results/weights/state_priority_chemical_weights.csv.gz",
        usecols=["state_code", "DTXSID", "state_weight"],
    )
    state_counts = state_weights.groupby("state_code").size().astype(int).to_dict()
    probability = load_probability(revision_root, 0.95)
    p = probability["p"].astype(float)
    reliability = probability["reliability"].astype(float)
    direct_n = probability["direct_n"].astype(float)
    species_index = {name: index for index, name in enumerate(probability["species"].astype(str))}
    chemical_index = {name: index for index, name in enumerate(probability["chemicals"].astype(str))}
    soft_index = soft.set_index(["state_code", "latin_name"])
    rows = []
    for record in selected.itertuples(index=False):
        state = str(record.state_code)
        name = str(record.latin_name)
        weights = state_weights[
            state_weights["state_code"].eq(state) & state_weights["DTXSID"].astype(str).isin(chemical_index)
        ].copy()
        weight_values = weights["state_weight"].to_numpy(float)
        weight_values /= weight_values.sum()
        columns = np.asarray([chemical_index[name_] for name_ in weights["DTXSID"].astype(str)], dtype=int)
        species_row = species_index[name]
        evidence = soft_index.loc[(state, name)]
        rows.append(
            {
                "state_code": state,
                "rank": int(record.rank),
                "latin_name": name,
                "candidate_status": str(evidence["soft_local_evidence_class"]),
                "marginal_direct_support_chemicals": int(np.sum(direct_n[species_row, columns] > 0)),
                "mean_tail_probability_state": float(weight_values @ p[species_row, columns]),
                "mean_reliability_state": float(weight_values @ reliability[species_row, columns]),
                "soft_local_species_weight": float(evidence["soft_local_species_weight"]),
            }
        )
    sequences = pd.DataFrame(rows).sort_values(["state_code", "rank"])
    if len(sequences) != 160:
        raise ValueError(f"Expected 160 localized Top-5 rows, found {len(sequences)}")
    return sequences, comparisons, state_counts


def source_records(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    records = []
    for label, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": label, "sha256": sha256(path), "bytes": path.stat().st_size})
    return records


def main() -> int:
    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    revision_root = args.revision_root.resolve()
    package_root = args.package_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_exporter(package_root)
    package = tomllib.loads((package_root / "COMPASS_PACKAGE.toml").read_text(encoding="utf-8"))

    candidate_path = analysis_root / "results/probability/candidate_universe_locked.csv"
    warning_path = analysis_root / "results/probability/rapid_warning_species_evidence_observed_only.csv"
    model_path = analysis_root / "results/toxicity/censored_model_cells.csv.gz"
    boundary_path = (
        args.boundary_zip.resolve()
        if args.boundary_zip
        else analysis_root / "data/external/cb_2024_us_state_500k.zip"
    )
    candidate = pd.read_csv(candidate_path)
    warning = pd.read_csv(warning_path)
    model = pd.read_csv(model_path, usecols=["latin_name", "n_broad_life_stages"], low_memory=False)
    life = (
        model.groupby("latin_name", as_index=False)["n_broad_life_stages"]
        .max()
        .rename(columns={"n_broad_life_stages": "broad_life_stage_category_count"})
    )
    testing = pd.read_csv(
        revision_root / "08_r1_exhibits/main_figure_05/figure_05d_strict_loo_species_priorities.csv"
    )
    frozen_sequence, frozen_curves = frozen_top20(revision_root)
    sequence = build_complete_order(analysis_root, revision_root, frozen_sequence, candidate)
    rhos = selected_rhos(revision_root)
    full_capture = build_full_capture(analysis_root, revision_root, sequence, rhos)
    validate_top20(full_capture, frozen_curves)
    contributions = r1_contributions(revision_root)
    national = base.build_national(sequence, candidate, warning, contributions, life, full_capture)
    catalog = base.build_catalog(candidate, warning, life, sequence, testing)
    coverage = base.build_coverage(sequence, full_capture)
    state_sequences, comparisons, state_counts = regional_inputs(analysis_root, revision_root)
    national_top5 = [row["scientific_name"] for row in national[:5]]
    regional = base.build_regional(state_sequences, comparisons, state_counts, national_top5)
    state_map = base.build_state_map(boundary_path)

    write_json(out / "national_sequence.json", national)
    write_json(out / "coverage.json", coverage)
    write_json(out / "species_catalog.json", catalog)
    write_json(out / "regional.json", regional)
    write_json(out / "state_map.json", state_map)
    base.write_downloads(out, national, coverage, catalog, regional)

    sources = source_records(
        [
            ("base/results/probability/candidate_universe_locked.csv", candidate_path),
            ("base/results/probability/rapid_warning_species_evidence_observed_only.csv", warning_path),
            ("base/results/toxicity/censored_model_cells.csv.gz", model_path),
            ("base/results/weights/national_priority_chemicals.csv", analysis_root / "results/weights/national_priority_chemicals.csv"),
            ("base/results/weights/state_priority_chemical_weights.csv.gz", analysis_root / "results/weights/state_priority_chemical_weights.csv.gz"),
            ("base/data/external/cb_2024_us_state_500k.zip", boundary_path),
            ("r1/figure_02a_top_k_curves.csv", revision_root / "08_r1_exhibits/main_figure_02/figure_02a_top_k_curves.csv"),
            ("r1/figure_02c_gain_loss.csv", revision_root / "08_r1_exhibits/main_figure_02/figure_02c_gain_loss.csv"),
            ("r1/ap05_rho_estimation_audit.csv", revision_root / "03_ap05_dependence/ap05_rho_estimation_audit.csv"),
            ("r1/ap04_state_support_and_fallback_ledger.csv", revision_root / "06_ap04_localization_sensitivity/ap04_state_support_and_fallback_ledger.csv"),
            ("r1/figure_05c_localized_top5_sequences.csv", revision_root / "08_r1_exhibits/main_figure_05/figure_05c_localized_top5_sequences.csv"),
            ("r1/figure_05d_strict_loo_species_priorities.csv", revision_root / "08_r1_exhibits/main_figure_05/figure_05d_strict_loo_species_priorities.csv"),
            *[
                (
                    f"r1/ap02_strict_loo_species_chemical_tail_probability_x{target_suffix(target)}.npz",
                    revision_root / "02_ap02_strict_loo" / f"ap02_strict_loo_species_chemical_tail_probability_x{target_suffix(target)}.npz",
                )
                for target, _label in TARGETS
            ],
        ]
    )
    source_digest = hashlib.sha256(
        "\n".join(f"{row['path']}:{row['sha256']}" for row in sources).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "site_version": package["site_version"],
        "analysis_version": package["analysis_version"],
        "base_analysis_version": package["base_analysis_version"],
        "manuscript_version": "R1 (2026-09-18)",
        "supporting_information_version": "R1 (2026-09-18)",
        "data_freeze_date": package["data_freeze_date"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis_identifier": source_digest,
        "git_commit": args.git_commit,
        "github_release": args.github_release,
        "zenodo_doi": None,
        "manuscript_title": MANUSCRIPT_TITLE,
        "authors": AUTHORS,
        "units": {
            "expected_capture_storage": "probability (0-1)",
            "expected_capture_display": "percent",
            "absolute_gain": "percentage points",
        },
        "definitions": {
            "sequence_basis": "The R1 national sequence was constructed from strict focal-species leave-one-out probabilities at the lower-5% target. Its frozen Top-20 is extended through all 2,144 candidates with the same deterministic greedy complementarity objective.",
            "cross_target_evaluation": "Every prefix of the fixed x=0.95 sequence is evaluated at the lower-5%, lower-10%, and lower-20% targets without reoptimizing membership, using the R1 strict-LOO matrices and target-specific AP05 dependence values.",
            "regional_target": "R1 regional panels were optimized at the lower-5% target for 32 supported states; PA, TN, and WV retain the national fallback because each has one eligible priority chemical.",
            "support_tiers": "Protective evidence volume is the number of distinct valid protective analysis contexts: very limited 1–4, limited 5–19, moderate 20–49, and strong 50 or more. It describes evidence volume, not sensitivity.",
            "extended_rank_scope": "Ranks 1–20 reproduce the frozen R1 Figure 2 analysis. Ranks 21–2,144 are a deterministic post hoc continuation for exploration under the same R1 objective and should not be read as equivalently validated biological rankings.",
            "state_map": "Interactive state boundaries derive from the U.S. Census Bureau 2024 1:500,000 cartographic file used in Figure 5. States without sufficient frozen priority-chemical support display the fixed national Top-5 as the documented fallback.",
        },
        "limitations": (
            "Precomputed R1 research outputs only; the site does not execute the full model, "
            "generate regulatory SSDs, or provide real-time risk assessment."
        ),
        "source_files": sources,
    }
    write_json(out / "metadata.json", metadata)
    manifest = [
        {
            "path": path.relative_to(out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    write_json(out / "downloads_manifest.json", manifest)
    print(
        json.dumps(
            {
                "analysis_version": package["analysis_version"],
                "national_top5": national_top5,
                "national_sequence_rows": len(national),
                "coverage_rows": len(coverage),
                "species_catalog_rows": len(catalog),
                "regional_rows": len(regional),
                "state_map_rows": len(state_map["states"]),
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
