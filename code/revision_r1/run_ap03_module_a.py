#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROTECTIVE = [
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
]
SEED = 20260622
N_BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def correlation_metrics(data: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[y_col], errors="coerce").to_numpy(float)
    return correlation_arrays(x, y)


def correlation_arrays(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    result: dict[str, float | int] = {"n": int(len(x)), "spearman_rho": np.nan, "kendall_tau": np.nan}
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        result["spearman_rho"] = float(spearmanr(x, y).statistic)
        result["kendall_tau"] = float(kendalltau(x, y).statistic)
    return result


def label_seed(label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((SEED + suffix) % (2**32 - 1))


def cluster_bootstrap_correlations(
    data: pd.DataFrame,
    x_columns: list[str],
    y_col: str,
    *,
    label: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    group_indices = [
        np.asarray(indices, dtype=int)
        for indices in data.groupby("dtxsid", sort=False).indices.values()
    ]
    if len(group_indices) < 3:
        return pd.DataFrame()
    arrays = {
        column: pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        for column in [*x_columns, y_col]
    }
    rng = np.random.default_rng(label_seed(label))
    rows: list[dict[str, object]] = []
    for draw in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        sample_indices = np.concatenate(
            [group_indices[int(index)] for index in sampled_groups]
        )
        record: dict[str, object] = {"analysis": label, "bootstrap": draw}
        for x_col in x_columns:
            metrics = correlation_arrays(
                arrays[x_col][sample_indices], arrays[y_col][sample_indices]
            )
            record[f"{x_col}__n"] = metrics["n"]
            record[f"{x_col}__spearman_rho"] = metrics["spearman_rho"]
            record[f"{x_col}__kendall_tau"] = metrics["kendall_tau"]
        if len(x_columns) == 2:
            first, second = x_columns
            record["delta_spearman_second_minus_first"] = (
                float(record[f"{second}__spearman_rho"])
                - float(record[f"{first}__spearman_rho"])
            )
            record["delta_kendall_second_minus_first"] = (
                float(record[f"{second}__kendall_tau"])
                - float(record[f"{first}__kendall_tau"])
            )
        rows.append(record)
    return pd.DataFrame(rows)


def interval(draws: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(draws, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 0
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), int(len(values))


def add_metric_intervals(
    record: dict[str, object], draws: pd.DataFrame, prefix: str
) -> dict[str, object]:
    for metric in ["spearman_rho", "kendall_tau"]:
        lower, upper, valid = interval(draws[f"{prefix}__{metric}"]) if not draws.empty else (np.nan, np.nan, 0)
        record[f"{metric}_ci95_lower"] = lower
        record[f"{metric}_ci95_upper"] = upper
        record[f"{metric}_bootstrap_valid"] = valid
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhbt-root", required=True)
    parser.add_argument("--ap02-dir", required=True)
    parser.add_argument("--ap05-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.bhbt_root).resolve()
    ap02 = Path(args.ap02_dir).resolve()
    ap05 = Path(args.ap05_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": root / "results/toxicity/censored_model_cells.csv.gz",
        "r0_paired_rows": root / "results/warning_hc5_bridge/top5_panel_apical_threshold_vs_reference_hc5.csv",
        "reference_hc5": root / "results/warning_hc5_bridge/chemical_medium_reference_hc5.csv",
        "shared_input_rows": root / "results/warning_hc5_bridge/hc5_excluding_top5_top5_apical_paired_rows.csv",
        "r0_bridge_manifest": root / "results/warning_hc5_bridge/warning_hc5_bridge_manifest.json",
        "ap02_gate": ap02 / "gate_ap02.json",
        "ap05_gate": ap05 / "gate_ap05.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AP03 module A input(s):\n" + "\n".join(missing))

    ap02_gate = json.loads(paths["ap02_gate"].read_text(encoding="utf-8"))
    ap05_gate = json.loads(paths["ap05_gate"].read_text(encoding="utf-8"))
    if ap02_gate.get("status") != "PASS" or ap05_gate.get("status") != "PASS":
        raise AssertionError("AP02 and AP05 gates must both pass before AP03 module A")
    top5 = list(ap02_gate["x95_disposition"]["selected_downstream_top5"])
    if len(top5) != 5 or len(set(top5)) != 5:
        raise AssertionError("The selected downstream Top-5 must contain five unique species")

    model = pd.read_csv(paths["model"], low_memory=False)
    valid = model["mle_status"].astype(str).str.startswith("identified") & np.isfinite(
        pd.to_numeric(model["mu_log10_umol_L"], errors="coerce")
    )
    protective = model[valid & model["effect_family"].isin(PROTECTIVE)].copy()
    protective["dtxsid"] = protective["dtxsid"].astype(str)
    reference = pd.read_csv(paths["reference_hc5"])
    reference["dtxsid"] = reference["dtxsid"].astype(str)

    member_min = (
        protective[protective["latin_name"].astype(str).isin(top5)]
        .groupby(["dtxsid", "medium_family", "latin_name"], as_index=False)
        .agg(
            measured_apical_min_log10=("mu_log10_umol_L", "min"),
            n_protective_model_cells=("mu_log10_umol_L", "size"),
            n_protective_effect_families=("effect_family", "nunique"),
        )
    )
    threshold_wide = member_min.pivot(
        index=["dtxsid", "medium_family"],
        columns="latin_name",
        values="measured_apical_min_log10",
    ).reindex(columns=top5)
    count_wide = member_min.pivot(
        index=["dtxsid", "medium_family"],
        columns="latin_name",
        values="n_protective_model_cells",
    ).reindex(columns=top5)

    rows: list[dict[str, object]] = []
    reference_index = reference.set_index(["dtxsid", "medium_family"])
    for key, values in threshold_wide.iterrows():
        if key not in reference_index.index:
            continue
        available = [species for species in top5 if pd.notna(values[species])]
        if not available:
            continue
        minimum = float(np.nanmin(values.to_numpy(float)))
        contributors = [
            species
            for species in available
            if np.isclose(float(values[species]), minimum, rtol=0.0, atol=1e-12)
        ]
        reference_row = reference_index.loc[key]
        record: dict[str, object] = {
            "dtxsid": key[0],
            "medium_family": key[1],
            "n_top5_species_measured": len(available),
            "available_top5_species": " | ".join(available),
            "available_member_minimum_log10": minimum,
            "minimum_contributors": " | ".join(contributors),
            "n_minimum_contributors": len(contributors),
        }
        for column in reference.columns:
            if column not in {"dtxsid", "medium_family"}:
                record[column] = reference_row[column]
        for species in top5:
            suffix = safe_name(species)
            record[f"measured__{suffix}"] = bool(pd.notna(values[species]))
            record[f"apical_min_log10__{suffix}"] = (
                float(values[species]) if pd.notna(values[species]) else np.nan
            )
            record[f"n_model_cells__{suffix}"] = (
                int(count_wide.loc[key, species]) if pd.notna(count_wide.loc[key, species]) else 0
            )
        rows.append(record)
    ledger = pd.DataFrame(rows).sort_values(["dtxsid", "medium_family"]).reset_index(drop=True)
    ledger.to_csv(out / "ap03a_top5_availability_and_contribution_ledger.csv", index=False)

    r0_rows = pd.read_csv(paths["r0_paired_rows"]).sort_values(["dtxsid", "medium_family"]).reset_index(drop=True)
    comparison = ledger.merge(
        r0_rows[
            [
                "dtxsid",
                "medium_family",
                "panel_min_apical_log10",
                "n_top5_species_with_apical",
            ]
        ],
        on=["dtxsid", "medium_family"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    r0_reproduction = {
        "row_count_expected_115": len(ledger) == 115,
        "row_keys_exact_match": bool(comparison["_merge"].eq("both").all()),
        "member_counts_exact_match": bool(
            comparison["n_top5_species_measured"]
            .eq(comparison["n_top5_species_with_apical"])
            .all()
        ),
        "panel_minimum_max_absolute_difference": float(
            np.nanmax(
                np.abs(
                    comparison["available_member_minimum_log10"].to_numpy(float)
                    - comparison["panel_min_apical_log10"].to_numpy(float)
                )
            )
        ),
    }
    r0_reproduction["pass"] = bool(
        r0_reproduction["row_count_expected_115"]
        and r0_reproduction["row_keys_exact_match"]
        and r0_reproduction["member_counts_exact_match"]
        and r0_reproduction["panel_minimum_max_absolute_difference"] <= 1e-12
    )

    bootstrap_frames: list[pd.DataFrame] = []
    contribution_rows: list[dict[str, object]] = []
    for species in top5:
        suffix = safe_name(species)
        measured = ledger[f"measured__{suffix}"].astype(bool)
        contributor_credit = ledger["minimum_contributors"].fillna("").map(
            lambda value: (
                1.0 / len(str(value).split(" | "))
                if species in str(value).split(" | ") and str(value)
                else 0.0
            )
        )
        rng = np.random.default_rng(label_seed(f"contributor::{species}"))
        draws = []
        for _ in range(N_BOOTSTRAP):
            index = rng.integers(0, len(ledger), size=len(ledger))
            draws.append(float(contributor_credit.iloc[index].mean()))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        contribution_rows.append(
            {
                "latin_name": species,
                "selected_top5_rank": top5.index(species) + 1,
                "n_rows_measured": int(measured.sum()),
                "fraction_rows_measured": float(measured.mean()),
                "n_rows_as_minimum_contributor": int(
                    ledger["minimum_contributors"].fillna("").map(
                        lambda value: species in str(value).split(" | ")
                    ).sum()
                ),
                "fractional_minimum_credit": float(contributor_credit.sum()),
                "fractional_contribution_fraction": float(contributor_credit.mean()),
                "fractional_contribution_fraction_ci95_lower": float(lower),
                "fractional_contribution_fraction_ci95_upper": float(upper),
                "bootstrap_draws": N_BOOTSTRAP,
            }
        )
    contribution = pd.DataFrame(contribution_rows)
    contribution.to_csv(out / "ap03a_member_contribution_summary.csv", index=False)

    count_rows: list[dict[str, object]] = []
    strata: list[tuple[str, pd.Series]] = [("all_eligible_rows", pd.Series(True, index=ledger.index))]
    strata.extend(
        (f"exactly_{count}_measured", ledger["n_top5_species_measured"].eq(count))
        for count in range(1, 6)
    )
    strata.extend(
        (f"at_least_{count}_measured", ledger["n_top5_species_measured"].ge(count))
        for count in range(2, 5)
    )
    for stratum, mask in strata:
        subset = ledger[mask].copy()
        for reference_col in [
            "reference_hc5_typical_log10",
            "reference_hc5_conservative_log10",
        ]:
            metrics = correlation_metrics(
                subset, "available_member_minimum_log10", reference_col
            )
            label = f"member_count::{stratum}::{reference_col}"
            draws = cluster_bootstrap_correlations(
                subset,
                ["available_member_minimum_log10"],
                reference_col,
                label=label,
            )
            if not draws.empty:
                bootstrap_frames.append(draws)
            record: dict[str, object] = {
                "stratum": stratum,
                "reference": reference_col,
                "n_rows": len(subset),
                "spearman_rho": metrics["spearman_rho"],
                "kendall_tau": metrics["kendall_tau"],
            }
            count_rows.append(add_metric_intervals(record, draws, "available_member_minimum_log10"))
    count_summary = pd.DataFrame(count_rows)
    count_summary.to_csv(out / "ap03a_member_count_concordance.csv", index=False)

    ablation_rows: list[dict[str, object]] = []
    focal_rows: list[dict[str, object]] = []
    for species in top5:
        suffix = safe_name(species)
        other_columns = [f"apical_min_log10__{safe_name(other)}" for other in top5 if other != species]
        ledger[f"minimum_without__{suffix}"] = ledger[other_columns].min(axis=1, skipna=True)
        comparable = ledger[ledger[f"minimum_without__{suffix}"].notna()].copy()
        changed = ~np.isclose(
            comparable["available_member_minimum_log10"].to_numpy(float),
            comparable[f"minimum_without__{suffix}"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        )
        for reference_col in [
            "reference_hc5_typical_log10",
            "reference_hc5_conservative_log10",
        ]:
            full_metrics = correlation_metrics(
                comparable, "available_member_minimum_log10", reference_col
            )
            ablated_metrics = correlation_metrics(
                comparable, f"minimum_without__{suffix}", reference_col
            )
            label = f"ablation::{species}::{reference_col}"
            draws = cluster_bootstrap_correlations(
                comparable,
                ["available_member_minimum_log10", f"minimum_without__{suffix}"],
                reference_col,
                label=label,
            )
            if not draws.empty:
                bootstrap_frames.append(draws)
            delta_s_low, delta_s_high, delta_s_valid = (
                interval(draws["delta_spearman_second_minus_first"])
                if not draws.empty
                else (np.nan, np.nan, 0)
            )
            delta_k_low, delta_k_high, delta_k_valid = (
                interval(draws["delta_kendall_second_minus_first"])
                if not draws.empty
                else (np.nan, np.nan, 0)
            )
            ablation_rows.append(
                {
                    "removed_member": species,
                    "reference": reference_col,
                    "n_matched_rows": len(comparable),
                    "n_rows_dropped_no_remaining_member": len(ledger) - len(comparable),
                    "n_rows_minimum_changed": int(changed.sum()),
                    "full_available_spearman": full_metrics["spearman_rho"],
                    "ablated_spearman": ablated_metrics["spearman_rho"],
                    "delta_spearman_ablated_minus_full": float(ablated_metrics["spearman_rho"])
                    - float(full_metrics["spearman_rho"]),
                    "delta_spearman_ci95_lower": delta_s_low,
                    "delta_spearman_ci95_upper": delta_s_high,
                    "delta_spearman_bootstrap_valid": delta_s_valid,
                    "full_available_kendall": full_metrics["kendall_tau"],
                    "ablated_kendall": ablated_metrics["kendall_tau"],
                    "delta_kendall_ablated_minus_full": float(ablated_metrics["kendall_tau"])
                    - float(full_metrics["kendall_tau"]),
                    "delta_kendall_ci95_lower": delta_k_low,
                    "delta_kendall_ci95_upper": delta_k_high,
                    "delta_kendall_bootstrap_valid": delta_k_valid,
                }
            )

        focal_subset = ledger[
            ledger[f"measured__{suffix}"].astype(bool)
            & ledger[f"minimum_without__{suffix}"].notna()
        ].copy()
        focal_col = f"apical_min_log10__{suffix}"
        for reference_col in [
            "reference_hc5_typical_log10",
            "reference_hc5_conservative_log10",
        ]:
            for estimator_label, estimator_col in [
                ("focal_member_only", focal_col),
                ("all_available_members", "available_member_minimum_log10"),
                ("other_available_members", f"minimum_without__{suffix}"),
            ]:
                metrics = correlation_metrics(focal_subset, estimator_col, reference_col)
                label = f"focal_matched::{species}::{reference_col}::{estimator_label}"
                draws = cluster_bootstrap_correlations(
                    focal_subset, [estimator_col], reference_col, label=label
                )
                if not draws.empty:
                    bootstrap_frames.append(draws)
                record = {
                    "focal_member": species,
                    "estimator": estimator_label,
                    "reference": reference_col,
                    "n_matched_rows": len(focal_subset),
                    "spearman_rho": metrics["spearman_rho"],
                    "kendall_tau": metrics["kendall_tau"],
                }
                focal_rows.append(add_metric_intervals(record, draws, estimator_col))

    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(out / "ap03a_member_ablation_matched.csv", index=False)
    focal = pd.DataFrame(focal_rows)
    focal.to_csv(out / "ap03a_focal_member_matched_coverage.csv", index=False)

    shared = pd.read_csv(paths["shared_input_rows"])
    shared["dtxsid"] = shared["dtxsid"].astype(str)
    shared_rows: list[dict[str, object]] = []
    shared_comparisons = [
        (
            "available-member minimum vs full-data typical HC5",
            "panel_min_apical_log10",
            "reference_hc5_typical_log10",
        ),
        (
            "available-member minimum vs HC5 excluding Top-5",
            "panel_min_apical_log10",
            "reference_hc5_non_w5_typical_log10",
        ),
        (
            "full-data typical HC5 vs HC5 excluding Top-5",
            "reference_hc5_typical_log10",
            "reference_hc5_non_w5_typical_log10",
        ),
    ]
    for name, x_col, y_col in shared_comparisons:
        metrics = correlation_metrics(shared, x_col, y_col)
        label = f"shared_input::{name}"
        draws = cluster_bootstrap_correlations(shared, [x_col], y_col, label=label)
        if not draws.empty:
            bootstrap_frames.append(draws)
        record = {
            "comparison": name,
            "n_rows": metrics["n"],
            "spearman_rho": metrics["spearman_rho"],
            "kendall_tau": metrics["kendall_tau"],
        }
        shared_rows.append(add_metric_intervals(record, draws, x_col))
    shared_summary = pd.DataFrame(shared_rows)
    shared_summary.to_csv(out / "ap03a_shared_input_sensitivity_with_intervals.csv", index=False)

    if bootstrap_frames:
        pd.concat(bootstrap_frames, ignore_index=True, sort=False).to_csv(
            out / "ap03a_bootstrap_draws.csv.gz", index=False, compression="gzip"
        )

    count_distribution = {
        str(count): int(ledger["n_top5_species_measured"].eq(count).sum())
        for count in range(1, 6)
    }
    daphnia_row = contribution.loc[contribution["latin_name"].eq("Daphnia magna")]
    daphnia_minimum_count = (
        int(daphnia_row.iloc[0]["n_rows_as_minimum_contributor"])
        if len(daphnia_row)
        else None
    )
    gate_checks = {
        "r0_pairing_reproduced": r0_reproduction["pass"],
        "eligible_rows_equal_115": len(ledger) == 115,
        "unique_chemicals_equal_115": ledger["dtxsid"].nunique() == 115,
        "all_rows_freshwater": set(ledger["medium_family"].astype(str)) == {"freshwater"},
        "measured_member_distribution_matches_review": count_distribution
        == {"1": 54, "2": 44, "3": 13, "4": 4, "5": 0},
        "daphnia_magna_minimum_count_matches_review": daphnia_minimum_count == 92,
        "no_missing_reference_values": bool(
            ledger[
                ["reference_hc5_typical_log10", "reference_hc5_conservative_log10"]
            ].notna().all().all()
        ),
        "ablation_completed_for_all_five_members": ablation["removed_member"].nunique() == 5,
        "bootstrap_draws_frozen": N_BOOTSTRAP == 2000,
    }
    status = "PASS" if all(gate_checks.values()) else "FAIL"
    gate = {
        "schema_version": 1,
        "gate": "A4_AP03_MODULE_A_AVAILABLE_MEMBER_VALIDATION",
        "status": status,
        "seed": SEED,
        "bootstrap_draws": N_BOOTSTRAP,
        "selected_top5": top5,
        "eligibility_boundary": "115 EPA-framework-eligible chemical-medium rows; 115 unique chemicals; freshwater only",
        "count_distribution": count_distribution,
        "daphnia_magna_minimum_contributor_rows": daphnia_minimum_count,
        "full_five_member_rows": int(count_distribution["5"]),
        "checks": gate_checks,
        "interpretation": (
            "Measured validation is an available-member minimum comparison on the 115-row "
            "eligible subset. It is distinct from model-predicted capture and is not a fully "
            "observed five-member panel validation."
        ),
        "next_stage": "A5_AP03_MODULE_B" if status == "PASS" else None,
    }
    (out / "gate_ap03a.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    input_manifest = {
        "schema_version": 1,
        "seed": SEED,
        "bootstrap_draws": N_BOOTSTRAP,
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
    (out / "ap03a_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    primary = count_summary[
        count_summary["stratum"].eq("all_eligible_rows")
        & count_summary["reference"].eq("reference_hc5_typical_log10")
    ].iloc[0]
    daphnia_ablation = ablation[
        ablation["removed_member"].eq("Daphnia magna")
        & ablation["reference"].eq("reference_hc5_typical_log10")
    ].iloc[0]
    report = [
        "# AP03 module A: measured available-member validation",
        "",
        f"Gate: `{status}`.",
        "",
        f"The selected Top-5 membership is: {' | '.join(top5)}.",
        "",
        "## Coverage boundary",
        "",
        f"The analysis contains {len(ledger)} EPA-framework-eligible chemical-medium rows "
        f"from {ledger['dtxsid'].nunique()} unique chemicals. Measured-member counts are "
        f"1: {count_distribution['1']}, 2: {count_distribution['2']}, "
        f"3: {count_distribution['3']}, 4: {count_distribution['4']}, and 5: {count_distribution['5']}.",
        f"Daphnia magna is a minimum contributor in {daphnia_minimum_count}/{len(ledger)} rows.",
        "",
        "## Primary available-member concordance",
        "",
        f"Against the typical full-data HC5, Spearman rho is {primary.spearman_rho:.6f} "
        f"(chemical-bootstrap 95% interval {primary.spearman_rho_ci95_lower:.6f} to "
        f"{primary.spearman_rho_ci95_upper:.6f}); Kendall tau is {primary.kendall_tau:.6f} "
        f"({primary.kendall_tau_ci95_lower:.6f} to {primary.kendall_tau_ci95_upper:.6f}).",
        "",
        "## Daphnia magna ablation on matched rows",
        "",
        f"After excluding Daphnia magna, {int(daphnia_ablation.n_matched_rows)} rows retain at "
        f"least one measured Top-5 member and {int(daphnia_ablation.n_rows_dropped_no_remaining_member)} "
        "rows are not scoreable. On the matched rows, the ablated-minus-full Spearman change is "
        f"{daphnia_ablation.delta_spearman_ablated_minus_full:.6f} "
        f"(95% interval {daphnia_ablation.delta_spearman_ci95_lower:.6f} to "
        f"{daphnia_ablation.delta_spearman_ci95_upper:.6f}).",
        "",
        "These measured-threshold results describe available-member concordance only. They do "
        "not evaluate complete five-species measurement coverage and do not replace the strict-LOO "
        "model-predicted expected-capture analysis.",
        "",
        "No manuscript or SI file was edited in this stage.",
    ]
    (out / "AP03A_AVAILABLE_MEMBER_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
