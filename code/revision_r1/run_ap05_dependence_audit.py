#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = (0.80, 0.90, 0.95)
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) / ordered_weights.sum()
    return float(ordered_values[np.searchsorted(cumulative, 0.5)])


def estimate_working_rho_with_pairs(
    rows: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    tail_column: str,
    variant: str,
    target: float,
) -> tuple[dict[str, float | int | bool], pd.DataFrame, pd.DataFrame]:
    top = (
        candidates.nlargest(min(40, len(candidates)), "n_contexts")
        .loc[:, ["latin_name", "n_contexts"]]
        .reset_index(drop=True)
    )
    top["support_rank"] = np.arange(1, len(top) + 1)
    support = top.set_index("latin_name")["n_contexts"].to_dict()
    support_rank = top.set_index("latin_name")["support_rank"].to_dict()
    pivot = rows[rows["latin_name"].isin(top["latin_name"])].pivot_table(
        index="context_id",
        columns="latin_name",
        values=tail_column,
        aggfunc="mean",
    )
    pair_rows: list[dict[str, object]] = []
    columns = pivot.columns.astype(str).tolist()
    for i, species_a in enumerate(columns):
        for species_b in columns[i + 1 :]:
            a = pivot[species_a]
            b = pivot[species_b]
            mask = a.notna() & b.notna()
            n_shared = int(mask.sum())
            if n_shared < 20:
                continue
            aa = a[mask].to_numpy(float)
            bb = b[mask].to_numpy(float)
            if np.std(aa) == 0 or np.std(bb) == 0:
                continue
            correlation = float(np.corrcoef(aa, bb)[0, 1])
            if not np.isfinite(correlation):
                continue
            pair_rows.append(
                {
                    "protection_target_x": target,
                    "variant": variant,
                    "species_a": species_a,
                    "species_b": species_b,
                    "species_a_support_rank": int(support_rank[species_a]),
                    "species_b_support_rank": int(support_rank[species_b]),
                    "species_a_n_contexts_r0": int(support[species_a]),
                    "species_b_n_contexts_r0": int(support[species_b]),
                    "n_shared_contexts": n_shared,
                    "pearson_tail_event_correlation": correlation,
                    "positive_correlation": bool(correlation > 0),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        summary: dict[str, float | int | bool] = {
            "rho_working": 0.15,
            "rho_median_all": 0.15,
            "rho_median_positive": 0.15,
            "rho_q25_all": np.nan,
            "rho_q75_all": np.nan,
            "n_pairs": 0,
            "n_positive_pairs": 0,
            "fallback_used": True,
        }
    else:
        values = pairs["pearson_tail_event_correlation"].to_numpy(float)
        weights = pairs["n_shared_contexts"].to_numpy(float)
        positive = values > 0
        median_all = weighted_median(values, weights)
        median_positive = weighted_median(values[positive], weights[positive]) if positive.any() else 0.0
        summary = {
            "rho_working": float(np.clip(median_positive, 0.0, 0.5)),
            "rho_median_all": float(median_all),
            "rho_median_positive": float(median_positive),
            "rho_q25_all": float(np.quantile(values, 0.25)),
            "rho_q75_all": float(np.quantile(values, 0.75)),
            "n_pairs": int(len(values)),
            "n_positive_pairs": int(positive.sum()),
            "fallback_used": False,
        }
    top.insert(0, "protection_target_x", target)
    top.insert(1, "variant", variant)
    return summary, pairs, top


def close(left: object, right: object, tolerance: float = TOLERANCE) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhbt-root", required=True)
    parser.add_argument("--ap02-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.bhbt_root).resolve()
    ap02 = Path(args.ap02_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "candidate_universe": root / "results/probability/candidate_universe_locked.csv",
        "r0_panel_manifest": root / "results/panels/panel_probability_manifest.json",
        "ap02_gate": ap02 / "gate_ap02.json",
        "ap02_context_rows": ap02 / "ap02_context_focal_species_loo_rows.csv.gz",
        "ap02_rho_diagnostic": ap02 / "ap02_rho_diagnostic.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AP05 input(s):\n" + "\n".join(missing))

    ap02_gate = json.loads(paths["ap02_gate"].read_text(encoding="utf-8"))
    if ap02_gate.get("status") != "PASS":
        raise AssertionError("AP02 gate must pass before AP05")
    if (
        ap02_gate.get("x95_disposition", {}).get("selected_downstream_probability_source")
        != "AP02_STRICT_LOO"
    ):
        raise AssertionError("AP05 expected AP02 strict-LOO as the selected downstream source")

    candidates = (
        pd.read_csv(paths["candidate_universe"])
        .sort_values("latin_name")
        .reset_index(drop=True)
    )
    rows = pd.read_csv(paths["ap02_context_rows"], low_memory=False)
    manifest = json.loads(paths["r0_panel_manifest"].read_text(encoding="utf-8"))
    ap02_rho = pd.read_csv(paths["ap02_rho_diagnostic"])

    input_manifest = {
        "schema_version": 1,
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
    (out / "ap05_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    audit_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    all_checks: list[bool] = []
    selected_rho: dict[str, float] = {}

    for target in TARGETS:
        label = int(target * 100)
        original_tail = f"tail_binary_original_x{label}"
        strict_tail = f"tail_binary_strict_loo_x{label}"
        rows[original_tail] = (
            rows["mu_log10_umol_L"] <= rows[f"original_cutoff_log10_x{label}"]
        ).astype(float)
        rows[strict_tail] = (
            rows["mu_log10_umol_L"] <= rows[f"loo_cutoff_log10_x{label}"]
        ).astype(float)

        original_summary, original_pairs, original_top = estimate_working_rho_with_pairs(
            rows,
            candidates,
            tail_column=original_tail,
            variant="R0_RECOMPUTED",
            target=target,
        )
        strict_rows = rows[rows["strict_loo_eligible"].astype(bool)].copy()
        strict_summary, strict_pairs, strict_top = estimate_working_rho_with_pairs(
            strict_rows,
            candidates,
            tail_column=strict_tail,
            variant="AP02_STRICT_LOO",
            target=target,
        )
        pair_frames.extend([original_pairs, strict_pairs])
        top_frames.extend([original_top, strict_top])

        manifest_row = manifest["dependence_audit_by_x"][str(target)]
        ap02_row = ap02_rho.loc[
            np.isclose(ap02_rho["protection_target_x"].astype(float), target)
        ].iloc[0]
        original_checks = {
            "rho_working": close(original_summary["rho_working"], manifest_row["rho_working"]),
            "rho_median_all": close(original_summary["rho_median_all"], manifest_row["rho_median_all"]),
            "rho_median_positive": close(
                original_summary["rho_median_positive"], manifest_row["rho_median_positive"]
            ),
            "rho_q25_all": close(original_summary["rho_q25_all"], manifest_row["rho_q25_all"]),
            "rho_q75_all": close(original_summary["rho_q75_all"], manifest_row["rho_q75_all"]),
            "n_pairs": int(original_summary["n_pairs"]) == int(manifest_row["n_pairs"]),
        }
        strict_checks = {
            "rho_working": close(
                strict_summary["rho_working"], ap02_row["strict_loo_rho_working"]
            ),
            "rho_median_all": close(
                strict_summary["rho_median_all"], ap02_row["strict_loo_rho_median_all"]
            ),
            "rho_median_positive": close(
                strict_summary["rho_median_positive"],
                ap02_row["strict_loo_rho_median_positive"],
            ),
            "n_pairs": int(strict_summary["n_pairs"]) == int(ap02_row["strict_loo_n_pairs"]),
        }
        original_pass = all(original_checks.values())
        strict_pass = all(strict_checks.values())
        all_checks.extend([original_pass, strict_pass])
        selected_rho[str(target)] = float(strict_summary["rho_working"])

        for variant, summary, comparator, checks, passed in [
            ("R0_RECOMPUTED", original_summary, "R0 panel manifest", original_checks, original_pass),
            ("AP02_STRICT_LOO", strict_summary, "AP02 rho diagnostic", strict_checks, strict_pass),
        ]:
            audit_rows.append(
                {
                    "protection_target_x": target,
                    "variant": variant,
                    **summary,
                    "comparison_source": comparator,
                    "comparison_checks_json": json.dumps(checks, sort_keys=True),
                    "comparison_pass": passed,
                    "selected_for_downstream": variant == "AP02_STRICT_LOO",
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "ap05_rho_estimation_audit.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(
        out / "ap05_pairwise_tail_event_correlations.csv.gz", index=False, compression="gzip"
    )
    pd.concat(top_frames, ignore_index=True).drop_duplicates().to_csv(
        out / "ap05_top40_species_support.csv", index=False
    )

    status = "PASS" if all(all_checks) else "FAIL"
    gate = {
        "schema_version": 1,
        "gate": "A3_AP05_DEPENDENCE_AUDIT",
        "status": status,
        "method": {
            "candidate_species_rule": "up to 40 candidates with highest R0 direct-context support",
            "pair_rule": "Pearson correlation of binary tail events for species pairs with at least 20 shared contexts",
            "summary_rule": "shared-context-count-weighted median of positive pairwise correlations",
            "copula_range": [0.0, 0.5],
            "fallback": 0.15,
            "fallback_condition": "no eligible species pair",
        },
        "r0_manifest_reproduction_pass": bool(
            audit.loc[audit["variant"].eq("R0_RECOMPUTED"), "comparison_pass"].all()
        ),
        "ap02_strict_loo_reproduction_pass": bool(
            audit.loc[audit["variant"].eq("AP02_STRICT_LOO"), "comparison_pass"].all()
        ),
        "fallback_used": bool(audit["fallback_used"].any()),
        "selected_downstream_probability_source": "AP02_STRICT_LOO",
        "selected_downstream_rho_by_x": selected_rho,
        "next_stage": "A4_AP03_MODULE_A" if status == "PASS" else None,
    }
    (out / "gate_ap05.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "# AP05 Gaussian-copula dependence audit",
        "",
        f"Gate: `{status}`.",
        "",
        "The working dependence parameter is data-derived when eligible species pairs exist. "
        "It is the shared-context-count-weighted median of positive Pearson correlations among "
        "binary tail-event patterns for up to 40 high-support species, with at least 20 shared "
        "contexts per pair, clipped to 0–0.5. The value 0.15 is reserved for the no-eligible-pair fallback.",
        "",
        "| x | variant | eligible pairs | positive pairs | median all | median positive | q25 | q75 | working rho | fallback |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit.itertuples(index=False):
        report_lines.append(
            f"| {row.protection_target_x:.2f} | {row.variant} | {int(row.n_pairs)} | "
            f"{int(row.n_positive_pairs)} | {row.rho_median_all:.6f} | "
            f"{row.rho_median_positive:.6f} | {row.rho_q25_all:.6f} | "
            f"{row.rho_q75_all:.6f} | {row.rho_working:.6f} | {row.fallback_used} |"
        )
    report_lines.extend(
        [
            "",
            "R0 values were independently recomputed and compared with the production panel manifest. "
            "The strict-LOO values were independently recomputed and compared with AP02. Because G2 "
            "selected strict LOO for the R1 primary analysis, the strict-LOO working rho values are "
            "the downstream dependence parameters.",
            "",
            "No manuscript or SI file was edited in this stage.",
        ]
    )
    (out / "AP05_DEPENDENCE_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
