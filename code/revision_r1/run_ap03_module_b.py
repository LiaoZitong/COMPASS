#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


SEED = 20260622
N_BOOTSTRAP = 2000
MORTALITY = "mortality_survival"
FRACTIONS = (0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def label_seed(label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((SEED + suffix) % (2**32 - 1))


def panel_score(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    return 1.0 - np.prod(1.0 - probabilities, axis=0)


def load_target_scores(path: Path, top5: list[str], variant: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as archive:
        p = np.asarray(archive["p"], dtype=float)
        species = archive["species"].astype(str)
        target_ids = archive["target_ids"].astype(str)
        chemicals = archive["chemicals"].astype(str)
        effects = archive["effects"].astype(str)
    species_index = {name: index for index, name in enumerate(species)}
    missing = [name for name in top5 if name not in species_index]
    if missing:
        raise AssertionError(f"Top-5 species missing from probability matrix: {missing}")
    indices = [species_index[name] for name in top5]
    scores = panel_score(p[indices, :])
    table = pd.DataFrame(
        {
            "variant": variant,
            "dtxsid": chemicals,
            "target_id": target_ids,
            "effect_family": effects,
            "panel_target_score": scores,
        }
    )
    arrays = {
        "species": species,
        "target_ids": target_ids,
        "chemicals": chemicals,
        "effects": effects,
    }
    return table, arrays


def aggregate_rankings(target_scores: pd.DataFrame, prefix: str) -> pd.DataFrame:
    duplicated = target_scores.duplicated(["dtxsid", "effect_family"], keep=False)
    if duplicated.any():
        raise AssertionError("Chemical-effect targets are not unique")
    family = target_scores.copy()
    family_counts = family.groupby("dtxsid")["effect_family"].nunique()
    nonmortality_counts = (
        family[family["effect_family"].ne(MORTALITY)]
        .groupby("dtxsid")["effect_family"]
        .nunique()
    )

    ordered = family.sort_values(
        ["dtxsid", "panel_target_score", "target_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    primary = ordered.groupby("dtxsid", as_index=False).first()[
        ["dtxsid", "target_id", "effect_family", "panel_target_score"]
    ].rename(
        columns={
            "target_id": f"{prefix}_primary_max_target_id",
            "effect_family": f"{prefix}_primary_max_effect_family",
            "panel_target_score": f"{prefix}_primary_max_score",
        }
    )
    balanced = (
        family.groupby("dtxsid", as_index=False)["panel_target_score"]
        .mean()
        .rename(columns={"panel_target_score": f"{prefix}_balanced_score"})
    )
    mortality_excluded = (
        family[family["effect_family"].ne(MORTALITY)]
        .groupby("dtxsid", as_index=False)["panel_target_score"]
        .mean()
        .rename(columns={"panel_target_score": f"{prefix}_mortality_excluded_score"})
    )
    result = primary.merge(balanced, on="dtxsid", validate="one_to_one")
    result = result.merge(mortality_excluded, on="dtxsid", how="left", validate="one_to_one")
    result["n_supported_effect_families"] = result["dtxsid"].map(family_counts).astype(int)
    result["n_supported_nonmortality_effect_families"] = (
        result["dtxsid"].map(nonmortality_counts).fillna(0).astype(int)
    )
    result[f"{prefix}_mortality_excluded_status"] = np.where(
        result[f"{prefix}_mortality_excluded_score"].notna(),
        "SCOREABLE",
        "NOT SCOREABLE",
    )
    return result


def deterministic_rank(data: pd.DataFrame, score_col: str) -> pd.Series:
    scoreable = data[data[score_col].notna()].sort_values(
        [score_col, "dtxsid"], ascending=[False, True], kind="mergesort"
    )
    return pd.Series(
        np.arange(1, len(scoreable) + 1, dtype=int),
        index=scoreable["dtxsid"].astype(str),
    )


def ordered_ids(data: pd.DataFrame, score_col: str) -> list[str]:
    return (
        data[data[score_col].notna()]
        .sort_values([score_col, "dtxsid"], ascending=[False, True], kind="mergesort")["dtxsid"]
        .astype(str)
        .tolist()
    )


def selected_ids(data: pd.DataFrame, score_col: str, n_select: int) -> set[str]:
    return set(ordered_ids(data, score_col)[:n_select])


def interval(values: list[float] | np.ndarray) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan, 0
    lower, upper = np.quantile(array, [0.025, 0.975])
    return float(lower), float(upper), int(len(array))


def pairwise_comparison(
    data: pd.DataFrame,
    score_a: str,
    score_b: str,
    *,
    scope: str,
    label: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    paired = data.dropna(subset=[score_a, score_b]).copy()
    x = paired[score_a].to_numpy(float)
    y = paired[score_b].to_numpy(float)
    n = len(paired)
    point: dict[str, object] = {
        "scope": scope,
        "comparison": label,
        "score_a": score_a,
        "score_b": score_b,
        "n_chemicals": n,
        "spearman_rho": float(spearmanr(x, y).statistic),
        "kendall_tau": float(kendalltau(x, y).statistic),
    }
    for fraction in FRACTIONS:
        count = max(1, int(math.ceil(n * fraction)))
        a = selected_ids(paired, score_a, count)
        b = selected_ids(paired, score_b, count)
        union = a | b
        point[f"top_{int(fraction * 100)}pct_n"] = count
        point[f"top_{int(fraction * 100)}pct_overlap"] = len(a & b)
        point[f"top_{int(fraction * 100)}pct_jaccard"] = len(a & b) / len(union)
    top100_n = min(100, n)
    a100 = selected_ids(paired, score_a, top100_n)
    b100 = selected_ids(paired, score_b, top100_n)
    point["top100_n"] = top100_n
    point["top100_overlap"] = len(a100 & b100)
    point["top100_overlap_fraction"] = len(a100 & b100) / top100_n

    rng = np.random.default_rng(label_seed(f"pairwise::{scope}::{label}"))
    draw_rows: list[dict[str, object]] = []
    for draw in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, size=n)
        sx = x[sample]
        sy = y[sample]
        record: dict[str, object] = {
            "analysis": f"pairwise::{scope}::{label}",
            "bootstrap": draw,
            "spearman_rho": float(spearmanr(sx, sy).statistic),
            "kendall_tau": float(kendalltau(sx, sy).statistic),
        }
        draw_unit = np.arange(n)
        order_a = np.lexsort((draw_unit, -sx))
        order_b = np.lexsort((draw_unit, -sy))
        for fraction in FRACTIONS:
            count = max(1, int(math.ceil(n * fraction)))
            sa = set(order_a[:count].tolist())
            sb = set(order_b[:count].tolist())
            record[f"top_{int(fraction * 100)}pct_jaccard"] = len(sa & sb) / len(sa | sb)
        sa100 = set(order_a[:top100_n].tolist())
        sb100 = set(order_b[:top100_n].tolist())
        record["top100_overlap_fraction"] = len(sa100 & sb100) / top100_n
        draw_rows.append(record)
    draws = pd.DataFrame(draw_rows)
    for metric in [
        "spearman_rho",
        "kendall_tau",
        "top_5pct_jaccard",
        "top_10pct_jaccard",
        "top_20pct_jaccard",
        "top100_overlap_fraction",
    ]:
        lower, upper, valid = interval(draws[metric])
        point[f"{metric}_ci95_lower"] = lower
        point[f"{metric}_ci95_upper"] = upper
        point[f"{metric}_bootstrap_valid"] = valid
    return point, draws


def recall_analysis(
    data: pd.DataFrame,
    score_col: str,
    high_hazard: set[str],
    *,
    scope: str,
    variant: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    scoreable = data[data[score_col].notna()].copy()
    scoreable_ids = set(scoreable["dtxsid"].astype(str))
    high_scoreable = high_hazard & scoreable_ids
    coverage = len(high_scoreable) / len(high_hazard)
    labels = scoreable["dtxsid"].astype(str).isin(high_hazard).to_numpy(bool)
    scores = scoreable[score_col].to_numpy(float)
    n = len(scoreable)
    budgets = [(f"top_{int(f * 100)}pct", max(1, int(math.ceil(n * f)))) for f in FRACTIONS]
    budgets.append(("top100", min(100, n)))
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(label_seed(f"recall::{scope}::{variant}"))
    draw_records: list[dict[str, object]] = []
    bootstrap_store: dict[str, list[float]] = {name: [] for name, _ in budgets}
    for draw in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, size=n)
        sample_scores = scores[sample]
        sample_labels = labels[sample]
        denominator = int(sample_labels.sum())
        order = np.lexsort((np.arange(n), -sample_scores))
        record: dict[str, object] = {
            "analysis": f"recall::{scope}::{variant}",
            "bootstrap": draw,
            "n_high_hazard_in_bootstrap": denominator,
        }
        for budget_name, count in budgets:
            conditional = (
                float(sample_labels[order[:count]].sum() / denominator)
                if denominator > 0
                else np.nan
            )
            record[f"{budget_name}_conditional_recall"] = conditional
            bootstrap_store[budget_name].append(conditional)
        draw_records.append(record)
    ranking = ordered_ids(scoreable, score_col)
    for budget_name, count in budgets:
        selected = set(ranking[:count])
        hits = len(selected & high_scoreable)
        conditional = hits / len(high_scoreable) if high_scoreable else np.nan
        overall = hits / len(high_hazard)
        low, high, valid = interval(bootstrap_store[budget_name])
        rows.append(
            {
                "scope": scope,
                "variant": variant,
                "score_column": score_col,
                "budget": budget_name,
                "n_scoreable_chemicals": n,
                "n_selected": count,
                "n_lower20_hazard_all": len(high_hazard),
                "n_lower20_hazard_scoreable": len(high_scoreable),
                "scoreable_hazard_coverage": coverage,
                "n_hazard_recovered": hits,
                "conditional_recall_scoreable": conditional,
                "conditional_recall_ci95_lower": low,
                "conditional_recall_ci95_upper": high,
                "conditional_recall_bootstrap_valid": valid,
                "recall_of_all_23": overall,
                "recall_of_all_23_ci95_lower": coverage * low,
                "recall_of_all_23_ci95_upper": coverage * high,
            }
        )
    return rows, pd.DataFrame(draw_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhbt-root", required=True)
    parser.add_argument("--ap02-dir", required=True)
    parser.add_argument("--ap03a-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.bhbt_root).resolve()
    ap02 = Path(args.ap02_dir).resolve()
    ap03a = Path(args.ap03a_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "r0_target_matrix": root / "results/probability/species_protective_target_tail_probability_x95.npz",
        "strict_loo_target_matrix": ap02 / "ap02_strict_loo_species_protective_target_tail_probability_x95.npz",
        "reference_hc5": root / "results/warning_hc5_bridge/top5_panel_apical_threshold_vs_reference_hc5.csv",
        "chemical_names": root / "results/chemistry/chemical_form_and_moa_audit.csv",
        "r0_bridge_manifest": root / "results/warning_hc5_bridge/warning_hc5_bridge_manifest.json",
        "ap02_gate": ap02 / "gate_ap02.json",
        "ap03a_gate": ap03a / "gate_ap03a.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AP03 module B input(s):\n" + "\n".join(missing))

    ap02_gate = json.loads(paths["ap02_gate"].read_text(encoding="utf-8"))
    ap03a_gate = json.loads(paths["ap03a_gate"].read_text(encoding="utf-8"))
    if ap02_gate.get("status") != "PASS" or ap03a_gate.get("status") != "PASS":
        raise AssertionError("AP02 and AP03 module A gates must pass before module B")
    r1_top5 = list(ap02_gate["x95_disposition"]["selected_downstream_top5"])
    r0_manifest = json.loads(paths["r0_bridge_manifest"].read_text(encoding="utf-8"))
    r0_top5 = list(r0_manifest["panel_top5"])
    if set(r0_top5) != set(r1_top5):
        raise AssertionError(
            "Module B requires an explicit cross-membership design because AP02 changed Top-5 membership"
        )

    r0_target, r0_arrays = load_target_scores(paths["r0_target_matrix"], r0_top5, "R0")
    r1_target, r1_arrays = load_target_scores(
        paths["strict_loo_target_matrix"], r1_top5, "R1_STRICT_LOO"
    )
    for key in ["species", "target_ids", "chemicals", "effects"]:
        if not np.array_equal(r0_arrays[key], r1_arrays[key]):
            raise AssertionError(f"R0 and strict-LOO target matrices differ in {key} ordering")
    effect_scores = r1_target.merge(
        r0_target[["dtxsid", "target_id", "effect_family", "panel_target_score"]].rename(
            columns={"panel_target_score": "r0_panel_target_score"}
        ),
        on=["dtxsid", "target_id", "effect_family"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={"panel_target_score": "r1_strict_loo_panel_target_score"})
    effect_scores.to_csv(out / "ap03b_effect_family_scores.csv", index=False)

    r0_rankings = aggregate_rankings(r0_target, "r0")
    r1_rankings = aggregate_rankings(r1_target, "r1")
    ranking = r1_rankings.merge(
        r0_rankings[
            [
                "dtxsid",
                "r0_primary_max_target_id",
                "r0_primary_max_effect_family",
                "r0_primary_max_score",
            ]
        ],
        on="dtxsid",
        validate="one_to_one",
    )
    names = pd.read_csv(paths["chemical_names"], low_memory=False)
    names["DTXSID"] = names["DTXSID"].astype(str)
    name_map = (
        names.dropna(subset=["PREFERRED_NAME"])
        .drop_duplicates("DTXSID")
        .set_index("DTXSID")["PREFERRED_NAME"]
    )
    ranking["chemical_name"] = ranking["dtxsid"].map(name_map)

    reference = pd.read_csv(paths["reference_hc5"])
    reference["dtxsid"] = reference["dtxsid"].astype(str)
    cutoff = float(reference["reference_hc5_typical_log10"].quantile(0.20))
    reference["lower20_high_hazard"] = reference["reference_hc5_typical_log10"].le(cutoff)
    high_hazard = set(reference.loc[reference["lower20_high_hazard"], "dtxsid"])
    reference_columns = [
        "dtxsid",
        "reference_hc5_typical_log10",
        "lower20_high_hazard",
    ]
    ranking = ranking.merge(reference[reference_columns], on="dtxsid", how="left", validate="one_to_one")
    ranking["hc5_label_status"] = np.where(
        ranking["reference_hc5_typical_log10"].notna(), "LABELED", "UNLABELED"
    )

    score_columns = {
        "r0_primary_max": "r0_primary_max_score",
        "r1_primary_max": "r1_primary_max_score",
        "r1_balanced": "r1_balanced_score",
        "r1_mortality_excluded": "r1_mortality_excluded_score",
    }
    for variant, column in score_columns.items():
        rank_map = deterministic_rank(ranking, column)
        ranking[f"{variant}_rank"] = ranking["dtxsid"].map(rank_map).astype("Int64")
    ranking = ranking.sort_values(
        ["r1_primary_max_rank", "dtxsid"], kind="mergesort"
    ).reset_index(drop=True)
    ranking.to_csv(out / "ap03b_chemical_followup_rankings.csv", index=False)

    common_mask = (
        ranking["n_supported_effect_families"].ge(2)
        & ranking["n_supported_nonmortality_effect_families"].ge(1)
    )
    common = ranking[common_mask].copy()
    common.to_csv(out / "ap03b_matched_support_common_set.csv", index=False)

    pairwise_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    pairs = [
        ("r0_primary_max_score", "r1_primary_max_score", "R0 primary vs R1 strict-LOO primary"),
        ("r1_primary_max_score", "r1_balanced_score", "R1 primary max vs balanced"),
        (
            "r1_primary_max_score",
            "r1_mortality_excluded_score",
            "R1 primary max vs mortality-excluded balanced",
        ),
        (
            "r1_balanced_score",
            "r1_mortality_excluded_score",
            "R1 balanced vs mortality-excluded balanced",
        ),
    ]
    for scope, frame in [("full_pairwise_scoreable", ranking), ("matched_support", common)]:
        for score_a, score_b, label in pairs:
            row, draws = pairwise_comparison(
                frame, score_a, score_b, scope=scope, label=label
            )
            pairwise_rows.append(row)
            bootstrap_frames.append(draws)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(out / "ap03b_pairwise_ranking_robustness.csv", index=False)

    recall_rows: list[dict[str, object]] = []
    for scope, frame in [("full_scoreable", ranking), ("matched_support", common)]:
        for variant, score_col in score_columns.items():
            rows_out, draws = recall_analysis(
                frame,
                score_col,
                high_hazard,
                scope=scope,
                variant=variant,
            )
            recall_rows.extend(rows_out)
            bootstrap_frames.append(draws)
    recall = pd.DataFrame(recall_rows)
    recall.to_csv(out / "ap03b_lower20_hazard_recall.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True, sort=False).to_csv(
        out / "ap03b_bootstrap_draws.csv.gz", index=False, compression="gzip"
    )

    r0_mortality_count = int(ranking["r0_primary_max_effect_family"].eq(MORTALITY).sum())
    r0_top100_mortality = int(
        ranking.sort_values("r0_primary_max_rank")
        .head(100)["r0_primary_max_effect_family"]
        .eq(MORTALITY)
        .sum()
    )
    r1_mortality_count = int(ranking["r1_primary_max_effect_family"].eq(MORTALITY).sum())
    r1_top100_mortality = int(
        ranking.sort_values("r1_primary_max_rank")
        .head(100)["r1_primary_max_effect_family"]
        .eq(MORTALITY)
        .sum()
    )
    r0_recall = recall[
        recall["scope"].eq("full_scoreable")
        & recall["variant"].eq("r0_primary_max")
        & recall["budget"].isin(["top_5pct", "top_10pct", "top_20pct"])
    ].set_index("budget")["recall_of_all_23"].to_dict()
    expected_recall = {
        "top_5pct": 6 / 23,
        "top_10pct": 9 / 23,
        "top_20pct": 13 / 23,
    }
    checks = {
        "target_rows_equal_912": bool(len(effect_scores) == 912),
        "chemical_universe_equal_639": bool(ranking["dtxsid"].nunique() == 639),
        "hc5_labeled_equal_115": bool(ranking["hc5_label_status"].eq("LABELED").sum() == 115),
        "lower20_hazard_equal_23": bool(len(high_hazard) == 23),
        "r0_mortality_max_equal_629": bool(r0_mortality_count == 629),
        "r0_top100_all_mortality": bool(r0_top100_mortality == 100),
        "r0_recall_reproduced": bool(all(
            np.isclose(float(r0_recall[key]), value) for key, value in expected_recall.items()
        )),
        "r1_primary_scoreable_equal_639": bool(ranking["r1_primary_max_score"].notna().sum() == 639),
        "r1_balanced_scoreable_equal_639": bool(ranking["r1_balanced_score"].notna().sum() == 639),
        "mortality_excluded_missing_is_nan": bool(
            ranking.loc[
                ranking["r1_mortality_excluded_status"].eq("NOT SCOREABLE"),
                "r1_mortality_excluded_score",
            ].isna().all()
        ),
        "matched_support_requires_two_families_and_nonmortality": bool(
            common["n_supported_effect_families"].ge(2).all()
            and common["n_supported_nonmortality_effect_families"].ge(1).all()
        ),
        "bootstrap_draws_frozen": bool(N_BOOTSTRAP == 2000),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    gate = {
        "schema_version": 1,
        "gate": "A5_AP03_MODULE_B_FOLLOWUP_RANKING_ROBUSTNESS",
        "status": status,
        "seed": SEED,
        "bootstrap_draws": N_BOOTSTRAP,
        "aggregation_contract": {
            "primary": "maximum panel target score across supported chemical-effect targets",
            "balanced": "equal-weight mean of panel target scores across supported protective effect families",
            "mortality_excluded": "equal-weight mean across supported non-mortality protective effect families; no support is NOT SCOREABLE and is not replaced by zero",
            "matched_support": "at least two supported protective effect families and at least one supported non-mortality family",
        },
        "denominators": {
            "all_ranked_chemicals": int(len(ranking)),
            "mortality_excluded_scoreable": int(
                ranking["r1_mortality_excluded_score"].notna().sum()
            ),
            "mortality_excluded_not_scoreable": int(
                ranking["r1_mortality_excluded_score"].isna().sum()
            ),
            "matched_support_common_set": int(len(common)),
            "hc5_labeled": int(ranking["hc5_label_status"].eq("LABELED").sum()),
            "lower20_hazard": int(len(high_hazard)),
        },
        "mortality_dominance": {
            "r0_primary_max_rows": r0_mortality_count,
            "r0_primary_max_top100": r0_top100_mortality,
            "r1_strict_loo_primary_max_rows": r1_mortality_count,
            "r1_strict_loo_primary_max_top100": r1_top100_mortality,
        },
        "checks": checks,
        "next_stage": "A6_AP04_LOCALIZATION_SENSITIVITY" if status == "PASS" else None,
    }
    (out / "gate_ap03b.json").write_text(
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
    (out / "ap03b_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    matched_primary_balanced = pairwise[
        pairwise["scope"].eq("matched_support")
        & pairwise["comparison"].eq("R1 primary max vs balanced")
    ].iloc[0]
    matched_primary_nonmort = pairwise[
        pairwise["scope"].eq("matched_support")
        & pairwise["comparison"].eq("R1 primary max vs mortality-excluded balanced")
    ].iloc[0]
    report = [
        "# AP03 module B: follow-up ranking robustness",
        "",
        f"Gate: `{status}`.",
        "",
        "## R0 audit",
        "",
        f"The R0 maximum aggregation is mortality/survival-led for {r0_mortality_count}/639 "
        f"chemicals and {r0_top100_mortality}/100 of the leading chemicals, reproducing the "
        "review baseline. The lower-20% HC5 target count is 23 and R0 recall at 5%, 10%, and "
        "20% follow-up budgets was reproduced exactly.",
        "",
        "## Strict-LOO R1 scoreability",
        "",
        f"The primary and balanced rankings score all 639 chemicals. The mortality-excluded "
        f"ranking scores {int(ranking['r1_mortality_excluded_score'].notna().sum())} chemicals; "
        f"the remaining {int(ranking['r1_mortality_excluded_score'].isna().sum())} are explicitly "
        "NOT SCOREABLE. The matched-support set contains "
        f"{len(common)} chemicals with at least two protective effect families and at least one "
        "non-mortality family.",
        "",
        "## Matched-support rank comparisons",
        "",
        f"Primary max versus balanced: Spearman rho {matched_primary_balanced.spearman_rho:.6f} "
        f"(95% interval {matched_primary_balanced.spearman_rho_ci95_lower:.6f} to "
        f"{matched_primary_balanced.spearman_rho_ci95_upper:.6f}); top-20% Jaccard "
        f"{matched_primary_balanced.top_20pct_jaccard:.6f}.",
        f"Primary max versus mortality-excluded balanced: Spearman rho "
        f"{matched_primary_nonmort.spearman_rho:.6f} "
        f"(95% interval {matched_primary_nonmort.spearman_rho_ci95_lower:.6f} to "
        f"{matched_primary_nonmort.spearman_rho_ci95_upper:.6f}); top-20% Jaccard "
        f"{matched_primary_nonmort.top_20pct_jaccard:.6f}.",
        "",
        "These rankings describe evidence-conditioned follow-up priority. Missing non-mortality "
        "support is not interpreted as low priority or low risk.",
        "",
        "No manuscript or SI file was edited in this stage.",
    ]
    (out / "AP03B_FOLLOWUP_ROBUSTNESS_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
