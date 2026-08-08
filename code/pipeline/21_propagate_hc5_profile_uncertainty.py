#!/usr/bin/env python3
"""Propagate record-level censored-likelihood uncertainty to HC5 comparisons.

This module does not predict unmeasured absolute toxicity.  It profiles the
location likelihood of already observed species/context cells with the pooled
sigma fixed at the fitted v16.5 value, then uses likelihood-weighted draws to
quantify uncertainty in reference and subset HC5 estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm, spearmanr


PROTECTIVE = {
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
}
GRID_SD = 6.0
GRID_SIZE = 401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def context_id(frame: pd.DataFrame) -> pd.Series:
    return frame[["dtxsid", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]].astype(str).agg("|".join, axis=1)


def log_interval_probability(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    delta = ndtr(upper) - ndtr(lower)
    return np.log(np.clip(delta, 1e-300, None))


def profile_for_cell(records: pd.DataFrame, estimate: float, sigma: float, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    grid = estimate + sigma * np.linspace(-GRID_SD, GRID_SD, GRID_SIZE)
    ll = np.zeros(GRID_SIZE, dtype=float)
    for censoring_class, group in records.groupby("censoring_class", sort=False):
        if censoring_class in {"exact", "approximate_exact"}:
            values = group["_log_point"].to_numpy(float)
            ll += (-0.5 * ((values[:, None] - grid[None, :]) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)).sum(axis=0)
        elif censoring_class == "left_censored":
            upper = group["_log_upper"].to_numpy(float)
            ll += log_ndtr((upper[:, None] - grid[None, :]) / sigma).sum(axis=0)
        elif censoring_class == "right_censored":
            lower = group["_log_lower"].to_numpy(float)
            ll += log_ndtr((grid[None, :] - lower[:, None]) / sigma).sum(axis=0)
        elif censoring_class == "interval_censored":
            lower = group["_log_lower"].to_numpy(float)
            upper = group["_log_upper"].to_numpy(float)
            ll += log_interval_probability((lower[:, None] - grid[None, :]) / sigma, (upper[:, None] - grid[None, :]) / sigma).sum(axis=0)
    relative = 2 * (ll - np.nanmax(ll))
    keep = relative >= -3.841458820694124
    if not keep.any():
        keep[np.nanargmax(ll)] = True
    lower_ci = float(grid[np.where(keep)[0][0]])
    upper_ci = float(grid[np.where(keep)[0][-1]])
    mass = np.exp(ll - np.nanmax(ll))
    mass /= mass.sum()
    sampled = rng.choice(grid, size=draws, replace=True, p=mass)
    return sampled, lower_ci, upper_ci


def summarise(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.5, 0.975]))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = root / "results" / "validation" / "hc5_profile_uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    estimates = pd.read_csv(root / "results/hc5_framework_comparison/hc5_framework_context_estimates.csv.gz")
    estimates = estimates[estimates.replicate.eq(0)].copy()
    context_ids = set(estimates.context_id.astype(str))
    model = pd.read_csv(root / "results/toxicity/censored_model_cells.csv.gz", low_memory=False)
    valid = model.mle_status.astype(str).str.startswith("identified") & model.effect_family.isin(PROTECTIVE)
    model = model.loc[valid].copy()
    model["context_id"] = context_id(model)
    model = model[model.context_id.isin(context_ids)].copy()
    records = pd.read_csv(root / "results/toxicity/censored_record_level_compact.csv.gz", low_memory=False)
    records = records[records.effect_family.isin(PROTECTIVE)].copy()
    records["context_id"] = context_id(records)
    records = records[records.context_id.isin(context_ids)].copy()
    key_columns = ["context_id", "latin_name"]
    record_groups = {key: group for key, group in records.groupby(key_columns, sort=False)}

    draw_map: dict[tuple[str, str], np.ndarray] = {}
    interval_rows: list[dict[str, object]] = []
    for row in model.sort_values(key_columns).itertuples(index=False):
        key = (row.context_id, row.latin_name)
        group = record_groups.get(key)
        if group is None or group.empty:
            continue
        sampled, lower_ci, upper_ci = profile_for_cell(group, float(row.mu_log10_umol_L), float(row.pooled_sigma_log10), args.draws, rng)
        draw_map[key] = sampled
        interval_rows.append({
            "context_id": row.context_id,
            "dtxsid": row.dtxsid,
            "effect_family": row.effect_family,
            "latin_name": row.latin_name,
            "mle_log10_umol_L": float(row.mu_log10_umol_L),
            "profile_ci025": lower_ci,
            "profile_ci975": upper_ci,
            "n_records": int(row.n_records),
            "pooled_sigma_log10": float(row.pooled_sigma_log10),
        })
    pd.DataFrame(interval_rows).to_csv(out / "profile_cell_intervals.csv.gz", index=False, compression="gzip")

    per_context_rows: list[dict[str, object]] = []
    metric_draws: dict[str, list[dict[str, float]]] = {}
    top5 = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    top5 = top5[(top5.universe.eq("priority")) & (top5.protection_target_x.eq(0.95)) & (top5.method.eq("data_driven"))].sort_values("rank").head(5).latin_name.astype(str).tolist()
    observed_capture: list[float] = []
    z05 = norm.ppf(0.05)

    for current_context, group in estimates.groupby("context_id", sort=True):
        reference_species = [name for name in model.loc[model.context_id.eq(current_context), "latin_name"].astype(str).unique() if (current_context, name) in draw_map]
        if len(reference_species) < 15:
            continue
        matrix = np.vstack([draw_map[(current_context, name)] for name in reference_species])
        reference = matrix.mean(axis=0) + z05 * matrix.std(axis=0, ddof=1)
        capture_species = [name for name in top5 if (current_context, name) in draw_map]
        if capture_species:
            cap = np.vstack([draw_map[(current_context, name)] for name in capture_species]) <= reference[None, :]
            observed_capture.append(float(cap.any(axis=0).mean()))
        for row in group.itertuples(index=False):
            selected = [name for name in str(row.selected_species).split(" | ") if (current_context, name) in draw_map]
            if len(selected) < 2:
                continue
            subset = np.vstack([draw_map[(current_context, name)] for name in selected])
            estimated = subset.mean(axis=0) + z05 * subset.std(axis=0, ddof=1)
            ratio = 10 ** (estimated - reference)
            low, median, high = summarise(ratio)
            per_context_rows.append({
                "context_id": current_context,
                "method": row.method,
                "n_reference_species": len(reference_species),
                "n_selected": len(selected),
                "reference_hc5_profile_q025": summarise(reference)[0],
                "reference_hc5_profile_q50": summarise(reference)[1],
                "reference_hc5_profile_q975": summarise(reference)[2],
                "subset_hc5_profile_q025": summarise(estimated)[0],
                "subset_hc5_profile_q50": summarise(estimated)[1],
                "subset_hc5_profile_q975": summarise(estimated)[2],
                "ratio_profile_q025": low,
                "ratio_profile_q50": median,
                "ratio_profile_q975": high,
            })
            metric_draws.setdefault(row.method, []).append({"reference": reference, "subset": estimated})

    context_frame = pd.DataFrame(per_context_rows)
    context_frame.to_csv(out / "framework_context_profile_intervals.csv.gz", index=False, compression="gzip")
    metric_rows: list[dict[str, object]] = []
    for method, values in metric_draws.items():
        refs = np.vstack([value["reference"] for value in values])
        subs = np.vstack([value["subset"] for value in values])
        ratios = 10 ** (subs - refs)
        per_draw: list[dict[str, float]] = []
        for draw_index in range(args.draws):
            ratio = ratios[:, draw_index]
            correlation = spearmanr(subs[:, draw_index], refs[:, draw_index]).statistic
            per_draw.append({
                "median_ratio": float(np.median(ratio)),
                "spearman_r": float(correlation),
                "within_factor2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
                "false_safe_factor2_rate": float(np.mean(ratio > 2.0)),
            })
        draw_frame = pd.DataFrame(per_draw)
        row: dict[str, object] = {"method": method, "n_contexts": len(values)}
        for column in draw_frame.columns:
            low, median, high = summarise(draw_frame[column].to_numpy(float))
            row[f"{column}_q025"] = low
            row[f"{column}_q50"] = median
            row[f"{column}_q975"] = high
        metric_rows.append(row)
    pd.DataFrame(metric_rows).sort_values("method").to_csv(out / "framework_profile_metrics.csv", index=False)
    capture_array = np.asarray(observed_capture, dtype=float)
    capture_payload = {
        "scope": "framework-eligible contexts with at least one fixed national Top-5 species observed",
        "n_contexts": int(capture_array.size),
        "mean_context_capture": float(capture_array.mean()) if capture_array.size else None,
        "capture_context_q025": float(np.quantile(capture_array, 0.025)) if capture_array.size else None,
        "capture_context_q975": float(np.quantile(capture_array, 0.975)) if capture_array.size else None,
    }
    (out / "top5_observed_context_capture.json").write_text(json.dumps(capture_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "version": "v16.5",
        "draws": args.draws,
        "seed": args.seed,
        "method": "fixed-sigma record-level profile likelihood with likelihood-weighted propagation",
        "scope": "identified protective species/context cells in the existing framework comparison",
        "boundary": "Intervals quantify censored-record and working-SSD estimation uncertainty for observed contexts. They are not a posterior for all borrowed probability cells and do not predict unmeasured absolute toxicity.",
        "top5": top5,
    }
    (out / "hc5_profile_uncertainty_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"profiled_cells": len(interval_rows), "framework_rows": len(context_frame), "methods": len(metric_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
