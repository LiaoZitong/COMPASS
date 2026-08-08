#!/usr/bin/env python3
"""Quantify the observed time-window and concentration lead of warning evidence.

The analysis is restricted to direct chemical × species × medium matches.  It
describes observed endpoint potential and evidence gaps; it does not turn a
warning threshold into a regulatory HC5 or infer a causal mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTECTIVE = {
    "mortality_survival", "immobilization_intoxication", "growth",
    "reproduction", "development_morphology",
}
WARNING = {"heartbeat", "behavior_feeding_avoidance", "other_physiology"}
TIME_ORDER = {
    "h00_02_ultra_early": 1,
    "h02_06_early": 2,
    "h06_24_same_day": 3,
    "h24_48": 4,
    "h48_96": 5,
    "h96_168": 6,
    "d07_14": 7,
    "gt14d": 8,
}
TIME_LABEL = {
    "h00_02_ultra_early": "0-2 h", "h02_06_early": "2-6 h",
    "h06_24_same_day": "6-24 h", "h24_48": "24-48 h",
    "h48_96": "48-96 h", "h96_168": "96 h-7 d",
    "d07_14": "7-14 d", "gt14d": ">14 d",
}
KEY = ["dtxsid", "latin_name", "medium_family"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--min-chemicals", type=int, default=10)
    return parser.parse_args()


def summarise_pairs(frame: pd.DataFrame) -> dict[str, float | int]:
    delta = frame.warning_log10_threshold.to_numpy(float) - frame.apical_log10_threshold.to_numpy(float)
    time_delta = frame.warning_time_rank.to_numpy(float) - frame.apical_time_rank.to_numpy(float)
    earlier = time_delta < 0
    lower = delta <= 0
    return {
        "n_pairs": int(len(frame)),
        "n_chemicals": int(frame.dtxsid.nunique()),
        "median_log10_warning_to_apical_ratio": float(np.median(delta)),
        "median_warning_to_apical_ratio": float(10 ** np.median(delta)),
        "warning_lower_concentration_fraction": float(lower.mean()),
        "warning_earlier_window_fraction": float(earlier.mean()),
        "earlier_and_lower_fraction": float((earlier & lower).mean()),
        "earlier_but_higher_fraction": float((earlier & ~lower).mean()),
        "not_earlier_but_lower_fraction": float((~earlier & lower).mean()),
        "not_earlier_and_higher_fraction": float((~earlier & ~lower).mean()),
    }


def chemical_cluster_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    chemicals = frame.dtxsid.astype(str).drop_duplicates().tolist()
    if len(chemicals) < 5:
        return pd.DataFrame()
    groups = {chemical: group for chemical, group in frame.groupby("dtxsid", sort=False)}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for draw in range(draws):
        sampled = [groups[chemicals[index]] for index in rng.integers(0, len(chemicals), size=len(chemicals))]
        summary = summarise_pairs(pd.concat(sampled, ignore_index=True))
        summary["bootstrap"] = draw
        rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = root / "results" / "warning_lead_potential"
    out.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(root / "results/toxicity/censored_model_cells.csv.gz", low_memory=False)
    model = model[
        model.mle_status.astype(str).str.startswith("identified")
        & np.isfinite(model.mu_log10_umol_L)
        & model.duration_window_v16.isin(TIME_ORDER)
    ].copy()
    model["dtxsid"] = model.dtxsid.astype(str)
    model["time_rank"] = model.duration_window_v16.map(TIME_ORDER).astype(int)
    model["time_window"] = model.duration_window_v16.map(TIME_LABEL)

    warning = model[model.effect_family.isin(WARNING)].copy()
    protective = model[model.effect_family.isin(PROTECTIVE)].copy()
    availability = pd.concat([
        warning.assign(evidence_layer="warning"),
        protective.assign(evidence_layer="protective"),
    ]).groupby(["evidence_layer", "effect_family", "duration_window_v16"], dropna=False).agg(
        n_model_cells=("mu_log10_umol_L", "size"),
        n_records=("n_records", "sum"),
        n_species=("latin_name", "nunique"),
        n_chemicals=("dtxsid", "nunique"),
        n_species_chemical_medium=("medium_family", "size"),
    ).reset_index()
    availability.to_csv(out / "warning_and_protective_evidence_availability.csv", index=False)

    # One conservative direct pair per chemical × species × medium × warning family:
    # earliest observed warning window, compared with the most sensitive protective
    # threshold and its earliest observed protective time window.
    warning = warning.sort_values(KEY + ["effect_family", "time_rank", "mu_log10_umol_L"])
    warning = warning.groupby(KEY + ["effect_family"], as_index=False).first()
    apical = protective.groupby(KEY, as_index=False).agg(
        apical_log10_threshold=("mu_log10_umol_L", "min"),
        apical_time_rank=("time_rank", "min"),
        n_apical_cells=("mu_log10_umol_L", "size"),
        n_apical_effect_families=("effect_family", "nunique"),
        n_apical_records=("n_records", "sum"),
    )
    apical["apical_time_window"] = apical.apical_time_rank.map({rank: label for key, rank in TIME_ORDER.items() for label in [TIME_LABEL[key]]})
    pairs = warning.merge(apical, on=KEY, how="inner", validate="many_to_one")
    pairs = pairs.rename(columns={
        "effect_family": "warning_effect_family",
        "mu_log10_umol_L": "warning_log10_threshold",
        "time_rank": "warning_time_rank",
        "time_window": "warning_time_window",
        "n_records": "n_warning_records",
    })
    pairs["log10_warning_to_apical_ratio"] = pairs.warning_log10_threshold - pairs.apical_log10_threshold
    pairs["warning_is_earlier"] = pairs.warning_time_rank < pairs.apical_time_rank
    pairs["warning_is_lower_concentration"] = pairs.log10_warning_to_apical_ratio <= 0
    pairs["lead_quadrant"] = np.select(
        [
            pairs.warning_is_earlier & pairs.warning_is_lower_concentration,
            pairs.warning_is_earlier & ~pairs.warning_is_lower_concentration,
            ~pairs.warning_is_earlier & pairs.warning_is_lower_concentration,
        ],
        ["earlier_and_lower", "earlier_but_higher", "not_earlier_but_lower"],
        default="not_earlier_and_higher",
    )
    pairs.to_csv(out / "warning_apical_direct_pairs.csv.gz", index=False, compression="gzip")

    summaries: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    groups = [("all_warning", pairs)] + [(str(name), group) for name, group in pairs.groupby("warning_effect_family", sort=True)]
    for name, group in groups:
        summary: dict[str, object] = {"scope": name, **summarise_pairs(group)}
        eligible = summary["n_pairs"] >= args.min_pairs and summary["n_chemicals"] >= args.min_chemicals
        summary["meets_minimum_evidence"] = bool(eligible)
        summaries.append(summary)
        if eligible:
            boot = chemical_cluster_bootstrap(group, args.bootstrap, args.seed)
            if not boot.empty:
                boot["scope"] = name
                bootstrap_frames.append(boot)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out / "warning_apical_lead_summary.csv", index=False)
    if bootstrap_frames:
        boot = pd.concat(bootstrap_frames, ignore_index=True)
        boot.to_csv(out / "warning_apical_lead_bootstrap.csv.gz", index=False, compression="gzip")
        interval_rows = []
        numeric = [column for column in boot.columns if column not in {"bootstrap", "scope", "n_pairs", "n_chemicals"}]
        for scope, group in boot.groupby("scope", sort=True):
            for metric in numeric:
                interval_rows.append({
                    "scope": scope, "metric": metric,
                    "q025": float(group[metric].quantile(0.025)),
                    "q50": float(group[metric].quantile(0.5)),
                    "q975": float(group[metric].quantile(0.975)),
                })
        pd.DataFrame(interval_rows).to_csv(out / "warning_apical_lead_bootstrap_intervals.csv", index=False)

    manifest = {
        "analysis": "direct warning-versus-protective matched evidence",
        "pair_key": "chemical × species × medium",
        "warning_definition": "earliest observed known-duration warning cell within warning effect family",
        "protective_definition": "minimum observed protective MLE across valid apical cells for the same pair",
        "time_metric": "ordered duration windows, not an exact response-time measurement",
        "minimum_evidence": {"pairs": args.min_pairs, "chemicals": args.min_chemicals},
        "bootstrap": {"draws": args.bootstrap, "cluster": "chemical", "seed": args.seed},
        "boundary": "Observed lower concentration or earlier window indicates screening potential only. It does not establish endpoint equivalence, a regulatory HC5, causality, or external validation.",
    }
    (out / "warning_apical_lead_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"direct_pairs": len(pairs), "scopes": len(summary_frame), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
