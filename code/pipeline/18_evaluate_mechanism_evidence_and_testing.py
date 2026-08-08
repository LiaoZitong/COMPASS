#!/usr/bin/env python3
"""Audit functional-MOA/form evidence and identify mechanism evidence gaps.

This module is intentionally diagnostic.  It reports OOF uncertainty and data
coverage and proposes evidence gaps for testing; it never changes the HC5 core
matrix or predicts an unmeasured absolute toxicity value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


PROTECTIVE = {
    "mortality_survival", "immobilization_intoxication", "growth",
    "reproduction", "development_morphology",
}
LAYERS = ("functional_moa", "chemical_form")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def safe_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
    }


def bootstrap_oof(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (target, effect, layer), group in frame.groupby(["protection_target_x", "effect_family", "candidate_layer"], sort=True):
        chemicals = group.dtxsid.astype(str).drop_duplicates().tolist()
        by_chemical = {chemical: subset for chemical, subset in group.groupby("dtxsid", sort=False)}
        if len(chemicals) < 5:
            continue
        for draw in range(draws):
            sampled = pd.concat([by_chemical[chemicals[index]] for index in rng.integers(0, len(chemicals), size=len(chemicals))], ignore_index=True)
            y = sampled.tail_binary.to_numpy(int)
            tax = safe_metrics(y, sampled.taxonomy_oof.to_numpy(float))
            candidate = safe_metrics(y, sampled.candidate_oof.to_numpy(float))
            rows.append({
                "protection_target_x": target,
                "effect_family": effect,
                "candidate_layer": layer,
                "bootstrap": draw,
                "n_rows": len(sampled),
                "n_chemicals": len(chemicals),
                "brier_improvement": tax["brier"] - candidate["brier"],
                "log_loss_improvement": tax["log_loss"] - candidate["log_loss"],
                "roc_auc_improvement": candidate["roc_auc"] - tax["roc_auc"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = root / "results" / "mechanism_evidence"
    out.mkdir(parents=True, exist_ok=True)
    validation = root / "results" / "validation"
    prediction_file = validation / "moa_simplified_oof_predictions.csv.gz"
    predictions = pd.read_csv(prediction_file, low_memory=False)
    long = []
    for layer in LAYERS:
        candidate_column = f"{layer}_oof"
        subset = predictions[["dtxsid", "effect_family", "protection_target_x", "tail_binary", "taxonomy_oof", candidate_column]].copy()
        subset = subset.rename(columns={candidate_column: "candidate_oof"})
        subset["candidate_layer"] = layer
        long.append(subset)
    long_frame = pd.concat(long, ignore_index=True)
    boot = bootstrap_oof(long_frame, args.bootstrap, args.seed)
    if not boot.empty:
        boot.to_csv(out / "mechanism_oof_cluster_bootstrap.csv.gz", index=False, compression="gzip")
        summaries = []
        for keys, group in boot.groupby(["protection_target_x", "effect_family", "candidate_layer"], sort=True):
            record = dict(zip(["protection_target_x", "effect_family", "candidate_layer"], keys))
            record["n_chemicals"] = int(group.n_chemicals.iloc[0])
            for column in ("brier_improvement", "log_loss_improvement", "roc_auc_improvement"):
                record[f"{column}_q025"] = float(group[column].quantile(0.025))
                record[f"{column}_q50"] = float(group[column].quantile(0.5))
                record[f"{column}_q975"] = float(group[column].quantile(0.975))
            summaries.append(record)
        pd.DataFrame(summaries).to_csv(out / "mechanism_oof_cluster_bootstrap_summary.csv", index=False)

    model = pd.read_csv(root / "results/toxicity/censored_model_cells.csv.gz", low_memory=False)
    model = model[model.mle_status.astype(str).str.startswith("identified") & model.effect_family.isin(PROTECTIVE)].copy()
    model["dtxsid"] = model.dtxsid.astype(str)
    chemistry = pd.read_csv(root / "results/chemistry/chemical_moa_hierarchy.csv.gz", low_memory=False)
    chemistry["DTXSID"] = chemistry.DTXSID.astype(str)
    form_column = "moa_level_form"
    functional_column = "moa_level_functional"
    model = model.merge(chemistry[["DTXSID", form_column, functional_column]], left_on="dtxsid", right_on="DTXSID", how="left")
    model["chemical_form"] = model[form_column].fillna("FORM::unresolved")
    model["functional_moa"] = model[functional_column].fillna("MOAFAM::unresolved")
    density_rows = []
    for layer in LAYERS:
        for (effect, mechanism), group in model.groupby(["effect_family", layer], sort=True):
            density_rows.append({
                "candidate_layer": layer,
                "effect_family": effect,
                "mechanism_group": mechanism,
                "n_model_cells": int(len(group)),
                "n_records": int(group.n_records.sum()),
                "n_species": int(group.latin_name.nunique()),
                "n_chemicals": int(group.dtxsid.nunique()),
                "n_studies": int(group.n_studies.sum()),
                "median_records_per_cell": float(group.n_records.median()),
                "unresolved_group": bool(str(mechanism).endswith("unresolved")),
            })
    density = pd.DataFrame(density_rows)
    density.to_csv(out / "mechanism_evidence_density.csv", index=False)

    priority = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    priority["DTXSID"] = priority.DTXSID.astype(str)
    chemical_density = model.groupby("dtxsid").agg(
        n_model_cells=("mu_log10_umol_L", "size"),
        n_species=("latin_name", "nunique"),
        n_effect_families=("effect_family", "nunique"),
        n_records=("n_records", "sum"),
        chemical_form=("chemical_form", "first"),
        functional_moa=("functional_moa", "first"),
    ).reset_index().merge(priority[["DTXSID", "national_weight"]], left_on="dtxsid", right_on="DTXSID", how="right")
    chemical_density[["n_model_cells", "n_species", "n_effect_families", "n_records"]] = chemical_density[["n_model_cells", "n_species", "n_effect_families", "n_records"]].fillna(0)
    chemical_density["mechanism_evidence_gap_score"] = chemical_density.national_weight.fillna(0) * (
        1 / (1 + chemical_density.n_model_cells) + 1 / (1 + chemical_density.n_species)
    )
    chemical_density = chemical_density.sort_values("mechanism_evidence_gap_score", ascending=False)
    chemical_density.to_csv(out / "mechanism_evidence_gap_priorities.csv", index=False)

    manifest = {
        "analysis": "mechanism evidence coverage, OOF uncertainty and test-allocation gaps",
        "layers": list(LAYERS),
        "bootstrap": {"draws": args.bootstrap, "cluster": "chemical", "seed": args.seed},
        "core_rule": "Functional MOA and chemical form remain outside the HC5 core unless the pre-specified whole-chemical entry gate passes.",
        "boundary": "This output quantifies evidence density and OOF robustness and prioritizes data gaps. It is not a causal MOA analysis and does not predict unmeasured absolute toxicity.",
    }
    (out / "mechanism_evidence_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"bootstrap_rows": int(len(boot)), "density_rows": int(len(density)), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
