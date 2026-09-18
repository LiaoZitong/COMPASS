#!/usr/bin/env python3
"""Recompute profile-likelihood and MNAR sensitivity from the strict-LOO layer.

The script is intentionally downstream of the approved strict focal-species
leave-one-out analysis.  It never substitutes the earlier all-species
reference probability matrix and does not modify any primary analysis input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, log_ndtr, logit, ndtr
from scipy.stats import norm, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--odds-ratios", default="0.25,0.5,1,2,4")
    parser.add_argument("--negative-ratio", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def context_id(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "dtxsid",
        "effect_family",
        "endpoint_band_v16",
        "duration_window_v16",
        "medium_family",
    ]
    return frame[columns].astype(str).agg("|".join, axis=1)


def log_interval_probability(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    delta = ndtr(upper) - ndtr(lower)
    return np.log(np.clip(delta, 1e-300, None))


def profile_for_cell(
    records: pd.DataFrame,
    estimate: float,
    sigma: float,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    grid = estimate + sigma * np.linspace(-GRID_SD, GRID_SD, GRID_SIZE)
    likelihood = np.zeros(GRID_SIZE, dtype=float)
    for censoring_class, group in records.groupby("censoring_class", sort=False):
        if censoring_class in {"exact", "approximate_exact"}:
            values = group["_log_point"].to_numpy(float)
            likelihood += (
                -0.5 * ((values[:, None] - grid[None, :]) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
            ).sum(axis=0)
        elif censoring_class == "left_censored":
            upper = group["_log_upper"].to_numpy(float)
            likelihood += log_ndtr((upper[:, None] - grid[None, :]) / sigma).sum(axis=0)
        elif censoring_class == "right_censored":
            lower = group["_log_lower"].to_numpy(float)
            likelihood += log_ndtr((grid[None, :] - lower[:, None]) / sigma).sum(axis=0)
        elif censoring_class == "interval_censored":
            lower = group["_log_lower"].to_numpy(float)
            upper = group["_log_upper"].to_numpy(float)
            likelihood += log_interval_probability(
                (lower[:, None] - grid[None, :]) / sigma,
                (upper[:, None] - grid[None, :]) / sigma,
            ).sum(axis=0)
    relative = 2 * (likelihood - np.nanmax(likelihood))
    keep = relative >= -3.841458820694124
    if not keep.any():
        keep[np.nanargmax(likelihood)] = True
    mass = np.exp(likelihood - np.nanmax(likelihood))
    mass /= mass.sum()
    sampled = rng.choice(grid, size=draws, replace=True, p=mass)
    return sampled, float(grid[np.flatnonzero(keep)[0]]), float(grid[np.flatnonzero(keep)[-1]])


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.5, 0.975]))


def apply_selection_odds_ratio(mar_probability: np.ndarray, odds_ratio: float) -> np.ndarray:
    """Apply the prespecified MNAR selection shift to MAR probabilities."""

    if odds_ratio <= 0:
        raise ValueError("Selection odds ratio must be positive")
    return expit(
        logit(np.clip(np.asarray(mar_probability, dtype=float), 1e-5, 1 - 1e-5))
        - np.log(odds_ratio)
    )


def build_features(
    species: pd.DataFrame,
    targets: pd.DataFrame,
    species_index: np.ndarray,
    target_index: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_species_contexts": np.log1p(species.iloc[species_index].n_contexts.to_numpy(float)),
            "log_species_chemicals": np.log1p(species.iloc[species_index].n_chemicals.to_numpy(float)),
            "eligible_states": species.iloc[species_index].eligible_states.to_numpy(float),
            "official_method_species": species.iloc[species_index].official_or_common_method_species.astype(int).to_numpy(),
            "taxonomic_class": species.iloc[species_index]["class"].fillna("unresolved").astype(str).to_numpy(),
            "support_tier": species.iloc[species_index].support_tier.fillna("unresolved").astype(str).to_numpy(),
            "effect_family": targets.iloc[target_index].effect_family.astype(str).to_numpy(),
            "chemical_form": targets.iloc[target_index].chemical_form.astype(str).to_numpy(),
            "functional_moa": targets.iloc[target_index].functional_moa.astype(str).to_numpy(),
            "priority_chemical": targets.iloc[target_index].priority_chemical.astype(int).to_numpy(),
        }
    )


def deterministic_greedy(
    probability: np.ndarray,
    weights: np.ndarray,
    species_names: list[str],
    k: int = 5,
) -> list[int]:
    selected: list[int] = []
    residual = np.ones(probability.shape[1], dtype=float)
    remaining = np.ones(probability.shape[0], dtype=bool)
    for _ in range(k):
        gains = (probability * residual[None, :] * weights[None, :]).sum(axis=1)
        gains[~remaining] = -np.inf
        best = float(np.nanmax(gains))
        tied = np.flatnonzero(np.isclose(gains, best, rtol=1e-12, atol=1e-15))
        chosen = min(tied.tolist(), key=lambda index: species_names[index])
        selected.append(chosen)
        remaining[chosen] = False
        residual *= 1 - probability[chosen]
    return selected


def copula_union(probability: np.ndarray, rho: float, nodes: int = 16) -> np.ndarray:
    matrix = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    points, weights = hermgauss(nodes)
    latent = points * np.sqrt(2)
    weights = weights / np.sqrt(np.pi)
    threshold = norm.ppf(1 - matrix)
    no_event = np.zeros(matrix.shape[1], dtype=float)
    shared_sd = math.sqrt(max(rho, 0.0))
    residual_sd = math.sqrt(max(1 - rho, 1e-9))
    for value, weight in zip(latent, weights):
        no_event += weight * np.prod(norm.cdf((threshold - shared_sd * value) / residual_sd), axis=0)
    return np.clip(1 - no_event, 0, 1)


def expected_capture(
    probability: np.ndarray,
    selected: list[int],
    weights: np.ndarray,
    rho: float,
) -> float:
    return float(np.dot(weights, copula_union(probability[np.asarray(selected, dtype=int)], rho)))


def prepare_paths(project_root: Path, revision_root: Path) -> dict[str, Path]:
    analysis_root = revision_root / "04_analysis" if (revision_root / "04_analysis").is_dir() else revision_root
    ap02 = analysis_root / "02_ap02_strict_loo"
    paths = {
        "strict_rows": ap02 / "ap02_context_focal_species_loo_rows.csv.gz",
        "strict_direct": ap02 / "ap02_direct_target_probability_delta_x95.csv.gz",
        "strict_matrix": ap02 / "ap02_strict_loo_species_chemical_tail_probability_x95.npz",
        "strict_target_matrix": ap02 / "ap02_strict_loo_species_protective_target_tail_probability_x95.npz",
        "rho": ap02 / "ap02_rho_diagnostic.csv",
        "numeric_freeze": analysis_root / "07_g3_cross_package_integration" / "g3_authoritative_numeric_freeze.json",
        "model": project_root / "results" / "toxicity" / "censored_model_cells.csv.gz",
        "records": project_root / "results" / "toxicity" / "censored_record_level_compact.csv.gz",
        "species": project_root / "results" / "probability" / "candidate_universe_locked.csv",
        "targets": project_root / "results" / "probability" / "protective_chemical_effect_targets.csv",
        "chemistry": project_root / "results" / "chemistry" / "chemical_moa_form_master.csv.gz",
        "priority": project_root / "results" / "weights" / "national_priority_chemicals.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing strict-LOO robustness input(s):\n" + "\n".join(missing))
    return paths


def run_profile(
    paths: dict[str, Path],
    output_dir: Path,
    draws: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    strict_rows = pd.read_csv(paths["strict_rows"], low_memory=False)
    strict_rows = strict_rows[strict_rows.strict_loo_eligible.astype(bool)].copy()
    context_ids = set(strict_rows.context_id.astype(str))

    model = pd.read_csv(paths["model"], low_memory=False)
    valid = model.mle_status.astype(str).str.startswith("identified") & model.effect_family.isin(PROTECTIVE)
    model = model.loc[valid].copy()
    model["context_id"] = context_id(model)
    model = model[model.context_id.isin(context_ids)].copy()

    records = pd.read_csv(paths["records"], low_memory=False)
    records = records[records.effect_family.isin(PROTECTIVE)].copy()
    records["context_id"] = context_id(records)
    records = records[records.context_id.isin(context_ids)].copy()
    record_groups = {
        (str(context), str(species)): group
        for (context, species), group in records.groupby(["context_id", "latin_name"], sort=False)
    }

    draw_map: dict[tuple[str, str], np.ndarray] = {}
    interval_rows: list[dict[str, object]] = []
    for row in model.sort_values(["context_id", "latin_name"]).itertuples(index=False):
        key = (str(row.context_id), str(row.latin_name))
        group = record_groups.get(key)
        if group is None or group.empty:
            continue
        sampled, lower_ci, upper_ci = profile_for_cell(
            group,
            float(row.mu_log10_umol_L),
            float(row.pooled_sigma_log10),
            draws,
            rng,
        )
        draw_map[key] = sampled
        interval_rows.append(
            {
                "context_id": row.context_id,
                "dtxsid": row.dtxsid,
                "effect_family": row.effect_family,
                "latin_name": row.latin_name,
                "mle_log10_umol_L": float(row.mu_log10_umol_L),
                "profile_ci025": lower_ci,
                "profile_ci975": upper_ci,
                "n_records": int(row.n_records),
                "pooled_sigma_log10": float(row.pooled_sigma_log10),
            }
        )
    pd.DataFrame(interval_rows).to_csv(
        output_dir / "strict_loo_profile_cell_intervals.csv.gz", index=False, compression="gzip"
    )

    selected_by_context: dict[str, list[str]] = {}
    for current_context, group in strict_rows.groupby("context_id", sort=True):
        ranked = group.sort_values(
            ["p_context_loo_x95", "latin_name"], ascending=[False, True], kind="mergesort"
        )
        names = [
            str(name)
            for name in ranked.latin_name
            if (str(current_context), str(name)) in draw_map
        ]
        selected_by_context[str(current_context)] = names[:5]

    context_output: list[dict[str, object]] = []
    reference_draws: list[np.ndarray] = []
    subset_draws: list[np.ndarray] = []
    z05 = norm.ppf(0.05)
    for current_context in sorted(context_ids):
        reference_species = sorted(
            name
            for name in model.loc[model.context_id.eq(current_context), "latin_name"].astype(str).unique()
            if (current_context, name) in draw_map
        )
        selected = selected_by_context.get(current_context, [])
        if len(reference_species) < 15 or len(selected) != 5:
            continue
        reference_matrix = np.vstack([draw_map[(current_context, name)] for name in reference_species])
        subset_matrix = np.vstack([draw_map[(current_context, name)] for name in selected])
        reference = reference_matrix.mean(axis=0) + z05 * reference_matrix.std(axis=0, ddof=1)
        subset = subset_matrix.mean(axis=0) + z05 * subset_matrix.std(axis=0, ddof=1)
        ratio = 10 ** (subset - reference)
        ratio_q = quantiles(ratio)
        context_output.append(
            {
                "context_id": current_context,
                "n_reference_species": len(reference_species),
                "n_selected": len(selected),
                "selected_species": " | ".join(selected),
                "selection_basis": "strict focal-species leave-one-out p_context_loo_x95",
                "ratio_profile_q025": ratio_q[0],
                "ratio_profile_q50": ratio_q[1],
                "ratio_profile_q975": ratio_q[2],
            }
        )
        reference_draws.append(reference)
        subset_draws.append(subset)

    if not context_output:
        raise RuntimeError("No strict-LOO profile contexts met the >=15 species and k=5 requirements")
    context_frame = pd.DataFrame(context_output)
    context_frame.to_csv(
        output_dir / "strict_loo_framework_context_profile_intervals.csv.gz",
        index=False,
        compression="gzip",
    )
    references = np.vstack(reference_draws)
    subsets = np.vstack(subset_draws)
    ratios = 10 ** (subsets - references)
    per_draw: list[dict[str, float]] = []
    for draw_index in range(draws):
        ratio = ratios[:, draw_index]
        per_draw.append(
            {
                "median_ratio": float(np.median(ratio)),
                "spearman_r": float(spearmanr(subsets[:, draw_index], references[:, draw_index]).statistic),
                "within_factor2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
                "false_safe_factor2_rate": float(np.mean(ratio > 2.0)),
            }
        )
    draw_frame = pd.DataFrame(per_draw)
    metric: dict[str, object] = {
        "method": "strict_loo_probability_sentinel_k5",
        "n_contexts": len(context_output),
    }
    for column in draw_frame.columns:
        low, center, high = quantiles(draw_frame[column].to_numpy(float))
        metric[f"{column}_q025"] = low
        metric[f"{column}_q50"] = center
        metric[f"{column}_q975"] = high
    pd.DataFrame([metric]).to_csv(output_dir / "strict_loo_framework_profile_metrics.csv", index=False)
    return {
        "profiled_cells": len(interval_rows),
        "profile_contexts": len(context_output),
        "draws": draws,
        "seed": seed,
        "method": "fixed-sigma record-level profile likelihood with strict-LOO context-specific k=5 selection",
        "boundary": (
            "Intervals quantify censored-record and working-SSD uncertainty within measured protective contexts. "
            "They do not identify uncertainty for unmeasured toxicity or convert the sentinel subset into a regulatory SSD."
        ),
    }


def prepare_target_features(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    species = pd.read_csv(paths["species"]).reset_index(names="species_index")
    targets = pd.read_csv(paths["targets"])
    chemistry = pd.read_csv(paths["chemistry"], low_memory=False)
    chemistry["DTXSID"] = chemistry.DTXSID.astype(str)
    form_column = "chemical_form_class" if "chemical_form_class" in chemistry else "chemical_form"
    moa_column = "functional_moa" if "functional_moa" in chemistry else "primary_moa"
    targets = targets.merge(
        chemistry[["DTXSID", form_column, moa_column]],
        left_on="dtxsid",
        right_on="DTXSID",
        how="left",
    ).rename(columns={form_column: "chemical_form", moa_column: "functional_moa"})
    targets["chemical_form"] = targets.chemical_form.fillna("unresolved")
    targets["functional_moa"] = targets.functional_moa.fillna("unresolved")
    priority_ids = set(pd.read_csv(paths["priority"]).DTXSID.astype(str))
    targets["priority_chemical"] = targets.dtxsid.astype(str).isin(priority_ids)
    targets = targets.reset_index(names="target_index")
    return species, targets


def run_mnar(
    paths: dict[str, Path],
    output_dir: Path,
    odds_ratios: list[float],
    seed: int,
    negative_ratio: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    species, targets = prepare_target_features(paths)
    species_lookup = dict(zip(species.latin_name.astype(str), species.species_index.astype(int)))
    target_lookup = {
        (str(row.dtxsid), str(row.effect_family)): int(row.target_index)
        for row in targets.itertuples(index=False)
    }

    direct = pd.read_csv(paths["strict_direct"], low_memory=False)
    direct = direct[direct.strict_loo_direct_contexts.astype(int).gt(0)].copy()
    direct["species_index"] = direct.latin_name.astype(str).map(species_lookup)
    direct["target_index"] = [
        target_lookup.get((str(dtxsid), str(effect)))
        for dtxsid, effect in zip(direct.dtxsid, direct.effect_family)
    ]
    direct = direct.dropna(subset=["species_index", "target_index"]).copy()
    direct[["species_index", "target_index"]] = direct[["species_index", "target_index"]].astype(int)
    observed_pairs = direct[["species_index", "target_index"]].drop_duplicates().copy()
    observed_pairs["observed"] = 1

    observed_key = set(
        (observed_pairs.species_index * len(targets) + observed_pairs.target_index).astype(np.int64)
    )
    total_pairs = len(species) * len(targets)
    n_negative = min(negative_ratio * len(observed_pairs), total_pairs - len(observed_pairs))
    negative_keys: set[int] = set()
    while len(negative_keys) < n_negative:
        candidate = rng.integers(
            0,
            total_pairs,
            size=max(10000, 2 * (n_negative - len(negative_keys))),
            endpoint=False,
        )
        negative_keys.update(int(value) for value in candidate if int(value) not in observed_key)
    negative_array = np.fromiter(sorted(negative_keys), dtype=np.int64, count=n_negative)
    negatives = pd.DataFrame(
        {
            "species_index": negative_array // len(targets),
            "target_index": negative_array % len(targets),
            "observed": 0,
        }
    )
    sampled = pd.concat([observed_pairs, negatives], ignore_index=True)
    sampled["sample_weight"] = np.where(
        sampled.observed.eq(1),
        1.0,
        (total_pairs - len(observed_pairs)) / max(n_negative, 1),
    )
    features = build_features(
        species,
        targets,
        sampled.species_index.to_numpy(int),
        sampled.target_index.to_numpy(int),
    )
    numeric = ["log_species_contexts", "log_species_chemicals", "eligible_states", "official_method_species"]
    categorical = ["taxonomic_class", "support_tier", "effect_family", "chemical_form", "functional_moa", "priority_chemical"]
    transformer = ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    estimator = Pipeline(
        [("features", transformer), ("model", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0))]
    )
    groups = targets.iloc[sampled.target_index.to_numpy(int)].dtxsid.astype(str).to_numpy()
    oof = np.zeros(len(sampled), dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(features, sampled.observed.to_numpy(int), groups):
        model = clone(estimator)
        model.fit(
            features.iloc[train],
            sampled.observed.iloc[train],
            model__sample_weight=sampled.sample_weight.iloc[train],
        )
        oof[test] = model.predict_proba(features.iloc[test])[:, 1]
    propensity_auc = float(
        roc_auc_score(sampled.observed, oof, sample_weight=sampled.sample_weight)
    )
    observed_mask = sampled.observed.to_numpy(bool)
    propensity = sampled.loc[observed_mask, ["species_index", "target_index"]].copy()
    propensity["propensity_oof"] = oof[observed_mask]
    propensity_by_pair = dict(
        zip(
            (propensity.species_index * len(targets) + propensity.target_index).astype(np.int64),
            propensity.propensity_oof,
        )
    )

    strict_rows = pd.read_csv(paths["strict_rows"], low_memory=False)
    strict_rows = strict_rows[strict_rows.strict_loo_eligible.astype(bool)].copy()
    strict_rows["species_index"] = strict_rows.latin_name.astype(str).map(species_lookup)
    strict_rows["target_index"] = [
        target_lookup.get((str(dtxsid), str(effect)))
        for dtxsid, effect in zip(strict_rows.dtxsid, strict_rows.effect_family)
    ]
    strict_rows = strict_rows.dropna(subset=["species_index", "target_index"]).copy()
    strict_rows[["species_index", "target_index"]] = strict_rows[["species_index", "target_index"]].astype(int)
    direct_probability = direct[
        ["latin_name", "dtxsid", "effect_family", "p_strict_loo"]
    ].copy()
    if direct_probability.duplicated(["latin_name", "dtxsid", "effect_family"]).any():
        raise RuntimeError("Strict-LOO direct probabilities are not unique by species and target")
    strict_rows = strict_rows.merge(
        direct_probability,
        on=["latin_name", "dtxsid", "effect_family"],
        how="left",
        validate="many_to_one",
    )
    if strict_rows.p_strict_loo.isna().any():
        raise RuntimeError("Strict-LOO context rows are missing direct-scale probabilities")
    strict_rows["pair_key"] = strict_rows.species_index * len(targets) + strict_rows.target_index
    strict_rows["propensity_oof"] = strict_rows.pair_key.map(propensity_by_pair).clip(1e-4, 1 - 1e-4)
    if strict_rows.propensity_oof.isna().any():
        raise RuntimeError("Strict-LOO context rows are missing pair-level propensity estimates")
    raw_weight = 1 / strict_rows.propensity_oof.to_numpy(float)
    lower_clip, upper_clip = np.quantile(raw_weight, [0.01, 0.99])
    ipw = np.clip(raw_weight, lower_clip, upper_clip)
    ipw /= ipw.mean()
    # The recalibration predictor must be on the same species-by-target scale as
    # the full matrix to which the fitted map is later applied.  Context-level
    # focal probabilities are conditional on the very context used to define
    # the binary outcome and therefore are not transportable to that matrix.
    predictor = logit(np.clip(strict_rows.p_strict_loo.to_numpy(float), 1e-5, 1 - 1e-5)).reshape(-1, 1)
    tail_binary = (
        strict_rows.mu_log10_umol_L.to_numpy(float)
        <= strict_rows.loo_cutoff_log10_x95.to_numpy(float)
    ).astype(int)
    outcome_model = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
    outcome_model.fit(predictor, tail_binary, sample_weight=ipw)
    intercept = float(outcome_model.intercept_[0])
    slope = float(outcome_model.coef_[0, 0])
    if not (np.isfinite(intercept) and np.isfinite(slope) and 0 < slope < 10):
        raise RuntimeError(
            f"Non-transportable strict-LOO MNAR recalibration: intercept={intercept}, slope={slope}"
        )
    strict_rows["ipw"] = ipw
    strict_rows["strict_loo_ipw_mar_probability"] = expit(intercept + slope * predictor.ravel())
    strict_rows[
        [
            "context_id",
            "dtxsid",
            "latin_name",
            "effect_family",
            "p_context_loo_x95",
            "p_strict_loo",
            "propensity_oof",
            "ipw",
            "strict_loo_ipw_mar_probability",
        ]
    ].to_csv(output_dir / "strict_loo_mnar_weighted_context_predictions.csv.gz", index=False, compression="gzip")

    with np.load(paths["strict_matrix"], allow_pickle=True) as matrix:
        probability = matrix["p"].astype(float)
        matrix_species = matrix["species"].astype(str).tolist()
        chemical_axis = matrix["chemicals"].astype(str).tolist()
    if matrix_species != species.latin_name.astype(str).tolist():
        raise RuntimeError("Candidate order differs between strict-LOO MNAR inputs and probability matrix")
    chemical_index = {chemical: index for index, chemical in enumerate(chemical_axis)}
    priority = pd.read_csv(paths["priority"])
    priority["DTXSID"] = priority.DTXSID.astype(str)
    priority = priority[priority.DTXSID.isin(chemical_index)].copy()
    priority = priority.sort_values("DTXSID", kind="mergesort")
    positions = np.asarray([chemical_index[value] for value in priority.DTXSID], dtype=int)
    weights = priority.national_weight.to_numpy(float)
    weights /= weights.sum()
    strict_priority = probability[:, positions]

    freeze = json.loads(paths["numeric_freeze"].read_text(encoding="utf-8"))
    official_names = [str(name) for name in freeze["strict_loo_primary"]["top5_order"]]
    official_indices = [matrix_species.index(name) for name in official_names]
    rho_frame = pd.read_csv(paths["rho"])
    rho = float(
        rho_frame.loc[np.isclose(rho_frame.protection_target_x, 0.95), "strict_loo_rho_working"].iloc[0]
    )
    official_coverage = expected_capture(strict_priority, official_indices, weights, rho)
    frozen_coverage = float(freeze["strict_loo_primary"]["final_expected_capture_by_target"]["0.95"])
    if not np.isclose(official_coverage, frozen_coverage, rtol=0, atol=1e-9):
        raise RuntimeError(
            f"Strict-LOO core coverage mismatch: calculated={official_coverage}, frozen={frozen_coverage}"
        )

    mar_probability = expit(intercept + slope * logit(np.clip(strict_priority, 1e-5, 1 - 1e-5)))
    scenario_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for odds_ratio in sorted(set(odds_ratios)):
        scenario_probability = apply_selection_odds_ratio(mar_probability, odds_ratio)
        selected = deterministic_greedy(scenario_probability, weights, matrix_species, 5)
        names = [matrix_species[index] for index in selected]
        coverage = expected_capture(scenario_probability, selected, weights, rho)
        overlap = len(set(names) & set(official_names)) / len(set(names) | set(official_names))
        scenario_rows.append(
            {
                "odds_ratio_tail_positive_testing": odds_ratio,
                "coverage_copula": coverage,
                "coverage_delta_vs_strict_loo_core": coverage - official_coverage,
                "top5_jaccard_vs_strict_loo_core": overlap,
                "top5_member_changed_vs_strict_loo_core": set(names) != set(official_names),
                "top5_rank_order_changed_vs_strict_loo_core": names != official_names,
                "strict_loo_core_coverage": official_coverage,
                "rho_strict_loo": rho,
            }
        )
        sequence_rows.extend(
            {
                "odds_ratio_tail_positive_testing": odds_ratio,
                "rank": rank,
                "latin_name": name,
                "scenario": "strict-LOO MNAR sensitivity; primary core unchanged",
            }
            for rank, name in enumerate(names, 1)
        )
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(output_dir / "strict_loo_mnar_sensitivity_scenarios.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(output_dir / "strict_loo_mnar_top5_sequences.csv", index=False)
    mar = scenarios.loc[np.isclose(scenarios.odds_ratio_tail_positive_testing, 1.0)].iloc[0]
    mar_selected = deterministic_greedy(mar_probability, weights, matrix_species, 5)
    mar_expected = expected_capture(mar_probability, mar_selected, weights, rho)
    if not np.isclose(float(mar.coverage_copula), mar_expected, rtol=0, atol=1e-12):
        raise RuntimeError("OR = 1 MNAR scenario does not reproduce the IPW/MAR anchor")
    member_flip = scenarios.loc[
        scenarios.top5_member_changed_vs_strict_loo_core,
        "odds_ratio_tail_positive_testing",
    ]
    return {
        "seed": seed,
        "target": "lower-5% strict focal-species leave-one-out measured-tail probability",
        "propensity_features_excluding_tail_outcome": numeric + categorical,
        "sampled_pairs": int(len(sampled)),
        "observed_pairs": int(len(observed_pairs)),
        "negative_pairs": int(len(negatives)),
        "strict_loo_context_rows": int(len(strict_rows)),
        "cross_fitted_propensity_auc": propensity_auc,
        "weight_clip_q01": float(lower_clip),
        "weight_clip_q99": float(upper_clip),
        "effective_outcome_sample_size": float(ipw.sum() ** 2 / np.square(ipw).sum()),
        "outcome_predictor": "logit of strict-LOO direct species-by-chemical-by-effect probability",
        "ipw_recalibration": {"intercept": intercept, "slope": slope},
        "strict_loo_rho": rho,
        "strict_loo_core_coverage": official_coverage,
        "or1_expected_capture": float(mar.coverage_copula),
        "or1_matches_ipw_mar_anchor": True,
        "top5_member_flip_threshold_within_tested_or_range": (
            None if member_flip.empty else float(member_flip.iloc[0])
        ),
        "top5_member_stability_interpretation": (
            "A null threshold means that the five selected members remained unchanged over the explicitly tested odds-ratio range; rank-order changes are reported separately."
        ),
        "odds_ratio_interpretation": (
            "OR > 1 assumes true tail-positive species-by-chemical-by-effect cells were more likely to be directly tested, "
            "lowering population tail probability after correction; OR < 1 represents the reverse sensitivity direction."
        ),
        "boundary": (
            "MNAR is not identifiable from these data. The scenarios bound sensitivity to explicit testing-selection assumptions "
            "around the strict-LOO core; they do not replace the primary matrix or predict unmeasured absolute toxicity."
        ),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    revision_root = args.revision_root.resolve()
    output_dir = args.output_dir.resolve()
    if revision_root not in output_dir.parents:
        raise RuntimeError(f"Output directory must remain inside the revision workspace: {output_dir}")
    if args.profile_draws < 100:
        raise ValueError("Profile propagation requires at least 100 draws")
    odds_ratios = [float(value) for value in args.odds_ratios.split(",")]
    if not odds_ratios or any(value <= 0 for value in odds_ratios):
        raise ValueError("All MNAR odds ratios must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = prepare_paths(project_root, revision_root)

    profile = run_profile(paths, output_dir, args.profile_draws, args.seed)
    mnar = run_mnar(paths, output_dir, odds_ratios, args.seed, args.negative_ratio)
    manifest = {
        "schema_version": 1,
        "analysis_layer": "strict_focal_species_leave_one_out_robustness",
        "dependency_order": [
            "approved strict-LOO context and species-chemical probability matrices",
            "record-level profile-likelihood propagation",
            "strict-LOO MNAR testing-selection sensitivity",
        ],
        "profile": profile,
        "mnar": mnar,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "strict_loo_robustness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_metrics = pd.read_csv(output_dir / "strict_loo_framework_profile_metrics.csv")
    scenarios = pd.read_csv(output_dir / "strict_loo_mnar_sensitivity_scenarios.csv")
    interval_columns = [
        ("median_ratio_q025", "median_ratio_q50", "median_ratio_q975"),
        ("spearman_r_q025", "spearman_r_q50", "spearman_r_q975"),
        ("within_factor2_q025", "within_factor2_q50", "within_factor2_q975"),
        ("false_safe_factor2_rate_q025", "false_safe_factor2_rate_q50", "false_safe_factor2_rate_q975"),
    ]
    interval_ordered = all(
        bool(
            (
                (profile_metrics[lower] <= profile_metrics[middle])
                & (profile_metrics[middle] <= profile_metrics[upper])
            ).all()
        )
        for lower, middle, upper in interval_columns
    )
    gate = {
        "status": "PASS",
        "profile_interval_ordered": interval_ordered,
        "profile_contexts": int(profile["profile_contexts"]),
        "mnar_coverage_bounded_0_1": bool(scenarios.coverage_copula.between(0, 1).all()),
        "mnar_odds_ratio_grid": scenarios.odds_ratio_tail_positive_testing.astype(float).tolist(),
        "or1_matches_ipw_mar_anchor": bool(mnar["or1_matches_ipw_mar_anchor"]),
        "strict_loo_core_coverage": float(mnar["strict_loo_core_coverage"]),
        "top5_member_flip_threshold_within_tested_or_range": mnar[
            "top5_member_flip_threshold_within_tested_or_range"
        ],
    }
    if not all(
        [
            gate["profile_interval_ordered"],
            gate["mnar_coverage_bounded_0_1"],
            gate["or1_matches_ipw_mar_anchor"],
        ]
    ):
        gate["status"] = "FAIL"
    (output_dir / "gate_strict_loo_robustness.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if gate["status"] != "PASS":
        raise RuntimeError(f"Strict-LOO robustness gate failed: {gate}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "profiled_cells": profile["profiled_cells"],
                "profile_contexts": profile["profile_contexts"],
                "mnar_observed_pairs": mnar["observed_pairs"],
                "strict_loo_core_coverage": mnar["strict_loo_core_coverage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
