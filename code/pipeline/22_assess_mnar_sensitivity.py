#!/usr/bin/env python3
"""Test-observation propensity and MNAR sensitivity analysis for measured-tail panels.

The propensity model deliberately excludes tail outcomes.  MNAR enters only as
an explicit, user-visible odds-ratio sensitivity parameter after MAR/IPW
calibration; this module never modifies the official v16.5 probability matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--odds-ratios", default="0.25,0.5,1,2,4")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--negative-ratio", type=int, default=10)
    return parser.parse_args()


def build_features(species: pd.DataFrame, targets: pd.DataFrame, species_index: np.ndarray, target_index: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
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
    return frame


def independent_greedy(probability: np.ndarray, weights: np.ndarray, species: list[str], k: int = 5) -> list[int]:
    selected: list[int] = []
    residual = np.ones(probability.shape[1], dtype=float)
    remaining = np.ones(probability.shape[0], dtype=bool)
    for _ in range(k):
        gains = (probability * residual[None, :] * weights[None, :]).sum(axis=1)
        gains[~remaining] = -np.inf
        best = np.nanmax(gains)
        tied = np.flatnonzero(np.isclose(gains, best))
        chosen = min(tied.tolist(), key=lambda index: species[index])
        selected.append(chosen)
        remaining[chosen] = False
        residual *= 1 - probability[chosen]
    return selected


def copula_coverage(probability: np.ndarray, selected: list[int], weights: np.ndarray, rho: float) -> float:
    nodes, node_weights = np.polynomial.hermite.hermgauss(24)
    latent = nodes * np.sqrt(2)
    weights_gh = node_weights / np.sqrt(np.pi)
    thresholds = norm.ppf(np.clip(probability[selected], 1e-6, 1 - 1e-6))
    denominator = np.sqrt(max(1 - rho, 1e-9))
    no_event = np.ones((len(latent), probability.shape[1]), dtype=float)
    for threshold in thresholds:
        conditional = norm.cdf((threshold[None, :] - np.sqrt(max(rho, 0)) * latent[:, None]) / denominator)
        no_event *= 1 - conditional
    union = 1 - (weights_gh[:, None] * no_event).sum(axis=0)
    return float(np.dot(weights, union))


def main() -> None:
    args = parse_args()
    odds_ratios = [float(value) for value in args.odds_ratios.split(",")]
    if not odds_ratios or any(value <= 0 for value in odds_ratios):
        raise ValueError("All MNAR odds ratios must be positive")
    root = Path(args.root)
    out = root / "results" / "validation" / "mnar_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    species = pd.read_csv(root / "results/probability/candidate_universe_locked.csv")
    species = species.reset_index(names="species_index")
    species_lookup = dict(zip(species.latin_name.astype(str), species.species_index.astype(int)))
    targets = pd.read_csv(root / "results/probability/protective_chemical_effect_targets.csv")
    chemistry = pd.read_csv(root / "results/chemistry/chemical_moa_form_master.csv.gz", low_memory=False)
    chemistry["DTXSID"] = chemistry.DTXSID.astype(str)
    form_column = "chemical_form_class" if "chemical_form_class" in chemistry else "chemical_form"
    moa_column = "functional_moa" if "functional_moa" in chemistry else "primary_moa"
    targets = targets.merge(
        chemistry[["DTXSID", form_column, moa_column]], left_on="dtxsid", right_on="DTXSID", how="left"
    ).rename(columns={form_column: "chemical_form", moa_column: "functional_moa"})
    targets["chemical_form"] = targets.chemical_form.fillna("unresolved")
    targets["functional_moa"] = targets.functional_moa.fillna("unresolved")
    priority = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    priority_ids = set(priority.DTXSID.astype(str))
    targets["priority_chemical"] = targets.dtxsid.astype(str).isin(priority_ids)
    targets = targets.reset_index(names="target_index")
    target_lookup = {(str(row.dtxsid), str(row.effect_family)): int(row.target_index) for row in targets.itertuples(index=False)}

    predictions = pd.read_csv(root / "results/validation/moa_simplified_oof_predictions.csv.gz", low_memory=False)
    predictions = predictions[predictions.protection_target_x.eq(0.95)].copy()
    predictions["species_index"] = predictions.latin_name.astype(str).map(species_lookup)
    predictions["target_index"] = [target_lookup.get((str(dtxsid), str(effect))) for dtxsid, effect in zip(predictions.dtxsid, predictions.effect_family)]
    predictions = predictions.dropna(subset=["species_index", "target_index"]).copy()
    predictions[["species_index", "target_index"]] = predictions[["species_index", "target_index"]].astype(int)
    observed_pairs = predictions[["species_index", "target_index"]].drop_duplicates().copy()
    observed_pairs["observed"] = 1
    observed_key = set((observed_pairs.species_index * len(targets) + observed_pairs.target_index).astype(np.int64))
    total_pairs = len(species) * len(targets)
    n_negative = min(args.negative_ratio * len(observed_pairs), total_pairs - len(observed_pairs))
    negative_keys: set[int] = set()
    while len(negative_keys) < n_negative:
        candidate = rng.integers(0, total_pairs, size=max(10000, 2 * (n_negative - len(negative_keys))), endpoint=False)
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
    sampled["sample_weight"] = np.where(sampled.observed.eq(1), 1.0, (total_pairs - len(observed_pairs)) / max(n_negative, 1))
    features = build_features(species, targets, sampled.species_index.to_numpy(int), sampled.target_index.to_numpy(int))
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
        model.fit(features.iloc[train], sampled.observed.iloc[train], model__sample_weight=sampled.sample_weight.iloc[train])
        oof[test] = model.predict_proba(features.iloc[test])[:, 1]
    propensity_auc = float(roc_auc_score(sampled.observed, oof, sample_weight=sampled.sample_weight))
    propensity_by_pair = sampled.loc[sampled.observed.eq(1), ["species_index", "target_index"]].copy()
    propensity_by_pair["propensity_oof"] = oof[sampled.observed.to_numpy(bool)]
    pair_key_to_propensity = dict(zip((propensity_by_pair.species_index * len(targets) + propensity_by_pair.target_index).astype(np.int64), propensity_by_pair.propensity_oof))
    predictions["pair_key"] = predictions.species_index * len(targets) + predictions.target_index
    predictions["propensity_oof"] = predictions.pair_key.map(pair_key_to_propensity).clip(1e-4, 1 - 1e-4)
    raw_weight = 1 / predictions.propensity_oof.to_numpy(float)
    lower, upper = np.quantile(raw_weight, [0.01, 0.99])
    ipw = np.clip(raw_weight, lower, upper)
    ipw /= ipw.mean()
    x = logit(np.clip(predictions.taxonomy_oof.to_numpy(float), 1e-5, 1 - 1e-5)).reshape(-1, 1)
    outcome_model = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
    outcome_model.fit(x, predictions.tail_binary.to_numpy(int), sample_weight=ipw)
    intercept = float(outcome_model.intercept_[0])
    slope = float(outcome_model.coef_[0, 0])
    p_mar_oof = expit(intercept + slope * x.ravel())
    predictions["ipw"] = ipw
    predictions["taxonomy_ipw_mar_probability"] = p_mar_oof
    predictions[["dtxsid", "latin_name", "effect_family", "context_id", "tail_binary", "taxonomy_oof", "propensity_oof", "ipw", "taxonomy_ipw_mar_probability"]].to_csv(out / "mnar_weighted_oof_predictions.csv.gz", index=False, compression="gzip")

    matrix = np.load(root / "results/probability/species_chemical_tail_probability_x95.npz", allow_pickle=True)
    p = matrix["p"].astype(float)
    matrix_species = matrix["species"].astype(str).tolist()
    if matrix_species != species.latin_name.astype(str).tolist():
        raise RuntimeError("Candidate order differs between MNAR inputs and probability matrix")
    chemical_axis = matrix["chemicals"].astype(str).tolist()
    chemical_index = {chemical: index for index, chemical in enumerate(chemical_axis)}
    priority = priority[priority.DTXSID.astype(str).isin(chemical_index)].copy()
    priority["DTXSID"] = priority.DTXSID.astype(str)
    priority = priority.sort_values("DTXSID")
    chemical_positions = np.array([chemical_index[chemical] for chemical in priority.DTXSID], dtype=int)
    weights = priority.national_weight.to_numpy(float)
    weights /= weights.sum()
    p_priority = p[:, chemical_positions]
    manifest = json.loads((root / "results/panels/panel_probability_manifest.json").read_text(encoding="utf-8"))
    rho = float(manifest["dependence_audit_by_x"]["0.95"]["rho_working"])
    official = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    official = official[(official.universe.eq("priority")) & (official.protection_target_x.eq(0.95)) & (official.method.eq("data_driven"))].sort_values("rank").head(5)
    official_names = official.latin_name.astype(str).tolist()
    official_indices = [matrix_species.index(name) for name in official_names]
    official_coverage = copula_coverage(p_priority, official_indices, weights, rho)
    p_mar = expit(intercept + slope * logit(np.clip(p_priority, 1e-5, 1 - 1e-5)))
    scenario_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for odds_ratio in sorted(set(odds_ratios)):
        p_scenario = expit(logit(np.clip(p_mar, 1e-5, 1 - 1e-5)) - np.log(odds_ratio))
        selected = independent_greedy(p_scenario, weights, matrix_species, 5)
        names = [matrix_species[index] for index in selected]
        coverage = copula_coverage(p_scenario, selected, weights, rho)
        overlap = len(set(names) & set(official_names)) / len(set(names) | set(official_names))
        scenario_rows.append({
            "odds_ratio_tail_positive_testing": odds_ratio,
            "coverage_copula": coverage,
            "coverage_delta_vs_official": coverage - official_coverage,
            "top5_jaccard_vs_official": overlap,
            "top5_member_changed_vs_official": set(names) != set(official_names),
            "top5_rank_order_changed_vs_official": names != official_names,
            "official_core_coverage": official_coverage,
        })
        sequence_rows.extend(
            {
                "odds_ratio_tail_positive_testing": odds_ratio,
                "rank": rank,
                "latin_name": name,
                "scenario": "MNAR sensitivity only; not official core panel",
            }
            for rank, name in enumerate(names, 1)
        )
    scenario_frame = pd.DataFrame(scenario_rows)
    scenario_frame.to_csv(out / "mnar_sensitivity_scenarios.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(out / "mnar_top5_sequences.csv", index=False)
    mar = scenario_frame.loc[np.isclose(scenario_frame.odds_ratio_tail_positive_testing, 1.0)].iloc[0]
    member_flip = scenario_frame.loc[scenario_frame.top5_member_changed_vs_official, "odds_ratio_tail_positive_testing"]
    diagnostics = {
        "version": "v16.5",
        "seed": args.seed,
        "target": "lower-5% measured-tail probability only",
        "propensity_features_excluding_tail_outcome": numeric + categorical,
        "sampled_pairs": int(len(sampled)),
        "observed_pairs": int(len(observed_pairs)),
        "negative_pairs": int(len(negatives)),
        "cross_fitted_propensity_auc": propensity_auc,
        "weight_clip_q01": float(lower),
        "weight_clip_q99": float(upper),
        "effective_outcome_sample_size": float(ipw.sum() ** 2 / np.square(ipw).sum()),
        "ipw_recalibration": {"intercept": intercept, "slope": slope},
        "odds_ratio_interpretation": "OR > 1 assumes true tail-positive species×chemical×effect cells were more likely to be directly tested, lowering population tail probability after correction; OR < 1 represents the reverse sensitivity direction.",
        "or1_matches_mar": bool(np.isclose(float(mar.coverage_copula), copula_coverage(p_mar, independent_greedy(p_mar, weights, matrix_species, 5), weights, rho))),
        "top5_member_flip_threshold_within_tested_or_range": None if member_flip.empty else float(member_flip.iloc[0]),
        "top5_member_stability_interpretation": "A null threshold means no Top-5 member changed over the explicitly tested odds-ratio range; rank-order changes are reported separately.",
        "boundary": "MNAR is not identifiable from this dataset. These scenarios quantify robustness to explicit selection assumptions and do not replace the official taxonomy/direct-evidence HC5 core matrix or predict absolute toxicity.",
    }
    (out / "mnar_sensitivity_manifest.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"observed_pairs": len(observed_pairs), "scenarios": len(scenario_frame), "propensity_auc": propensity_auc}, ensure_ascii=False))


if __name__ == "__main__":
    main()
