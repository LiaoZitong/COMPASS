#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr


PROTECTIVE = [
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
]
TARGETS = (0.80, 0.90, 0.95)
SEED = 20260622


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_probability_module(root: Path):
    path = root / "src/pipeline/06_estimate_uncalibrated_tail_probabilities.py"
    spec = importlib.util.spec_from_file_location("r0_probability_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def leave_one_out_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sample mean and sample SD after excluding each value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    total = float(values.sum())
    total_sq = float(np.square(values).sum())
    remaining_n = n - 1
    means = (total - values) / remaining_n
    centered_ss = (total_sq - np.square(values)) - remaining_n * np.square(means)
    variances = np.maximum(centered_ss / (remaining_n - 1), 0.0)
    return means, np.sqrt(variances)


def prepare_context_rows(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = model["mle_status"].astype(str).str.startswith("identified") & np.isfinite(
        pd.to_numeric(model["mu_log10_umol_L"], errors="coerce")
    )
    protective_all = model[valid & model["effect_family"].isin(PROTECTIVE)].copy()
    protective_all["context_id"] = protective_all[
        ["dtxsid", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]
    ].astype(str).agg("|".join, axis=1)

    context = (
        protective_all.groupby("context_id", as_index=False)
        .agg(
            dtxsid=("dtxsid", "first"),
            effect_family=("effect_family", "first"),
            endpoint_band_v16=("endpoint_band_v16", "first"),
            duration_window_v16=("duration_window_v16", "first"),
            medium_family=("medium_family", "first"),
            n_species=("latin_name", "nunique"),
            ssd_mean=("mu_log10_umol_L", "mean"),
            ssd_sd=("mu_log10_umol_L", "std"),
            n_records=("n_records", "sum"),
            n_left=("n_left", "sum"),
            n_right=("n_right", "sum"),
            n_interval=("n_interval", "sum"),
            n_approx=("n_approx", "sum"),
        )
    )
    context = context[
        context["n_species"].ge(5)
        & context["ssd_sd"].notna()
        & context["ssd_sd"].gt(0.05)
    ].copy()
    context["censored_fraction"] = (
        context[["n_left", "n_right", "n_interval"]].sum(axis=1)
        / context["n_records"].clip(lower=1)
    )
    q1, q2 = context["ssd_sd"].quantile([0.33, 0.67])
    context["ssd_width_class"] = np.select(
        [context["ssd_sd"].le(q1), context["ssd_sd"].le(q2)],
        ["narrow", "moderate"],
        default="wide",
    )
    context["original_n_stratum"] = np.select(
        [context["n_species"].eq(5), context["n_species"].between(6, 9)],
        ["n5", "n6_9"],
        default="n10_plus",
    )
    context["censoring_stratum"] = pd.cut(
        context["censored_fraction"],
        bins=[-1e-12, 0.0, 0.25, 0.50, 1.0],
        labels=["none", "gt0_to_25pct", "gt25_to_50pct", "gt50pct"],
        include_lowest=True,
    ).astype(str)

    rows = protective_all[protective_all["context_id"].isin(context["context_id"])].merge(
        context,
        on=[
            "context_id",
            "dtxsid",
            "effect_family",
            "endpoint_band_v16",
            "duration_window_v16",
            "medium_family",
        ],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_context"),
    )

    duplicate = rows.duplicated(["context_id", "latin_name"], keep=False)
    if duplicate.any():
        raise AssertionError("Model cells are not unique by context_id and latin_name")

    loo_parts = []
    for _context_id, group in rows.groupby("context_id", sort=False):
        means, sds = leave_one_out_stats(group["mu_log10_umol_L"].to_numpy(float))
        part = pd.DataFrame(
            {
                "row_index": group.index.to_numpy(),
                "loo_ssd_mean": means,
                "loo_ssd_sd": sds,
            }
        )
        loo_parts.append(part)
    loo = pd.concat(loo_parts, ignore_index=True).set_index("row_index")
    rows["loo_ssd_mean"] = loo["loo_ssd_mean"].reindex(rows.index).to_numpy()
    rows["loo_ssd_sd"] = loo["loo_ssd_sd"].reindex(rows.index).to_numpy()
    rows["loo_n_species"] = rows["n_species"].astype(int) - 1
    rows["strict_loo_eligible"] = (
        rows["n_species"].ge(6)
        & rows["loo_n_species"].ge(5)
        & rows["loo_ssd_sd"].notna()
        & rows["loo_ssd_sd"].gt(0.05)
    )
    rows["n5_diagnostic"] = rows["n_species"].eq(5)
    return protective_all, rows


def add_target_probabilities(rows: pd.DataFrame, target: float) -> pd.DataFrame:
    out = rows.copy()
    q0 = 1.0 - target
    zq = norm.ppf(q0)
    out["original_cutoff_log10"] = out["ssd_mean"] + zq * out["ssd_sd"]
    out["loo_cutoff_log10"] = out["loo_ssd_mean"] + zq * out["loo_ssd_sd"]
    original_uncertainty = np.sqrt(
        out["se_mu_log10"].fillna(0.35).astype(float) ** 2
        + (out["ssd_sd"] / np.sqrt(out["n_species"])) ** 2
    ).clip(lower=0.12)
    loo_uncertainty = np.sqrt(
        out["se_mu_log10"].fillna(0.35).astype(float) ** 2
        + (out["loo_ssd_sd"] / np.sqrt(out["loo_n_species"])) ** 2
    ).clip(lower=0.12)
    out["p_context_original"] = norm.cdf(
        (out["original_cutoff_log10"] - out["mu_log10_umol_L"]) / original_uncertainty
    )
    out["p_context_loo"] = norm.cdf(
        (out["loo_cutoff_log10"] - out["mu_log10_umol_L"]) / loo_uncertainty
    )
    out["tail_binary_original"] = (
        out["mu_log10_umol_L"] <= out["original_cutoff_log10"]
    ).astype(float)
    out["tail_binary_loo"] = (
        out["mu_log10_umol_L"] <= out["loo_cutoff_log10"]
    ).astype(float)
    out["p_context_delta"] = out["p_context_loo"] - out["p_context_original"]
    return out


def build_probability_matrices(
    mm: pd.DataFrame,
    protective_all: pd.DataFrame,
    candidates: pd.DataFrame,
    traits: pd.DataFrame,
    chem: pd.DataFrame,
    targets: pd.DataFrame,
    chemicals: list[str],
    effect_probability_table: pd.DataFrame,
    calibration: dict[str, object],
    target: float,
    probability_module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q0 = 1.0 - target
    candidates = candidates.reset_index(drop=True).copy()
    species = candidates["latin_name"].astype(str).tolist()
    species_index = {name: index for index, name in enumerate(species)}
    effects = sorted(PROTECTIVE)
    effect_index = {name: index for index, name in enumerate(effects)}

    chem = chem.copy()
    chem["DTXSID"] = chem["DTXSID"].astype(str)
    chem["mechanism_group"] = np.where(
        chem["primary_moa"].fillna("MOA_UNRESOLVED").ne("MOA_UNRESOLVED"),
        chem["primary_moa"],
        "FORM::" + chem["chemical_form_class"].fillna("unresolved").astype(str),
    )
    chem_mech = chem.set_index("DTXSID")["mechanism_group"].to_dict()
    mechanisms = sorted({chem_mech.get(c, "MOA_UNRESOLVED") for c in chemicals})
    mechanism_index = {name: index for index, name in enumerate(mechanisms)}

    mm = mm.copy()
    mm["mechanism_group"] = mm["dtxsid"].map(chem_mech).fillna("MOA_UNRESOLVED")
    direct = (
        mm[mm["latin_name"].isin(species)]
        .groupby(["latin_name", "dtxsid", "effect_family"])["p_context"]
        .agg(["sum", "count"])
        .reset_index()
    )
    direct_key = {
        (str(row.latin_name), str(row.dtxsid), str(row.effect_family)): (
            float(row["sum"]),
            int(row["count"]),
        )
        for _, row in direct.iterrows()
    }

    evidence_moa = (
        mm.groupby(
            ["latin_name", "genus", "family", "class", "mechanism_group", "effect_family"]
        )["p_context"]
        .agg(["sum", "count"])
        .reset_index()
    )
    evidence_tax = (
        mm.groupby(["latin_name", "genus", "family", "class", "effect_family"])["p_context"]
        .agg(["sum", "count"])
        .reset_index()
    )
    maps_moa: dict[str, dict[tuple[object, ...], tuple[float, int]]] = {}
    maps_tax: dict[str, dict[tuple[object, ...], tuple[float, int]]] = {}
    for level in ["latin_name", "genus", "family", "class"]:
        grouped_moa = (
            evidence_moa.groupby([level, "mechanism_group", "effect_family"])[["sum", "count"]]
            .sum()
            .reset_index()
        )
        maps_moa[level] = {
            (row[level], row.mechanism_group, row.effect_family): (
                float(row["sum"]),
                int(row["count"]),
            )
            for _, row in grouped_moa.iterrows()
        }
        grouped_tax = (
            evidence_tax.groupby([level, "effect_family"])[["sum", "count"]]
            .sum()
            .reset_index()
        )
        maps_tax[level] = {
            (row[level], row.effect_family): (float(row["sum"]), int(row["count"]))
            for _, row in grouped_tax.iterrows()
        }

    s_count = len(species)
    prior_moa = np.full((s_count, len(mechanisms), len(effects)), q0, dtype=float)
    rel_moa = np.zeros_like(prior_moa)
    prior_tax = np.full((s_count, len(effects)), q0, dtype=float)
    rel_tax = np.zeros_like(prior_tax)
    level_caps = {"latin_name": 0.80, "genus": 0.50, "family": 0.30, "class": 0.15}
    level_weights = [("latin_name", 4.0), ("genus", 2.0), ("family", 1.0), ("class", 0.5)]
    for i, row in candidates.iterrows():
        for effect, effect_i in effect_index.items():
            values: list[float] = []
            weights: list[float] = []
            reliabilities: list[float] = []
            for level, base_weight in level_weights:
                key = row.latin_name if level == "latin_name" else row[level]
                value_sum, n = maps_tax[level].get((key, effect), (0.0, 0))
                if n:
                    values.append((value_sum + 10 * q0) / (n + 10))
                    weights.append(base_weight * min(n, 20))
                    reliabilities.append(level_caps[level] * (1 - np.exp(-n / 10.0)))
            if values:
                prior_tax[i, effect_i] = np.average(values, weights=weights)
                rel_tax[i, effect_i] = max(reliabilities)

            for mechanism, mechanism_i in mechanism_index.items():
                values = []
                weights = []
                reliabilities = []
                for level, base_weight in level_weights:
                    key = row.latin_name if level == "latin_name" else row[level]
                    value_sum, n = maps_moa[level].get((key, mechanism, effect), (0.0, 0))
                    if n:
                        values.append((value_sum + 10 * q0) / (n + 10))
                        weights.append(base_weight * min(n, 20))
                        reliabilities.append(level_caps[level] * (1 - np.exp(-n / 10.0)))
                if values:
                    prior_moa[i, mechanism_i, effect_i] = np.average(values, weights=weights)
                    rel_moa[i, mechanism_i, effect_i] = max(reliabilities)

    neighbors = probability_module.trait_neighbors(traits, species)
    before_moa = prior_moa.copy()
    before_moa_rel = rel_moa.copy()
    before_tax = prior_tax.copy()
    before_tax_rel = rel_tax.copy()
    for i, neighbor_indices in enumerate(neighbors):
        if len(neighbor_indices) >= 3:
            prior_moa[i] = 0.70 * prior_moa[i] + 0.30 * np.nanmean(before_moa[neighbor_indices], axis=0)
            rel_moa[i] = np.maximum(
                0.70 * before_moa_rel[i], 0.10 * np.nanmean(before_moa_rel[neighbor_indices], axis=0)
            )
            prior_tax[i] = 0.70 * prior_tax[i] + 0.30 * np.nanmean(before_tax[neighbor_indices], axis=0)
            rel_tax[i] = np.maximum(
                0.70 * before_tax_rel[i], 0.10 * np.nanmean(before_tax_rel[neighbor_indices], axis=0)
            )

    by_target_cal = calibration.get("by_protection_target", {})
    target_cal = by_target_cal.get(str(target)) or by_target_cal.get(f"{target:.2f}") or {}
    validated_moa_groups = set(target_cal.get("validated_moa_groups", []))

    targets = targets.reset_index(drop=True).copy()
    target_mechanism_indices = np.array(
        [mechanism_index[str(value)] for value in targets["mechanism_group"]], dtype=int
    )
    target_effect_indices = np.array(
        [effect_index[str(value)] for value in targets["effect_family"]], dtype=int
    )
    p_target = np.empty((s_count, len(targets)), dtype=np.float32)
    r_target = np.empty_like(p_target)
    n_target = np.zeros((s_count, len(targets)), dtype=np.uint16)
    for target_i, row in targets.iterrows():
        mechanism_i = target_mechanism_indices[target_i]
        effect_i = target_effect_indices[target_i]
        group_key = f"{row.mechanism_group}||{row.effect_family}"
        use_moa = group_key in validated_moa_groups
        if use_moa:
            prior_col = prior_moa[:, mechanism_i, effect_i]
            reliability_prior = rel_moa[:, mechanism_i, effect_i]
        else:
            prior_col = prior_tax[:, effect_i]
            reliability_prior = rel_tax[:, effect_i]
        p_col = q0 + reliability_prior * (prior_col - q0)
        r_col = reliability_prior.copy()
        direct_n_col = np.zeros(s_count, dtype=np.uint16)
        for species_i, species_name in enumerate(species):
            value_sum, n = direct_key.get(
                (species_name, str(row.dtxsid), str(row.effect_family)), (0.0, 0)
            )
            if n:
                raw = (value_sum + 2 * prior_col[species_i]) / (n + 2)
                reliability_direct = 1 - np.exp(-n / 3.0)
                p_col[species_i] = q0 + reliability_direct * (raw - q0)
                r_col[species_i] = reliability_direct
                direct_n_col[species_i] = n
        if target_cal:
            calibration_key = "moa_taxonomy_calibration" if use_moa else "taxonomy_calibration"
            parameters = target_cal.get(calibration_key)
            if parameters is not None:
                missing = direct_n_col == 0
                logit = np.log(
                    np.clip(p_col, 1e-6, 1 - 1e-6)
                    / np.clip(1 - p_col, 1e-6, 1 - 1e-6)
                )
                calibrated = 1 / (
                    1
                    + np.exp(
                        -(
                            float(parameters["intercept"])
                            + float(parameters["coefficient"]) * logit
                        )
                    )
                )
                p_col[missing] = calibrated[missing]
        p_target[:, target_i] = np.clip(p_col, 0.001, 0.999).astype(np.float32)
        r_target[:, target_i] = r_col.astype(np.float32)
        n_target[:, target_i] = direct_n_col

    effect_weight_map = {
        mechanism: group.set_index("effect_family")["effect_probability_within_moa"]
        .reindex(effects)
        .to_numpy(float)
        for mechanism, group in effect_probability_table.groupby("mechanism_group")
    }
    target_lookup = {
        (str(row.dtxsid), str(row.effect_family)): int(index)
        for index, row in targets.iterrows()
    }
    p_chemical = np.empty((s_count, len(chemicals)), dtype=np.float32)
    r_chemical = np.empty_like(p_chemical)
    n_chemical = np.zeros((s_count, len(chemicals)), dtype=np.uint16)
    for chemical_i, chemical in enumerate(chemicals):
        mechanism = chem_mech.get(chemical, "MOA_UNRESOLVED")
        mechanism_i = mechanism_index[mechanism]
        effect_weights = effect_weight_map.get(
            mechanism, np.full(len(effects), 1.0 / len(effects))
        )
        chemical_p = np.zeros((s_count, len(effects)), dtype=float)
        chemical_r = np.zeros_like(chemical_p)
        chemical_n = np.zeros((s_count, len(effects)), dtype=np.uint16)
        for effect, effect_i in effect_index.items():
            key = (chemical, effect)
            if key in target_lookup:
                index = target_lookup[key]
                chemical_p[:, effect_i] = p_target[:, index]
                chemical_r[:, effect_i] = r_target[:, index]
                chemical_n[:, effect_i] = n_target[:, index]
                continue
            group_key = f"{mechanism}||{effect}"
            use_moa = group_key in validated_moa_groups
            if use_moa:
                prior_col = prior_moa[:, mechanism_i, effect_i]
                reliability_col = rel_moa[:, mechanism_i, effect_i]
                calibration_key = "moa_taxonomy_calibration"
            else:
                prior_col = prior_tax[:, effect_i]
                reliability_col = rel_tax[:, effect_i]
                calibration_key = "taxonomy_calibration"
            borrowed = q0 + reliability_col * (prior_col - q0)
            parameters = target_cal.get(calibration_key) if target_cal else None
            if parameters is not None:
                logit = np.log(
                    np.clip(borrowed, 1e-6, 1 - 1e-6)
                    / np.clip(1 - borrowed, 1e-6, 1 - 1e-6)
                )
                borrowed = 1 / (
                    1
                    + np.exp(
                        -(
                            float(parameters["intercept"])
                            + float(parameters["coefficient"]) * logit
                        )
                    )
                )
            chemical_p[:, effect_i] = np.clip(borrowed, 0.001, 0.999)
            chemical_r[:, effect_i] = reliability_col
        p_chemical[:, chemical_i] = (chemical_p @ effect_weights).astype(np.float32)
        r_chemical[:, chemical_i] = (chemical_r @ effect_weights).astype(np.float32)
        n_chemical[:, chemical_i] = np.clip(
            chemical_n.astype(np.uint32).sum(axis=1), 0, 65535
        ).astype(np.uint16)

    return p_target, r_target, n_target, p_chemical, r_chemical, n_chemical


def expected_capture(probability_module, p: np.ndarray, seq: list[int], weights: np.ndarray, rho: float) -> float:
    q = probability_module.copula_union(p[np.asarray(seq, dtype=int)], rho)
    return float(weights @ q)


def make_strata_summary(rows: pd.DataFrame, target: float) -> pd.DataFrame:
    strict = rows[rows["strict_loo_eligible"]].copy()
    output: list[dict[str, object]] = []
    definitions = {
        "overall": pd.Series("all", index=strict.index),
        "original_n_stratum": strict["original_n_stratum"],
        "censoring_stratum": strict["censoring_stratum"],
        "ssd_width_class": strict["ssd_width_class"],
    }
    for dimension, values in definitions.items():
        for level, group in strict.groupby(values, sort=True):
            delta = group["p_context_delta"].to_numpy(float)
            absolute = np.abs(delta)
            output.append(
                {
                    "protection_target_x": target,
                    "stratification": dimension,
                    "stratum": str(level),
                    "n_species_context_rows": len(group),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "mean_absolute_delta": float(absolute.mean()),
                    "median_absolute_delta": float(np.median(absolute)),
                    "q95_absolute_delta": float(np.quantile(absolute, 0.95)),
                    "max_absolute_delta": float(absolute.max()),
                }
            )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bhbt-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.bhbt_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    probability_module = load_probability_module(root)

    paths = {
        "model": root / "results/toxicity/censored_model_cells.csv.gz",
        "candidates": root / "results/probability/candidate_universe_locked.csv",
        "targets": root / "results/probability/protective_chemical_effect_targets.csv",
        "traits": root / "results/chemistry/species_traits_taxonomy.csv.gz",
        "chem": root / "results/chemistry/chemical_moa_form_master.csv.gz",
        "effect_probability": root / "results/probability/moa_effect_family_probability.csv",
        "all_chemicals": root / "results/weights/all_protective_chemicals_uniform.csv",
        "priority": root / "results/weights/national_priority_chemicals.csv",
        "calibration": root / "results/validation/probability_calibration_parameters.json",
        "panel_manifest": root / "results/panels/panel_probability_manifest.json",
    }
    for target in TARGETS:
        label = int(target * 100)
        paths[f"target_npz_x{label}"] = root / f"results/probability/species_protective_target_tail_probability_x{label}.npz"
        paths[f"chemical_npz_x{label}"] = root / f"results/probability/species_chemical_tail_probability_x{label}.npz"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AP02 input(s):\n" + "\n".join(missing))

    model = pd.read_csv(paths["model"], low_memory=False)
    candidates = pd.read_csv(paths["candidates"]).sort_values("latin_name").reset_index(drop=True)
    targets = pd.read_csv(paths["targets"]).reset_index(drop=True)
    traits = pd.read_csv(paths["traits"], low_memory=False)
    chem = pd.read_csv(paths["chem"], low_memory=False)
    effect_probability_table = pd.read_csv(paths["effect_probability"])
    all_chemicals = pd.read_csv(paths["all_chemicals"])["DTXSID"].astype(str).tolist()
    priority = pd.read_csv(paths["priority"])
    priority["DTXSID"] = priority["DTXSID"].astype(str)
    panel_manifest = json.loads(paths["panel_manifest"].read_text(encoding="utf-8"))
    calibration_file = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    calibration = panel_manifest.get("whole_chemical_probability_calibration")
    if not isinstance(calibration, dict):
        raise AssertionError(
            "The production panel manifest does not embed its probability calibration contract"
        )

    protective_all, context_rows = prepare_context_rows(model)
    species = candidates["latin_name"].astype(str).tolist()
    species_index = {name: index for index, name in enumerate(species)}
    target_ids = targets["target_id"].astype(str).to_numpy(object)

    input_manifest = {
        "schema_version": 1,
        "seed": SEED,
        "probability_calibration_source": (
            "results/panels/panel_probability_manifest.json::"
            "whole_chemical_probability_calibration"
        ),
        "standalone_calibration_file_used_for_computation": False,
        "standalone_vs_embedded_calibration_equal": calibration_file == calibration,
        "standalone_validated_moa_group_counts": {
            key: len(value.get("validated_moa_groups", []))
            for key, value in calibration_file.get("by_protection_target", {}).items()
        },
        "embedded_validated_moa_group_counts": {
            key: len(value.get("validated_moa_groups", []))
            for key, value in calibration.get("by_protection_target", {}).items()
        },
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
    (out / "ap02_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    context_output = context_rows[
        [
            "context_id",
            "dtxsid",
            "effect_family",
            "endpoint_band_v16",
            "duration_window_v16",
            "medium_family",
            "latin_name",
            "mu_log10_umol_L",
            "se_mu_log10",
            "n_species",
            "ssd_mean",
            "ssd_sd",
            "loo_n_species",
            "loo_ssd_mean",
            "loo_ssd_sd",
            "strict_loo_eligible",
            "n5_diagnostic",
            "original_n_stratum",
            "censored_fraction",
            "censoring_stratum",
            "ssd_width_class",
        ]
    ].copy()

    reconstruction: dict[str, dict[str, object]] = {}
    rank_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    rho_rows: list[dict[str, object]] = []
    strata_frames: list[pd.DataFrame] = []
    x95_gate: dict[str, object] | None = None
    optimized_sequences: dict[str, dict[float, list[str]]] = {"baseline": {}, "strict_loo": {}}

    for target in TARGETS:
        label = int(target * 100)
        rows_x = add_target_probabilities(context_rows, target)
        context_output[f"original_cutoff_log10_x{label}"] = rows_x["original_cutoff_log10"]
        context_output[f"loo_cutoff_log10_x{label}"] = rows_x["loo_cutoff_log10"]
        context_output[f"p_context_original_x{label}"] = rows_x["p_context_original"]
        context_output[f"p_context_loo_x{label}"] = rows_x["p_context_loo"]
        context_output[f"p_context_delta_x{label}"] = rows_x["p_context_delta"]
        strata_frames.append(make_strata_summary(rows_x, target))

        original_mm = rows_x.copy()
        original_mm["p_context"] = original_mm["p_context_original"]
        original_mm["tail_binary"] = original_mm["tail_binary_original"]
        strict_mm = rows_x[rows_x["strict_loo_eligible"]].copy()
        strict_mm["p_context"] = strict_mm["p_context_loo"]
        strict_mm["tail_binary"] = strict_mm["tail_binary_loo"]

        (
            rebuilt_target,
            _rebuilt_target_rel,
            rebuilt_target_n,
            rebuilt_chemical,
            _rebuilt_chemical_rel,
            rebuilt_chemical_n,
        ) = build_probability_matrices(
            original_mm,
            protective_all,
            candidates,
            traits,
            chem,
            targets,
            all_chemicals,
            effect_probability_table,
            calibration,
            target,
            probability_module,
        )
        with np.load(paths[f"target_npz_x{label}"], allow_pickle=True) as baseline_target_npz:
            baseline_target = np.asarray(baseline_target_npz["p"], dtype=np.float32)
            baseline_target_n = np.asarray(baseline_target_npz["direct_n"], dtype=np.uint16)
            baseline_target_species = baseline_target_npz["species"].astype(str).tolist()
            baseline_target_ids = baseline_target_npz["target_ids"].astype(str).tolist()
        with np.load(paths[f"chemical_npz_x{label}"], allow_pickle=True) as baseline_chemical_npz:
            baseline_chemical = np.asarray(baseline_chemical_npz["p"], dtype=np.float32)
            baseline_chemical_n = np.asarray(baseline_chemical_npz["direct_n"], dtype=np.uint16)
            baseline_chemical_species = baseline_chemical_npz["species"].astype(str).tolist()
            baseline_chemical_names = baseline_chemical_npz["chemicals"].astype(str).tolist()

        ordering_ok = (
            baseline_target_species == species
            and baseline_chemical_species == species
            and baseline_target_ids == target_ids.astype(str).tolist()
            and baseline_chemical_names == all_chemicals
        )
        target_max_abs = float(np.max(np.abs(rebuilt_target.astype(float) - baseline_target.astype(float))))
        chemical_max_abs = float(np.max(np.abs(rebuilt_chemical.astype(float) - baseline_chemical.astype(float))))
        target_n_match = bool(np.array_equal(rebuilt_target_n, baseline_target_n))
        chemical_n_match = bool(np.array_equal(rebuilt_chemical_n, baseline_chemical_n))
        reconstruction_pass = (
            ordering_ok
            and target_max_abs <= 2e-6
            and chemical_max_abs <= 2e-6
            and target_n_match
            and chemical_n_match
        )
        reconstruction[f"x{label}"] = {
            "ordering_match": ordering_ok,
            "target_probability_max_absolute_difference": target_max_abs,
            "chemical_probability_max_absolute_difference": chemical_max_abs,
            "target_direct_n_exact_match": target_n_match,
            "chemical_direct_n_exact_match": chemical_n_match,
            "pass": reconstruction_pass,
        }
        if not reconstruction_pass:
            target_delta = rebuilt_target.astype(float) - baseline_target.astype(float)
            chemical_delta = rebuilt_chemical.astype(float) - baseline_chemical.astype(float)
            diagnostic_rows: list[dict[str, object]] = []
            for matrix_name, delta, rebuilt, baseline, direct_n, column_names in [
                (
                    "protective_target",
                    target_delta,
                    rebuilt_target,
                    baseline_target,
                    baseline_target_n,
                    baseline_target_ids,
                ),
                (
                    "chemical",
                    chemical_delta,
                    rebuilt_chemical,
                    baseline_chemical,
                    baseline_chemical_n,
                    baseline_chemical_names,
                ),
            ]:
                flat_order = np.argsort(np.abs(delta), axis=None)[-100:][::-1]
                row_indices, col_indices = np.unravel_index(flat_order, delta.shape)
                for row_i, col_i in zip(row_indices, col_indices):
                    diagnostic_rows.append(
                        {
                            "protection_target_x": target,
                            "matrix": matrix_name,
                            "latin_name": species[int(row_i)],
                            "column_id": column_names[int(col_i)],
                            "direct_n": int(direct_n[int(row_i), int(col_i)]),
                            "rebuilt_probability": float(rebuilt[int(row_i), int(col_i)]),
                            "baseline_probability": float(baseline[int(row_i), int(col_i)]),
                            "signed_difference": float(delta[int(row_i), int(col_i)]),
                            "absolute_difference": float(abs(delta[int(row_i), int(col_i)])),
                        }
                    )
            pd.DataFrame(diagnostic_rows).to_csv(
                out / f"ap02_baseline_reconstruction_largest_differences_x{label}.csv",
                index=False,
            )
            reconstruction[f"x{label}"]["difference_by_support"] = {
                "target_direct_mean_absolute_difference": float(
                    np.mean(np.abs(target_delta)[baseline_target_n > 0])
                ),
                "target_borrowed_mean_absolute_difference": float(
                    np.mean(np.abs(target_delta)[baseline_target_n == 0])
                ),
                "chemical_direct_mean_absolute_difference": float(
                    np.mean(np.abs(chemical_delta)[baseline_chemical_n > 0])
                ),
                "chemical_borrowed_mean_absolute_difference": float(
                    np.mean(np.abs(chemical_delta)[baseline_chemical_n == 0])
                ),
            }
            (out / "ap02_baseline_reconstruction_gate.json").write_text(
                json.dumps(reconstruction, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise AssertionError(f"R0 reconstruction failed for x={target}")

        del rebuilt_target, rebuilt_target_n, rebuilt_chemical, rebuilt_chemical_n

        (
            loo_target,
            loo_target_rel,
            loo_target_n,
            loo_chemical,
            loo_chemical_rel,
            loo_chemical_n,
        ) = build_probability_matrices(
            strict_mm,
            protective_all,
            candidates,
            traits,
            chem,
            targets,
            all_chemicals,
            effect_probability_table,
            calibration,
            target,
            probability_module,
        )

        np.savez_compressed(
            out / f"ap02_strict_loo_species_protective_target_tail_probability_x{label}.npz",
            p=loo_target,
            reliability=loo_target_rel,
            direct_n=loo_target_n,
            species=np.asarray(species, dtype=object),
            target_ids=target_ids,
            chemicals=targets["dtxsid"].astype(str).to_numpy(object),
            effects=targets["effect_family"].astype(str).to_numpy(object),
            mechanisms=targets["mechanism_group"].astype(str).to_numpy(object),
        )
        np.savez_compressed(
            out / f"ap02_strict_loo_species_chemical_tail_probability_x{label}.npz",
            p=loo_chemical,
            reliability=loo_chemical_rel,
            direct_n=loo_chemical_n,
            species=np.asarray(species, dtype=object),
            chemicals=np.asarray(all_chemicals, dtype=object),
        )

        changed_direct = (baseline_target_n > 0) | (loo_target_n > 0)
        species_i, target_i = np.where(changed_direct)
        direct_delta = pd.DataFrame(
            {
                "latin_name": np.asarray(species, dtype=object)[species_i],
                "target_id": target_ids[target_i],
                "dtxsid": targets["dtxsid"].astype(str).to_numpy(object)[target_i],
                "effect_family": targets["effect_family"].astype(str).to_numpy(object)[target_i],
                "original_direct_contexts": baseline_target_n[species_i, target_i],
                "strict_loo_direct_contexts": loo_target_n[species_i, target_i],
                "p_original": baseline_target[species_i, target_i],
                "p_strict_loo": loo_target[species_i, target_i],
                "p_delta": loo_target[species_i, target_i] - baseline_target[species_i, target_i],
            }
        )
        direct_delta.to_csv(
            out / f"ap02_direct_target_probability_delta_x{label}.csv.gz",
            index=False,
            compression="gzip",
        )

        chemical_index = {name: index for index, name in enumerate(all_chemicals)}
        priority_rows = priority[priority["DTXSID"].isin(chemical_index)].copy()
        priority_indices = np.asarray(
            [chemical_index[name] for name in priority_rows["DTXSID"]], dtype=int
        )
        weights = priority_rows["national_weight"].to_numpy(float)
        weights /= weights.sum()
        baseline_priority = baseline_chemical[:, priority_indices].astype(float)
        loo_priority = loo_chemical[:, priority_indices].astype(float)
        baseline_mean = baseline_priority @ weights
        loo_mean = loo_priority @ weights
        baseline_rank = pd.Series(baseline_mean).rank(ascending=False, method="average")
        loo_rank = pd.Series(loo_mean).rank(ascending=False, method="average")
        rank_spearman = float(spearmanr(baseline_mean, loo_mean).statistic)
        rank_kendall = float(kendalltau(baseline_mean, loo_mean).statistic)

        base_seq = probability_module.deterministic_greedy(
            baseline_priority, weights, species, 20
        )
        loo_seq = probability_module.deterministic_greedy(loo_priority, weights, species, 20)
        optimized_sequences["baseline"][target] = [species[index] for index in base_seq]
        optimized_sequences["strict_loo"][target] = [species[index] for index in loo_seq]
        top5_base = [species[index] for index in base_seq[:5]]
        top5_loo = [species[index] for index in loo_seq[:5]]
        baseline_rho = float(panel_manifest["dependence_audit_by_x"][str(target)]["rho_working"])
        loo_rho_info = probability_module.estimate_working_rho(strict_mm, candidates)
        loo_rho = float(loo_rho_info["rho_working"])
        base_capture = expected_capture(
            probability_module, baseline_priority, base_seq[:5], weights, baseline_rho
        )
        loo_capture_fixed_rho = expected_capture(
            probability_module, loo_priority, loo_seq[:5], weights, baseline_rho
        )
        loo_capture_updated_rho = expected_capture(
            probability_module, loo_priority, loo_seq[:5], weights, loo_rho
        )
        top5_jaccard = len(set(top5_base) & set(top5_loo)) / len(set(top5_base) | set(top5_loo))
        top20_overlap = len(set(base_seq[:20]) & set(loo_seq[:20]))

        species_summary = candidates[["latin_name", "n_contexts", "n_chemicals", "support_tier"]].copy()
        species_summary["protection_target_x"] = target
        species_summary["priority_weighted_mean_probability_original"] = baseline_mean
        species_summary["priority_weighted_mean_probability_strict_loo"] = loo_mean
        species_summary["probability_delta"] = loo_mean - baseline_mean
        species_summary["rank_original"] = baseline_rank.to_numpy(float)
        species_summary["rank_strict_loo"] = loo_rank.to_numpy(float)
        species_summary["rank_change_strict_loo_minus_original"] = (
            loo_rank - baseline_rank
        ).to_numpy(float)
        species_summary.to_csv(out / f"ap02_species_ranking_delta_x{label}.csv", index=False)

        rank_rows.append(
            {
                "protection_target_x": target,
                "n_species": len(species),
                "rank_spearman": rank_spearman,
                "rank_kendall": rank_kendall,
                "top5_membership_jaccard": top5_jaccard,
                "top5_order_identical": top5_base == top5_loo,
                "top20_membership_overlap": top20_overlap,
                "top5_original": " | ".join(top5_base),
                "top5_strict_loo": " | ".join(top5_loo),
                "expected_capture_original": base_capture,
                "expected_capture_strict_loo_fixed_r0_rho": loo_capture_fixed_rho,
                "expected_capture_strict_loo_updated_rho": loo_capture_updated_rho,
                "absolute_capture_difference_fixed_r0_rho": loo_capture_fixed_rho - base_capture,
                "relative_capture_difference_fixed_r0_rho": (
                    loo_capture_fixed_rho / base_capture - 1 if base_capture else np.nan
                ),
            }
        )
        rho_rows.append(
            {
                "protection_target_x": target,
                "r0_rho_working": baseline_rho,
                "strict_loo_rho_working": loo_rho,
                "strict_loo_n_pairs": int(loo_rho_info["n_pairs"]),
                "strict_loo_rho_median_all": float(loo_rho_info["rho_median_all"]),
                "strict_loo_rho_median_positive": float(loo_rho_info["rho_median_positive"]),
            }
        )
        panel_rows.extend(
            [
                {
                    "protection_target_x": target,
                    "analysis": "optimized_for_same_target",
                    "variant": "R0",
                    "rho": baseline_rho,
                    "top5": " | ".join(top5_base),
                    "expected_capture": base_capture,
                },
                {
                    "protection_target_x": target,
                    "analysis": "optimized_for_same_target",
                    "variant": "strict_LOO_fixed_R0_rho",
                    "rho": baseline_rho,
                    "top5": " | ".join(top5_loo),
                    "expected_capture": loo_capture_fixed_rho,
                },
                {
                    "protection_target_x": target,
                    "analysis": "optimized_for_same_target",
                    "variant": "strict_LOO_updated_rho",
                    "rho": loo_rho,
                    "top5": " | ".join(top5_loo),
                    "expected_capture": loo_capture_updated_rho,
                },
            ]
        )

        if math.isclose(target, 0.95):
            membership_changed = set(top5_base) != set(top5_loo)
            order_changed = top5_base != top5_loo
            capture_absolute = abs(loo_capture_fixed_rho - base_capture)
            primary_update = membership_changed or capture_absolute >= 0.020 or rank_spearman < 0.90
            prominent_sensitivity = (
                not primary_update
                and (
                    order_changed
                    or capture_absolute >= 0.005
                    or rank_spearman < 0.95
                )
            )
            if primary_update:
                disposition = "PRIMARY_ANALYSIS_UPDATE_REQUIRED"
                selected_source = "AP02_STRICT_LOO"
            elif prominent_sensitivity:
                disposition = "WORDING_AND_PROMINENT_SENSITIVITY_UPDATE_REQUIRED"
                selected_source = "R0_PRIMARY_WITH_AP02_SENSITIVITY"
            else:
                disposition = "SENSITIVITY_LAYER_RETENTION_SUFFICIENT"
                selected_source = "R0_PRIMARY_WITH_AP02_SENSITIVITY"
            x95_gate = {
                "membership_changed": membership_changed,
                "order_changed": order_changed,
                "rank_spearman": rank_spearman,
                "expected_capture_original": base_capture,
                "expected_capture_strict_loo_fixed_r0_rho": loo_capture_fixed_rho,
                "absolute_expected_capture_difference": capture_absolute,
                "disposition": disposition,
                "selected_downstream_probability_source": selected_source,
                "selected_downstream_top5": top5_loo if primary_update else top5_base,
            }

        del (
            baseline_target,
            baseline_target_n,
            baseline_chemical,
            baseline_chemical_n,
            loo_target,
            loo_target_rel,
            loo_target_n,
            loo_chemical,
            loo_chemical_rel,
            loo_chemical_n,
            baseline_priority,
            loo_priority,
        )

    context_output.to_csv(
        out / "ap02_context_focal_species_loo_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    context_output[context_output["n5_diagnostic"]].to_csv(
        out / "ap02_original_n5_remaining4_diagnostic.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(strata_frames, ignore_index=True).to_csv(
        out / "ap02_context_probability_delta_strata.csv", index=False
    )
    pd.DataFrame(rank_rows).to_csv(out / "ap02_species_rank_and_panel_comparison.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(out / "ap02_rho_diagnostic.csv", index=False)

    if x95_gate is None:
        raise AssertionError("x=0.95 disposition gate was not generated")
    baseline_fixed = optimized_sequences["baseline"][0.95][:5]
    loo_fixed = optimized_sequences["strict_loo"][0.95][:5]
    priority_names = priority["DTXSID"].astype(str).tolist()
    for target in TARGETS:
        label = int(target * 100)
        with np.load(paths[f"chemical_npz_x{label}"], allow_pickle=True) as baseline_npz:
            baseline_p = baseline_npz["p"].astype(float)
            chemical_names = baseline_npz["chemicals"].astype(str).tolist()
        with np.load(
            out / f"ap02_strict_loo_species_chemical_tail_probability_x{label}.npz",
            allow_pickle=True,
        ) as loo_npz:
            loo_p = loo_npz["p"].astype(float)
        chemical_index = {name: index for index, name in enumerate(chemical_names)}
        present_priority = priority[priority["DTXSID"].isin(chemical_index)].copy()
        column_indices = np.asarray(
            [chemical_index[name] for name in present_priority["DTXSID"]], dtype=int
        )
        weights = present_priority["national_weight"].to_numpy(float)
        weights /= weights.sum()
        baseline_rho = float(panel_manifest["dependence_audit_by_x"][str(target)]["rho_working"])
        loo_rho = float(
            next(row["strict_loo_rho_working"] for row in rho_rows if row["protection_target_x"] == target)
        )
        base_indices = [species_index[name] for name in baseline_fixed]
        loo_indices = [species_index[name] for name in loo_fixed]
        panel_rows.extend(
            [
                {
                    "protection_target_x": target,
                    "analysis": "fixed_x95_sequence_cross_target",
                    "variant": "R0_fixed_x95_sequence",
                    "rho": baseline_rho,
                    "top5": " | ".join(baseline_fixed),
                    "expected_capture": expected_capture(
                        probability_module,
                        baseline_p[:, column_indices],
                        base_indices,
                        weights,
                        baseline_rho,
                    ),
                },
                {
                    "protection_target_x": target,
                    "analysis": "fixed_x95_sequence_cross_target",
                    "variant": "strict_LOO_fixed_x95_sequence_updated_rho",
                    "rho": loo_rho,
                    "top5": " | ".join(loo_fixed),
                    "expected_capture": expected_capture(
                        probability_module,
                        loo_p[:, column_indices],
                        loo_indices,
                        weights,
                        loo_rho,
                    ),
                },
            ]
        )
        del baseline_p, loo_p
    pd.DataFrame(panel_rows).to_csv(out / "ap02_panel_expected_capture_comparison.csv", index=False)

    reconstruction_gate = {
        "schema_version": 1,
        "tolerance": 2e-6,
        "by_target": reconstruction,
        "overall_pass": all(record["pass"] for record in reconstruction.values()),
    }
    (out / "ap02_baseline_reconstruction_gate.json").write_text(
        json.dumps(reconstruction_gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gate = {
        "schema_version": 1,
        "gate": "G2_AP02_STRICT_FOCAL_SPECIES_LOO",
        "status": "PASS" if reconstruction_gate["overall_pass"] else "FAIL",
        "seed": SEED,
        "baseline_reconstruction": reconstruction_gate,
        "strict_loo_context_contract": {
            "eligible_original_contexts": int(context_rows["context_id"].nunique()),
            "original_n5_diagnostic_contexts": int(
                context_rows.loc[context_rows["n5_diagnostic"], "context_id"].nunique()
            ),
            "original_n_ge6_contexts": int(
                context_rows.loc[context_rows["n_species"].ge(6), "context_id"].nunique()
            ),
            "strict_loo_species_context_rows": int(context_rows["strict_loo_eligible"].sum()),
            "strict_loo_rule": "original n>=6, remaining n>=5, and recomputed remaining-species SD>0.05 log10",
        },
        "x95_disposition": x95_gate,
        "next_stage": "A3_AP05_DEPENDENCE_AUDIT",
    }
    (out / "gate_ap02.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# AP02 strict focal-species leave-one-out report

## Gate

`{gate['status']}` — the independent implementation reproduced the frozen R0 probability matrices within the prespecified tolerance before generating strict-LOO results.

## Context contract

- Original eligible contexts: {gate['strict_loo_context_contract']['eligible_original_contexts']:,}
- Original n = 5 contexts retained only as diagnostics: {gate['strict_loo_context_contract']['original_n5_diagnostic_contexts']:,}
- Original n >= 6 contexts eligible to contribute to strict LOO: {gate['strict_loo_context_contract']['original_n_ge6_contexts']:,}
- Strict-LOO species-context rows after the recomputed-SD gate: {gate['strict_loo_context_contract']['strict_loo_species_context_rows']:,}

## Prespecified disposition at x = 0.95

- Disposition: `{x95_gate['disposition']}`
- Downstream probability source: `{x95_gate['selected_downstream_probability_source']}`
- Top-5 membership changed: `{x95_gate['membership_changed']}`
- Top-5 order changed: `{x95_gate['order_changed']}`
- All-species priority-weighted rank Spearman: {x95_gate['rank_spearman']:.6f}
- R0 expected capture: {x95_gate['expected_capture_original']:.6f}
- Strict-LOO expected capture with fixed R0 rho: {x95_gate['expected_capture_strict_loo_fixed_r0_rho']:.6f}
- Absolute expected-capture difference: {x95_gate['absolute_expected_capture_difference']:.6f}
- Selected downstream Top-5: {' | '.join(x95_gate['selected_downstream_top5'])}

This report records analysis routing only. Manuscript claims will be revised after AP03–AP04 complete and the cross-package gate reconciles all dependent results.
"""
    (out / "AP02_STRICT_LOO_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
