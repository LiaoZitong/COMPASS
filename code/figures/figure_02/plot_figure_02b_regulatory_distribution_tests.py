#!/usr/bin/env python3
"""Exploratory Figure 2 panel b candidates using sampled regulatory combinations.

This script is intentionally separate from plot_figure_02.py. It keeps the
main manuscript figure untouched and renders review-only panel b alternatives.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import PALETTE, TARGET_COLORS, apply_style, export_figure, figure_dir_from_script, project_root_from_script, write_csv  # noqa: E402
from plot_figure_02b_matched_k_test import build_matched_k_table, continuous_cmap, text_color_for_value  # noqa: E402


TARGETS = [0.80, 0.90, 0.95]
TARGET_LABELS = {0.80: "HC20-tail", 0.90: "HC10-tail", 0.95: "HC5-tail"}
METHOD_SPECS = {
    "taxonomy_diversity_baseline": {
        "label": "Taxonomic diversity k=5",
        "k": 5,
        "reference_k": 5,
        "color": "#6BAED6",
        "sampling_note": "five different taxonomic classes where feasible",
    },
    "random_all_species_probability": {
        "label": "Random 5-species k=5",
        "k": 5,
        "reference_k": 5,
        "color": "#969696",
        "sampling_note": "unconstrained random five-species panels",
    },
    "EPA_WQC_taxonomic_requirements": {
        "label": "EPA WQC proxy k=8",
        "k": 8,
        "reference_k": 8,
        "color": "#2166AC",
        "sampling_note": "EPA freshwater slot-feasible panels",
    },
    "Canada_Type_A_composition": {
        "label": "CCME Type A proxy k=7",
        "k": 7,
        "reference_k": 7,
        "color": "#4393C3",
        "sampling_note": "3 fish + 3 invertebrates + 1 plant/alga",
    },
    "US_EPA_WET_species": {
        "label": "EPA WET panel k=8",
        "k": 8,
        "reference_k": 8,
        "curve_method": "EPA_WET_fixed_method_species",
        "color": "#737373",
        "sampling_note": "fixed standard/method-test species panel; no sampled combinations",
    },
}
N_SAMPLES = 2000
RANDOM_SEED = 20260701
HERMITE_NODES = 12
SAMPLING_MODEL = "support-weighted feasible-panel sampling; without replacement"


def _clean_taxon(value: object) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def copula_union(p: np.ndarray, rho: float, nodes: int = HERMITE_NODES) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    if p.ndim == 1:
        p = p[None, :]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - p)
    shared_sd = math.sqrt(max(rho, 0.0))
    residual_sd = math.sqrt(max(1.0 - rho, 1e-8))
    no_event = np.zeros(p.shape[1], dtype=float)
    for zz, ww in zip(z, w):
        conditional_no_event = norm.cdf((threshold - shared_sd * zz) / residual_sd)
        no_event += ww * np.prod(conditional_no_event, axis=0)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def copula_union_many(p: np.ndarray, rho: float, nodes: int = HERMITE_NODES) -> np.ndarray:
    """Evaluate many fixed panels at once; p has shape panel x species x chemical."""

    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    if p.ndim != 3:
        raise ValueError("p must have shape panel x species x chemical")
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - p)
    shared_sd = math.sqrt(max(rho, 0.0))
    residual_sd = math.sqrt(max(1.0 - rho, 1e-8))
    no_event = np.zeros((p.shape[0], p.shape[2]), dtype=float)
    for zz, ww in zip(z, w):
        conditional_no_event = norm.cdf((threshold - shared_sd * zz) / residual_sd)
        no_event += ww * np.prod(conditional_no_event, axis=1)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def target_suffix(target: float) -> str:
    return f"{int(round(target * 100)):02d}"


def choose_from_pool(pool: list[int], candidates: pd.DataFrame, rng: np.random.Generator) -> int:
    weights = np.log1p(candidates.iloc[pool]["n_contexts"].to_numpy(float))
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return int(rng.choice(pool))
    weights = weights / weights.sum()
    return int(rng.choice(pool, p=weights))


def choose_from_mask(
    pool_mask: np.ndarray,
    family_codes: np.ndarray,
    used_family_codes: np.ndarray,
    support_weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    preferred = pool_mask & ~used_family_codes[family_codes]
    if preferred.any():
        pool_mask = preferred
    pool = np.flatnonzero(pool_mask)
    if len(pool) == 0:
        raise ValueError("No feasible species for one regulatory slot")
    weights = support_weights[pool].astype(float)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return int(rng.choice(pool))
    weights = weights / weights.sum()
    return int(rng.choice(pool, p=weights))


def sample_panel_from_masks(
    slot_masks: list[np.ndarray],
    family_codes: np.ndarray,
    support_weights: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    selected: list[int] = []
    available = np.ones(len(family_codes), dtype=bool)
    used_family_codes = np.zeros(int(family_codes.max()) + 1, dtype=bool)
    for mask in slot_masks:
        chosen = choose_from_mask(mask & available, family_codes, used_family_codes, support_weights, rng)
        selected.append(chosen)
        available[chosen] = False
        used_family_codes[family_codes[chosen]] = True
    return selected


def sample_random_panel(n_species: int, k: int, rng: np.random.Generator) -> list[int]:
    return [int(x) for x in rng.choice(n_species, k, replace=False)]


def sample_taxonomic_class_panel(
    class_codes: np.ndarray,
    support_weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    selected: list[int] = []
    available = np.ones(len(class_codes), dtype=bool)
    used_class_codes = np.zeros(int(class_codes.max()) + 1, dtype=bool)
    strict_slots = min(k, int(np.unique(class_codes).size))
    for rank in range(k):
        if rank < strict_slots:
            pool_mask = available & ~used_class_codes[class_codes]
            if not pool_mask.any():
                pool_mask = available
        else:
            pool_mask = available
        pool = np.flatnonzero(pool_mask)
        weights = support_weights[pool].astype(float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            chosen = int(rng.choice(pool))
        else:
            chosen = int(rng.choice(pool, p=weights / weights.sum()))
        selected.append(chosen)
        available[chosen] = False
        used_class_codes[class_codes[chosen]] = True
    return selected


def cleaned_columns(candidates: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "class": candidates["class"].map(_clean_taxon).to_numpy(str),
        "tax_order": candidates["tax_order"].map(_clean_taxon).to_numpy(str),
        "family": candidates["family"].map(_clean_taxon).to_numpy(str),
        "phylum_division": candidates["phylum_division"].map(_clean_taxon).to_numpy(str),
        "kingdom": candidates["kingdom"].map(_clean_taxon).to_numpy(str),
    }


def epa_wqc_slot_masks(candidates: pd.DataFrame) -> list[np.ndarray]:
    c = cleaned_columns(candidates)
    fish = (c["phylum_division"] == "chordata") & np.isin(c["class"], ["actinopterygii", "teleostei", "osteichthyes"])
    chordate = c["phylum_division"] == "chordata"
    planktonic_crustacean = np.isin(c["class"], ["branchiopoda", "maxillopoda", "copepoda"]) | np.isin(
        c["tax_order"], ["cladocera", "calanoida", "cyclopoida"]
    )
    benthic_crustacean = (c["class"] == "malacostraca") | np.isin(c["tax_order"], ["amphipoda", "isopoda", "decapoda"])
    insect = c["class"] == "insecta"
    other_phylum = ~np.isin(c["phylum_division"], ["arthropoda", "chordata", ""])
    return [
        fish & (c["family"] == "salmonidae"),
        fish & (c["family"] != "salmonidae"),
        chordate,
        planktonic_crustacean,
        benthic_crustacean,
        insect,
        other_phylum,
        np.ones(len(candidates), dtype=bool),
    ]


def ccme_type_a_slot_masks(candidates: pd.DataFrame) -> list[np.ndarray]:
    c = cleaned_columns(candidates)
    fish = (c["phylum_division"] == "chordata") & np.isin(c["class"], ["actinopterygii", "teleostei", "osteichthyes"])
    invertebrate = (c["kingdom"] == "animalia") & (c["phylum_division"] != "chordata")
    plant_alga = np.isin(c["kingdom"], ["plantae", "chromista"]) | np.isin(
        c["class"], ["chlorophyceae", "bacillariophyceae", "cyanophyceae"]
    )
    return [fish, fish, fish, invertebrate, invertebrate, invertebrate, plant_alga]


SlotPredicate = Callable[[pd.Series, set[str]], bool]


def sample_panel(
    candidates: pd.DataFrame,
    slots: list[SlotPredicate],
    rng: np.random.Generator,
) -> list[int]:
    selected: list[int] = []
    available = set(range(len(candidates)))
    used_families: set[str] = set()
    for predicate in slots:
        pool = [j for j in available if predicate(candidates.iloc[j], used_families)]
        if not pool:
            raise ValueError("No feasible species for one regulatory slot")
        preferred = [j for j in pool if _clean_taxon(candidates.iloc[j]["family"]) not in used_families]
        if preferred:
            pool = preferred
        chosen = choose_from_pool(pool, candidates, rng)
        selected.append(chosen)
        available.remove(chosen)
        used_families.add(_clean_taxon(candidates.iloc[chosen]["family"]))
    return selected


def is_fish(row: pd.Series) -> bool:
    return _clean_taxon(row["phylum_division"]) == "chordata" and _clean_taxon(row["class"]) in {
        "actinopterygii",
        "teleostei",
        "osteichthyes",
    }


def is_chordate(row: pd.Series) -> bool:
    return _clean_taxon(row["phylum_division"]) == "chordata"


def is_planktonic_crustacean(row: pd.Series) -> bool:
    return _clean_taxon(row["class"]) in {"branchiopoda", "maxillopoda", "copepoda"} or _clean_taxon(row["tax_order"]) in {
        "cladocera",
        "calanoida",
        "cyclopoida",
    }


def is_benthic_crustacean(row: pd.Series) -> bool:
    return _clean_taxon(row["class"]) == "malacostraca" or _clean_taxon(row["tax_order"]) in {
        "amphipoda",
        "isopoda",
        "decapoda",
    }


def is_insect(row: pd.Series) -> bool:
    return _clean_taxon(row["class"]) == "insecta"


def is_other_phylum(row: pd.Series) -> bool:
    return _clean_taxon(row["phylum_division"]) not in {"arthropoda", "chordata", ""}


def is_invertebrate(row: pd.Series) -> bool:
    return _clean_taxon(row["kingdom"]) == "animalia" and _clean_taxon(row["phylum_division"]) != "chordata"


def is_plant_alga(row: pd.Series) -> bool:
    return _clean_taxon(row["kingdom"]) in {"plantae", "chromista"} or _clean_taxon(row["class"]) in {
        "chlorophyceae",
        "bacillariophyceae",
        "cyanophyceae",
    }


def epa_wqc_slots() -> list[SlotPredicate]:
    return [
        lambda r, used: is_fish(r) and _clean_taxon(r["family"]) == "salmonidae",
        lambda r, used: is_fish(r) and _clean_taxon(r["family"]) != "salmonidae",
        lambda r, used: is_chordate(r) and _clean_taxon(r["family"]) not in used,
        lambda r, used: is_planktonic_crustacean(r),
        lambda r, used: is_benthic_crustacean(r),
        lambda r, used: is_insect(r),
        lambda r, used: is_other_phylum(r),
        lambda r, used: _clean_taxon(r["family"]) not in used,
    ]


def ccme_type_a_slots() -> list[SlotPredicate]:
    return [
        is_fish,
        is_fish,
        is_fish,
        is_invertebrate,
        is_invertebrate,
        is_invertebrate,
        is_plant_alga,
    ]


def load_candidates(root: Path) -> pd.DataFrame:
    candidates = pd.read_csv(root / "results/probability/species_probability_summary_x80.csv")
    return candidates.reset_index(drop=True)


def load_priority_probability(root: Path, target: float, curves: pd.DataFrame, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.load(root / f"results/probability/species_chemical_tail_probability_x{target_suffix(target)}.npz", allow_pickle=True)
    species = np.asarray(z["species"], dtype=object).astype(str)
    if not np.array_equal(species, candidates["latin_name"].astype(str).to_numpy()):
        reordered = candidates.set_index("latin_name").reindex(species)
        if reordered.isna().any(axis=None):
            raise ValueError("Candidate taxonomy cannot be aligned to probability matrix species order")
        candidates.iloc[:, :] = reordered.reset_index().iloc[:, :]

    p = z["p"].astype(np.float32)
    chemicals = np.asarray(z["chemicals"], dtype=object).astype(str)
    chemical_index = {chemical: i for i, chemical in enumerate(chemicals)}
    weights = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    weighted = weights[weights["DTXSID"].isin(chemical_index)].copy()
    weighted["national_weight"] = weighted["national_weight"].astype(float)
    weighted = weighted[weighted["national_weight"].gt(0)]
    indices = np.asarray([chemical_index[c] for c in weighted["DTXSID"]], dtype=int)
    w = weighted["national_weight"].to_numpy(dtype=float)
    w = w / w.sum()
    p_priority = p[:, indices]
    rho_hit = curves[
        curves["universe"].eq("priority")
        & curves["method"].eq("data_driven")
        & curves["protection_target_x"].eq(target)
    ]["rho"].dropna()
    rho = float(rho_hit.iloc[0]) if len(rho_hit) else 0.0
    return p_priority, w, rho


def load_fixed_top5_cross_target(root: Path) -> pd.DataFrame:
    fixed = pd.read_csv(root / "results/panels/fixed_x95_sequence_cross_target_curves.csv")
    fixed = fixed[
        fixed["sequence_basis_target"].eq(0.95)
        & fixed["fixed_sequence"].astype(bool)
        & fixed["panel_size"].eq(5)
    ].copy()
    if set(fixed["evaluation_target"].round(2)) != {0.80, 0.90, 0.95}:
        raise ValueError("Fixed x=0.95 Top-5 cross-target references are incomplete")
    if fixed["species_prefix"].nunique() != 1:
        raise ValueError("Fixed Top-5 membership differs across evaluation targets")
    return fixed.set_index("evaluation_target")


def sample_regulatory_panels(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(root / "results/panels/national_k1_20_coverage_curves.csv")
    fixed_top5 = load_fixed_top5_cross_target(root)
    candidates = load_candidates(root)
    rng = np.random.default_rng(RANDOM_SEED)

    sampled_panels: list[dict[str, object]] = []
    panel_indices: dict[str, list[list[int]]] = {}
    family_codes = pd.factorize(candidates["family"].map(_clean_taxon), sort=True)[0].astype(int)
    class_codes = pd.factorize(candidates["class"].map(_clean_taxon), sort=True)[0].astype(int)
    support_weights = np.log1p(candidates["n_contexts"].to_numpy(float))
    for method in METHOD_SPECS:
        spec = METHOD_SPECS[method]
        if method == "US_EPA_WET_species":
            continue
        sampling_model = (
            "uniform sampling without replacement"
            if method == "random_all_species_probability"
            else SAMPLING_MODEL
        )
        if method == "EPA_WQC_taxonomic_requirements":
            slot_masks = epa_wqc_slot_masks(candidates)
        elif method == "Canada_Type_A_composition":
            slot_masks = ccme_type_a_slot_masks(candidates)
        else:
            slot_masks = []
        panels: list[list[int]] = []
        for sample_id in range(N_SAMPLES):
            if method == "random_all_species_probability":
                selected = sample_random_panel(len(candidates), int(spec["k"]), rng)
            elif method == "taxonomy_diversity_baseline":
                selected = sample_taxonomic_class_panel(class_codes, support_weights, int(spec["k"]), rng)
            else:
                selected = sample_panel_from_masks(slot_masks, family_codes, support_weights, rng)
            panels.append(selected)
            sampled_panels.append(
                {
                    "method": method,
                    "method_label": spec["label"],
                    "method_k": spec["k"],
                    "sample_id": sample_id,
                    "panel_species": " | ".join(candidates.iloc[selected]["latin_name"].astype(str).tolist()),
                    "sampling_note": spec["sampling_note"],
                    "sampling_model": sampling_model,
                    "random_seed": RANDOM_SEED,
                }
            )
        panel_indices[method] = panels

    sampled_rows: list[dict[str, object]] = []
    panel_meta = pd.DataFrame(sampled_panels)
    fixed_rows: list[dict[str, object]] = []
    for target in TARGETS:
        p_priority, w, rho = load_priority_probability(root, target, curves, candidates)
        top5 = float(fixed_top5.loc[target, "expected_capture"])
        for method, panels in panel_indices.items():
            spec = METHOD_SPECS[method]
            optimized_k = float(
                curves[
                    curves["universe"].eq("priority")
                    & curves["method"].eq("data_driven")
                    & curves["k"].eq(spec["reference_k"])
                    & curves["protection_target_x"].eq(target)
                ]["expected_weighted_joint_coverage"].iloc[0]
            )
            if method == "random_all_species_probability":
                deterministic = np.nan
            else:
                curve_method = str(spec.get("curve_method", method))
                deterministic = float(
                    curves[
                        curves["universe"].eq("priority")
                        & curves["method"].eq(curve_method)
                        & curves["k"].eq(spec["k"])
                        & curves["protection_target_x"].eq(target)
                    ]["expected_weighted_joint_coverage"].iloc[0]
                )
            panel_array = np.asarray(panels, dtype=int)
            q_many = copula_union_many(p_priority[panel_array, :], rho)
            coverage_values = q_many @ w
            for sample_id, coverage in enumerate(coverage_values):
                sampled_rows.append(
                    {
                        "method": method,
                        "method_label": spec["label"],
                        "method_k": spec["k"],
                        "sample_id": sample_id,
                        "protection_target_x": target,
                        "coverage": float(coverage),
                        "fixed_top5_coverage": top5,
                        "target_specific_optimized_at_comparator_k": optimized_k,
                        "deterministic_proxy_coverage": deterministic,
                        "delta_vs_fixed_top5": coverage - top5,
                        "delta_vs_target_specific_optimized_at_comparator_k": coverage - optimized_k,
                        "copula_rho": rho,
                        "sampling_model": (
                            "uniform sampling without replacement"
                            if method == "random_all_species_probability"
                            else SAMPLING_MODEL
                        ),
                        "random_seed": RANDOM_SEED,
                    }
                )
        wet_spec = METHOD_SPECS["US_EPA_WET_species"]
        wet_curve_method = str(wet_spec.get("curve_method", "US_EPA_WET_species"))
        wet_deterministic = float(
            curves[
                curves["universe"].eq("priority")
                & curves["method"].eq(wet_curve_method)
                & curves["k"].eq(wet_spec["k"])
                & curves["protection_target_x"].eq(target)
            ]["expected_weighted_joint_coverage"].iloc[0]
        )
        wet_optimized = float(
            curves[
                curves["universe"].eq("priority")
                & curves["method"].eq("data_driven")
                & curves["k"].eq(wet_spec["reference_k"])
                & curves["protection_target_x"].eq(target)
            ]["expected_weighted_joint_coverage"].iloc[0]
        )
        fixed_rows.append(
            {
                "method": "US_EPA_WET_species",
                "method_label": wet_spec["label"],
                "method_k": wet_spec["k"],
                "protection_target_x": target,
                "sampled_n": 0,
                "sampled_mean": np.nan,
                "sampled_median": np.nan,
                "sampled_q025": np.nan,
                "sampled_q975": np.nan,
                "sampled_min": np.nan,
                "sampled_max": np.nan,
                "fixed_top5_coverage": top5,
                "target_specific_optimized_at_comparator_k": wet_optimized,
                "deterministic_proxy_coverage": wet_deterministic,
                "copula_rho": rho,
                "sampling_note": wet_spec["sampling_note"],
                "sampling_model": "fixed panel; no sampling",
                "random_seed": np.nan,
            }
        )
    sampled = pd.DataFrame(sampled_rows).merge(
        panel_meta,
        on=["method", "method_label", "method_k", "sample_id", "sampling_model", "random_seed"],
    )
    summary = (
        sampled.groupby(["method", "method_label", "method_k", "protection_target_x"], as_index=False)
        .agg(
            sampled_n=("coverage", "size"),
            sampled_mean=("coverage", "mean"),
            sampled_median=("coverage", "median"),
            sampled_q025=("coverage", lambda x: float(np.quantile(x, 0.025))),
            sampled_q975=("coverage", lambda x: float(np.quantile(x, 0.975))),
            sampled_min=("coverage", "min"),
            sampled_max=("coverage", "max"),
            fixed_top5_coverage=("fixed_top5_coverage", "first"),
            target_specific_optimized_at_comparator_k=("target_specific_optimized_at_comparator_k", "first"),
            deterministic_proxy_coverage=("deterministic_proxy_coverage", "first"),
            copula_rho=("copula_rho", "first"),
            sampling_note=("sampling_note", "first"),
            sampling_model=("sampling_model", "first"),
            random_seed=("random_seed", "first"),
        )
    )
    summary["deterministic_delta_vs_fixed_top5"] = summary["deterministic_proxy_coverage"] - summary["fixed_top5_coverage"]
    summary["deterministic_delta_vs_target_specific_optimized_at_comparator_k"] = (
        summary["deterministic_proxy_coverage"] - summary["target_specific_optimized_at_comparator_k"]
    )
    if fixed_rows:
        fixed = pd.DataFrame(fixed_rows)
        fixed["deterministic_delta_vs_fixed_top5"] = fixed["deterministic_proxy_coverage"] - fixed["fixed_top5_coverage"]
        fixed["deterministic_delta_vs_target_specific_optimized_at_comparator_k"] = (
            fixed["deterministic_proxy_coverage"] - fixed["target_specific_optimized_at_comparator_k"]
        )
        summary = pd.concat([summary, fixed], ignore_index=True, sort=False)
    return sampled, summary


def render_v2_scatter(sampled: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> list[str]:
    apply_style(font_size=7.0)
    fig, ax = plt.subplots(figsize=(4.15, 3.05))
    rng = np.random.default_rng(RANDOM_SEED + 11)
    method_order = [
        "taxonomy_diversity_baseline",
        "random_all_species_probability",
        "Canada_Type_A_composition",
        "EPA_WQC_taxonomic_requirements",
        "US_EPA_WET_species",
    ]
    x_positions = {method: i for i, method in enumerate(method_order)}
    target_offsets = {0.80: -0.155, 0.90: 0.0, 0.95: 0.155}
    bar_width = 0.12

    for target in TARGETS:
        target_summary = summary[summary["protection_target_x"].eq(target)]
        top5 = float(target_summary["fixed_top5_coverage"].iloc[0])
        color = TARGET_COLORS[target]
        ax.axhline(top5, color=color, lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax.text(
            len(method_order) - 0.48,
            top5,
            f"{TARGET_LABELS[target]} {top5:.3f}",
            ha="right",
            va="bottom",
            fontsize=4.4,
            color=color,
        )

    for method in method_order:
        spec = METHOD_SPECS[method]
        x0 = x_positions[method]
        for target in TARGETS:
            row = summary[
                summary["method"].eq(method)
                & summary["protection_target_x"].eq(target)
            ].iloc[0]
            x = x0 + target_offsets[target]
            color = TARGET_COLORS[target]
            if int(row["sampled_n"]) > 0:
                data = sampled[
                    sampled["method"].eq(method)
                    & sampled["protection_target_x"].eq(target)
                ]["coverage"].to_numpy(float)
                jitter = rng.normal(0, 0.022, size=len(data))
                ax.scatter(
                    np.full(len(data), x) + jitter,
                    data,
                    s=2.4,
                    alpha=0.075,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                    zorder=2,
                )
                ax.bar(
                    x,
                    row["sampled_median"],
                    width=bar_width,
                    color=color,
                    alpha=0.32,
                    edgecolor=color,
                    linewidth=0.65,
                    zorder=3,
                )
                ax.vlines(x, row["sampled_q025"], row["sampled_q975"], color=color, lw=0.95, zorder=4)
                ax.hlines(row["sampled_median"], x - bar_width / 2, x + bar_width / 2, color=color, lw=1.05, zorder=5)
                sampled_mean = float(row["sampled_mean"])
                ax.scatter(
                    [x],
                    [sampled_mean],
                    marker="o",
                    s=17,
                    color="white",
                    edgecolor=PALETTE["neutral_dark"],
                    linewidth=0.7,
                    zorder=6,
                )
            else:
                fixed_panel_value = float(row["deterministic_proxy_coverage"])
                ax.bar(
                    x,
                    fixed_panel_value,
                    width=bar_width,
                    color=color,
                    alpha=0.32,
                    edgecolor=color,
                    linewidth=0.65,
                    zorder=3,
                )

    ax.set_xticks([x_positions[m] for m in method_order])
    ax.set_xticklabels(
        [
            "Taxonomic\ndiversity\nk=5",
            "Random\nk=5",
            "CCME\nk=7",
            "WQC\nk=8",
            "WET fixed\nk=8",
        ],
        rotation=0,
        fontsize=5.25,
    )
    ax.set_xlim(-0.48, len(method_order) - 0.42)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Coverage", fontsize=6.5)
    ax.set_title("Feasible-combination distributions", fontsize=8.0, pad=7)
    ax.grid(axis="y", color="#E6E6E6", lw=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.80], edgecolor=TARGET_COLORS[0.80], alpha=0.32, label="HC20 median"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.90], edgecolor=TARGET_COLORS[0.90], alpha=0.32, label="HC10 median"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.95], edgecolor=TARGET_COLORS[0.95], alpha=0.32, label="HC5 median"),
        mpl.lines.Line2D([0], [0], color=PALETTE["neutral_dark"], lw=0.95, label="95% sampled interval"),
        mpl.lines.Line2D([0], [0], marker="o", color=PALETTE["neutral_dark"], markerfacecolor="white", markersize=4.2, lw=0, label="sampled mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.1, bbox_to_anchor=(0.53, 0.015))
    fig.text(
        0.98,
        0.125,
        f"n={N_SAMPLES} per sampled method x target; WET fixed has no sampled combinations",
        ha="right",
        va="bottom",
        fontsize=4.9,
        color=PALETTE["neutral_dark"],
    )
    fig.tight_layout(rect=(0, 0.20, 1, 0.98))
    return export_figure(fig, out_dir / "figure_02b_test_v2_regulatory_scatter")


def render_v3_heatmap_intervals(root: Path, summary: pd.DataFrame, out_dir: Path) -> list[str]:
    apply_style(font_size=7.0)
    matched = build_matched_k_table(root)

    interval_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        interval_rows.append(
            {
                "row_label": row.method_label,
                "protection_target_x": float(row.protection_target_x),
                "interval_low": float(row.sampled_q025),
                "interval_high": float(row.sampled_q975),
                "interval_source": "sampled feasible combinations",
            }
        )
    intervals = pd.DataFrame(interval_rows)
    matched = matched.merge(intervals, on=["row_label", "protection_target_x"], how="left")

    row_order = matched[["row_order", "row_label"]].drop_duplicates().sort_values("row_order")
    rows = row_order["row_label"].tolist()
    values = matched.pivot(index="row_label", columns="protection_target_x", values="coverage").reindex(rows)[TARGETS].to_numpy(float)
    deltas = (
        matched.pivot(
            index="row_label",
            columns="protection_target_x",
            values="delta_vs_target_specific_optimized_at_comparator_k",
        )
        .reindex(rows)[TARGETS]
        .to_numpy(float)
    )
    lows = matched.pivot(index="row_label", columns="protection_target_x", values="interval_low").reindex(rows)[TARGETS].to_numpy(float)
    highs = matched.pivot(index="row_label", columns="protection_target_x", values="interval_high").reindex(rows)[TARGETS].to_numpy(float)

    fig, ax = plt.subplots(figsize=(5.65, 3.35))
    cmap = continuous_cmap("bhbt_matched_k_blues_v3", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], rotation=28, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.8)
    ax.set_title("Matched-k comparison with sampled intervals", pad=8)

    for i, row_label in enumerate(rows):
        is_reference = row_label.startswith("Optimized")
        for j, _target in enumerate(TARGETS):
            color = text_color_for_value(values[i, j], cmap, norm)
            label = f"{values[i, j]:.3f}" if is_reference else f"{values[i, j]:.3f}\n$\\Delta$ {deltas[i, j]:+.3f}"
            ax.text(j, i - (0.08 if np.isfinite(lows[i, j]) else 0.0), label, ha="center", va="center", fontsize=5.7, color=color, linespacing=1.05)
            if np.isfinite(lows[i, j]) and np.isfinite(highs[i, j]):
                line_color = "white" if color == "white" else PALETTE["neutral_dark"]
                x_low = j - 0.36 + 0.72 * float(norm(lows[i, j]))
                x_high = j - 0.36 + 0.72 * float(norm(highs[i, j]))
                y = i + 0.30
                ax.plot([x_low, x_high], [y, y], color=line_color, lw=1.0, solid_capstyle="round", zorder=5)
                ax.plot([x_low, x_low], [y - 0.035, y + 0.035], color=line_color, lw=0.9, zorder=5)
                ax.plot([x_high, x_high], [y - 0.035, y + 0.035], color=line_color, lw=0.9, zorder=5)

    ax.hlines([2.5, 4.5], -0.5, len(TARGETS) - 0.5, colors="white", linewidths=2.2)
    ax.hlines([2.5, 4.5], -0.5, len(TARGETS) - 0.5, colors="#D9D9D9", linewidths=0.7)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("Expected weighted coverage\n(higher = darker)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    ax.text(
        1.0,
        -0.21,
        r"$\Delta$ = row minus matched optimized Top-k; small bar = sampled 95% interval",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=PALETTE["neutral_dark"],
    )
    fig.tight_layout(pad=0.6)
    return export_figure(fig, out_dir / "figure_02b_test_v3_matched_k_intervals")


def main() -> None:
    root = project_root_from_script(SCRIPT)
    out_dir = figure_dir_from_script(SCRIPT)
    sampled, summary = sample_regulatory_panels(root)
    sampled_name = write_csv(sampled, out_dir / "figure_02b_regulatory_sampled_combinations.csv")
    summary_name = write_csv(summary, out_dir / "figure_02b_regulatory_sampled_summary.csv")
    outputs = []
    outputs.extend(render_v2_scatter(sampled, summary, out_dir))
    outputs.extend(render_v3_heatmap_intervals(root, summary, out_dir))
    print({"source_tables": [sampled_name, summary_name], "outputs": outputs})


if __name__ == "__main__":
    main()
