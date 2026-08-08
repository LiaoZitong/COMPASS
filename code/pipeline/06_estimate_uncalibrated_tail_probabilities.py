#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import kurtosis, norm, skew

PROTECTIVE = [
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
]
WARNING_BANDS = {"heartbeat_early_warning", "other_early_warning"}
WARNING_WINDOWS = {"h00_02_ultra_early", "h02_06_early", "h06_24_same_day"}
EPA_WET_METHOD_SPECIES = [
    "Pimephales promelas", "Cyprinella leedsi", "Ceriodaphnia dubia",
    "Oncorhynchus mykiss", "Salvelinus fontinalis", "Daphnia pulex",
    "Daphnia magna", "Raphidocelis subcapitata",
]
EPA_ADDITIONAL_STANDARD_METHOD_SPECIES = [
    "Lepomis macrochirus", "Cyprinodon variegatus", "Hyalella azteca",
    "Chironomus dilutus",
]
TRAITS = [
    "trait_body_length_cm", "trait_body_mass_g", "trait_trophic_level",
    "trait_longevity_y", "trait_maturity_y", "trait_generation_time_y",
    "trait_habitat_breadth", "trait_diet_breadth",
]


def copula_union(p: np.ndarray, rho: float, nodes: int = 16) -> np.ndarray:
    """Equicorrelated one-factor Gaussian-copula union probability."""
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    if p.ndim == 1:
        p = p[None, :]
    if p.shape[0] == 1:
        return p[0]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1 - p)
    sr = math.sqrt(max(rho, 0.0))
    sd = math.sqrt(max(1 - rho, 1e-8))
    no_hit = np.zeros(p.shape[1], dtype=float)
    for zz, ww in zip(z, w):
        no_hit += ww * np.prod(norm.cdf((threshold - sr * zz) / sd), axis=0)
    return np.clip(1 - no_hit, 0, 1)


def copula_union_sequence(p: np.ndarray, rho: float, nodes: int = 16) -> np.ndarray:
    """Joint union probabilities for every prefix of an ordered panel in one pass."""
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    if p.ndim == 1:
        p = p[None, :]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1 - p)
    sr = math.sqrt(max(rho, 0.0))
    sd = math.sqrt(max(1 - rho, 1e-8))
    no_hit = np.zeros_like(p, dtype=float)
    for zz, ww in zip(z, w):
        conditional_no_hit = norm.cdf((threshold - sr * zz) / sd)
        no_hit += ww * np.cumprod(conditional_no_hit, axis=0)
    return np.clip(1 - no_hit, 0, 1)


def deterministic_greedy(p: np.ndarray, w: np.ndarray, names: list[str], kmax: int = 20) -> list[int]:
    """Deterministic full-vector greedy optimization over every eligible species."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    no_hit = np.ones(p.shape[1], dtype=float)
    remaining = np.arange(p.shape[0], dtype=int)
    selected: list[int] = []
    for _ in range(min(kmax, p.shape[0])):
        gains = p[remaining] @ (w * no_hit)
        best = float(np.nanmax(gains))
        tied_pos = np.flatnonzero(np.isclose(gains, best, rtol=0, atol=1e-14))
        j = min((int(remaining[q]) for q in tied_pos), key=lambda q: names[q])
        selected.append(j)
        no_hit *= 1 - p[j]
        remaining = remaining[remaining != j]
    return selected

def taxonomy_sequence(candidates: pd.DataFrame, kmax: int = 20) -> list[int]:
    """Select a taxonomy-diversity baseline.

    Selection takes as many new taxonomic classes as possible before repeating
    a class. If all classes have already been represented, later positions
    continue the broader phylum/order/family diversity heuristic.
    """

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    seen = {x: set() for x in ["class", "tax_order", "family", "phylum_division"]}
    support = np.log1p(candidates.n_contexts.to_numpy(float))
    strict_class_slots = min(kmax, candidates["class"].nunique(dropna=True))
    for rank in range(min(kmax, len(remaining))):
        scores = []
        for j in remaining:
            row = candidates.iloc[j]
            if rank < strict_class_slots and row["class"] in seen["class"]:
                continue
            score = (
                4 * (row["class"] not in seen["class"])
                + 2 * (row.tax_order not in seen["tax_order"])
                + 1 * (row.family not in seen["family"])
                + 0.5 * (row.phylum_division not in seen["phylum_division"])
                + 0.05 * support[j]
            )
            scores.append((j, score))
        if not scores:
            scores = []
            for j in remaining:
                row = candidates.iloc[j]
                score = (
                    4 * (row["class"] not in seen["class"])
                    + 2 * (row.tax_order not in seen["tax_order"])
                    + 1 * (row.family not in seen["family"])
                    + 0.5 * (row.phylum_division not in seen["phylum_division"])
                    + 0.05 * support[j]
                )
                scores.append((j, score))
        best = max(s for _, s in scores)
        tied = [j for j, s in scores if np.isclose(s, best)]
        j = min(tied, key=lambda q: candidates.iloc[q].latin_name)
        selected.append(j)
        row = candidates.iloc[j]
        for field in seen:
            seen[field].add(row[field])
        remaining.remove(j)
    return selected



def _clean_taxon(value: object) -> str:
    return str(value).strip().lower() if pd.notna(value) else ""


def _pick_best(candidates: pd.DataFrame, available: set[int], predicate, used_families: set[str]) -> int | None:
    pool = [j for j in available if predicate(candidates.iloc[j])]
    if not pool:
        return None
    def key(j: int):
        row = candidates.iloc[j]
        family_new = _clean_taxon(row.family) not in used_families
        return (family_new, float(np.log1p(row.n_contexts)), -len(row.latin_name), row.latin_name)
    return max(pool, key=key)


def _append_slot(selected: list[int], available: set[int], candidates: pd.DataFrame, predicate, used_families: set[str]) -> None:
    j = _pick_best(candidates, available, predicate, used_families)
    if j is not None:
        selected.append(j)
        available.remove(j)
        used_families.add(_clean_taxon(candidates.iloc[j].family))


def epa_wqc_sequence(candidates: pd.DataFrame, kmax: int = 20) -> list[int]:
    """Operationalize EPA 1985 freshwater eight-family minimum as a baseline.

    The guideline specifies taxonomic/functional slots rather than a fixed species list.
    After the eight slots, the sequence is extended by the generic taxonomy-diversity rule.
    """
    selected: list[int] = []
    available = set(range(len(candidates)))
    used_families: set[str] = set()
    is_fish = lambda r: _clean_taxon(r.phylum_division) == "chordata" and _clean_taxon(r["class"]) in {"actinopterygii", "teleostei", "osteichthyes"}
    is_chordate = lambda r: _clean_taxon(r.phylum_division) == "chordata"
    is_plank_crust = lambda r: _clean_taxon(r["class"]) in {"branchiopoda", "maxillopoda", "copepoda"} or _clean_taxon(r.tax_order) in {"cladocera", "calanoida", "cyclopoida"}
    is_benthic_crust = lambda r: _clean_taxon(r["class"]) == "malacostraca" or _clean_taxon(r.tax_order) in {"amphipoda", "isopoda", "decapoda"}
    is_insect = lambda r: _clean_taxon(r["class"]) == "insecta"
    is_other_phylum = lambda r: _clean_taxon(r.phylum_division) not in {"arthropoda", "chordata", ""}
    _append_slot(selected, available, candidates, lambda r: is_fish(r) and _clean_taxon(r.family) == "salmonidae", used_families)
    _append_slot(selected, available, candidates, lambda r: is_fish(r) and _clean_taxon(r.family) != "salmonidae", used_families)
    _append_slot(selected, available, candidates, lambda r: is_chordate(r) and _clean_taxon(r.family) not in used_families, used_families)
    _append_slot(selected, available, candidates, is_plank_crust, used_families)
    _append_slot(selected, available, candidates, is_benthic_crust, used_families)
    _append_slot(selected, available, candidates, is_insect, used_families)
    _append_slot(selected, available, candidates, is_other_phylum, used_families)
    _append_slot(selected, available, candidates, lambda r: _clean_taxon(r.family) not in used_families, used_families)
    fill = taxonomy_sequence(candidates, kmax)
    for j in fill:
        if j not in selected:
            selected.append(j)
        if len(selected) >= kmax:
            break
    return selected[:kmax]


def canada_type_a_sequence(candidates: pd.DataFrame, kmax: int = 20) -> list[int]:
    """Operationalize Canada's Type A minimum composition: 3 fish, 3 invertebrates, 1 plant/alga."""
    selected: list[int] = []
    available = set(range(len(candidates)))
    used_families: set[str] = set()
    is_fish = lambda r: _clean_taxon(r.phylum_division) == "chordata" and _clean_taxon(r["class"]) in {"actinopterygii", "teleostei", "osteichthyes"}
    is_invert = lambda r: _clean_taxon(r.kingdom) == "animalia" and _clean_taxon(r.phylum_division) != "chordata"
    is_plant_alga = lambda r: _clean_taxon(r.kingdom) in {"plantae", "chromista"} or _clean_taxon(r["class"]) in {"chlorophyceae", "bacillariophyceae", "cyanophyceae"}
    for _ in range(3):
        _append_slot(selected, available, candidates, is_fish, used_families)
    for _ in range(3):
        _append_slot(selected, available, candidates, is_invert, used_families)
    _append_slot(selected, available, candidates, is_plant_alga, used_families)
    fill = taxonomy_sequence(candidates, kmax)
    for j in fill:
        if j not in selected:
            selected.append(j)
        if len(selected) >= kmax:
            break
    return selected[:kmax]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    c = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(c, 0.5)])


def estimate_working_rho(mm: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, float | int]:
    """Estimate a conservative common dependence parameter from shared binary tail events.

    The selected working rho is the overlap-weighted median of positive pairwise Pearson
    correlations. Negative correlations are retained in the audit but not used in the
    one-factor positive-dependence copula. This is conservative for union expected capture.
    """
    top = candidates.nlargest(min(40, len(candidates)), "n_contexts").latin_name.tolist()
    pv = mm[mm.latin_name.isin(top)].pivot_table(
        index="context_id", columns="latin_name", values="tail_binary", aggfunc="mean"
    )
    rows = []
    cols = pv.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = pv[cols[i]]
            b = pv[cols[j]]
            mask = a.notna() & b.notna()
            n = int(mask.sum())
            if n < 20:
                continue
            aa = a[mask].to_numpy(float)
            bb = b[mask].to_numpy(float)
            if np.std(aa) == 0 or np.std(bb) == 0:
                continue
            r = float(np.corrcoef(aa, bb)[0, 1])
            if np.isfinite(r):
                rows.append((r, n))
    if not rows:
        return {"rho_working": 0.15, "rho_median_all": 0.15, "rho_median_positive": 0.15, "n_pairs": 0}
    vals = np.array([r for r, _ in rows], float)
    weights = np.array([n for _, n in rows], float)
    pos = vals > 0
    med_all = weighted_median(vals, weights)
    med_pos = weighted_median(vals[pos], weights[pos]) if pos.any() else 0.0
    rho = float(np.clip(med_pos, 0.0, 0.5))
    return {
        "rho_working": rho,
        "rho_median_all": float(med_all),
        "rho_median_positive": float(med_pos),
        "rho_q25_all": float(np.quantile(vals, 0.25)),
        "rho_q75_all": float(np.quantile(vals, 0.75)),
        "n_pairs": int(len(vals)),
    }


def support_tier(n: pd.Series) -> pd.Series:
    return pd.cut(
        n,
        bins=[0, 4, 19, 49, np.inf],
        labels=["very_sparse_1_4", "sparse_5_19", "moderate_20_49", "strong_50_plus"],
        include_lowest=True,
        right=True,
    ).astype(str)


def build_candidate_tables(
    m: pd.DataFrame, occurrence: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use every formal species with >=1 valid protective context in the national model.

    Support is never a hard eligibility cutoff. Sparse species remain in the probability and
    ranking universe, receive stronger shrinkage/wider uncertainty, and are explicitly routed
    to active testing. State occurrence/NAS are used only downstream as soft-local
    state-panel relevance weights.
    """
    sup = (
        m.groupby("latin_name")
        .agg(
            n_contexts=("context_id", "nunique"),
            n_chemicals=("dtxsid", "nunique"),
            class_=("class", "first"),
            tax_order=("tax_order", "first"),
            family=("family", "first"),
            genus=("genus", "first"),
            phylum_division=("phylum_division", "first"),
            kingdom=("kingdom", "first"),
        )
        .reset_index()
        .rename(columns={"class_": "class"})
    )
    formal_name = sup.latin_name.str.match(r"^[A-Z][a-z-]+ [a-z][A-Za-z.-]+$", na=False)
    sup = sup[formal_name].copy()
    breadth = (
        occurrence[occurrence.eligible_state_candidate.fillna(False).astype(bool)]
        .groupby("latin_name").state_code.nunique().rename("eligible_states").reset_index()
    )
    sup = sup.merge(breadth, on="latin_name", how="left")
    sup.eligible_states = sup.eligible_states.fillna(0).astype(int)
    sup["support_tier"] = support_tier(sup.n_contexts)
    sup["candidate_basis"] = ">=1 valid protective context; no hard support exclusion"
    sup["state_occurrence_available"] = sup.eligible_states >= 1
    sup["official_wet_method_species"] = sup.latin_name.isin(EPA_WET_METHOD_SPECIES)
    sup["official_or_common_method_species"] = sup.latin_name.isin(
        EPA_WET_METHOD_SPECIES + EPA_ADDITIONAL_STANDARD_METHOD_SPECIES
    )
    sup["panel_eligible"] = True

    under = sup[sup.n_contexts < 20].copy()
    breadth_norm = under.eligible_states / max(under.eligible_states.max(), 1)
    support_uncertainty = 1 / np.sqrt(under.n_contexts.clip(lower=1))
    under["under_tested_priority_seed"] = (
        support_uncertainty * (0.25 + 0.75 * breadth_norm)
    )
    under["routing"] = "retained_in_panel_model_and_high_priority_for_active_testing"

    all_evidence = sup.copy()
    all_evidence["random_universe_basis"] = ">=1 valid protective context and formal binomial name"
    return (
        sup.sort_values("latin_name").reset_index(drop=True),
        under.sort_values("under_tested_priority_seed", ascending=False).reset_index(drop=True),
        all_evidence.sort_values("latin_name").reset_index(drop=True),
    )

def trait_neighbors(traits: pd.DataFrame, species: list[str]) -> list[list[int]]:
    """Vectorized weak trait-neighbour graph using pairwise nan-aware distances.

    A pair is eligible only when at least two continuous traits are jointly observed.
    This is a weak prior only; no absolute toxicity value is predicted.
    """
    from sklearn.metrics.pairwise import nan_euclidean_distances
    tt = traits.drop_duplicates("latin_name").set_index("latin_name").reindex(species)
    x = tt.reindex(columns=TRAITS).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x = np.log1p(np.clip(x, 0, None))
    means = np.nanmean(x, axis=0)
    sds = np.nanstd(x, axis=0)
    sds[~np.isfinite(sds) | (sds == 0)] = 1.0
    x = (x - means) / sds
    observed = np.isfinite(x).astype(np.int16)
    shared = observed @ observed.T
    dist = nan_euclidean_distances(x)
    dist[shared < 2] = np.inf
    np.fill_diagonal(dist, np.inf)
    neighbors: list[list[int]] = []
    for i in range(len(species)):
        order = np.argsort(dist[i], kind="stable")
        order = order[np.isfinite(dist[i, order])][:10]
        neighbors.append(order.astype(int).tolist())
    return neighbors

def effect_target_weights(
    targets: pd.DataFrame,
    chemical_weights: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each chemical's total weight equally across its observed adverse effect families."""
    t = targets[targets.dtxsid.isin(chemical_weights.index)].copy()
    n_eff = t.groupby("dtxsid").effect_family.transform("nunique")
    weight = t.dtxsid.map(chemical_weights).to_numpy(float) / n_eff.to_numpy(float)
    if weight.sum() <= 0:
        raise ValueError("No target weight after chemical/effect expansion")
    return t.index.to_numpy(int), weight / weight.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chem", required=True)
    ap.add_argument("--traits", required=True)
    ap.add_argument("--occurrence", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--out-prob", required=True)
    ap.add_argument("--out-panels", required=True)
    ap.add_argument("--calibration-json", default="")
    ap.add_argument("--target", type=float, choices=[0.80, 0.90, 0.95], default=None, help="Run one protection target per process to bound memory")
    args = ap.parse_args()

    out_prob = Path(args.out_prob)
    out_panels = Path(args.out_panels)
    out_prob.mkdir(parents=True, exist_ok=True)
    out_panels.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir)
    calibration = None
    if args.calibration_json:
        calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))

    model = pd.read_csv(args.model, low_memory=False)
    chem = pd.read_csv(args.chem, low_memory=False)
    traits = pd.read_csv(args.traits, low_memory=False)
    occurrence = pd.read_csv(args.occurrence, low_memory=False)

    valid = model.mle_status.astype(str).str.startswith("identified") & np.isfinite(model.mu_log10_umol_L)
    protective = model[valid & model.effect_family.isin(PROTECTIVE)].copy()
    protective["context_id"] = protective[
        ["dtxsid", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]
    ].astype(str).agg("|".join, axis=1)

    context = (
        protective.groupby("context_id")
        .agg(
            dtxsid=("dtxsid", "first"), effect_family=("effect_family", "first"),
            endpoint_band_v16=("endpoint_band_v16", "first"),
            duration_window_v16=("duration_window_v16", "first"),
            medium_family=("medium_family", "first"),
            n_species=("latin_name", "nunique"),
            ssd_mean=("mu_log10_umol_L", "mean"),
            ssd_sd=("mu_log10_umol_L", "std"),
        )
        .reset_index()
    )
    context = context[(context.n_species >= 5) & context.ssd_sd.notna() & (context.ssd_sd > 0.05)].copy()
    protective = protective[protective.context_id.isin(context.context_id)].merge(
        context[["context_id", "n_species", "ssd_mean", "ssd_sd"]], on="context_id"
    )

    shape = protective.groupby("context_id").mu_log10_umol_L.agg(
        ssd_skew=lambda x: skew(x, bias=False),
        ssd_excess_kurtosis=lambda x: kurtosis(x, bias=False),
    ).reset_index()
    context = context.merge(shape, on="context_id")
    q1, q2 = context.ssd_sd.quantile([0.33, 0.67])
    context["ssd_width_class"] = np.select(
        [context.ssd_sd <= q1, context.ssd_sd <= q2], ["narrow", "moderate"], default="wide"
    )
    context["ssd_shape_class"] = np.select(
        [context.ssd_skew < -0.5, context.ssd_skew > 0.5],
        ["left_skew_sensitive_tail", "right_skew_resistant_tail"],
        default="approximately_symmetric",
    )
    context.to_csv(out_prob / "context_ssd_shape_summary.csv", index=False)

    candidates, under_tested, all_evidence_species = build_candidate_tables(protective, occurrence)
    candidates.to_csv(out_prob / "candidate_universe_locked.csv", index=False)
    under_tested.to_csv(out_prob / "under_tested_species_registry.csv", index=False)
    all_evidence_species.to_csv(out_prob / "all_evidence_species_universe.csv", index=False)
    candidate_hash = hashlib.sha256("\n".join(candidates.latin_name).encode()).hexdigest()
    species = candidates.latin_name.tolist()
    s_count = len(species)
    s_index = {s: i for i, s in enumerate(species)}

    chem["DTXSID"] = chem.DTXSID.astype(str)
    chem["mechanism_group"] = np.where(
        chem.primary_moa.fillna("MOA_UNRESOLVED").ne("MOA_UNRESOLVED"),
        chem.primary_moa,
        "FORM::" + chem.chemical_form_class.fillna("unresolved").astype(str),
    )
    chem_mech = chem.set_index("DTXSID").mechanism_group.to_dict()
    protective["mechanism_group"] = protective.dtxsid.map(chem_mech).fillna("MOA_UNRESOLVED")

    all_chem = pd.read_csv(weights_dir / "all_protective_chemicals_uniform.csv")
    chemicals = all_chem.DTXSID.astype(str).tolist()
    c_index = {c: i for i, c in enumerate(chemicals)}
    priority = pd.read_csv(weights_dir / "national_priority_chemicals.csv")
    priority = priority[priority.DTXSID.astype(str).isin(c_index)].copy()
    priority["DTXSID"] = priority.DTXSID.astype(str)
    priority_weights = priority.set_index("DTXSID").national_weight.astype(float)
    priority_weights /= priority_weights.sum()
    uniform_weights = pd.Series(1 / len(chemicals), index=chemicals)

    effects = sorted(PROTECTIVE)
    effect_index = {e: i for i, e in enumerate(effects)}
    mechanisms = sorted(set(chem_mech.get(c, "MOA_UNRESOLVED") for c in chemicals))
    mechanism_index = {m: i for i, m in enumerate(mechanisms)}
    neighbors = trait_neighbors(traits, species)

    targets = (
        context[context.dtxsid.isin(chemicals)][["dtxsid", "effect_family"]]
        .drop_duplicates()
        .sort_values(["dtxsid", "effect_family"])
        .reset_index(drop=True)
    )
    targets["target_id"] = targets.dtxsid + "|" + targets.effect_family
    targets["mechanism_group"] = targets.dtxsid.map(chem_mech).fillna("MOA_UNRESOLVED")
    targets.to_csv(out_prob / "protective_chemical_effect_targets.csv", index=False)
    target_chem_idx = np.array([c_index[c] for c in targets.dtxsid], int)
    target_mech_idx = np.array([mechanism_index[m] for m in targets.mechanism_group], int)
    target_eff_idx = np.array([effect_index[e] for e in targets.effect_family], int)

    # MOA/form-specific probability that each protective effect family is represented.
    # Add-one smoothing prevents an unobserved family from receiving an impossible zero weight.
    effect_counts = (
        protective.groupby(["mechanism_group", "effect_family"]).size()
        .rename("n_cells").reset_index()
    )
    full_effect_rows = []
    for mech in mechanisms:
        sub = effect_counts[effect_counts.mechanism_group == mech].set_index("effect_family").n_cells
        denom = float(sub.sum() + len(effects))
        for eff in effects:
            nn = int(sub.get(eff, 0))
            full_effect_rows.append({
                "mechanism_group": mech, "effect_family": eff, "n_cells": nn,
                "effect_probability_within_moa": (nn + 1.0) / denom,
            })
    effect_probability_table = pd.DataFrame(full_effect_rows)
    effect_probability_table.to_csv(out_prob / "moa_effect_family_probability.csv", index=False)
    effect_weight_map = {
        mech: grp.set_index("effect_family").effect_probability_within_moa.reindex(effects).to_numpy(float)
        for mech, grp in effect_probability_table.groupby("mechanism_group")
    }

    targets_run = [args.target] if args.target is not None else [0.80, 0.90, 0.95]
    curves: list[dict] = []
    sequence_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    daphnia_rows: list[dict] = []
    rho_audit: dict[str, dict] = {}

    for x in targets_run:
        q0 = 1 - x
        zq = norm.ppf(q0)
        mm = protective.copy()
        mm["hc"] = mm.ssd_mean + zq * mm.ssd_sd
        uncertainty = np.sqrt(
            mm.se_mu_log10.fillna(0.35) ** 2 + (mm.ssd_sd / np.sqrt(mm.n_species)) ** 2
        ).clip(lower=0.12)
        mm["p_context"] = norm.cdf((mm.hc - mm.mu_log10_umol_L) / uncertainty)
        mm["tail_binary"] = (mm.mu_log10_umol_L <= mm.hc).astype(float)

        direct = (
            mm[mm.latin_name.isin(species)]
            .groupby(["latin_name", "dtxsid", "effect_family"])
            .p_context.agg(["sum", "count"])
            .reset_index()
        )
        direct_key = {
            (r.latin_name, r.dtxsid, r.effect_family): (float(r["sum"]), int(r["count"]))
            for _, r in direct.iterrows()
        }

        # Probability-only borrowing is estimated twice: taxonomy-only and MOA-conditioned taxonomy.
        # The whole-chemical validation file decides which mechanism×effect strata are allowed to use MOA.
        evidence_moa = (
            mm.groupby(["latin_name", "genus", "family", "class", "mechanism_group", "effect_family"])
            .p_context.agg(["sum", "count"]).reset_index()
        )
        evidence_tax = (
            mm.groupby(["latin_name", "genus", "family", "class", "effect_family"])
            .p_context.agg(["sum", "count"]).reset_index()
        )
        maps_moa: dict[str, dict] = {}
        maps_tax: dict[str, dict] = {}
        for level in ["latin_name", "genus", "family", "class"]:
            gm = evidence_moa.groupby([level, "mechanism_group", "effect_family"])[["sum", "count"]].sum().reset_index()
            maps_moa[level] = {
                (r[level], r.mechanism_group, r.effect_family): (float(r["sum"]), int(r["count"]))
                for _, r in gm.iterrows()
            }
            gt = evidence_tax.groupby([level, "effect_family"])[["sum", "count"]].sum().reset_index()
            maps_tax[level] = {
                (r[level], r.effect_family): (float(r["sum"]), int(r["count"]))
                for _, r in gt.iterrows()
            }

        prior_moa = np.full((s_count, len(mechanisms), len(effects)), q0, dtype=float)
        rel_moa = np.zeros_like(prior_moa)
        prior_tax = np.full((s_count, len(effects)), q0, dtype=float)
        rel_tax = np.zeros_like(prior_tax)
        level_caps = {"latin_name": 0.80, "genus": 0.50, "family": 0.30, "class": 0.15}
        level_weights = [("latin_name", 4.0), ("genus", 2.0), ("family", 1.0), ("class", 0.5)]
        for i, row in candidates.iterrows():
            for eff, ei in effect_index.items():
                values, weights, reliabilities = [], [], []
                for level, base_weight in level_weights:
                    key = row.latin_name if level == "latin_name" else row[level]
                    sm, nn = maps_tax[level].get((key, eff), (0.0, 0))
                    if nn:
                        values.append((sm + 10 * q0) / (nn + 10))
                        weights.append(base_weight * min(nn, 20))
                        reliabilities.append(level_caps[level] * (1 - np.exp(-nn / 10.0)))
                if values:
                    prior_tax[i, ei] = np.average(values, weights=weights)
                    rel_tax[i, ei] = max(reliabilities)
                for mech, mi in mechanism_index.items():
                    values, weights, reliabilities = [], [], []
                    for level, base_weight in level_weights:
                        key = row.latin_name if level == "latin_name" else row[level]
                        sm, nn = maps_moa[level].get((key, mech, eff), (0.0, 0))
                        if nn:
                            values.append((sm + 10 * q0) / (nn + 10))
                            weights.append(base_weight * min(nn, 20))
                            reliabilities.append(level_caps[level] * (1 - np.exp(-nn / 10.0)))
                    if values:
                        prior_moa[i, mi, ei] = np.average(values, weights=weights)
                        rel_moa[i, mi, ei] = max(reliabilities)

        # Weak trait prior, never more than 30% and only with >=3 eligible neighbours.
        before_moa = prior_moa.copy(); before_moa_rel = rel_moa.copy()
        before_tax = prior_tax.copy(); before_tax_rel = rel_tax.copy()
        trait_cells = 0
        for i, nbs in enumerate(neighbors):
            if len(nbs) >= 3:
                prior_moa[i] = 0.70 * prior_moa[i] + 0.30 * np.nanmean(before_moa[nbs], axis=0)
                rel_moa[i] = np.maximum(0.70 * before_moa_rel[i], 0.10 * np.nanmean(before_moa_rel[nbs], axis=0))
                prior_tax[i] = 0.70 * prior_tax[i] + 0.30 * np.nanmean(before_tax[nbs], axis=0)
                rel_tax[i] = np.maximum(0.70 * before_tax_rel[i], 0.10 * np.nanmean(before_tax_rel[nbs], axis=0))
                trait_cells += prior_moa.shape[1] * prior_moa.shape[2]

        by_target_cal = calibration.get("by_protection_target", {}) if calibration is not None else {}
        target_cal = by_target_cal.get(str(x)) or by_target_cal.get(f"{x:.2f}") or {}
        validated_moa_groups = set(target_cal.get("validated_moa_groups", []))

        p_target = np.empty((s_count, len(targets)), dtype=float)
        r_target = np.empty_like(p_target)
        n_target = np.zeros_like(p_target)
        borrowed_model = np.full(len(targets), "taxonomy_only", object)
        for t, row in targets.iterrows():
            mi = target_mech_idx[t]
            ei = target_eff_idx[t]
            group_key = f"{row.mechanism_group}||{row.effect_family}"
            use_moa = group_key in validated_moa_groups
            if use_moa:
                prior_col = prior_moa[:, mi, ei]
                reliability_prior = rel_moa[:, mi, ei]
                borrowed_model[t] = "validated_moa_gate"
            else:
                prior_col = prior_tax[:, ei]
                reliability_prior = rel_tax[:, ei]
            p_col = q0 + reliability_prior * (prior_col - q0)
            r_col = reliability_prior.copy()
            for i, sp in enumerate(species):
                sm, nn = direct_key.get((sp, row.dtxsid, row.effect_family), (0.0, 0))
                if nn:
                    raw = (sm + 2 * prior_col[i]) / (nn + 2)
                    reliability_direct = 1 - np.exp(-nn / 3.0)
                    p_col[i] = q0 + reliability_direct * (raw - q0)
                    r_col[i] = reliability_direct
                    n_target[i, t] = nn

            # Calibrate borrowed cells using the model selected by the validation-informed gate.
            if target_cal:
                cal_key = "moa_taxonomy_calibration" if use_moa else "taxonomy_calibration"
                params = target_cal.get(cal_key)
                if params is not None:
                    missing = n_target[:, t] == 0
                    logit = np.log(np.clip(p_col, 1e-6, 1-1e-6) / np.clip(1-p_col, 1e-6, 1-1e-6))
                    calibrated = 1 / (1 + np.exp(-(float(params["intercept"]) + float(params["coefficient"]) * logit)))
                    p_col[missing] = calibrated[missing]
            p_target[:, t] = np.clip(p_col, 0.001, 0.999)
            r_target[:, t] = r_col

        pd.DataFrame({
            "target_id": targets.target_id,
            "dtxsid": targets.dtxsid,
            "effect_family": targets.effect_family,
            "mechanism_group": targets.mechanism_group,
            "borrowed_probability_model": borrowed_model,
        }).to_csv(out_prob / f"target_probability_model_gate_x{int(x*100)}.csv", index=False)

        # Chemical-level probabilities cover the complete 5,530-chemical surveillance universe.
        # Direct chemical×effect targets override borrowed probabilities. For sparse chemicals,
        # validated MOA gates, taxonomy priors and MOA-specific effect-family weights provide
        # probability-only borrowing; no absolute LC/EC/HC value is predicted.
        p_chemical = np.empty((s_count, len(chemicals)), dtype=float)
        r_chemical = np.empty_like(p_chemical)
        n_chemical = np.zeros_like(p_chemical)
        target_lookup = {(row.dtxsid, row.effect_family): int(i) for i, row in targets.iterrows()}
        chemical_audit_rows = []
        for c, ci in c_index.items():
            mech = chem_mech.get(c, "MOA_UNRESOLVED")
            mi = mechanism_index[mech]
            ew = effect_weight_map.get(mech, np.full(len(effects), 1 / len(effects)))
            pc = np.zeros((s_count, len(effects)), dtype=float)
            rc = np.zeros_like(pc)
            nc = np.zeros_like(pc)
            direct_effects = 0
            moa_effects = 0
            for eff, ei in effect_index.items():
                target_key = (c, eff)
                if target_key in target_lookup:
                    tt = target_lookup[target_key]
                    pc[:, ei] = p_target[:, tt]
                    rc[:, ei] = r_target[:, tt]
                    nc[:, ei] = n_target[:, tt]
                    direct_effects += 1
                    continue
                group_key = f"{mech}||{eff}"
                use_moa = group_key in validated_moa_groups
                if use_moa:
                    prior_col = prior_moa[:, mi, ei]
                    rel_col = rel_moa[:, mi, ei]
                    moa_effects += 1
                    cal_key = "moa_taxonomy_calibration"
                else:
                    prior_col = prior_tax[:, ei]
                    rel_col = rel_tax[:, ei]
                    cal_key = "taxonomy_calibration"
                borrowed = q0 + rel_col * (prior_col - q0)
                if target_cal and target_cal.get(cal_key) is not None:
                    params = target_cal[cal_key]
                    logit = np.log(np.clip(borrowed, 1e-6, 1-1e-6) / np.clip(1-borrowed, 1e-6, 1-1e-6))
                    borrowed = 1 / (1 + np.exp(-(float(params["intercept"]) + float(params["coefficient"]) * logit)))
                pc[:, ei] = np.clip(borrowed, 0.001, 0.999)
                rc[:, ei] = rel_col
            p_chemical[:, ci] = pc @ ew
            r_chemical[:, ci] = rc @ ew
            n_chemical[:, ci] = nc.sum(axis=1)
            chemical_audit_rows.append({
                "dtxsid": c, "mechanism_group": mech,
                "n_directly_estimable_effect_families": direct_effects,
                "n_validated_moa_borrowed_effect_families": moa_effects,
                "effect_weight_entropy": float(-(ew * np.log(np.clip(ew, 1e-12, 1))).sum()),
                "mean_species_probability_reliability": float(r_chemical[:, ci].mean()),
                "fraction_species_with_any_direct_support": float((n_chemical[:, ci] > 0).mean()),
            })
        pd.DataFrame(chemical_audit_rows).to_csv(
            out_prob / f"chemical_probability_evidence_audit_x{int(x*100)}.csv", index=False
        )
        p_chemical = p_chemical.astype(np.float32, copy=False)
        r_chemical = r_chemical.astype(np.float32, copy=False)
        n_chemical = np.clip(n_chemical, 0, 65535).astype(np.uint16, copy=False)

        rho_info = estimate_working_rho(mm, candidates)
        rho = float(rho_info["rho_working"])
        rho_audit[str(x)] = rho_info

        np.savez_compressed(
            out_prob / f"species_protective_target_tail_probability_x{int(x*100)}.npz",
            p=p_target.astype(np.float32), reliability=r_target.astype(np.float32), direct_n=np.clip(n_target,0,65535).astype(np.uint16),
            species=np.array(species, object), target_ids=targets.target_id.to_numpy(object),
            chemicals=targets.dtxsid.to_numpy(object), effects=targets.effect_family.to_numpy(object),
            mechanisms=targets.mechanism_group.to_numpy(object),
        )
        np.savez_compressed(
            out_prob / f"species_chemical_tail_probability_x{int(x*100)}.npz",
            p=p_chemical, reliability=r_chemical, direct_n=n_chemical,
            species=np.array(species, object), chemicals=np.array(chemicals, object),
            mechanisms=np.array(mechanisms, object),
        )
        import gc
        del p_target, r_target, n_target, prior_moa, rel_moa, prior_tax, rel_tax, before_moa, before_moa_rel, before_tax, before_tax_rel
        gc.collect()

        if x in [0.80, 0.90, 0.95]:
            pidx = np.array([c_index[c] for c in priority_weights.index], int)
            summary = candidates.copy()
            summary["mean_tail_probability_all"] = p_chemical.mean(axis=1)
            summary["mean_tail_probability_priority"] = p_chemical[:, pidx].mean(axis=1)
            summary["mean_reliability"] = r_chemical.mean(axis=1)
            summary["direct_chemical_fraction"] = (n_chemical > 0).mean(axis=1)
            summary["support_tier"] = support_tier(summary.n_contexts)
            summary["direct_tail_hits"] = mm.groupby("latin_name").tail_binary.sum().reindex(summary.latin_name).fillna(0).to_numpy()
            alpha = summary.direct_tail_hits.to_numpy(float) + 2 * q0
            beta = summary.n_contexts.to_numpy(float) - summary.direct_tail_hits.to_numpy(float) + 2 * (1-q0)
            from scipy.stats import beta as beta_dist
            summary["support_adjusted_tail_mean"] = alpha / (alpha + beta)
            summary["support_adjusted_tail_q025"] = beta_dist.ppf(0.025, alpha, beta)
            summary["support_adjusted_tail_q975"] = beta_dist.ppf(0.975, alpha, beta)
            summary.to_csv(out_prob / f"species_probability_summary_x{int(x*100)}.csv", index=False)
            if x == 0.95:
                mm.groupby(["mechanism_group", "class"]).agg(
                mean_tail_probability=("p_context", "mean"),
                n_contexts=("context_id", "nunique"),
                n_species=("latin_name", "nunique"),
            ).reset_index().to_csv(out_prob / "moa_taxon_sensitivity_summary.csv", index=False)
            ef = mm.groupby(["mechanism_group", "effect_family"]).size().rename("n_cells").reset_index()
            ef["effect_probability_within_moa"] = ef.n_cells / ef.groupby("mechanism_group").n_cells.transform("sum")
            ef.to_csv(out_prob / "moa_effect_family_probability_observed.csv", index=False)
            c2 = context.copy()
            c2["mechanism_group"] = c2.dtxsid.map(chem_mech).fillna("MOA_UNRESOLVED")
            c2.groupby("mechanism_group").agg(
                n_contexts=("context_id", "size"),
                median_ssd_sd=("ssd_sd", "median"),
                wide_fraction=("ssd_width_class", lambda q: float((q == "wide").mean())),
                left_skew_fraction=("ssd_shape_class", lambda q: float((q == "left_skew_sensitive_tail").mean())),
            ).reset_index().to_csv(out_prob / "moa_ssd_shape_summary.csv", index=False)

            # Rapid-warning evidence remains a parallel, observed-only layer; it is not mixed into protective measured-tail capture.
            warning = model[
                valid
                & model.endpoint_band_v16.isin(WARNING_BANDS)
                ].copy()
            warning["warning_context_id"] = warning[
                ["dtxsid", "effect_family", "duration_window_v16", "medium_family"]
            ].astype(str).agg("|".join, axis=1)
            wc = warning.groupby("warning_context_id").latin_name.nunique()
            warning = warning[warning.warning_context_id.isin(wc[wc >= 3].index)]
            warning_summary = warning.groupby("latin_name").agg(
                warning_contexts=("warning_context_id", "nunique"),
                warning_chemicals=("dtxsid", "nunique"),
                warning_effect_families=("effect_family", "nunique"),
                warning_median_log10_umol_L=("mu_log10_umol_L", "median"),
            ).reset_index()
            warning["warning_response_window"] = warning.duration_window_v16.map({
                "h00_02_ultra_early": "h00_02_ultra_early",
                "h02_06_early": "h02_06_early",
                "h06_24_same_day": "h06_24_same_day",
                "h24_48": "h24_48",
                "h48_96": "h48_168",
                "h96_168": "h48_168",
                "d07_14": "gt7d",
                "gt14d": "gt7d",
                "duration_unknown": "duration_unknown",
            }).fillna("duration_unknown")
            by_window = warning.groupby(["latin_name", "warning_response_window"]).size().unstack(fill_value=0).reset_index()
            warning_summary = warning_summary.merge(by_window, on="latin_name", how="left")
            warning_summary.to_csv(out_prob / "rapid_warning_species_evidence_observed_only.csv", index=False)
            summary[["latin_name", "mean_tail_probability_priority", "n_contexts", "n_chemicals"]].merge(
                warning_summary, on="latin_name", how="outer"
            ).to_csv(out_prob / "protective_vs_warning_species_contributions.csv", index=False)

        # Primary national curves use complete chemical-level universes:
        # 362 exposure-priority chemicals and all 5,530 protective-evidence chemicals.
        # Effect-resolved targets remain available as an evidence audit, but do not truncate
        # the chemical universe shown in the main k×protection curves.
        universes = {
            "priority": priority_weights,
            "all_chemicals": uniform_weights,
        }
        # Regulatory/taxonomic baseline sequences do not depend on chemical weights.
        wet_seq = [s_index[s] for s in EPA_WET_METHOD_SPECIES if s in s_index]
        epa_wqc_seq = epa_wqc_sequence(candidates, 20)
        canada_seq = canada_type_a_sequence(candidates, 20)
        tax_seq = taxonomy_sequence(candidates, 20)
        for universe, chemical_weights in universes.items():
            chem_names = [c for c in chemical_weights.index if c in c_index]
            chem_idx = np.array([c_index[c] for c in chem_names], int)
            cw = chemical_weights.reindex(chem_names).to_numpy(float)
            cw = cw / cw.sum()
            p_univ = p_chemical[:, chem_idx]
            r_univ = r_chemical[:, chem_idx]
            n_univ = n_chemical[:, chem_idx]
            data_seq = deterministic_greedy(p_univ, cw, species, 20)
            for method, seq in {
                "data_driven": data_seq,
                "EPA_WET_fixed_method_species": wet_seq,
                "EPA_WQC_taxonomic_requirements": epa_wqc_seq,
                "Canada_Type_A_composition": canada_seq,
                "taxonomy_diversity_baseline": tax_seq,
            }.items():
                for rank, j in enumerate(seq, 1):
                    row = candidates.iloc[j]
                    sequence_rows.append({
                        "universe": universe, "protection_target_x": x, "method": method,
                        "rank": rank, "latin_name": species[j],
                        "n_observed_contexts": int(row.n_contexts),
                        "n_observed_chemicals": int(row.n_chemicals),
                        "official_or_common_method_species": bool(row.official_or_common_method_species),
                        "eligible_states": int(row.eligible_states),
                        "candidate_universe_sha256": candidate_hash,
                    })
                seq_arr = np.asarray(seq[:20], int)
                pp_seq = p_univ[seq_arr]
                q_seq = copula_union_sequence(pp_seq, rho)
                simple_seq = np.maximum.accumulate(pp_seq, axis=0)
                rel_seq = np.maximum.accumulate(r_univ[seq_arr], axis=0)
                direct_seq = np.logical_or.accumulate(n_univ[seq_arr] > 0, axis=0)
                for k in range(1, len(seq_arr) + 1):
                    q = q_seq[k-1]
                    simple = simple_seq[k-1]
                    max_rel = rel_seq[k-1]
                    any_direct = direct_seq[k-1]
                    curves.append({
                        "universe": universe, "protection_target_x": x, "method": method, "k": k,
                        "expected_weighted_joint_coverage": float(cw @ q),
                        "expected_weighted_simple_best_species": float(cw @ simple),
                        "joint_gain_over_simple": float(cw @ (q - simple)),
                        "weighted_fraction_q_ge_0_5": float(cw[q >= 0.5].sum()),
                        "weighted_fraction_q_ge_0_8": float(cw[q >= 0.8].sum()),
                        "weighted_fraction_q_ge_0_95": float(cw[q >= 0.95].sum()),
                        "weighted_mean_max_probability_reliability": float(cw @ max_rel),
                        "weighted_fraction_any_direct_support": float(cw[any_direct].sum()),
                        "rho": rho, "n_chemical_effect_targets": int(len(targets)),
                        "n_chemicals": int(len(chem_names)),
                        "selection_objective": "chemical-level measured-tail expected capture; deterministic greedy independence surrogate; final evaluation by Gaussian copula",
                    })

        if x == 0.90:
            for scenario in [c for c in priority.columns if re.fullmatch(r"w[235]{3}", c)]:
                cw = priority.set_index("DTXSID")[scenario].astype(float)
                cw /= cw.sum()
                chem_names = [c for c in cw.index if c in c_index]
                cidx = np.array([c_index[c] for c in chem_names], int)
                ww = cw.reindex(chem_names).to_numpy(float); ww = ww / ww.sum()
                seq = deterministic_greedy(p_chemical[:, cidx], ww, species, 20)
                for k in [5, 10, 20]:
                    q = copula_union(p_chemical[np.ix_(seq[:k], cidx)], rho)
                    sensitivity_rows.append({
                        "scenario": scenario, "k": k,
                        "expected_joint_coverage": float(ww @ q),
                        "top_species": " | ".join(species[j] for j in seq[:k]),
                    })

        if "Daphnia magna" in s_index:
            di = s_index["Daphnia magna"]
            rank_values = [r["rank"] for r in sequence_rows if r["universe"] == "priority" and r["protection_target_x"] == x and r["method"] == "data_driven" and r["latin_name"] == "Daphnia magna"]
            pidx_dm = np.array([c_index[c] for c in priority_weights.index], int)
            daphnia_rows.append({
                "protection_target_x": x,
                "deterministic_panel_rank": int(rank_values[0]) if rank_values else np.nan,
                "n_contexts": int(candidates.iloc[di].n_contexts),
                "n_chemicals": int(candidates.iloc[di].n_chemicals),
                "support_tier": str(candidates.iloc[di].support_tier),
                "mean_tail_probability_all": float(p_chemical[di].mean()),
                "mean_tail_probability_priority": float(p_chemical[di, pidx_dm].mean()),
                "direct_fraction_all": float((n_chemical[di] > 0).mean()),
                "candidate_universe_sha256": candidate_hash,
            })

        # Release per-target dense matrices before constructing the next protection target.
        del p_univ, r_univ, n_univ, pp_seq, q_seq, simple_seq, rel_seq, direct_seq
        del p_chemical, r_chemical, n_chemical, mm, direct, evidence_moa, evidence_tax, maps_moa, maps_tax
        import gc
        gc.collect()

    pd.DataFrame(sequence_rows).to_csv(out_panels / "national_panel_sequences.csv", index=False)
    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(out_panels / "national_k1_20_coverage_curves.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out_panels / "priority_weight_panel_sensitivity.csv", index=False)
    curve_df[(curve_df.protection_target_x == 0.90) & curve_df.k.isin([5, 10, 20])].to_csv(
        out_panels / "x90_method_comparison.csv", index=False
    )

    pd.DataFrame(daphnia_rows).to_csv(out_panels / "daphnia_magna_audit.csv", index=False)

    manifest = {
        "version": "v16.5",
        "candidate_species": s_count,
        "all_evidence_species_for_random_and_optimization": int(len(all_evidence_species)),
        "candidate_universe_rule": "all formal binomial species with >=1 valid protective context; no hard support cutoff and no standard-species exception; support controls shrinkage, uncertainty and testing priority; occurrence/NAS are downstream soft-local state-panel relevance weights",
        "candidate_universe_sha256": candidate_hash,
        "contexts": int(len(context)),
        "chemical_effect_targets": int(len(targets)),
        "all_chemical_universe": int(len(chemicals)),
        "priority_chemical_universe": int(len(priority_weights)),
        "dependence_audit_by_x": rho_audit,
        "probability_definition": "effect-family-specific probability that a species threshold is at or below a context-specific working lower-tail threshold; direct evidence is support-shrunk; missing cells borrow MOA × effect then genus/family/class and at most 30% trait-neighbour probability",
        "no_absolute_toxicity_prediction": True,
        "evidence_layers": {
            "protective": "apical adverse effect families; direct chemical×effect evidence is retained, then combined to complete chemical-level probabilities using MOA-specific effect-family weights; main national curves evaluate all 362 priority and all 5,530 surveillance chemicals",
            "rapid_warning": "heartbeat/behavior/physiology across 0-2 h, 2-6 h, 6-24 h, 24-48 h, 48 h-7 d and >7 d, observed-only parallel evidence; never mixed with protective measured-tail capture",
            "mechanism_support": "MOA/AOP informs probability priors and effect/SSD-shape audits only",
        },
        "sparse_extreme_control": "direct evidence uses two prior-equivalent observations and reliability 1-exp(-n/3); probability priors shrink to 1-x",
        "coverage_definition": "Legacy field name for Gaussian-copula expected capture: the probability that at least one panel member lies in the chemical-level working lower tail; chemical probabilities are effect-family-weighted and evidence reliability is reported alongside expected capture",
        "selection_definition": "deterministic greedy independence surrogate with lexical tie-breaking; final reported expected capture uses fixed copula working dependence",
        "whole_chemical_probability_calibration": calibration,
    }
    (out_panels / "panel_probability_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
