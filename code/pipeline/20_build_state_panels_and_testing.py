#!/usr/bin/env python3
"""Build state-level regional-localization panels and testing priorities.

The regional-localization workflow is operationalized at state level because
chemical-monitoring and species-relevance inputs are most complete at that scale.
It does not use state occurrence as a hard field-deployment gate. For each
supported state-level application, it re-optimizes the lower-5% measured-tail
panel against state-priority chemical weights and a soft species relevance
weight. The species
weight combines direct state occurrence, adjacent-state occurrence, broad
occurrence support, and evidence/method-support representativeness. Therefore a
localized panel is a regional protection-oriented screening panel whose
state-level implementation remains subject to ecological and operational review.
The same stage independently writes national species-by-candidate-MOA testing
priorities for main Figure 5c and separate state-level testing-priority
patterns for the SI.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


def copula_union(P: np.ndarray, rho: float = 0.15, nodes: int = 16) -> np.ndarray:
    P = np.clip(np.asarray(P, float), 1e-7, 1 - 1e-7)
    if P.ndim == 1:
        P = P[None, :]
    if P.shape[0] == 1:
        return P[0]
    x, w = hermgauss(nodes)
    z = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)
    t = norm.ppf(1 - P)
    sr = math.sqrt(rho)
    sd = math.sqrt(1 - rho)
    no = np.zeros(P.shape[1])
    for zz, ww in zip(z, w):
        no += ww * np.prod(norm.cdf((t - sr * zz) / sd), axis=0)
    return 1 - no


def greedy(P: np.ndarray, w: np.ndarray, kmax: int, names: list[str] | None = None) -> list[int]:
    """Deterministic greedy ranking for the independent-union surrogate."""

    no = np.ones(P.shape[1])
    selected: list[int] = []
    remaining = list(range(P.shape[0]))
    names = names or [str(i) for i in range(P.shape[0])]
    for _ in range(min(kmax, len(remaining))):
        rr = np.asarray(remaining, int)
        gains = P[rr] @ (w * no)
        best = np.nanmax(gains)
        tied = [remaining[i] for i, gain in enumerate(gains) if np.isclose(gain, best, rtol=0, atol=1e-14)]
        j = min(tied, key=lambda q: names[q])
        selected.append(j)
        no *= 1 - P[j]
        remaining.remove(j)
    return selected


def normalized_positive(values: pd.Series | np.ndarray, floor: float = 0.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, floor, None)
    mx = float(x.max()) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=float)
    return x / mx


def readable_moa_term(value: object) -> str:
    """Return a short, reader-facing MOA term without ontology prefixes.

    The source annotation table can contain labels such as ``MIE::Activation,
    AhR`` or ``MOAFAM::...``. Figure 5 and its source data need directly
    readable terms, so this function removes ontology prefixes and collapses
    common detailed MIE labels into manuscript-scale mechanism families.
    """

    if not isinstance(value, str) or value.strip() == "":
        return "Unassigned"
    v = value.strip()
    if v.upper() in {"MOA_UNRESOLVED", "UNRESOLVED", "NONE", "NAN"}:
        return "Unassigned"
    if "::" in v:
        _, rest = v.split("::", 1)
    else:
        rest = v
    rest = rest.replace("_", " ").strip()
    low = rest.lower()
    if any(token in low for token in ("acetylcholinesterase", "sodium channel", "ionotrop", "glutamate", "neuro")):
        return "neurotransmission"
    if any(token in low for token in ("thyro", "deiodinase", "symporter")):
        return "thyroid axis"
    if any(token in low for token in ("deposition of energy", "mitochond", "oxidative phosphorylation", "nadh")):
        return "energy metabolism"
    if any(token in low for token in ("photosystem", "photosynthesis")):
        return "photosynthesis"
    if any(token in low for token in ("ahr", "androgen", "estrogen", "ppar", "receptor", "cyp")):
        return "receptor/xenobiotic signaling"
    if "histone deacetylase" in low:
        return "histone deacetylase inhibition"
    if "vegfr2" in low:
        return "VEGFR2 inhibition"
    if "calcineurin" in low:
        return "calcineurin activity inhibition"
    if any(token in low for token in ("alkylation", "reactive", "oxid", "strand breaks")):
        return "reactive chemistry"
    if "," in rest:
        parts = [part.strip() for part in rest.split(",") if part.strip()]
        if len(parts) >= 2:
            rest = f"{parts[1]} {parts[0]}".strip()
        elif parts:
            rest = parts[0]
    return rest if rest else "Unassigned"


def build_species_moa_testing_matrix(
    pairs: pd.DataFrame,
    species_priority: pd.DataFrame,
    chemical_priority: pd.DataFrame,
    *,
    scope: str,
    priority_col: str = "pair_testing_priority",
    top_species_n: int = 10,
    top_moa_n: int = 8,
) -> pd.DataFrame:
    """Aggregate national species-chemical priorities to a candidate-MOA matrix."""

    if pairs.empty:
        return pd.DataFrame()
    pairs = pairs.copy()
    pairs["moa_term"] = pairs["moa_term"].fillna("Unassigned").astype(str)
    pairs = pairs[~pairs["moa_term"].str.contains("Unassigned|UNRESOLVED", case=False, na=False)].copy()
    if pairs.empty:
        return pd.DataFrame()

    species_totals = (
        pairs.groupby("latin_name", as_index=False)
        .agg(
            species_total_testing_priority=(priority_col, "sum"),
            species_total_missing_pairs=("DTXSID", "size"),
        )
        .sort_values(
            ["species_total_testing_priority", "species_total_missing_pairs", "latin_name"],
            ascending=[False, False, True],
        )
    )
    moa_totals = (
        pairs.groupby("moa_term", as_index=False)
        .agg(
            candidate_moa_total_testing_priority=(priority_col, "sum"),
            candidate_moa_total_missing_pairs=("DTXSID", "size"),
        )
        .sort_values(
            ["candidate_moa_total_testing_priority", "candidate_moa_total_missing_pairs", "moa_term"],
            ascending=[False, False, True],
        )
    )
    top_species = species_totals.head(top_species_n)["latin_name"].astype(str).tolist()
    top_moa = moa_totals.head(top_moa_n)["moa_term"].astype(str).tolist()
    chem_counts = chemical_priority.groupby("moa_term")["DTXSID"].nunique().to_dict()
    d = pairs[pairs["latin_name"].isin(top_species) & pairs["moa_term"].isin(top_moa)].copy()
    summary = (
        d.groupby(["latin_name", "moa_term"], as_index=False)
        .agg(
            n_missing_pairs=("DTXSID", "size"),
            sum_priority=(priority_col, "sum"),
            mean_tail_probability=("tail_probability", "mean"),
            mean_reliability=("reliability", "mean"),
        )
    )
    full = pd.MultiIndex.from_product([top_species, top_moa], names=["latin_name", "moa_term"]).to_frame(index=False)
    summary = full.merge(summary, on=["latin_name", "moa_term"], how="left")
    for col in ["n_missing_pairs", "sum_priority", "mean_tail_probability", "mean_reliability"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)
    summary["n_priority_chemicals_in_moa"] = summary["moa_term"].map(chem_counts).fillna(summary["n_missing_pairs"]).astype(float)
    summary["missingness_density"] = summary["n_missing_pairs"] / summary["n_priority_chemicals_in_moa"].replace(0, np.nan)
    summary["missingness_density"] = summary["missingness_density"].fillna(0.0)
    summary = summary.merge(species_priority[["latin_name", "species_testing_priority"]], on="latin_name", how="left")
    summary = summary.merge(
        species_totals[["latin_name", "species_total_testing_priority"]],
        on="latin_name",
        how="left",
    )
    summary = summary.merge(
        moa_totals[["moa_term", "candidate_moa_total_testing_priority"]],
        on="moa_term",
        how="left",
    )
    summary["candidate_moa_term"] = summary["moa_term"]
    summary["national_testing_priority_score"] = summary["sum_priority"]
    summary["testing_scope"] = scope
    summary["score_aggregation"] = (
        "sum of national species-chemical pair testing-priority scores across missing direct-evidence chemicals"
    )
    summary["score_components"] = (
        "national exposure weight x probability uncertainty x evidence uncertainty x "
        "candidate-MOA/chemical-form gap"
    )
    return summary


def state_neighbors() -> dict[str, set[str]]:
    """Contiguous-US neighbor graph plus DC-free state codes used for soft occurrence support."""

    pairs = [
        ("AL", "FL"), ("AL", "GA"), ("AL", "MS"), ("AL", "TN"),
        ("AZ", "CA"), ("AZ", "CO"), ("AZ", "NV"), ("AZ", "NM"), ("AZ", "UT"),
        ("AR", "LA"), ("AR", "MS"), ("AR", "MO"), ("AR", "OK"), ("AR", "TN"), ("AR", "TX"),
        ("CA", "NV"), ("CA", "OR"),
        ("CO", "KS"), ("CO", "NE"), ("CO", "NM"), ("CO", "OK"), ("CO", "UT"), ("CO", "WY"),
        ("CT", "MA"), ("CT", "NY"), ("CT", "RI"),
        ("DE", "MD"), ("DE", "NJ"), ("DE", "PA"),
        ("FL", "GA"),
        ("GA", "NC"), ("GA", "SC"), ("GA", "TN"),
        ("ID", "MT"), ("ID", "NV"), ("ID", "OR"), ("ID", "UT"), ("ID", "WA"), ("ID", "WY"),
        ("IL", "IN"), ("IL", "IA"), ("IL", "KY"), ("IL", "MO"), ("IL", "WI"),
        ("IN", "KY"), ("IN", "MI"), ("IN", "OH"),
        ("IA", "MN"), ("IA", "MO"), ("IA", "NE"), ("IA", "SD"), ("IA", "WI"),
        ("KS", "MO"), ("KS", "NE"), ("KS", "OK"),
        ("KY", "MO"), ("KY", "OH"), ("KY", "TN"), ("KY", "VA"), ("KY", "WV"),
        ("LA", "MS"), ("LA", "TX"),
        ("ME", "NH"),
        ("MD", "PA"), ("MD", "VA"), ("MD", "WV"),
        ("MA", "NH"), ("MA", "NY"), ("MA", "RI"), ("MA", "VT"),
        ("MI", "OH"), ("MI", "WI"),
        ("MN", "ND"), ("MN", "SD"), ("MN", "WI"),
        ("MS", "TN"),
        ("MO", "NE"), ("MO", "OK"), ("MO", "TN"),
        ("MT", "ND"), ("MT", "SD"), ("MT", "WY"),
        ("NE", "SD"), ("NE", "WY"),
        ("NV", "OR"), ("NV", "UT"),
        ("NH", "VT"),
        ("NJ", "NY"), ("NJ", "PA"),
        ("NM", "OK"), ("NM", "TX"), ("NM", "UT"),
        ("NY", "PA"), ("NY", "VT"),
        ("NC", "SC"), ("NC", "TN"), ("NC", "VA"),
        ("ND", "SD"),
        ("OH", "PA"), ("OH", "WV"),
        ("OK", "TX"),
        ("OR", "WA"),
        ("PA", "WV"),
        ("SD", "WY"),
        ("TN", "VA"),
        ("UT", "WY"),
        ("VA", "WV"),
    ]
    out: dict[str, set[str]] = {}
    for a, b in pairs:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def build_soft_local_weights(occ: pd.DataFrame, cand: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    """Species-state weights for soft-localized protection relevance.

    The score is intentionally a relevance weight, not a hard native-status claim.
    It combines exact state occurrence, occurrence probability/monitorability,
    adjacent-state evidence, broad state occurrence support, and candidate
    method-support/evidence support.
    """

    occ = occ.copy()
    occ["state_code"] = occ["state_code"].astype(str)
    occ["latin_name"] = occ["latin_name"].astype(str)
    occ["eligible_bool"] = occ["eligible_state_candidate"].fillna(False).astype(bool)
    occ["exact_bool"] = occ["exact_species_occurrence"].fillna(False).astype(bool)
    occ["occurrence_prob"] = pd.to_numeric(occ.get("occurrence_prob", 0), errors="coerce").fillna(0).clip(0, 1)
    occ["monitorability_score"] = pd.to_numeric(occ.get("monitorability_score", 0), errors="coerce").fillna(0).clip(0, 1)

    states = sorted(occ["state_code"].dropna().astype(str).unique().tolist())
    species_set = set(species)
    state_species = {
        st: set(g.loc[g["eligible_bool"] & g["latin_name"].isin(species_set), "latin_name"].astype(str))
        for st, g in occ.groupby("state_code")
    }
    neighbor_map = state_neighbors()
    state_species_neighbor: dict[str, set[str]] = {}
    for st in states:
        nearby = set()
        for nb in neighbor_map.get(st, set()):
            nearby |= state_species.get(nb, set())
        state_species_neighbor[st] = nearby

    exact_scores = (
        occ[occ["latin_name"].isin(species_set)]
        .groupby(["state_code", "latin_name"], as_index=False)
        .agg(
            exact_state_occurrence=("exact_bool", "max"),
            eligible_state_candidate=("eligible_bool", "max"),
            occurrence_prob=("occurrence_prob", "max"),
            monitorability_score=("monitorability_score", "max"),
        )
    )
    exact_scores["state_occurrence_score"] = np.where(
        exact_scores["eligible_state_candidate"],
        np.maximum(exact_scores["occurrence_prob"], 0.65),
        np.where(exact_scores["exact_state_occurrence"], np.maximum(exact_scores["occurrence_prob"], 0.35), 0.0),
    )
    exact_lookup = exact_scores.set_index(["state_code", "latin_name"]).to_dict("index")

    cmeta = cand.set_index("latin_name").reindex(species).copy()
    eligible_states = pd.to_numeric(cmeta["eligible_states"], errors="coerce").fillna(0)
    n_contexts = pd.to_numeric(cmeta["n_contexts"], errors="coerce").fillna(0)
    n_chemicals = pd.to_numeric(cmeta["n_chemicals"], errors="coerce").fillna(0)
    official = cmeta["official_or_common_method_species"].fillna(False).astype(bool).to_numpy()
    occurrence_breadth = normalized_positive(np.log1p(eligible_states), floor=0)
    support_score = 0.5 * normalized_positive(np.log1p(n_contexts), floor=0) + 0.5 * normalized_positive(np.log1p(n_chemicals), floor=0)
    method_support_score = np.clip(0.55 + 0.35 * support_score + 0.10 * official.astype(float), 0.35, 1.0)

    rows = []
    for st in states:
        neighbor_species = state_species_neighbor.get(st, set())
        for i, sp in enumerate(species):
            rec = exact_lookup.get((st, sp), {})
            state_score = float(rec.get("state_occurrence_score", 0.0))
            neighbor_score = 0.28 if sp in neighbor_species else 0.0
            broad_score = 0.18 * float(occurrence_breadth[i])
            local_relevance = max(0.08, state_score, neighbor_score, broad_score)
            if bool(rec.get("eligible_state_candidate", False)):
                evidence_class = "state_occurrence"
            elif neighbor_score > 0:
                evidence_class = "neighbor_occurrence"
            elif broad_score > 0.02:
                evidence_class = "regional_breadth"
            else:
                evidence_class = "model_transfer_low_weight"
            species_weight = float(np.clip(local_relevance * method_support_score[i], 0.03, 1.0))
            rows.append(
                {
                    "state_code": st,
                    "latin_name": sp,
                    "state_occurrence_score": state_score,
                    "neighbor_occurrence_score": neighbor_score,
                    "occurrence_breadth_score": float(occurrence_breadth[i]),
                    "method_support_representativeness_score": float(method_support_score[i]),
                    "soft_local_species_weight": species_weight,
                    "soft_local_evidence_class": evidence_class,
                    "eligible_state_candidate": bool(rec.get("eligible_state_candidate", False)),
                    "exact_state_occurrence": bool(rec.get("exact_state_occurrence", False)),
                }
            )
    return pd.DataFrame(rows)


def weighted_panel_probability(P_state: np.ndarray, species_weights: np.ndarray) -> np.ndarray:
    """Species-relevance-weighted chemical tail-capture contribution matrix."""

    return P_state * np.asarray(species_weights, dtype=float)[:, None]


def evaluate_sequence(
    P_state: np.ndarray,
    species_weights: np.ndarray,
    chemical_weights: np.ndarray,
    seq: list[int] | np.ndarray,
    rho: float,
) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros(P_state.shape[1], dtype=float)
    local_p = weighted_panel_probability(P_state[np.asarray(seq, int)], species_weights[np.asarray(seq, int)])
    return copula_union(local_p, rho=rho)


def national_top_sequence(root: Path, target_x: float, kmax: int, species_index: dict[str, int]) -> list[int]:
    national = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    seq = national[
        national["universe"].eq("priority")
        & national["method"].eq("data_driven")
        & np.isclose(national["protection_target_x"].astype(float), target_x)
        & national["rank"].between(1, kmax)
    ].sort_values("rank")
    return [species_index[s] for s in seq["latin_name"].astype(str).tolist() if s in species_index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--occurrence", required=True)
    ap.add_argument("--state-weights", required=True)
    ap.add_argument("--candidate-summary", required=True)
    ap.add_argument("--priority", required=True)
    ap.add_argument("--chem", required=True)
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--out-testing", required=True)
    ap.add_argument("--under-tested-registry", default="")
    args = ap.parse_args()

    out_state = Path(args.out_state)
    out_testing = Path(args.out_testing)
    out_state.mkdir(parents=True, exist_ok=True)
    out_testing.mkdir(parents=True, exist_ok=True)

    z = np.load(args.prob, allow_pickle=True)
    P = np.asarray(z["p"], dtype=float)
    R = np.asarray(z["reliability"], dtype=float)
    N = np.asarray(z["direct_n"])
    species = z["species"].astype(str).tolist()
    chemicals = z["chemicals"].astype(str).tolist()
    chemical_index = {c: i for i, c in enumerate(chemicals)}
    species_index = {s: i for i, s in enumerate(species)}

    occ = pd.read_csv(args.occurrence, low_memory=False)
    state_weights = pd.read_csv(args.state_weights, low_memory=False)
    cand = pd.read_csv(args.candidate_summary)
    priority = pd.read_csv(args.priority)
    chem = pd.read_csv(args.chem, low_memory=False)

    target_x = 0.95 if "x95" in Path(args.prob).name else (0.90 if "x90" in Path(args.prob).name else 0.80)
    manifest_path = Path(args.prob).parents[1] / "panels" / "panel_probability_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rho = float(manifest["dependence_audit_by_x"][str(target_x)]["rho_working"])

    soft_weights = build_soft_local_weights(occ, cand, species)
    soft_weights.to_csv(out_state / "state_soft_local_species_weights.csv", index=False)

    national_seq_20 = national_top_sequence(Path.cwd(), target_x, 20, species_index)
    national_seq_5 = national_seq_20[:5]
    rows: list[dict] = []
    curves: list[dict] = []
    comparison_rows: list[dict] = []

    for state_code, group in state_weights.groupby("state_code"):
        state_code = str(state_code)
        state_chemicals = [x for x in group["DTXSID"].astype(str) if x in chemical_index]
        if len(state_chemicals) < 3:
            continue
        chemical_idx = np.array([chemical_index[x] for x in state_chemicals], dtype=int)
        w = group.set_index("DTXSID")["state_weight"].reindex(state_chemicals).fillna(0).clip(lower=0).to_numpy(float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.repeat(1 / len(state_chemicals), len(state_chemicals))
        else:
            w = w / w.sum()

        sw = (
            soft_weights[soft_weights["state_code"].eq(state_code)]
            .set_index("latin_name")
            .reindex(species)
            .fillna(
                {
                    "soft_local_species_weight": 0.03,
                    "state_occurrence_score": 0.0,
                    "neighbor_occurrence_score": 0.0,
                    "occurrence_breadth_score": 0.0,
                    "method_support_representativeness_score": 0.35,
                    "soft_local_evidence_class": "model_transfer_low_weight",
                    "eligible_state_candidate": False,
                    "exact_state_occurrence": False,
                }
            )
        )
        species_weight = sw["soft_local_species_weight"].to_numpy(float)
        candidate_idx = np.where(species_weight >= 0.05)[0]
        if len(candidate_idx) < 3:
            candidate_idx = np.argsort(-species_weight)[: max(3, min(20, len(species_weight)))]
        candidate_idx = np.array(sorted(candidate_idx, key=lambda i: species[i]), dtype=int)

        P_state = P[np.ix_(candidate_idx, chemical_idx)]
        candidate_weights = species_weight[candidate_idx]
        objective_matrix = weighted_panel_probability(P_state, candidate_weights)
        seq_local = greedy(objective_matrix, w, 20, names=[species[i] for i in candidate_idx])
        seq_global = [int(candidate_idx[j]) for j in seq_local]

        state_q5 = evaluate_sequence(P[np.ix_(np.arange(len(species)), chemical_idx)], species_weight, w, seq_global[:5], rho)
        national_q5 = evaluate_sequence(P[np.ix_(np.arange(len(species)), chemical_idx)], species_weight, w, national_seq_5, rho)
        hard_local_national = [
            j for j in national_seq_5
            if bool(sw.loc[species[j], "eligible_state_candidate"]) if species[j] in sw.index
        ]
        hard_local_q5 = evaluate_sequence(P[np.ix_(np.arange(len(species)), chemical_idx)], species_weight, w, hard_local_national, rho)
        comparison_rows.append(
            {
                "state_code": state_code,
                "k": 5,
                "n_priority_chemicals": len(state_chemicals),
                "n_candidate_species": len(candidate_idx),
                "state_soft_local_expected_weighted_joint_coverage_x95": float(w @ state_q5),
                "national_top5_soft_local_expected_weighted_joint_coverage_x95": float(w @ national_q5),
                "national_top5_hard_local_overlap_expected_weighted_joint_coverage_x95": float(w @ hard_local_q5),
                "state_gain_over_national_top5_percent_points": float(100 * (w @ (state_q5 - national_q5))),
                "state_gain_over_hard_local_overlap_percent_points": float(100 * (w @ (state_q5 - hard_local_q5))),
                "national_top5_soft_local_species": "; ".join(species[j] for j in national_seq_5),
                "national_top5_hard_local_overlap_species": "; ".join(species[j] for j in hard_local_national) or "none",
                "state_panel_policy": "soft-localized objective: chemical weights are state priority weights; species relevance is a soft state/neighbor/breadth/method-support weight, not a hard occurrence gate",
                "comparison_universe": "same state-priority chemicals, same soft-local species relevance weights, k=5",
            }
        )

        for rank, j in enumerate(seq_global, 1):
            rows.append(
                {
                    "state_code": state_code,
                    "rank": rank,
                    "latin_name": species[j],
                    "marginal_direct_support_chemicals": int((N[j, chemical_idx] > 0).sum()),
                    "mean_tail_probability_state": float(P[j, chemical_idx].mean()),
                    "mean_reliability_state": float(R[j, chemical_idx].mean()),
                    "soft_local_species_weight": float(species_weight[j]),
                    "state_occurrence_score": float(sw.loc[species[j], "state_occurrence_score"]),
                    "neighbor_occurrence_score": float(sw.loc[species[j], "neighbor_occurrence_score"]),
                    "occurrence_breadth_score": float(sw.loc[species[j], "occurrence_breadth_score"]),
                    "method_support_representativeness_score": float(sw.loc[species[j], "method_support_representativeness_score"]),
                    "candidate_status": str(sw.loc[species[j], "soft_local_evidence_class"]),
                }
            )

        for k in range(1, len(seq_global) + 1):
            q = evaluate_sequence(P[np.ix_(np.arange(len(species)), chemical_idx)], species_weight, w, seq_global[:k], rho)
            curves.append(
                {
                    "state_code": state_code,
                    "k": k,
                    "n_priority_chemicals": len(state_chemicals),
                    "n_candidate_species": len(candidate_idx),
                    f"expected_weighted_joint_coverage_x{int(target_x*100)}": float(w @ q),
                    "weighted_fraction_q_ge_0_5": float(w[q >= 0.5].sum()),
                    "weighted_fraction_q_ge_0_8": float(w[q >= 0.8].sum()),
                    "weighted_fraction_q_ge_0_95": float(w[q >= 0.95].sum()),
                    "state_panel_policy": "soft_localized_v2",
                }
            )

    seq_df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    comp_df = pd.DataFrame(comparison_rows)
    seq_df.to_csv(out_state / f"state_x{int(target_x*100)}_panel_sequences.csv", index=False)
    curve_df.to_csv(out_state / f"state_x{int(target_x*100)}_k1_20_coverage.csv", index=False)
    comp_df.to_csv(out_state / f"state_x{int(target_x*100)}_k5_soft_local_comparison.csv", index=False)

    # National testing priorities for main Figure 5c.
    pidx = np.array([chemical_index[c] for c in priority.DTXSID.astype(str) if c in chemical_index])
    pchems = [c for c in priority.DTXSID.astype(str) if c in chemical_index]
    w = priority.set_index("DTXSID").national_weight.reindex(pchems).fillna(0).to_numpy(float)
    w = w / w.sum()
    PP = P[:, pidx]
    NN = N[:, pidx]
    RR = R[:, pidx]
    cmeta = cand.set_index("latin_name").reindex(species)
    occ_breadth = cmeta.eligible_states.fillna(0).to_numpy() / max(cmeta.eligible_states.max(), 1)
    unc = (PP * (1 - PP)) @ w
    under = 1 / np.sqrt(1 + (NN > 0).sum(axis=1))
    mech_gap = ((NN == 0) * PP).mean(axis=1)
    score = unc * under * (0.25 + 0.75 * occ_breadth) * (0.5 + 0.5 * mech_gap)
    sp = pd.DataFrame(
        {
            "latin_name": species,
            "species_testing_priority": score,
            "weighted_probability_uncertainty": unc,
            "under_tested_score": under,
            "occurrence_breadth_score": occ_breadth,
            "moa_probability_gap": mech_gap,
            "n_direct_priority_chemicals": (NN > 0).sum(axis=1),
            "n_observed_contexts": cmeta.n_contexts.to_numpy(),
            "n_observed_chemicals": cmeta.n_chemicals.to_numpy(),
        }
    ).sort_values("species_testing_priority", ascending=False)
    sp.to_csv(out_testing / "national_species_testing_priorities.csv", index=False)

    mpr = priority.set_index("DTXSID").reindex(pchems).reset_index()
    nsp = (NN > 0).sum(axis=0)
    species_gap = 1 / np.sqrt(1 + nsp)
    mean_unc = (PP * (1 - PP)).mean(axis=0)
    mech = chem.set_index("DTXSID").reindex(pchems)
    mg = mech.primary_moa.fillna("MOA_UNRESOLVED").astype(str)
    moa_terms = mg.map(readable_moa_term)
    group_support = pd.Series(nsp, index=pchems).groupby(moa_terms.values).transform("sum").to_numpy()
    moa_gap = 1 / np.sqrt(1 + group_support)
    form_terms = mech.chemical_form_class.fillna("unresolved chemical form").astype(str)
    form_support = pd.Series(nsp, index=pchems).groupby(form_terms.values).transform("sum").to_numpy()
    form_gap = 1 / np.sqrt(1 + form_support)
    moa_gap_norm = normalized_positive(moa_gap)
    form_gap_norm = normalized_positive(form_gap)
    mechanism_form_gap_multiplier = 0.5 + 0.25 * moa_gap_norm + 0.25 * form_gap_norm
    chem_score = (
        w
        * species_gap
        * (0.5 + 0.5 * mean_unc / 0.25)
        * mechanism_form_gap_multiplier
    )
    ch = pd.DataFrame(
        {
            "DTXSID": pchems,
            "preferred_name": mpr["PREFERRED_NAME"],
            "national_weight": w,
            "chemical_testing_priority": chem_score,
            "n_candidate_species_with_direct_data": nsp,
            "species_data_gap": species_gap,
            "mean_probability_uncertainty": mean_unc,
            "primary_moa": mg.to_numpy(),
            "moa_term": moa_terms.to_numpy(),
            "chemical_form_class": mech.chemical_form_class.to_numpy(),
            "moa_group_gap": moa_gap,
            "chemical_form_gap": form_gap,
            "mechanism_form_gap_multiplier": mechanism_form_gap_multiplier,
        }
    ).sort_values("chemical_testing_priority", ascending=False)
    ch["chemical_testing_priority_norm"] = ch.chemical_testing_priority / ch.chemical_testing_priority.max()
    ch.to_csv(out_testing / "national_chemical_moa_testing_priorities.csv", index=False)

    pair: list[dict[str, object]] = []
    top_c = ch.head(200).DTXSID.tolist()
    pweight = dict(zip(pchems, w))
    name = chem.set_index("DTXSID").PREFERRED_NAME.to_dict()
    moam = chem.set_index("DTXSID").primary_moa.to_dict()
    candidate_moa = dict(zip(ch.DTXSID.astype(str), ch.moa_term.astype(str)))
    chemical_form = dict(zip(ch.DTXSID.astype(str), ch.chemical_form_class.fillna("unresolved chemical form").astype(str)))
    gap_multiplier = dict(zip(ch.DTXSID.astype(str), ch.mechanism_form_gap_multiplier.astype(float)))
    for c in top_c:
        j = chemical_index[c]
        missing_species_idx = np.where(N[:, j] == 0)[0]
        for i in missing_species_idx:
            s = species[i]
            if N[i, j] > 0:
                continue
            val = (
                pweight.get(c, 0)
                * P[i, j]
                * (1 - P[i, j])
                * (1 - R[i, j])
                * gap_multiplier.get(c, 0.5)
            )
            pair.append(
                {
                    "latin_name": s,
                    "DTXSID": c,
                    "preferred_name": name.get(c, ""),
                    "primary_moa": moam.get(c, "MOA_UNRESOLVED"),
                    "moa_term": candidate_moa.get(c, readable_moa_term(moam.get(c, "MOA_UNRESOLVED"))),
                    "candidate_moa_term": candidate_moa.get(
                        c, readable_moa_term(moam.get(c, "MOA_UNRESOLVED"))
                    ),
                    "chemical_form_class": chemical_form.get(c, "unresolved chemical form"),
                    "mechanism_form_gap_multiplier": gap_multiplier.get(c, 0.5),
                    "pair_testing_priority": val,
                    "tail_probability": P[i, j],
                    "reliability": R[i, j],
                    "testing_scope": "national",
                    "reason": (
                        "national exposure x probability uncertainty x evidence uncertainty x "
                        "missing direct species-chemical evidence x candidate-MOA/chemical-form gap"
                    ),
                }
            )
    pair_all_df = pd.DataFrame(pair).sort_values("pair_testing_priority", ascending=False)
    pair_df = pair_all_df.head(8000).copy()
    pair_df.to_csv(out_testing / "national_species_chemical_testing_pairs.csv", index=False)
    national_species_moa = build_species_moa_testing_matrix(
        pair_all_df,
        sp,
        ch,
        scope="national",
        priority_col="pair_testing_priority",
        top_species_n=10,
        top_moa_n=8,
    )
    national_species_moa.to_csv(out_testing / "national_species_moa_testing_matrix.csv", index=False)
    reps = ch.sort_values(["moa_term", "chemical_testing_priority"], ascending=[True, False]).groupby(
        "moa_term", as_index=False
    ).head(3)
    reps.to_csv(out_testing / "national_moa_representative_chemicals.csv", index=False)

    # State-level testing priorities are retained for the SI. They use the
    # same evidence-acquisition logic as the national matrix but replace the
    # national chemical weights with state-priority chemical weights and add the
    # soft-local species relevance weight.
    chem_name = chem.set_index("DTXSID").PREFERRED_NAME.to_dict()
    chem_primary_moa = chem.set_index("DTXSID").primary_moa.to_dict()
    state_moa_rows: list[dict[str, object]] = []
    state_pair_rows: list[dict[str, object]] = []
    state_species_moa_rows: list[dict[str, object]] = []
    for state_code, group in state_weights.groupby("state_code"):
        state_code = str(state_code)
        state_chemicals = [x for x in group["DTXSID"].astype(str) if x in chemical_index]
        if not state_chemicals:
            continue
        chemical_idx = np.array([chemical_index[x] for x in state_chemicals], dtype=int)
        state_w = group.set_index("DTXSID")["state_weight"].reindex(state_chemicals).fillna(0).clip(lower=0).to_numpy(float)
        if not np.isfinite(state_w).all() or state_w.sum() <= 0:
            state_w = np.repeat(1 / len(state_chemicals), len(state_chemicals))
        else:
            state_w = state_w / state_w.sum()
        sw = (
            soft_weights[soft_weights["state_code"].eq(state_code)]
            .set_index("latin_name")
            .reindex(species)
            .fillna({"soft_local_species_weight": 0.03, "soft_local_evidence_class": "model_transfer_low_weight"})
        )
        species_weight = sw["soft_local_species_weight"].to_numpy(float)
        local_uncertainty = P[:, chemical_idx] * (1 - P[:, chemical_idx]) * (1 - R[:, chemical_idx]) * (N[:, chemical_idx] == 0)
        state_species_score = species_weight * (local_uncertainty @ state_w)
        top_local_species_idx = np.argsort(-state_species_score)[: min(25, len(species))]
        state_pair_candidates: list[dict[str, object]] = []
        for local_col, chem_id in enumerate(state_chemicals):
            j = chemical_idx[local_col]
            moa_term = readable_moa_term(chem_primary_moa.get(chem_id, "MOA_UNRESOLVED"))
            if moa_term == "Unassigned":
                continue
            missing = N[:, j] == 0
            base = species_weight * P[:, j] * (1 - P[:, j]) * (1 - R[:, j]) * missing
            priority_value = float(state_w[local_col] * base.sum())
            if priority_value <= 0:
                continue
            weighted_tail_denominator = float((species_weight * missing).sum())
            state_moa_rows.append(
                {
                    "state_code": state_code,
                    "DTXSID": chem_id,
                    "preferred_name": chem_name.get(chem_id, ""),
                    "primary_moa": chem_primary_moa.get(chem_id, "MOA_UNRESOLVED"),
                    "moa_term": moa_term,
                    "state_weight": float(state_w[local_col]),
                    "state_moa_testing_priority": priority_value,
                    "n_missing_species_pairs": int(missing.sum()),
                    "soft_weighted_missing_species_sum": weighted_tail_denominator,
                    "mean_tail_probability_missing": float(np.average(P[:, j], weights=species_weight * missing))
                    if weighted_tail_denominator > 0
                    else 0.0,
                    "mean_reliability_missing": float(np.average(R[:, j], weights=species_weight * missing))
                    if weighted_tail_denominator > 0
                    else 0.0,
                    "testing_scope": "state-level",
                    "reason": "state exposure x soft-local species relevance x decision uncertainty x missing evidence",
                }
            )
            for i in top_local_species_idx:
                if N[i, j] > 0:
                    continue
                pair_priority = float(state_w[local_col] * species_weight[i] * P[i, j] * (1 - P[i, j]) * (1 - R[i, j]))
                if pair_priority <= 0:
                    continue
                state_pair_candidates.append(
                    {
                        "state_code": state_code,
                        "latin_name": species[i],
                        "DTXSID": chem_id,
                        "preferred_name": chem_name.get(chem_id, ""),
                        "primary_moa": chem_primary_moa.get(chem_id, "MOA_UNRESOLVED"),
                        "moa_term": moa_term,
                        "state_pair_testing_priority": pair_priority,
                        "tail_probability": float(P[i, j]),
                        "reliability": float(R[i, j]),
                        "soft_local_species_weight": float(species_weight[i]),
                        "state_weight": float(state_w[local_col]),
                        "testing_scope": "state-level",
                    }
                )
        if state_pair_candidates:
            state_pairs = pd.DataFrame(state_pair_candidates).sort_values("state_pair_testing_priority", ascending=False).head(100)
            state_pair_rows.extend(state_pairs.to_dict("records"))
            summary = (
                state_pairs.groupby(["state_code", "latin_name", "moa_term"], as_index=False)
                .agg(
                    n_missing_pairs=("DTXSID", "size"),
                    sum_priority=("state_pair_testing_priority", "sum"),
                    mean_tail_probability=("tail_probability", "mean"),
                    mean_reliability=("reliability", "mean"),
                    mean_soft_local_species_weight=("soft_local_species_weight", "mean"),
                )
            )
            state_species_moa_rows.extend(summary.to_dict("records"))

    state_moa = pd.DataFrame(state_moa_rows)
    if not state_moa.empty:
        state_moa_summary = (
            state_moa.groupby(["state_code", "moa_term"], as_index=False)
            .agg(
                n_priority_chemicals=("DTXSID", "nunique"),
                n_missing_species_pairs=("n_missing_species_pairs", "sum"),
                sum_priority=("state_moa_testing_priority", "sum"),
                mean_tail_probability=("mean_tail_probability_missing", "mean"),
                mean_reliability=("mean_reliability_missing", "mean"),
            )
            .sort_values(["state_code", "sum_priority"], ascending=[True, False])
        )
    else:
        state_moa_summary = pd.DataFrame(
            columns=[
                "state_code",
                "moa_term",
                "n_priority_chemicals",
                "n_missing_species_pairs",
                "sum_priority",
                "mean_tail_probability",
                "mean_reliability",
            ]
        )
    state_moa.to_csv(out_testing / "state_chemical_moa_testing_priorities.csv", index=False)
    state_moa_summary.to_csv(out_testing / "state_moa_testing_priorities.csv", index=False)
    state_pair_df = pd.DataFrame(state_pair_rows)
    if not state_pair_df.empty:
        state_pair_df = state_pair_df.sort_values(["state_code", "state_pair_testing_priority"], ascending=[True, False])
    state_pair_df.to_csv(out_testing / "state_species_chemical_testing_pairs.csv", index=False)
    pd.DataFrame(state_species_moa_rows).to_csv(out_testing / "state_species_moa_testing_matrix.csv", index=False)

    if args.under_tested_registry and Path(args.under_tested_registry).exists():
        ut = pd.read_csv(args.under_tested_registry)
        warning_path = Path(args.under_tested_registry).with_name("rapid_warning_species_evidence_observed_only.csv")
        if warning_path.exists():
            warning = pd.read_csv(warning_path)[["latin_name", "warning_contexts", "warning_chemicals"]]
            ut = ut.merge(warning, on="latin_name", how="left")
        for col in ["warning_contexts", "warning_chemicals"]:
            if col not in ut:
                ut[col] = 0
            ut[col] = ut[col].fillna(0)
        warn_norm = np.log1p(ut.warning_contexts) / max(np.log1p(ut.warning_contexts).max(), 1)
        ut["active_testing_priority"] = ut.under_tested_priority_seed * (0.8 + 0.2 * warn_norm)
        ut.sort_values("active_testing_priority", ascending=False).to_csv(
            out_testing / "national_under_tested_species_priorities.csv", index=False
        )
    else:
        ut = pd.DataFrame()

    man = {
        "protection_target_x": target_x,
        "state_panels": int(seq_df.state_code.nunique()) if not seq_df.empty else 0,
        "state_panel_policy": (
            "regional-localization objective operationalized at state level: "
            "selection maximizes state-priority chemical weights multiplied by soft species relevance "
            "(state occurrence, neighbor occurrence, occurrence breadth, evidence and method support)."
        ),
        "state_comparison_policy": (
            "fixed national Top-5 and localized Top-5 are evaluated on the same state-priority chemicals "
            "and soft species-relevance weights"
        ),
        "mean_state_gain_over_national_top5_percent_points": float(
            comp_df.state_gain_over_national_top5_percent_points.mean()
        )
        if not comp_df.empty
        else None,
        "median_state_gain_over_national_top5_percent_points": float(
            comp_df.state_gain_over_national_top5_percent_points.median()
        )
        if not comp_df.empty
        else None,
        "species_testing_rows": len(sp),
        "chemical_testing_rows": len(ch),
        "pair_testing_rows": int(len(pair_df)),
        "national_species_moa_testing_matrix_rows": int(len(national_species_moa)),
        "state_chemical_moa_testing_rows": int(len(state_moa)),
        "state_moa_testing_rows": int(len(state_moa_summary)),
        "state_species_chemical_testing_pair_rows": int(len(state_pair_df)),
        "two_dimensions": [
            "regional localization demonstrated through state-level localized Top-5 panels",
            "national species-by-candidate-MOA testing priorities for main Figure 5c",
            "state-level MOA testing priorities for SI Figure S8",
            "sparse-species testing priorities",
            "chemical/candidate-MOA priorities with readable terms and chemical-form gaps",
        ],
        "under_tested_species_rows": len(ut),
    }
    (out_testing / "testing_and_state_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(json.dumps(man, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
