#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

PROTECTIVE = [
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
]
TARGETS = [0.80, 0.90, 0.95]
LEVELS = [("latin_name", 4.0, 0.80), ("genus", 2.0, 0.50), ("family", 1.0, 0.30), ("class", 0.5, 0.15)]


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

def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    out["average_precision"] = float(average_precision_score(y, p)) if np.sum(y) > 0 else np.nan
    return out


def fit_grouped_calibration(raw: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    raw_logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    calibrated = np.full(len(y), np.nan)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(raw_logit, y, groups):
        fit = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
        fit.fit(raw_logit[train], y[train])
        calibrated[test] = fit.predict_proba(raw_logit[test])[:, 1]
    final = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
    final.fit(raw_logit, y)
    params = {
        "method": "Platt logistic calibration on raw probability logit",
        "grouped_crossfit": "5-fold GroupKFold by whole chemical",
        "intercept": float(final.intercept_[0]),
        "coefficient": float(final.coef_[0, 0]),
        "apply_to": "borrowed cells with direct_n==0; direct chemical evidence remains support-shrunk",
    }
    return calibrated, params


def leaveout_probability(m: pd.DataFrame, q0: float, use_moa: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized whole-chemical leave-out probability using taxonomy, optionally MOA-conditioned."""
    n_rows = len(m)
    weighted_value_sum = np.zeros(n_rows, dtype=float)
    total_weight = np.zeros(n_rows, dtype=float)
    support = np.zeros(n_rows, dtype=int)
    reliability = np.zeros(n_rows, dtype=float)

    for level, base_weight, cap in LEVELS:
        keys = [level, "effect_family"]
        chemical_keys = ["dtxsid", level, "effect_family"]
        if use_moa:
            keys.insert(1, "mechanism_group")
            chemical_keys.insert(2, "mechanism_group")

        total = (
            m.groupby(keys, dropna=False).p_context
            .agg(total_sum="sum", total_count="count")
            .reset_index()
        )
        by_chemical = (
            m.groupby(chemical_keys, dropna=False).p_context
            .agg(chemical_sum="sum", chemical_count="count")
            .reset_index()
        )
        frame = m[chemical_keys].copy()
        frame["__row_id"] = np.arange(n_rows)
        frame = frame.merge(total, on=keys, how="left", sort=False)
        frame = frame.merge(by_chemical, on=chemical_keys, how="left", sort=False)
        frame = frame.sort_values("__row_id")

        sm = frame.total_sum.fillna(0).to_numpy(float) - frame.chemical_sum.fillna(0).to_numpy(float)
        nn = frame.total_count.fillna(0).to_numpy(int) - frame.chemical_count.fillna(0).to_numpy(int)
        valid = nn > 0
        values = np.full(n_rows, q0, dtype=float)
        values[valid] = (sm[valid] + 10 * q0) / (nn[valid] + 10)
        weights = np.zeros(n_rows, dtype=float)
        weights[valid] = base_weight * np.minimum(nn[valid], 20)
        rel = np.zeros(n_rows, dtype=float)
        rel[valid] = cap * (1 - np.exp(-nn[valid] / 10.0))

        weighted_value_sum += values * weights
        total_weight += weights
        support = np.maximum(support, nn)
        reliability = np.maximum(reliability, rel)

    prior = np.full(n_rows, q0, dtype=float)
    mask = total_weight > 0
    prior[mask] = weighted_value_sum[mask] / total_weight[mask]
    prediction = q0 + reliability * (prior - q0)
    return prediction, support, reliability


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--chem", required=True)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "results/validation"
    out.mkdir(parents=True, exist_ok=True)

    model = pd.read_csv(args.model, low_memory=False)
    chem = pd.read_csv(args.chem, low_memory=False)
    candidates = pd.read_csv(root / "results/probability/candidate_universe_locked.csv")
    species = candidates.latin_name.astype(str).tolist()
    species_set = set(species)

    valid = model.mle_status.astype(str).str.startswith("identified") & np.isfinite(model.mu_log10_umol_L)
    base = model[valid & model.effect_family.isin(PROTECTIVE) & model.latin_name.isin(species_set)].copy()
    base["context_id"] = base[
        ["dtxsid", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]
    ].astype(str).agg("|".join, axis=1)
    cs = base.groupby("context_id").agg(
        n_species=("latin_name", "nunique"),
        ssd_mean=("mu_log10_umol_L", "mean"),
        ssd_sd=("mu_log10_umol_L", "std"),
    ).reset_index()
    cs = cs[(cs.n_species >= 5) & cs.ssd_sd.notna() & (cs.ssd_sd > 0.05)]
    base = base[base.context_id.isin(cs.context_id)].merge(cs, on="context_id").reset_index(drop=True)

    chem["DTXSID"] = chem.DTXSID.astype(str)
    chem["mechanism_group"] = np.where(
        chem.primary_moa.fillna("MOA_UNRESOLVED").ne("MOA_UNRESOLVED"),
        chem.primary_moa,
        "FORM::" + chem.chemical_form_class.fillna("unresolved").astype(str),
    )
    mechanism = chem.set_index("DTXSID").mechanism_group.to_dict()
    base["mechanism_group"] = base.dtxsid.map(mechanism).fillna("MOA_UNRESOLVED")

    all_prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    calibration_rows: list[pd.DataFrame] = []
    ranking_rows: list[dict] = []
    gate_group_rows: list[dict] = []
    calibration_parameters: dict[str, dict] = {
        "version": "v16.5",
        "description": "whole-chemical grouped cross-fitted calibration for probability-only MOA+taxonomy transfer across the full evidence-bearing species universe",
        "by_protection_target": {},
    }

    for x in TARGETS:
        print(f"holdout target {x}: start", flush=True)
        q0 = 1 - x
        m = base.copy()
        m["hc"] = m.ssd_mean + norm.ppf(q0) * m.ssd_sd
        unc = np.sqrt(m.se_mu_log10.fillna(0.35) ** 2 + (m.ssd_sd / np.sqrt(m.n_species)) ** 2).clip(lower=0.12)
        m["p_context"] = np.clip(norm.cdf((m.hc - m.mu_log10_umol_L) / unc), 1e-6, 1 - 1e-6)
        m["tail_binary"] = (m.mu_log10_umol_L <= m.hc).astype(int)
        y = m.tail_binary.to_numpy(int)
        groups = m.dtxsid.astype(str).to_numpy()

        print(f"holdout target {x}: prepared {len(m)} rows", flush=True)
        tax_raw, tax_support, tax_rel = leaveout_probability(m, q0, use_moa=False)
        print(f"holdout target {x}: taxonomy done", flush=True)
        moa_raw, moa_support, moa_rel = leaveout_probability(m, q0, use_moa=True)
        print(f"holdout target {x}: MOA+taxonomy done", flush=True)
        tax_cal, tax_params = fit_grouped_calibration(tax_raw, y, groups)
        moa_cal, moa_params = fit_grouped_calibration(moa_raw, y, groups)
        print(f"holdout target {x}: calibration done", flush=True)

        # Validation-informed MOA gate: use MOA conditioning only in mechanism×effect strata
        # where the cross-fitted MOA model improves Brier score over taxonomy-only.
        gate_frame = m[["mechanism_group", "effect_family", "tail_binary"]].copy()
        gate_frame["taxonomy_cal"] = tax_cal
        gate_frame["moa_cal"] = moa_cal
        validated_groups: list[str] = []
        for (mechanism_group, effect_family), group in gate_frame.groupby(["mechanism_group", "effect_family"]):
            if len(group) < 50:
                continue
            tax_brier = brier_score_loss(group.tail_binary, group.taxonomy_cal)
            moa_brier = brier_score_loss(group.tail_binary, group.moa_cal)
            use_moa = bool(moa_brier < tax_brier)
            group_key = f"{mechanism_group}||{effect_family}"
            if use_moa:
                validated_groups.append(group_key)
            gate_group_rows.append({
                "protection_target_x": x,
                "mechanism_group": mechanism_group,
                "effect_family": effect_family,
                "n_rows": len(group),
                "taxonomy_brier": tax_brier,
                "moa_taxonomy_brier": moa_brier,
                "delta_brier_moa_minus_taxonomy": moa_brier - tax_brier,
                "validated_use_moa": use_moa,
            })
        group_keys = (m.mechanism_group.astype(str) + "||" + m.effect_family.astype(str)).to_numpy()
        adaptive_cal = np.where(np.isin(group_keys, validated_groups), moa_cal, tax_cal)
        calibration_parameters["by_protection_target"][str(x)] = {
            "protection_target_x": x,
            "taxonomy_calibration": tax_params,
            "moa_taxonomy_calibration": moa_params,
            "validated_moa_groups": sorted(validated_groups),
            "gate_min_rows": 50,
            "gate_rule": "use MOA-conditioned prior only when OOF Brier is lower than taxonomy-only within mechanism×effect stratum",
        }

        for name, pred in [
            ("taxonomy_probability_LOCO_raw", tax_raw),
            ("taxonomy_probability_LOCO_calibrated_OOF", tax_cal),
            ("MOA_taxonomy_probability_LOCO_raw", moa_raw),
            ("MOA_taxonomy_probability_LOCO_calibrated_OOF", moa_cal),
            ("validated_MOA_gate_probability_OOF", adaptive_cal),
            (f"exchangeable_{q0:.2f}_baseline", np.full(len(m), q0)),
        ]:
            metric_rows.append({"protection_target_x": x, "scope": "overall", "model": name, **safe_metrics(y, pred)})
        for effect, group in m.groupby("effect_family"):
            if len(group) < 100:
                continue
            idx = group.index.to_numpy()
            metric_rows.append({
                "protection_target_x": x,
                "scope": f"effect::{effect}",
                "model": "MOA_taxonomy_probability_LOCO_calibrated_OOF",
                **safe_metrics(group.tail_binary.to_numpy(), moa_cal[idx]),
            })

        support_counts = m.groupby("latin_name").context_id.nunique()
        support_tier_map = pd.cut(
            support_counts,
            bins=[0, 4, 19, 49, np.inf],
            labels=["very_sparse_1_4", "sparse_5_19", "moderate_20_49", "strong_50_plus"],
            include_lowest=True,
        ).astype(str).to_dict()
        row_tiers = m.latin_name.map(support_tier_map).fillna("unclassified")
        for tier in ["very_sparse_1_4", "sparse_5_19", "moderate_20_49", "strong_50_plus"]:
            mask_tier = row_tiers.eq(tier).to_numpy()
            if mask_tier.sum() < 50:
                continue
            for name, pred in [
                ("taxonomy_probability_LOCO_calibrated_OOF", tax_cal),
                ("MOA_taxonomy_probability_LOCO_calibrated_OOF", moa_cal),
                ("validated_MOA_gate_probability_OOF", adaptive_cal),
            ]:
                metric_rows.append({
                    "protection_target_x": x,
                    "scope": f"support_tier::{tier}",
                    "model": name,
                    **safe_metrics(y[mask_tier], pred[mask_tier]),
                })

        pred_frame = m[[
            "dtxsid", "latin_name", "effect_family", "context_id", "tail_binary",
            "p_context", "mechanism_group",
        ]].copy()
        pred_frame["protection_target_x"] = x
        pred_frame["taxonomy_loco_probability"] = tax_raw
        pred_frame["taxonomy_loco_probability_calibrated_oof"] = tax_cal
        pred_frame["moa_taxonomy_loco_probability"] = moa_raw
        pred_frame["moa_taxonomy_loco_probability_calibrated_oof"] = moa_cal
        pred_frame["validated_moa_gate_probability_oof"] = adaptive_cal
        pred_frame["loco_support"] = moa_support
        pred_frame["loco_reliability"] = moa_rel
        all_prediction_frames.append(pred_frame)

        bins = pd.qcut(pd.Series(moa_cal).rank(method="first"), q=10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"probability_bin": bins, "predicted": moa_cal, "observed": y, "support": moa_support})
        cal = cal.groupby("probability_bin").agg(
            n=("observed", "size"),
            predicted_mean=("predicted", "mean"),
            observed_rate=("observed", "mean"),
            support_median=("support", "median"),
        ).reset_index()
        cal["protection_target_x"] = x
        calibration_rows.append(cal)

        for (chemical, effect), group in pred_frame.groupby(["dtxsid", "effect_family"]):
            g = group.groupby("latin_name").agg(
                y=("tail_binary", "max"),
                moa_pred=("moa_taxonomy_loco_probability", "mean"),
                tax_pred=("taxonomy_loco_probability", "mean"),
                observed_p=("p_context", "mean"),
            ).reset_index()
            true_n = int(g.y.sum())
            if len(g) < 5 or true_n == 0:
                continue
            true = set(g[g.y == 1].latin_name)
            top_moa = set(g.nlargest(min(5, len(g)), "moa_pred").latin_name)
            top_tax = set(g.nlargest(min(5, len(g)), "tax_pred").latin_name)
            ranking_rows.append({
                "protection_target_x": x,
                "dtxsid": chemical,
                "effect_family": effect,
                "n_species": len(g),
                "n_true_tail": true_n,
                "moa_tail_recall_at_5": len(top_moa & true) / true_n,
                "taxonomy_tail_recall_at_5": len(top_tax & true) / true_n,
                "moa_tail_precision_at_5": len(top_moa & true) / len(top_moa),
                "taxonomy_tail_precision_at_5": len(top_tax & true) / len(top_tax),
                "spearman_moa_vs_observed_p": g[["moa_pred", "observed_p"]].corr(method="spearman").iloc[0, 1],
                "spearman_taxonomy_vs_observed_p": g[["tax_pred", "observed_p"]].corr(method="spearman").iloc[0, 1],
            })
        print(f"holdout target {x}: complete", flush=True)

    print("writing holdout outputs", flush=True)
    predictions = pd.concat(all_prediction_frames, ignore_index=True)
    predictions.to_csv(out / "leave_chemical_out_predictions.csv.gz", index=False, compression="gzip")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "leave_chemical_out_metrics.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(out / "leave_chemical_out_calibration.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(out / "leave_chemical_out_ranking_metrics.csv", index=False)
    pd.DataFrame(gate_group_rows).to_csv(out / "moa_gate_validation_by_group.csv", index=False)
    (out / "probability_calibration_parameters.json").write_text(
        json.dumps(calibration_parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Explicit MOA ablation summary for the rare-tail probability task.
    ablation = []
    for x in TARGETS:
        sub = metrics[(metrics.protection_target_x == x) & (metrics.scope == "overall")].set_index("model")
        moa = sub.loc["MOA_taxonomy_probability_LOCO_calibrated_OOF"]
        tax = sub.loc["taxonomy_probability_LOCO_calibrated_OOF"]
        gated = sub.loc["validated_MOA_gate_probability_OOF"]
        base_line = sub.loc[f"exchangeable_{1-x:.2f}_baseline"]
        ablation.append({
            "protection_target_x": x,
            "moa_plus_taxonomy_brier": moa.brier,
            "taxonomy_only_brier": tax.brier,
            "exchangeable_brier": base_line.brier,
            "delta_brier_moa_vs_taxonomy": moa.brier - tax.brier,
            "moa_plus_taxonomy_roc_auc": moa.roc_auc,
            "taxonomy_only_roc_auc": tax.roc_auc,
            "delta_auc_moa_vs_taxonomy": moa.roc_auc - tax.roc_auc,
            "validated_gate_brier": gated.brier,
            "validated_gate_roc_auc": gated.roc_auc,
            "n_validated_moa_groups": len(calibration_parameters["by_protection_target"][str(x)]["validated_moa_groups"]),
            "interpretation": "Unconditional MOA conditioning is not accepted globally when delta Brier is positive or delta AUC is negative; only validation-improving mechanism×effect strata enter the selective gate",
        })
    pd.DataFrame(ablation).to_csv(out / "moa_probability_ablation.csv", index=False)

    # Preserve the earlier absolute-potency benchmark only as a non-comparable reference.
    historical = pd.DataFrame([
        {"historical_task": "absolute_log_toxicity_whole_chemical_read_across", "method": "baseline", "rmse_log10": 1.7606043752071794, "pearson_r": 0.2823011397617642, "coverage_fraction": 1.0},
        {"historical_task": "absolute_log_toxicity_whole_chemical_read_across", "method": "MOA_nearest_neighbors", "rmse_log10": 1.6571741714342865, "pearson_r": 0.3957745805283878, "coverage_fraction": 0.5766682442025556},
        {"historical_task": "absolute_log_toxicity_whole_chemical_read_across", "method": "structure_nearest_neighbors", "rmse_log10": 1.580180905609827, "pearson_r": 0.5499151114404676, "coverage_fraction": 0.8226060892885313},
    ])
    historical["comparability_note"] = "Different target and support: predicts absolute log toxicity using chemical neighbours; retained for algorithm-history interpretation, not used in v16.5 panel probabilities."
    historical.to_csv(out / "historical_moa_absolute_readacross_reference.csv", index=False)

    print("starting panel stability", flush=True)
    # Panel stability uses the same complete chemical-level probability matrix and 362-chemical
    # national weights as the formal v16.5 HC10 priority curve.
    zprob = np.load(root / "results/probability/species_chemical_tail_probability_x90.npz", allow_pickle=True)
    p = zprob["p"]
    p_species = zprob["species"].astype(str).tolist()
    chemical_axis = zprob["chemicals"].astype(str).tolist()
    if p_species != species:
        raise RuntimeError("Candidate universe order mismatch between validation and probability output")
    c_index = {c: i for i, c in enumerate(chemical_axis)}
    priority = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    priority["DTXSID"] = priority.DTXSID.astype(str)
    priority = priority[priority.DTXSID.isin(c_index)].copy()
    chemicals = priority.DTXSID.tolist()
    cidx = np.array([c_index[c] for c in chemicals], int)
    p_priority = p[:, cidx]
    base_prob = priority.national_weight.to_numpy(float)
    base_prob /= base_prob.sum()

    seq1 = deterministic_greedy(p_priority, base_prob, species, 20)
    seq2 = deterministic_greedy(p_priority, base_prob, species, 20)
    deterministic = seq1 == seq2
    sequence_hash = hashlib.sha256("\n".join(species[i] for i in seq1).encode()).hexdigest()

    rng = np.random.default_rng(args.seed)
    inclusion = np.zeros((args.bootstrap, len(species)), dtype=bool)
    ranks = np.full((args.bootstrap, len(species)), np.nan)
    for b in range(args.bootstrap):
        counts = rng.multinomial(len(chemicals), base_prob)
        boot_w = counts / max(counts.sum(), 1)
        if boot_w.sum() == 0:
            continue
        seq = deterministic_greedy(p_priority, boot_w, species, 20)
        for rank, j in enumerate(seq, 1):
            inclusion[b, j] = True
            ranks[b, j] = rank

    stability = candidates.copy()
    stability["top20_inclusion_probability"] = inclusion.mean(axis=0)
    stability["median_rank_when_selected"] = np.nanmedian(ranks, axis=0)
    stability["rank_q025_when_selected"] = np.nanquantile(ranks, 0.025, axis=0)
    stability["rank_q975_when_selected"] = np.nanquantile(ranks, 0.975, axis=0)
    stability["full_data_rank"] = np.nan
    for rank, j in enumerate(seq1, 1):
        stability.loc[j, "full_data_rank"] = rank
    stability.sort_values(
        ["top20_inclusion_probability", "median_rank_when_selected"], ascending=[False, True]
    ).to_csv(out / "panel_bootstrap_stability.csv", index=False)

    daphnia = stability[stability.latin_name == "Daphnia magna"]
    daphnia_record = daphnia.to_dict("records")[0] if len(daphnia) else {}
    summary = {
        "version": "v16.5",
        "whole_chemical_holdout": {
            "definition": "remove all direct observations of each target chemical before estimating sensitivity-tail probability priors; no absolute toxicity prediction",
            "targets": TARGETS,
            "n_prediction_rows_per_target": int(len(base)),
            "n_chemical_effect_ranking_groups_all_targets": int(len(ranking_rows)),
            "models_compared": ["taxonomy_only", "MOA_plus_taxonomy", "exchangeable_prior"],
            "calibration": calibration_parameters,
            "historical_algorithm_difference": "earlier MOA result predicted absolute log toxicity with MOA-nearest chemical neighbours; v16.5 estimates measured lower-tail membership at x = 0.80/0.90/0.95 and therefore reports different metrics",
        },
        "panel_stability": {
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
            "same_input_deterministic_rerun": bool(deterministic),
            "full_sequence_sha256": sequence_hash,
            "protection_target_x": 0.90,
            "daphnia_magna": daphnia_record,
        },
    }
    (out / "validation_and_stability_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
