#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


FIXED_SEQUENCE_BASIS_TARGET = 0.95
EVALUATION_TARGETS = (0.95, 0.90, 0.80)
TAIL_LABELS = {0.95: "lower-5%", 0.90: "lower-10%", 0.80: "lower-20%"}


def target_suffix(target: float) -> str:
    return f"{int(round(target * 100)):02d}"


def target_key(target: float) -> str:
    return f"{target:.2f}".rstrip("0").rstrip(".")


def copula_union(P: np.ndarray, rho: float, nodes: int = 16) -> np.ndarray:
    P = np.clip(np.asarray(P, float), 1e-8, 1 - 1e-8)
    if P.ndim == 1:
        P = P[None, :]
    if len(P) == 0:
        return np.zeros(P.shape[1])
    if len(P) == 1:
        return P[0]
    x, w = hermgauss(nodes)
    z = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1 - P)
    no_event = np.zeros(P.shape[1])
    shared_sd = math.sqrt(max(rho, 0))
    residual_sd = math.sqrt(max(1 - rho, 1e-8))
    for zz, ww in zip(z, w):
        no_event += ww * np.prod(norm.cdf((threshold - shared_sd * zz) / residual_sd), axis=0)
    return np.clip(1 - no_event, 0, 1)


def greedy(P: np.ndarray, weights: np.ndarray, k: int = 20) -> list[int]:
    no_event = np.ones(P.shape[1])
    selected: list[int] = []
    for _ in range(k):
        gain = (P * (weights * no_event)[None, :]).sum(axis=1)
        gain[selected] = -np.inf
        index = int(np.argmax(gain))
        selected.append(index)
        no_event *= 1 - P[index]
    return selected


def build_fixed_sequence_cross_target_curves(
    root: Path,
    fixed_sequence: list[str],
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    rows: list[dict[str, object]] = []
    sequence_source = (
        "results/panels/national_panel_sequences.csv; universe=priority; "
        "protection_target_x=0.95; method=data_driven"
    )
    fixed_sequence_text = " | ".join(fixed_sequence)

    for target in EVALUATION_TARGETS:
        probability_path = root / f"results/probability/species_chemical_tail_probability_x{target_suffix(target)}.npz"
        probability = np.load(probability_path, allow_pickle=True)
        species = np.asarray(probability["species"], dtype=object).astype(str)
        chemicals = np.asarray(probability["chemicals"], dtype=object).astype(str)
        species_index = {name: index for index, name in enumerate(species)}
        chemical_index = {name: index for index, name in enumerate(chemicals)}
        missing_species = [name for name in fixed_sequence if name not in species_index]
        if missing_species:
            raise ValueError(f"Fixed x=0.95 sequence is absent from x={target:g} probability matrix: {missing_species}")

        target_weights = weights[weights["DTXSID"].isin(chemical_index)].copy()
        target_weights = target_weights[target_weights["national_weight"].astype(float).gt(0)]
        if len(target_weights) != 362:
            raise ValueError(f"Expected 362 national priority chemicals at x={target:g}, found {len(target_weights)}")
        normalized_weights = target_weights["national_weight"].to_numpy(float)
        normalized_weights /= normalized_weights.sum()
        chemical_columns = np.asarray([chemical_index[name] for name in target_weights["DTXSID"]], dtype=int)
        probability_matrix = probability["p"].astype(float)[:, chemical_columns]
        rho = float(manifest["dependence_audit_by_x"][target_key(target)]["rho_working"])
        ordered_indices = [species_index[name] for name in fixed_sequence]

        for panel_size in range(1, min(20, len(ordered_indices)) + 1):
            prefix = fixed_sequence[:panel_size]
            joint_capture = copula_union(probability_matrix[ordered_indices[:panel_size]], rho)
            expected_capture = float(normalized_weights @ joint_capture)
            rows.append(
                {
                    "sequence_basis_target": FIXED_SEQUENCE_BASIS_TARGET,
                    "evaluation_target": target,
                    "protection_target_x": target,
                    "measured_tail": TAIL_LABELS[target],
                    "panel_size": panel_size,
                    "k": panel_size,
                    "species_prefix": " | ".join(prefix),
                    "expected_capture": expected_capture,
                    "expected_weighted_joint_coverage": expected_capture,
                    "fixed_sequence": True,
                    "full_fixed_sequence": fixed_sequence_text,
                    "sequence_source": sequence_source,
                    "probability_source": probability_path.relative_to(root).as_posix(),
                    "chemical_weight_source": "results/weights/national_priority_chemicals.csv",
                    "copula_parameter_source": "results/panels/panel_probability_manifest.json",
                    "rho_working": rho,
                    "n_chemicals": len(target_weights),
                }
            )

    curves = pd.DataFrame(rows).sort_values(["evaluation_target", "panel_size"], ascending=[False, True])
    audit = curves[curves["panel_size"].eq(5)].copy()
    audit["fixed_top5_species"] = audit["species_prefix"]
    audit["text_percent_1dp"] = audit["expected_capture"] * 100
    audit["figure_value_3dp"] = audit["expected_capture"].round(3)
    audit = audit[
        [
            "sequence_basis_target",
            "evaluation_target",
            "measured_tail",
            "fixed_top5_species",
            "expected_capture",
            "text_percent_1dp",
            "figure_value_3dp",
            "rho_working",
            "n_chemicals",
            "sequence_source",
            "probability_source",
            "chemical_weight_source",
            "copula_parameter_source",
        ]
    ]
    return curves, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prob-npz", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    probability = np.load(args.prob_npz, allow_pickle=True)
    p = probability["p"].astype(float)
    species = probability["species"].astype(str)
    chemicals = probability["chemicals"].astype(str)
    chemical_index = {chemical: index for index, chemical in enumerate(chemicals)}
    species_index = {name: index for index, name in enumerate(species)}
    candidates = pd.read_csv(root / "results/probability/candidate_universe_locked.csv").set_index("latin_name")
    sequence_path = root / "results/panels/national_panel_sequences.csv"
    curves_path = root / "results/panels/national_k1_20_coverage_curves.csv"
    sequence = pd.read_csv(sequence_path)
    curves = pd.read_csv(curves_path)
    manifest_path = root / "results/panels/panel_probability_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rho = float(manifest["dependence_audit_by_x"]["0.95"]["rho_working"])

    universes: dict[str, tuple[list[str], np.ndarray]] = {}
    priority = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")
    priority = priority[priority["DTXSID"].astype(str).isin(chemical_index)].copy()
    priority["w"] = priority["national_weight"] / priority["national_weight"].sum()
    universes["priority"] = (priority["DTXSID"].astype(str).tolist(), priority["w"].to_numpy(float))
    all_chemicals = pd.read_csv(root / "results/weights/all_protective_chemicals_uniform.csv")
    all_chemicals = all_chemicals[all_chemicals["DTXSID"].astype(str).isin(chemical_index)].copy()
    all_chemicals["w"] = 1 / len(all_chemicals)
    universes["all_chemicals"] = (
        all_chemicals["DTXSID"].astype(str).tolist(),
        all_chemicals["w"].to_numpy(float),
    )

    new_sequences: list[dict[str, object]] = []
    new_curves: list[dict[str, object]] = []
    fixed_priority_sequence: list[str] | None = None
    for universe, (chemical_names, weights) in universes.items():
        chemical_columns = np.asarray([chemical_index[name] for name in chemical_names])
        probability_matrix = p[:, chemical_columns]
        selected = greedy(probability_matrix, weights, 20)
        names = species[selected].tolist()
        if universe == "priority":
            fixed_priority_sequence = names
        for rank, name in enumerate(names, 1):
            candidate = candidates.loc[name] if name in candidates.index else pd.Series(dtype=object)
            new_sequences.append(
                {
                    "universe": universe,
                    "protection_target_x": 0.95,
                    "method": "data_driven",
                    "rank": rank,
                    "latin_name": name,
                    "n_observed_contexts": candidate.get("n_contexts", np.nan),
                    "n_observed_chemicals": candidate.get("n_chemicals", np.nan),
                    "official_or_common_method_species": candidate.get("official_or_common_method_species", False),
                    "eligible_states": candidate.get("eligible_states", 0),
                    "candidate_universe_sha256": sequence["candidate_universe_sha256"].iloc[0],
                }
            )
        methods = {"data_driven": names}
        comparator_rows = sequence[
            sequence["universe"].eq(universe)
            & sequence["protection_target_x"].eq(0.95)
            & ~sequence["method"].eq("data_driven")
        ]
        for method, group in comparator_rows.groupby("method"):
            methods[method] = group.sort_values("rank")["latin_name"].astype(str).tolist()
        for method, method_names in methods.items():
            valid = [species_index[name] for name in method_names if name in species_index]
            for panel_size in range(1, min(20, len(valid)) + 1):
                joint_capture = copula_union(probability_matrix[valid[:panel_size]], rho)
                new_curves.append(
                    {
                        "universe": universe,
                        "protection_target_x": 0.95,
                        "method": method,
                        "k": panel_size,
                        "expected_weighted_joint_coverage": float(weights @ joint_capture),
                        "n_chemicals": len(chemical_names),
                        "rho": rho,
                    }
                )

    sequence = sequence[~(sequence["protection_target_x"].eq(0.95) & sequence["method"].eq("data_driven"))]
    sequence = pd.concat([sequence, pd.DataFrame(new_sequences)], ignore_index=True)
    sequence.to_csv(sequence_path, index=False)
    curves = curves[~curves["protection_target_x"].eq(0.95)]
    curves = pd.concat([curves, pd.DataFrame(new_curves)], ignore_index=True)
    curves.to_csv(curves_path, index=False)

    if fixed_priority_sequence is None:
        raise RuntimeError("The x=0.95 priority sequence was not generated")
    fixed_curves, fixed_audit = build_fixed_sequence_cross_target_curves(
        root,
        fixed_priority_sequence,
        manifest,
    )
    fixed_curves.to_csv(root / "results/panels/fixed_x95_sequence_cross_target_curves.csv", index=False)
    fixed_audit.to_csv(root / "results/panels/fixed_national_top5_cross_target_audit.csv", index=False)

    manifest["version"] = "v16.5"
    manifest["hc5_core_probability"] = (
        "taxonomy/direct evidence calibration only; no MOA layer passed the simplified HC5 core-entry gate"
    )
    manifest["fixed_sequence_cross_target_output"] = {
        "sequence_basis_target": 0.95,
        "evaluation_targets": [0.95, 0.90, 0.80],
        "sequence_file": "results/panels/national_panel_sequences.csv",
        "curve_file": "results/panels/fixed_x95_sequence_cross_target_curves.csv",
        "top5_audit_file": "results/panels/fixed_national_top5_cross_target_audit.csv",
        "chemical_weight_source": "results/weights/national_priority_chemicals.csv",
        "n_priority_chemicals": 362,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(fixed_audit.to_string(index=False))


if __name__ == "__main__":
    main()
