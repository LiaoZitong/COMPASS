#!/usr/bin/env python3
"""Render the R1 Figure 2 from the approved AP02/AP05 analysis layer.

The script preserves the R0 comparator-panel identities but recomputes every
displayed probability-derived quantity from the strict focal-species LOO
matrices and the re-estimated target-specific Gaussian-copula dependence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[3]
ANALYSIS_ROOT = PACKAGE_ROOT / "results" / "revision_r1"
BHBT_ROOT = PACKAGE_ROOT
AP02_DIR = ANALYSIS_ROOT / "02_ap02_strict_loo"
AP05_DIR = ANALYSIS_ROOT / "03_ap05_dependence"
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "main_figure_02"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import (  # noqa: E402
    PALETTE,
    TARGET_COLORS,
    apply_style,
    clean_axis,
    export_figure,
    short_species,
)


TARGETS = (0.80, 0.90, 0.95)
TARGET_LABELS = {0.80: "Lower 20%", 0.90: "Lower 10%", 0.95: "Lower 5%"}
EXPECTED_TOP5 = [
    "Gastrophryne carolinensis",
    "Neocloeon triangulifer",
    "Hyalella azteca",
    "Daphnia ambigua",
    "Daphnia magna",
]
WET_PANEL = [
    "Pimephales promelas",
    "Cyprinella leedsi",
    "Ceriodaphnia dubia",
    "Oncorhynchus mykiss",
    "Salvelinus fontinalis",
    "Daphnia pulex",
    "Daphnia magna",
    "Raphidocelis subcapitata",
]
METHOD_ORDER = [
    "random_all_species_probability",
    "taxonomy_diversity_baseline",
    "Canada_Type_A_composition",
    "EPA_WQC_taxonomic_requirements",
]
METHOD_LABELS = {
    "random_all_species_probability": "Random\nk = 5",
    "taxonomy_diversity_baseline": "Taxonomic\ndiversity k = 5",
    "Canada_Type_A_composition": "CCME Type A\nproxy k = 7",
    "EPA_WQC_taxonomic_requirements": "EPA WQC\nproxy k = 8",
    "US_EPA_WET_species": "EPA WET\npanel k = 8",
    "R1_STRICT_LOO_TOP5": "R1 national\nTop-5",
}
FIGURE_BASENAME = "figure_02_r1_strict_loo_performance_complementarity"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copula_union_sequence(p: np.ndarray, rho: float, nodes: int = 12) -> np.ndarray:
    """Return cumulative union probability for prefixes of a species sequence."""

    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    if p.ndim == 1:
        p = p[None, :]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - p)
    shared_sd = math.sqrt(max(float(rho), 0.0))
    residual_sd = math.sqrt(max(1.0 - float(rho), 1e-8))
    no_event = np.zeros_like(p, dtype=float)
    for zz, ww in zip(z, w):
        conditional_no = norm.cdf((threshold - shared_sd * zz) / residual_sd)
        no_event += ww * np.cumprod(conditional_no, axis=0)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def copula_union_many(p: np.ndarray, rho: float, nodes: int = 12) -> np.ndarray:
    """Return union probabilities for many fixed panels.

    Parameters
    ----------
    p
        Array with shape panel x species x chemical.
    """

    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - p)
    shared_sd = math.sqrt(max(float(rho), 0.0))
    residual_sd = math.sqrt(max(1.0 - float(rho), 1e-8))
    no_event = np.zeros((p.shape[0], p.shape[2]), dtype=float)
    for zz, ww in zip(z, w):
        conditional_no = norm.cdf((threshold - shared_sd * zz) / residual_sd)
        no_event += ww * np.prod(conditional_no, axis=1)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def deterministic_greedy(p: np.ndarray, weights: np.ndarray, species: np.ndarray, k: int) -> list[int]:
    """Frozen independent-union greedy selection used by AP02."""

    selected: list[int] = []
    remaining = set(range(p.shape[0]))
    no_event = np.ones(p.shape[1], dtype=float)
    for _ in range(min(k, p.shape[0])):
        candidates = sorted(remaining, key=lambda idx: str(species[idx]))
        gains = p[np.asarray(candidates, dtype=int)] @ (weights * no_event)
        best_gain = float(np.max(gains))
        tied = [candidates[i] for i, gain in enumerate(gains) if np.isclose(gain, best_gain, rtol=0, atol=1e-14)]
        best = min(tied, key=lambda idx: str(species[idx]))
        selected.append(best)
        remaining.remove(best)
        no_event *= 1.0 - p[best]
    return selected


def chemical_domain_label(value: str) -> str:
    value = str(value).strip()
    mapping = {
        "MIE::Deposition of Energy": "Energy metabolism",
        "MIE::Binding to voltage-gated sodium channel": "Sodium-channel binding",
        "MIE::Activation, AhR": "AhR activation",
        "MIE::Agonism, Androgen receptor": "Androgen-receptor agonism",
        "MIE::Alkylation, Protein": "Protein alkylation",
        "MIE::Binding of antagonist, PPAR alpha": "PPAR-alpha antagonist",
        "MIE::Inhibition, Na+/I- symporter (NIS)": "NIS inhibition",
        "MIE::Histone deacetylase inhibition": "HDAC inhibition",
        "MIE::Inhibition, VegfR2": "VEGFR2 inhibition",
        "MIE::Thyroperoxidase, Inhibition": "Thyroperoxidase inhibition",
    }
    if value in mapping:
        return mapping[value]
    if value.startswith("MIE::"):
        return value.split("::", 1)[1].replace(", ", " ")
    if value.startswith("FORM::"):
        return value.split("::", 1)[1].replace("_", " ").capitalize()
    if value.startswith("MOAFAM::"):
        return value.split("::", 1)[1].replace("_", " ").capitalize()
    return value.replace("_", " ")[:34] if value else "Unassigned"


def load_rhos() -> dict[float, float]:
    frame = pd.read_csv(AP05_DIR / "ap05_rho_estimation_audit.csv")
    frame = frame[frame["variant"].eq("AP02_STRICT_LOO") & frame["selected_for_downstream"].astype(bool)].copy()
    rhos = dict(zip(frame["protection_target_x"].astype(float), frame["rho_working"].astype(float)))
    if set(rhos) != set(TARGETS):
        raise ValueError(f"Expected three selected AP05 rho values, found {rhos}")
    return rhos


def load_weights(chemicals: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    weights = pd.read_csv(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv")
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    index = {str(value): idx for idx, value in enumerate(chemicals)}
    present = weights[weights["DTXSID"].isin(index)].copy()
    present_indices = np.asarray([index[value] for value in present["DTXSID"]], dtype=int)
    vector = present["national_weight"].to_numpy(float)
    vector /= vector.sum()
    return present_indices, vector, present


def load_probability(target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    label = int(round(target * 100))
    z = np.load(AP02_DIR / f"ap02_strict_loo_species_chemical_tail_probability_x{label}.npz", allow_pickle=True)
    species = np.asarray(z["species"], dtype=object)
    chemicals = np.asarray(z["chemicals"], dtype=object)
    priority_indices, weights, weight_frame = load_weights(chemicals)
    return z["p"][:, priority_indices].astype(float), species, chemicals[priority_indices], weights, weight_frame


def build_panel_a_and_c(rhos: dict[float, float]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    p95, species, _, weights, _ = load_probability(0.95)
    sequence_indices = deterministic_greedy(p95, weights, species, 20)
    sequence = [str(species[idx]) for idx in sequence_indices]
    if sequence[:5] != EXPECTED_TOP5:
        raise AssertionError(f"AP02 Top-5 mismatch: {sequence[:5]}")

    rows: list[dict[str, object]] = []
    cached: dict[float, np.ndarray] = {}
    for target in TARGETS:
        p, target_species, _, target_weights, _ = load_probability(target)
        if list(target_species) != list(species) or not np.allclose(target_weights, weights):
            raise AssertionError("Strict-LOO species or priority-weight alignment differs across targets")
        cumulative = copula_union_sequence(p[np.asarray(sequence_indices)], rhos[target]) @ weights
        cached[target] = cumulative
        for rank, (latin_name, coverage) in enumerate(zip(sequence, cumulative), start=1):
            rows.append(
                {
                    "sequence_basis_target": 0.95,
                    "protection_target_x": target,
                    "rank": rank,
                    "latin_name": latin_name,
                    "expected_weighted_joint_coverage": float(coverage),
                    "rho_working": rhos[target],
                    "probability_source": "AP02_STRICT_LOO",
                }
            )
    panel_a = pd.DataFrame(rows)

    capture_x95 = float(cached[0.95][4])
    if not np.isclose(capture_x95, 0.337764, atol=5e-6):
        raise AssertionError(f"Unexpected strict-LOO x=0.95 Top-5 capture: {capture_x95:.9f}")

    top5_indices = sequence_indices[:5]
    full_capture = capture_x95
    rows_c: list[dict[str, object]] = []
    for rank, idx in enumerate(top5_indices, start=1):
        retained = [other for other in top5_indices if other != idx]
        without_capture = float(copula_union_sequence(p95[np.asarray(retained)], rhos[0.95])[-1] @ weights)
        cumulative = float(cached[0.95][rank - 1])
        previous = float(cached[0.95][rank - 2]) if rank > 1 else 0.0
        rows_c.append(
            {
                "rank": rank,
                "latin_name": str(species[idx]),
                "species_label": short_species(str(species[idx])),
                "cumulative_coverage": cumulative,
                "incremental_gain": cumulative - previous,
                "leave_one_out_coverage": without_capture,
                "leave_one_out_loss": full_capture - without_capture,
                "full_top5_coverage": full_capture,
                "protection_target_x": 0.95,
                "rho_working": rhos[0.95],
            }
        )
    return panel_a, pd.DataFrame(rows_c), sequence


def score_panel_batch(
    panel_species: list[list[str]],
    probability: np.ndarray,
    species_index: dict[str, int],
    weights: np.ndarray,
    rho: float,
    batch_size: int = 96,
) -> np.ndarray:
    scores = np.empty(len(panel_species), dtype=float)
    for start in range(0, len(panel_species), batch_size):
        stop = min(start + batch_size, len(panel_species))
        indices = np.asarray([[species_index[name] for name in panel] for panel in panel_species[start:stop]], dtype=int)
        union = copula_union_many(probability[indices], rho)
        scores[start:stop] = union @ weights
    return scores


def build_panel_b(rhos: dict[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = BHBT_ROOT / "outputs" / "figures" / "figure_02" / "figure_02b_feasible_combination_samples.csv"
    sampled = pd.read_csv(source_path)
    sampled = sampled[sampled["method"].isin(METHOD_ORDER)].copy()
    sampled["coverage_r0_archived"] = sampled["coverage"]
    sampled["coverage"] = np.nan
    sampled["probability_source"] = "AP02_STRICT_LOO"
    sampled["rho_source"] = "AP05_STRICT_LOO"

    summary_rows: list[dict[str, object]] = []
    for target in TARGETS:
        probability, species, _, weights, _ = load_probability(target)
        species_index = {str(name): idx for idx, name in enumerate(species)}
        top5_indices = [species_index[name] for name in EXPECTED_TOP5]
        top5_coverage = float(copula_union_sequence(probability[np.asarray(top5_indices)], rhos[target])[-1] @ weights)

        for method in METHOD_ORDER:
            mask = sampled["method"].eq(method) & np.isclose(sampled["protection_target_x"], target)
            group = sampled.loc[mask].sort_values("sample_id")
            panels = [[part.strip() for part in value.split(" | ")] for value in group["panel_species"].astype(str)]
            missing = sorted({name for panel in panels for name in panel if name not in species_index})
            if missing:
                raise KeyError(f"Comparator species absent from strict-LOO matrix: {missing[:5]}")
            values = score_panel_batch(panels, probability, species_index, weights, rhos[target])
            sampled.loc[group.index, "coverage"] = values
            summary_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "method_k": len(panels[0]),
                    "protection_target_x": target,
                    "sampled_n": len(values),
                    "sampled_mean": float(np.mean(values)),
                    "sampled_median": float(np.median(values)),
                    "sampled_q025": float(np.quantile(values, 0.025)),
                    "sampled_q975": float(np.quantile(values, 0.975)),
                    "sampled_min": float(np.min(values)),
                    "sampled_max": float(np.max(values)),
                    "fixed_top5_coverage": top5_coverage,
                    "rho_working": rhos[target],
                    "interval_definition": "2.5-97.5% of the preserved seeded feasible-panel distribution",
                }
            )

        missing_wet = [name for name in WET_PANEL if name not in species_index]
        if missing_wet:
            raise KeyError(f"EPA WET species absent from strict-LOO matrix: {missing_wet}")
        wet_indices = [species_index[name] for name in WET_PANEL]
        wet_coverage = float(copula_union_sequence(probability[np.asarray(wet_indices)], rhos[target])[-1] @ weights)
        summary_rows.append(
            {
                "method": "US_EPA_WET_species",
                "method_label": METHOD_LABELS["US_EPA_WET_species"],
                "method_k": len(WET_PANEL),
                "protection_target_x": target,
                "sampled_n": 0,
                "sampled_mean": np.nan,
                "sampled_median": wet_coverage,
                "sampled_q025": np.nan,
                "sampled_q975": np.nan,
                "sampled_min": np.nan,
                "sampled_max": np.nan,
                "fixed_top5_coverage": top5_coverage,
                "rho_working": rhos[target],
                "interval_definition": "fixed standard/method-species panel; no sampled interval",
            }
        )
        summary_rows.append(
            {
                "method": "R1_STRICT_LOO_TOP5",
                "method_label": METHOD_LABELS["R1_STRICT_LOO_TOP5"],
                "method_k": 5,
                "protection_target_x": target,
                "sampled_n": 0,
                "sampled_mean": np.nan,
                "sampled_median": top5_coverage,
                "sampled_q025": np.nan,
                "sampled_q975": np.nan,
                "sampled_min": np.nan,
                "sampled_max": np.nan,
                "fixed_top5_coverage": top5_coverage,
                "rho_working": rhos[target],
                "interval_definition": "fixed R1 strict-LOO national Top-5; no sampled interval",
            }
        )

    if sampled["coverage"].isna().any():
        raise AssertionError("Some preserved comparator panels were not rescored")
    return pd.DataFrame(summary_rows), sampled


def build_panel_d(sequence: list[str]) -> pd.DataFrame:
    z = np.load(AP02_DIR / "ap02_strict_loo_species_protective_target_tail_probability_x95.npz", allow_pickle=True)
    probability = z["p"].astype(float)
    direct_n = z["direct_n"].astype(float)
    species = np.asarray(z["species"], dtype=object)
    chemicals = np.asarray(z["chemicals"], dtype=object)
    mechanisms = np.asarray(z["mechanisms"], dtype=object)
    weights = pd.read_csv(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv")[["DTXSID", "national_weight"]]
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    weight_map = weights.set_index("DTXSID")["national_weight"].to_dict()
    meta = pd.DataFrame(
        {
            "target_index": np.arange(len(chemicals)),
            "chemical": chemicals.astype(str),
            "domain_raw": mechanisms.astype(str),
            "domain": [chemical_domain_label(value) for value in mechanisms],
            "weight": [float(weight_map.get(str(value), 0.0)) for value in chemicals],
        }
    )
    domain_order = (
        meta.groupby("domain", as_index=False)
        .agg(domain_weight=("weight", "sum"), n_targets=("target_index", "size"))
        .sort_values(["domain_weight", "domain"], ascending=[False, True])
        .head(8)["domain"]
        .tolist()
    )
    species_index = {str(name): idx for idx, name in enumerate(species)}
    rows: list[dict[str, object]] = []
    for rank, latin_name in enumerate(sequence[:5], start=1):
        idx = species_index[latin_name]
        for domain in domain_order:
            mask = meta["domain"].eq(domain).to_numpy()
            local_weights = meta.loc[mask, "weight"].to_numpy(float)
            if local_weights.sum() <= 0:
                local_weights = np.ones(mask.sum(), dtype=float)
            rows.append(
                {
                    "rank": rank,
                    "latin_name": latin_name,
                    "species_label": short_species(latin_name),
                    "domain": domain,
                    "weighted_mean_tail_probability": float(np.average(probability[idx, mask], weights=local_weights)),
                    "direct_support_targets": int((direct_n[idx, mask] > 0).sum()),
                    "weighted_direct_support_fraction": float(np.average((direct_n[idx, mask] > 0).astype(float), weights=local_weights)),
                    "n_targets": int(mask.sum()),
                    "n_chemicals": int(meta.loc[mask, "chemical"].nunique()),
                    "domain_weight": float(local_weights.sum()),
                    "probability_source": "AP02_STRICT_LOO",
                }
            )
    return pd.DataFrame(rows)


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-26, 15),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=11,
        clip_on=False,
    )


def draw(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, panel_d: pd.DataFrame) -> list[str]:
    apply_style(font_size=7.3)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.25, 5.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.32, hspace=0.45)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    marker_styles = {0.80: "o", 0.90: "s", 0.95: "D"}
    for target in TARGETS:
        group = panel_a[np.isclose(panel_a["protection_target_x"], target)].sort_values("rank")
        ax_a.plot(
            group["rank"],
            group["expected_weighted_joint_coverage"],
            color=TARGET_COLORS[target],
            marker=marker_styles[target],
            ms=4.1,
            lw=1.25,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=TARGET_LABELS[target],
        )
        value = float(group.loc[group["rank"].eq(5), "expected_weighted_joint_coverage"].iloc[0])
        ax_a.text(5.3, value, f"{value:.3f}", va="center", fontsize=6.4, color=TARGET_COLORS[target])
    ax_a.axvline(5, color=PALETTE["neutral"], lw=0.8, ls=(0, (3, 2)))
    ax_a.axvline(10, color=PALETTE["neutral_light"], lw=0.8, ls=(0, (3, 2)))
    marker_box = dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.45)
    ax_a.text(4.55, 0.035, "Fixed national\nTop-5", ha="center", va="bottom", fontsize=6.4, color=PALETTE["neutral_dark"], linespacing=0.95, bbox=marker_box)
    ax_a.text(10.75, 0.035, "Extended national\nTop-10", ha="center", va="bottom", fontsize=6.4, color=PALETTE["neutral_dark"], linespacing=0.95, bbox=marker_box)
    ax_a.set(xlim=(1, 20), ylim=(0, 1.01), xticks=[1, 5, 10, 15, 20])
    ax_a.set_xlabel("Panel size (k)")
    ax_a.set_ylabel("National chemical-priority-\nweighted expected capture")
    ax_a.set_title("Expected capture by panel size", loc="left", fontsize=8.2)
    ax_a.legend(loc="lower right", fontsize=6.7, handlelength=1.7)
    clean_axis(ax_a, grid=True)
    aligned_panel_label(ax_a, "a")

    methods = [*METHOD_ORDER, "US_EPA_WET_species"]
    base_x = np.arange(len(methods), dtype=float)
    offsets = {0.80: -0.155, 0.90: 0.0, 0.95: 0.155}
    bar_width = 0.12
    target_reference_labels = {0.80: "Lower-20%", 0.90: "Lower-10%", 0.95: "Lower-5%"}
    for target in TARGETS:
        group = panel_b[np.isclose(panel_b["protection_target_x"], target)].set_index("method")
        x_values = base_x + offsets[target]
        medians = np.asarray([float(group.loc[method, "sampled_median"]) for method in methods])
        lower = np.asarray([
            float(group.loc[method, "sampled_q025"]) if pd.notna(group.loc[method, "sampled_q025"]) else value
            for method, value in zip(methods, medians)
        ])
        upper = np.asarray([
            float(group.loc[method, "sampled_q975"]) if pd.notna(group.loc[method, "sampled_q975"]) else value
            for method, value in zip(methods, medians)
        ])
        sampled_mask = np.asarray([int(group.loc[method, "sampled_n"]) > 0 for method in methods])
        ax_b.bar(x_values, medians, width=bar_width, color=TARGET_COLORS[target], alpha=0.26, edgecolor=TARGET_COLORS[target], linewidth=0.6, zorder=3)
        ax_b.vlines(x_values[sampled_mask], lower[sampled_mask], upper[sampled_mask], color=PALETTE["neutral_dark"], lw=0.85, zorder=8)
        ax_b.hlines(lower[sampled_mask], x_values[sampled_mask] - bar_width * 0.36, x_values[sampled_mask] + bar_width * 0.36, color=PALETTE["neutral_dark"], lw=0.85, zorder=8)
        ax_b.hlines(upper[sampled_mask], x_values[sampled_mask] - bar_width * 0.36, x_values[sampled_mask] + bar_width * 0.36, color=PALETTE["neutral_dark"], lw=0.85, zorder=8)
        sampled_means = np.asarray([float(group.loc[method, "sampled_mean"]) if int(group.loc[method, "sampled_n"]) > 0 else np.nan for method in methods])
        ax_b.scatter(x_values[sampled_mask], sampled_means[sampled_mask], marker="o", s=15, facecolor="white", edgecolor=PALETTE["neutral_dark"], linewidth=0.65, zorder=10)
        fixed_mask = ~sampled_mask
        ax_b.hlines(medians[fixed_mask], x_values[fixed_mask] - bar_width / 2, x_values[fixed_mask] + bar_width / 2, color=TARGET_COLORS[target], lw=1.0, zorder=9)
        top5_value = float(group.loc["R1_STRICT_LOO_TOP5", "sampled_median"])
        ax_b.axhline(top5_value, color=TARGET_COLORS[target], lw=1.25, ls=(0, (4, 2)), zorder=1)
        label_offset = {0.80: 0.020, 0.90: 0.027, 0.95: 0.032}[target]
        ax_b.text(len(methods) - 0.18, min(top5_value + label_offset, 1.0), f"{target_reference_labels[target]} Top-5 {top5_value:.3f}", ha="right", va="bottom", fontsize=5.6, fontweight="bold", color=TARGET_COLORS[target], bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.35))
    ax_b.set_xticks(base_x)
    ax_b.set_xticklabels([METHOD_LABELS[method] for method in methods], fontsize=5.9, linespacing=0.92)
    ax_b.set_xlim(-0.48, len(methods) - 0.23)
    ax_b.set_ylim(0, 1.01)
    ax_b.set_ylabel("Expected capture")
    ax_b.set_title("Comparator performance across tail targets", loc="left", fontsize=8.2)
    clean_axis(ax_b, grid=False)
    ax_b.yaxis.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    handles_b = [
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.80], edgecolor=TARGET_COLORS[0.80], alpha=0.28, label="Lower-20%"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.90], edgecolor=TARGET_COLORS[0.90], alpha=0.28, label="Lower-10%"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.95], edgecolor=TARGET_COLORS[0.95], alpha=0.28, label="Lower-5%"),
        mpl.lines.Line2D([0], [0], color=PALETTE["neutral_dark"], lw=1.0, label="95% feasible-panel interval"),
        mpl.lines.Line2D([0], [0], marker="o", color=PALETTE["neutral_dark"], markerfacecolor="white", markersize=4.0, lw=0, label="Sampled mean"),
    ]
    ax_b.legend(handles=handles_b, loc="center left", bbox_to_anchor=(1.01, 0.50), fontsize=5.4, frameon=False, handlelength=1.25, handletextpad=0.45, borderaxespad=0.0, labelspacing=0.38)
    aligned_panel_label(ax_b, "b")

    x = np.arange(len(panel_c))
    ax_c.bar(x, panel_c["incremental_gain"], color=PALETTE["blue_light"], edgecolor=PALETTE["blue"], lw=0.5, label="Incremental gain", zorder=2)
    ax_c.plot(x, panel_c["leave_one_out_loss"], color=PALETTE["red"], marker="o", lw=1.25, ms=3.2, label="Leave-one-out loss", zorder=3)
    ax_c.axhline(0, color=PALETTE["neutral_dark"], lw=0.65)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(panel_c["species_label"], rotation=34, ha="right", fontsize=6.5, fontstyle="italic")
    ax_c.set_ylabel("Change in lower-5% expected capture")
    ax_c.set_title("Non-redundant Top-5 contributions", loc="left", fontsize=8.2)
    ax_c.legend(loc="upper right", fontsize=6.6)
    clean_axis(ax_c, grid=False)
    ax_c.yaxis.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    aligned_panel_label(ax_c, "c")

    species_order = panel_d.sort_values("rank")["species_label"].drop_duplicates().tolist()
    domain_order = panel_d.groupby("domain")["domain_weight"].max().sort_values(ascending=False).index.tolist()
    lookup = panel_d.set_index(["species_label", "domain"])
    vmin = float(panel_d["weighted_mean_tail_probability"].min())
    vmax = float(panel_d["weighted_mean_tail_probability"].max())
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "r1_probability", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )
    color_norm = mpl.colors.Normalize(vmin=max(0, vmin * 0.95), vmax=vmax * 1.03)
    max_support = max(int(panel_d["direct_support_targets"].max()), 1)
    for row_i, species_label in enumerate(species_order):
        for col_i, domain in enumerate(domain_order):
            row = lookup.loc[(species_label, domain)]
            support = int(row["direct_support_targets"])
            size = 15 + 50 * math.sqrt(support / max_support)
            ax_d.scatter(
                col_i,
                row_i,
                s=size,
                facecolor=cmap(color_norm(float(row["weighted_mean_tail_probability"]))),
                edgecolor="white",
                linewidth=0.35,
            )
    ax_d.set_xticks(np.arange(len(domain_order)))
    ax_d.set_xticklabels(domain_order, rotation=43, ha="right", fontsize=5.8)
    ax_d.set_yticks(np.arange(len(species_order)))
    ax_d.set_yticklabels(species_order, fontsize=6.4, fontstyle="italic")
    ax_d.set_xlim(-0.6, len(domain_order) - 0.4)
    ax_d.set_ylim(len(species_order) - 0.4, -0.6)
    ax_d.set_title("Species-by-domain complementarity", loc="left", fontsize=8.2)
    ax_d.set_xlabel("Priority chemical domain")
    ax_d.set_ylabel("Fixed national Top-5 species")
    clean_axis(ax_d, grid=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), ax=ax_d, fraction=0.045, pad=0.025)
    colorbar.set_label("Mean tail probability", fontsize=6.3)
    colorbar.ax.tick_params(labelsize=5.7)
    legend_support = sorted(set([0, max(1, int(round(max_support / 2))), max_support]))
    handles = [
        ax_d.scatter([], [], s=15 + 50 * math.sqrt(value / max_support), facecolor="#9ECAE1", edgecolor=PALETTE["neutral_dark"], linewidth=0.35)
        for value in legend_support
    ]
    ax_d.legend(
        handles,
        [str(value) for value in legend_support],
        title="Measured\ntarget cells",
        fontsize=5.6,
        title_fontsize=5.8,
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        borderaxespad=0,
        handletextpad=0.55,
        labelspacing=0.55,
    )
    aligned_panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.09, right=0.89, top=0.95, bottom=0.09)
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(
    outputs: list[str],
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_b_samples: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    sequence: list[str],
    rhos: dict[float, float],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "figure_02a_top_k_curves.csv": panel_a,
        "figure_02b_comparator_summary.csv": panel_b,
        "figure_02b_preserved_panels_rescored.csv.gz": panel_b_samples,
        "figure_02c_gain_loss.csv": panel_c,
        "figure_02d_species_domain_complementarity.csv": panel_d,
    }
    for name, frame in tables.items():
        path = OUT_DIR / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)

    caption = (
        "Figure 2. Strict leave-one-out estimates preserve the membership of the national Top-5 while updating its order and expected capture. "
        "a, National chemical-priority-weighted expected capture for prefixes of the x = 0.95 sequence evaluated at the lower-20%, lower-10%, and lower-5% measured-tail targets. "
        "b, Rescored distributions for the same seeded random and rule-constrained comparator panels used in R0, together with the fixed EPA WET method-species panel; dashed horizontal lines show the R1 national Top-5 at each target. Bars show sampled medians (or the fixed EPA WET value), vertical lines show the 2.5th-97.5th percentiles of feasible-panel distributions, and open circles show sampled means. The EPA WQC and CCME comparators are study operationalizations of composition rules. "
        "c, Incremental gain during sequence construction and the expected-capture loss after removing each member from the completed Top-5. "
        "d, Complementarity across the eight highest-weight chemical domains; color denotes weighted mean lower-tail probability and point area denotes the number of directly measured species-target cells. "
        "All probability-derived quantities use the strict focal-species leave-one-out matrices and target-specific dependence estimates. Expected capture is a probabilistic measured-tail screening quantity; confirmatory apical evidence remains necessary for formal protection assessments."
    )
    contract = "\n".join(
        [
            "# R1 Figure 2 contract",
            "",
            "Core conclusion: strict focal-species leave-one-out estimation preserves the national Top-5 membership while updating order and expected-capture magnitudes; complementary contributions remain distributed across species and chemical domains.",
            "",
            "- Panel a: the R0 curve-and-reference-line layout is retained; only strict-LOO/AP05 values and the sequence order are updated.",
            "- Panel b: the R0 bar, feasible-interval, sampled-mean, and fixed-Top-5 benchmark grammar is retained; all probabilities and dependence parameters are recomputed from AP02/AP05.",
            "- Panel c: the R0 incremental-gain bars plus deletion-loss line are retained on one absolute expected-capture scale.",
            "- Panel d: direct measured support is encoded by point area and strict-LOO tail probability by color.",
            "- Scope: comparator distributions are feasible-panel distributions, not confidence intervals or agency-endorsed sentinel lists.",
            "- Style: R0 two-by-two quantitative grid, Times New Roman typography, restrained blue/teal/red/gold palette, editable text in SVG/PDF.",
            "",
            f"Manuscript caption: {caption}",
            "",
        ]
    )
    (OUT_DIR / "figure_contract.md").write_text(contract, encoding="utf-8")

    all_files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure 2",
        "version": "R1_AP02_AP05",
        "title": "Strict-LOO national panel performance and complementarity",
        "core_conclusion": "Strict leave-one-out estimation preserves Top-5 membership while updating order and expected-capture magnitudes.",
        "manuscript_caption": caption,
        "fixed_x95_sequence": sequence,
        "target_specific_rho": {str(key): value for key, value in rhos.items()},
        "panel_map": {
            "a": "R0 panel-size curves and Top-5/Top-10 guides with strict-LOO/AP05 values",
            "b": "R0 comparator bars, sampled intervals/means, and fixed Top-5 benchmarks rescored under AP02/AP05",
            "c": "R0 incremental-gain bars and leave-one-out-loss line with strict-LOO values",
            "d": "Strict-LOO species-by-domain complementarity",
        },
        "input_dependencies": [
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x80.npz"),
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x90.npz"),
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz"),
            str(AP02_DIR / "ap02_strict_loo_species_protective_target_tail_probability_x95.npz"),
            str(AP05_DIR / "ap05_rho_estimation_audit.csv"),
            str(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv"),
            str(BHBT_ROOT / "outputs" / "figures" / "figure_02" / "figure_02b_feasible_combination_samples.csv"),
        ],
        "outputs": all_files,
        "qa": {
            "top5_assertion": EXPECTED_TOP5,
            "x95_top5_capture_assertion": 0.337764,
            "editable_svg_text": True,
            "pdf_truetype_text": True,
            "png_dpi": 300,
            "tiff_dpi": 600,
            "red_green_not_sole_encoding": True,
        },
    }
    manifest["sha256"] = {name: sha256_file(OUT_DIR / name) for name in all_files if (OUT_DIR / name).exists()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rhos = load_rhos()
    panel_a, panel_c, sequence = build_panel_a_and_c(rhos)
    panel_b, panel_b_samples = build_panel_b(rhos)
    panel_d = build_panel_d(sequence)
    outputs = draw(panel_a, panel_b, panel_c, panel_d)
    write_outputs(outputs, panel_a, panel_b, panel_b_samples, panel_c, panel_d, sequence, rhos)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(OUT_DIR),
                "top5": sequence[:5],
                "x95_top5_capture": float(panel_a[(panel_a["protection_target_x"].eq(0.95)) & (panel_a["rank"].eq(5))]["expected_weighted_joint_coverage"].iloc[0]),
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
