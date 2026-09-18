#!/usr/bin/env python3
"""Render R1 SI Figure S3: guild context, comparator audit, and Top-5 support."""

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
FIG2_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "main_figure_02"
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "si_figure_s3"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, apply_style, clean_axis, export_figure, short_species  # noqa: E402


TOP5 = [
    "Gastrophryne carolinensis",
    "Neocloeon triangulifer",
    "Hyalella azteca",
    "Daphnia ambigua",
    "Daphnia magna",
]
METHOD_ORDER = [
    "sensitivity_weighted_guild_scaffold",
    "EPA_WQC_taxonomic_requirements",
    "Canada_Type_A_composition",
    "taxonomy_diversity_baseline",
]
METHOD_LABELS = {
    "sensitivity_weighted_guild_scaffold": "Sensitivity-weighted\nguild scaffold",
    "EPA_WQC_taxonomic_requirements": "EPA WQC proxy",
    "Canada_Type_A_composition": "CCME Type A proxy",
    "taxonomy_diversity_baseline": "Taxonomic diversity",
    "data_driven": "Optimized",
}
FIGURE_BASENAME = "figure_s3_r1_taxonomic_support"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copula_union_sequence(probability: np.ndarray, rho: float, nodes: int = 12) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    if probability.ndim == 1:
        probability = probability[None, :]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - probability)
    shared = math.sqrt(max(rho, 0.0))
    residual = math.sqrt(max(1.0 - rho, 1e-8))
    no_event = np.zeros_like(probability)
    for zz, ww in zip(z, w):
        no_event += ww * np.cumprod(norm.cdf((threshold - shared * zz) / residual), axis=0)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def copula_union_many(probability: np.ndarray, rho: float, nodes: int = 12) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - probability)
    shared = math.sqrt(max(rho, 0.0))
    residual = math.sqrt(max(1.0 - rho, 1e-8))
    no_event = np.zeros((probability.shape[0], probability.shape[2]), dtype=float)
    for zz, ww in zip(z, w):
        no_event += ww * np.prod(norm.cdf((threshold - shared * zz) / residual), axis=1)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def deterministic_greedy(probability: np.ndarray, weights: np.ndarray, species: np.ndarray, k: int) -> list[int]:
    selected: list[int] = []
    remaining = set(range(probability.shape[0]))
    no_event = np.ones(probability.shape[1], dtype=float)
    for _ in range(k):
        candidates = sorted(remaining, key=lambda idx: str(species[idx]))
        gains = probability[np.asarray(candidates)] @ (weights * no_event)
        maximum = float(np.max(gains))
        tied = [candidates[i] for i, value in enumerate(gains) if np.isclose(value, maximum, rtol=0, atol=1e-14)]
        best = min(tied, key=lambda idx: str(species[idx]))
        selected.append(best)
        remaining.remove(best)
        no_event *= 1.0 - probability[best]
    return selected


def load_strict_layer() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    z = np.load(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz", allow_pickle=True)
    species = np.asarray(z["species"], dtype=object)
    chemicals = np.asarray(z["chemicals"], dtype=object).astype(str)
    index = {name: idx for idx, name in enumerate(chemicals)}
    priority = pd.read_csv(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv")
    priority["DTXSID"] = priority["DTXSID"].astype(str)
    present = priority[priority["DTXSID"].isin(index)].copy()
    indices = np.asarray([index[name] for name in present["DTXSID"]], dtype=int)
    weights = present["national_weight"].to_numpy(float)
    weights /= weights.sum()
    rho_table = pd.read_csv(AP05_DIR / "ap05_rho_estimation_audit.csv")
    rho = float(
        rho_table[
            rho_table["variant"].eq("AP02_STRICT_LOO")
            & rho_table["selected_for_downstream"].astype(bool)
            & np.isclose(rho_table["protection_target_x"], 0.95)
        ]["rho_working"].iloc[0]
    )
    return z["p"][:, indices].astype(float), species, weights, rho


def build_panel_a() -> pd.DataFrame:
    source = pd.read_csv(BHBT_ROOT / "manuscript" / "si_figures" / "figure_s3_taxonomic_support_source_data.csv")
    panel = source[source["panel"].eq("a_guild_enrichment")].copy()
    panel = panel[~panel["guild"].eq("audit_wide_baseline")].copy()
    return panel.sort_values("hc5_hit_enrichment_vs_all", ascending=True)


def build_panel_b(probability: np.ndarray, species: np.ndarray, weights: np.ndarray, rho: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = BHBT_ROOT / "manuscript" / "si_figures" / "figure_s4b_feasible_combination_samples.csv"
    samples = pd.read_csv(sample_path)
    samples = samples[samples["method"].isin(METHOD_ORDER) & samples["k"].isin([5, 10])].copy()
    species_index = {str(name): idx for idx, name in enumerate(species)}
    samples["expected_capture_r0_archived"] = samples["expected_capture"]
    samples["expected_capture"] = np.nan

    summaries: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for k in [5, 10]:
            mask = samples["method"].eq(method) & samples["k"].eq(k)
            group = samples.loc[mask].sort_values("sample_id")
            panels = [[item.strip() for item in value.split(" | ")] for value in group["panel_species"].astype(str)]
            missing = sorted({name for panel in panels for name in panel if name not in species_index})
            if missing:
                raise KeyError(f"Species absent from strict-LOO matrix: {missing[:5]}")
            scores = np.empty(len(panels), dtype=float)
            for start in range(0, len(panels), 96):
                stop = min(start + 96, len(panels))
                indices = np.asarray([[species_index[name] for name in panel] for panel in panels[start:stop]], dtype=int)
                scores[start:stop] = copula_union_many(probability[indices], rho) @ weights
            samples.loc[group.index, "expected_capture"] = scores
            summaries.append(
                {
                    "method": method,
                    "framework": METHOD_LABELS[method],
                    "k": k,
                    "sampled_n": len(scores),
                    "sampled_mean": float(np.mean(scores)),
                    "sampled_median": float(np.median(scores)),
                    "sampled_q025": float(np.quantile(scores, 0.025)),
                    "sampled_q975": float(np.quantile(scores, 0.975)),
                    "fixed_panel_value": np.nan,
                    "rho_working": rho,
                    "probability_source": "AP02_STRICT_LOO",
                }
            )

    sequence = deterministic_greedy(probability, weights, species, 10)
    names = [str(species[idx]) for idx in sequence]
    if names[:5] != TOP5:
        raise AssertionError(f"Strict-LOO Top-5 mismatch: {names[:5]}")
    cumulative = copula_union_sequence(probability[np.asarray(sequence)], rho) @ weights
    for k in [5, 10]:
        summaries.append(
            {
                "method": "data_driven",
                "framework": METHOD_LABELS["data_driven"],
                "k": k,
                "sampled_n": 0,
                "sampled_mean": np.nan,
                "sampled_median": float(cumulative[k - 1]),
                "sampled_q025": np.nan,
                "sampled_q975": np.nan,
                "fixed_panel_value": float(cumulative[k - 1]),
                "rho_working": rho,
                "probability_source": "AP02_STRICT_LOO",
            }
        )
    if samples["expected_capture"].isna().any():
        raise AssertionError("Some SI Figure S3 comparator panels were not rescored")
    return pd.DataFrame(summaries), samples


def build_panel_c(probability: np.ndarray, species: np.ndarray, weights: np.ndarray, rho: float) -> pd.DataFrame:
    species_index = {str(name): idx for idx, name in enumerate(species)}
    top_indices = [species_index[name] for name in TOP5]
    full = float(copula_union_sequence(probability[np.asarray(top_indices)], rho)[-1] @ weights)
    candidate = pd.read_csv(BHBT_ROOT / "results" / "probability" / "candidate_universe_locked.csv").set_index("latin_name")
    rows: list[dict[str, object]] = []
    for rank, (name, idx) in enumerate(zip(TOP5, top_indices), start=1):
        retained = [other for other in top_indices if other != idx]
        without = float(copula_union_sequence(probability[np.asarray(retained)], rho)[-1] @ weights)
        rows.append(
            {
                "rank": rank,
                "latin_name": name,
                "species_label": short_species(name),
                "single_species_capture": float(probability[idx] @ weights),
                "leave_one_out_loss": full - without,
                "n_contexts": int(candidate.loc[name, "n_contexts"]),
                "support_tier": str(candidate.loc[name, "support_tier"]),
                "full_top5_capture": full,
                "rho_working": rho,
                "probability_source": "AP02_STRICT_LOO",
            }
        )
    return pd.DataFrame(rows)


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-22, 13), textcoords="offset points", ha="left", va="top", fontweight="bold", fontsize=10, clip_on=False)


def draw(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> list[str]:
    apply_style(font_size=7.1)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), gridspec_kw={"width_ratios": [1.0, 1.08, 0.95], "wspace": 0.52})
    ax_a, ax_b, ax_c = axes

    guild_labels = {
        "cladoceran_branchiopod": "Cladoceran/\nbranchiopod",
        "other_crustacean": "Other crustacean",
        "amphibian": "Amphibian",
        "aquatic_insect": "Aquatic insect",
        "other_invertebrate": "Other invertebrate",
        "mollusk": "Mollusk",
        "primary_producer_algae_plant": "Producer/alga/plant",
        "fish": "Fish",
        "other_or_unresolved": "Other/unresolved",
        "other_vertebrate": "Other vertebrate",
    }
    colors = [PALETTE["red"] if value >= 1.5 else PALETTE["gold"] if value >= 1.1 else PALETTE["blue2"] for value in panel_a["hc5_hit_enrichment_vs_all"]]
    y = np.arange(len(panel_a))
    ax_a.barh(y, panel_a["hc5_hit_enrichment_vs_all"], color=colors, height=0.72)
    ax_a.axvline(1, color=PALETTE["neutral"], lw=0.85, ls=(0, (3, 2)))
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([guild_labels.get(value, value) for value in panel_a["guild"]], fontsize=5.7)
    ax_a.set_xlabel("Fold enrichment vs audit-wide hit rate")
    ax_a.set_title("Lower-tail enrichment by guild", loc="left", fontsize=8.0)
    ax_a.set_xlim(0, max(2.6, float(panel_a["hc5_hit_enrichment_vs_all"].max()) + 0.45))
    for yy, row in enumerate(panel_a.itertuples(index=False)):
        ax_a.text(float(row.hc5_hit_enrichment_vs_all) + 0.04, yy, f"{100*float(row.hc5_hit_rate):.1f}%\nn={int(row.n_species)}", va="center", fontsize=5.2)
    clean_axis(ax_a, grid=False)
    aligned_panel_label(ax_a, "a")

    methods = ["data_driven", *METHOD_ORDER]
    y = np.arange(len(methods))
    offsets = {5: -0.17, 10: 0.17}
    colors_k = {5: PALETTE["blue2"], 10: PALETTE["teal"]}
    for k in [5, 10]:
        for idx, method in enumerate(methods):
            row = panel_b[panel_b["method"].eq(method) & panel_b["k"].eq(k)].iloc[0]
            value = float(row["sampled_median"])
            if int(row["sampled_n"]) > 0:
                low = value - float(row["sampled_q025"])
                high = float(row["sampled_q975"]) - value
                ax_b.errorbar(value * 100, idx + offsets[k], xerr=np.asarray([[low * 100], [high * 100]]), fmt="s", ms=4.2, color=colors_k[k], capsize=1.7, lw=0.9)
            else:
                ax_b.scatter(value * 100, idx + offsets[k], s=28, facecolor="white", edgecolor=colors_k[k], marker="s", linewidth=1.0)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([METHOD_LABELS[method] for method in methods], fontsize=5.7)
    ax_b.set_xlabel("Lower-5% expected capture (%)")
    ax_b.set_title("Matched-size comparator audit", loc="left", fontsize=8.0, pad=27)
    max_x = float(panel_b["sampled_q975"].fillna(panel_b["sampled_median"]).max()) * 100
    ax_b.set_xlim(0, max_x + 5)
    handles = [
        ax_b.scatter([], [], marker="s", s=24, color=colors_k[5], label="k = 5"),
        ax_b.scatter([], [], marker="s", s=24, color=colors_k[10], label="k = 10"),
    ]
    ax_b.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=5.7, ncol=2, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)
    clean_axis(ax_b, grid=False)
    ax_b.xaxis.grid(True, color="#E6E6E6", lw=0.45, zorder=0)
    aligned_panel_label(ax_b, "b")

    y = np.arange(len(panel_c))
    height = 0.34
    ax_c.barh(y - height / 2, panel_c["single_species_capture"] * 100, height, color=PALETTE["teal"], label="Single-species capture")
    ax_c.barh(y + height / 2, panel_c["leave_one_out_loss"] * 100, height, color=PALETTE["blue2"], label="Leave-one-out loss")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([f"{label}\nn={n}" for label, n in zip(panel_c["species_label"], panel_c["n_contexts"])], fontsize=5.6, fontstyle="italic")
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Expected-capture contribution (%)")
    ax_c.set_title("Strict-LOO Top-5 support", loc="left", fontsize=8.0, pad=27)
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=5.5, ncol=1, borderaxespad=0.0)
    clean_axis(ax_c, grid=False)
    ax_c.xaxis.grid(True, color="#E6E6E6", lw=0.45, zorder=0)
    aligned_panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.18)
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(outputs: list[str], panel_a: pd.DataFrame, panel_b: pd.DataFrame, samples: pd.DataFrame, panel_c: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "figure_s3a_guild_enrichment.csv": panel_a,
        "figure_s3b_strict_loo_comparator_summary.csv": panel_b,
        "figure_s3b_preserved_panels_rescored.csv.gz": samples,
        "figure_s3c_strict_loo_top5_support.csv": panel_c,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, compression="gzip" if name.endswith(".gz") else None)
    caption = (
        "Figure S3. Taxonomic context, matched-size comparator audit, and strict-LOO support for the fixed national Top-5. "
        "a, Enrichment of direct lower-tail observations by broad aquatic guild relative to the audit-wide hit rate; percentages are guild-specific hit rates and n is the number of species. "
        "b, Lower-5% expected capture for the optimized sequence and the preserved seeded feasible-panel distributions for four study-defined comparator rules at k = 5 and k = 10. Points are medians and horizontal intervals are 2.5th-97.5th percentiles; open symbols denote fixed optimized sequences. EPA WQC and CCME Type A labels denote study operationalizations of composition rules. "
        "c, Priority-weighted single-species capture and leave-one-out loss for the R1 national Top-5; n gives the number of observed protective contexts. Panels b-c use strict focal-species leave-one-out probabilities and the updated x = 0.95 dependence estimate."
    )
    (OUT_DIR / "figure_contract.md").write_text(
        "# R1 Figure S3 contract\n\n"
        "Core conclusion: direct evidence remains taxonomically uneven, but the strict-LOO national Top-5 retains complementary contributions and outperforms sampled rule-constrained panels at matched sizes.\n\n"
        "Panel b preserves the R0 sampled panel identities and recomputes their expected capture from AP02/AP05. Comparator intervals describe feasible-panel distributions, not confidence intervals.\n\n"
        "The panel-b and panel-c legends occupy dedicated bands between their titles and plotting regions; typography is Times New Roman.\n\n"
        f"Manuscript caption: {caption}\n",
        encoding="utf-8",
    )
    files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure S3",
        "version": "R1_AP02_AP05",
        "manuscript_caption": caption,
        "input_dependencies": [
            str(BHBT_ROOT / "manuscript" / "si_figures" / "figure_s3_taxonomic_support_source_data.csv"),
            str(BHBT_ROOT / "manuscript" / "si_figures" / "figure_s4b_feasible_combination_samples.csv"),
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz"),
            str(AP05_DIR / "ap05_rho_estimation_audit.csv"),
        ],
        "outputs": files,
        "qa": {
            "top5": TOP5,
            "x95_top5_capture": float(panel_c["full_top5_capture"].iloc[0]),
            "sampled_comparator_rows": int(len(samples)),
            "editable_svg_text": True,
            "png_dpi": 300,
            "tiff_dpi": 600,
        },
    }
    manifest["sha256"] = {name: sha256_file(OUT_DIR / name) for name in files if (OUT_DIR / name).exists()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probability, species, weights, rho = load_strict_layer()
    panel_a = build_panel_a()
    panel_b, samples = build_panel_b(probability, species, weights, rho)
    panel_c = build_panel_c(probability, species, weights, rho)
    outputs = draw(panel_a, panel_b, panel_c)
    write_outputs(outputs, panel_a, panel_b, samples, panel_c)
    print(json.dumps({"status": "PASS", "output_dir": str(OUT_DIR), "outputs": outputs, "top5_capture": float(panel_c["full_top5_capture"].iloc[0])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
