#!/usr/bin/env python3
"""Figure 2: Top-5 expected-capture performance and complementarity."""

from __future__ import annotations

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
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import (  # noqa: E402
    PALETTE,
    TARGET_COLORS,
    apply_style,
    clean_axis,
    export_figure,
    figure_dir_from_script,
    project_root_from_script,
    save_contract_note,
    short_method,
    short_species,
    write_combined_source,
    write_csv,
    write_manifest,
)
from plot_figure_02b_regulatory_distribution_tests import (  # noqa: E402
    METHOD_SPECS as PANEL_B_METHOD_SPECS,
    N_SAMPLES as PANEL_B_N_SAMPLES,
    sample_regulatory_panels as build_panel_b_feasible,
)


FIGURE_ID = "figure_02"
TITLE = "Fixed national Top-5 improves expected capture through complementary species contributions"
MANUSCRIPT_CAPTION = (
    "Figure 2. The fixed national Top-5 provides a compact configuration with complementary expected lower-tail "
    "capture. a, National chemical-priority-weighted expected capture for prefixes of the x = 0.95 national sequence "
    "evaluated at the lower-20%, lower-10% and lower-5% measured-tail targets. The first five species define the fixed national Top-5 used in "
    "subsequent analyses, whereas the national Top-10 represents an extended-coverage configuration. Chemical-priority "
    "weights were derived in this study from 2024 Water Quality Portal records processed using EPA TADA workflows41, "
    "42; they represent monitoring-derived screening weights rather than a regulatory priority list or measured "
    "exposure distribution. b, Expected-capture comparisons across the "
    "three measured-tail targets with random "
    "five-species panels, taxonomic-diversity k = 5 feasible panels, CCME Type A proxy k = 7 feasible panels, EPA WQC "
    "proxy k = 8 feasible panels and the fixed EPA WET k = 8 panel. Bars show sampled medians for sampled comparator "
    "distributions and the fixed EPA WET value where no sampled distribution exists; vertical lines show sampled "
    "2.5-97.5% feasible-combination intervals, and open circles show sampled means. EPA and CCME panels are "
    "operationalized study proxies and are not agency-endorsed sentinel lists. c, Incremental gain from adding each "
    "species and leave-one-out loss, defined as the reduction in expected capture after removing that species from the "
    "completed Top-5 panel. d, Species-by-chemical-domain complementarity. Expected capture represents probabilistic "
    "measured-tail coverage for sentinel selection; formal ecosystem-protection benchmarks continue to require "
    "confirmatory apical evidence and established SSD or water-quality-criterion derivation."
)
RANDOM_BASELINE_REPLICATES = 500
RANDOM_BASELINE_SEED = 20260622
RANDOM_BASELINE_PANEL_SIZE = 5
RANDOM_BASELINE_NODES = 12


def copula_union_sequence(p: np.ndarray, rho: float, nodes: int = RANDOM_BASELINE_NODES) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    if p.ndim == 1:
        p = p[None, :]
    x, w = hermgauss(nodes)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)
    threshold = norm.ppf(1.0 - p)
    shared_sd = math.sqrt(max(rho, 0.0))
    residual_sd = math.sqrt(max(1.0 - rho, 1e-8))
    no_event = np.zeros_like(p, dtype=float)
    for zz, ww in zip(z, w):
        conditional_no_event = norm.cdf((threshold - shared_sd * zz) / residual_sd)
        no_event += ww * np.cumprod(conditional_no_event, axis=0)
    return np.clip(1.0 - no_event, 0.0, 1.0)


def target_suffix(target: float) -> str:
    return f"{int(round(target * 100)):02d}"


def target_key(target: float) -> str:
    return f"{target:.2f}".rstrip("0").rstrip(".")


def text_color_for_value(value: float, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.46 else PALETTE["neutral_dark"]


def continuous_cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=256)


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-28, 18),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=13.0,
        color="black",
        clip_on=False,
    )


def support_marker_size(value: float) -> float:
    return 18.0 + 8.5 * np.sqrt(max(float(value), 0.0))


def size_legend_values(max_value: int) -> list[int]:
    if max_value <= 0:
        return [0]
    if max_value <= 10:
        mid = max(1, int(round(max_value / 2)))
        return sorted(set([0, mid, max_value]))
    if max_value <= 50:
        mid = int(round(max_value / 2 / 5) * 5)
        return sorted(set([0, max(5, mid), max_value]))
    return sorted(set([0, 25, max_value]))


def chemical_domain_label(value: str) -> str:
    """Readable Figure 2 domain labels; do not collapse distinct MIEs by verb."""

    if not isinstance(value, str) or value.strip() == "":
        return "Unassigned"
    v = value.strip()
    if v.startswith("MIE::"):
        detail = v.split("::", 1)[1].strip()
        mapping = {
            "Deposition of Energy": "Energy metabolism",
            "Binding to voltage-gated sodium channel": "Sodium-channel binding",
            "Activation, AhR": "AhR activation",
            "Agonism, Androgen receptor": "Androgen-receptor agonism",
            "Alkylation, Protein": "Protein alkylation",
            "Binding of antagonist, PPAR alpha": "PPAR-alpha antagonist",
            "Inhibition, Na+/I- symporter (NIS)": "NIS inhibition",
            "Histone deacetylase inhibition": "HDAC inhibition",
            "Inhibition, VegfR2": "VEGFR2 inhibition",
            "Thyroperoxidase, Inhibition": "Thyroperoxidase inhibition",
        }
        return mapping.get(detail, detail.replace(", ", " ").replace("  ", " "))
    if v.startswith("FORM::"):
        form = v.split("::", 1)[1].replace("_", " ")
        mapping = {
            "discrete nonmetal chemical": "Discrete non-metal form",
            "organic salt or multicomponent": "Organic salt/multicomponent",
            "carbon containing metal compound": "Carbon-metal form",
            "elemental metal or single metal species": "Elemental metal form",
            "inorganic metal compound": "Inorganic metal form",
            "metal salt or ionic complex": "Metal salt/ionic complex",
            "organometallic or metal complex": "Organometallic/metal complex",
        }
        return mapping.get(form, form)
    if v.startswith("MOAFAM::"):
        return v.split("::", 1)[1].replace("_", " ")
    return v.replace("_", " ")[:34]


def build_panel_a(root: Path) -> pd.DataFrame:
    curves = pd.read_csv(root / "results/panels/fixed_x95_sequence_cross_target_curves.csv")
    curves = curves[curves["sequence_basis_target"].eq(0.95) & curves["fixed_sequence"].astype(bool)].copy()
    if curves.groupby("protection_target_x")["full_fixed_sequence"].nunique().max() != 1:
        raise ValueError("Figure 2a requires one locked x=0.95 sequence across all evaluation targets")
    return curves


def build_panel_b(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled, summary = build_panel_b_feasible(root)
    summary = summary.copy()
    sampled = sampled.copy()
    summary["representative_selection_note"] = np.where(
        summary["method"].isin(["EPA_WQC_taxonomic_requirements", "Canada_Type_A_composition"]),
        "Regulatory frameworks define taxonomic/composition slots; the mean point summarizes seeded feasible combinations under this study proxy, not an agency-endorsed species list.",
        np.where(
            summary["method"].isin(["US_EPA_WET_species", "EPA_WET_fixed_method_species"]),
            "Fixed EPA WET standard/method species panel; no sampled mean or sampled-combination interval.",
            "Study-defined comparator distribution.",
        ),
    )
    sampled["representative_selection_note"] = "Sampled feasible panel composition under the stated method rule; not a regulatory prescription."
    return summary, sampled


def build_panel_b_stats(panel_b: pd.DataFrame) -> pd.DataFrame:
    method_labels = {
        "random_all_species_probability": "Random k=5",
        "taxonomy_diversity_baseline": "Taxonomic diversity k=5",
        "Canada_Type_A_composition": "CCME Type A proxy k=7",
        "EPA_WQC_taxonomic_requirements": "EPA WQC proxy k=8",
        "US_EPA_WET_species": "EPA WET panel k=8",
    }
    target_labels = {0.80: "lower-20%", 0.90: "lower-10%", 0.95: "lower-5%"}
    method_order = {method: order for order, method in enumerate(method_labels)}
    target_order = {target: order for order, target in enumerate(target_labels)}

    stats = panel_b.copy()
    stats["comparator"] = stats["method"].map(method_labels).fillna(stats["method_label"])
    stats["target_label"] = stats["protection_target_x"].map(target_labels)
    stats["sampled_mean_marker"] = np.where(stats["sampled_n"].fillna(0).astype(int) > 0, stats["sampled_mean"], np.nan)
    stats["fixed_panel_value"] = np.where(
        stats["sampled_n"].fillna(0).astype(int) == 0,
        stats["deterministic_proxy_coverage"],
        np.nan,
    )
    stats["interval_definition"] = np.where(
        stats["sampled_n"].fillna(0).astype(int) > 0,
        "sampled 2.5-97.5% feasible-combination interval",
        "fixed panel; no sampled interval",
    )
    stats["_method_order"] = stats["method"].map(method_order)
    stats["_target_order"] = stats["protection_target_x"].map(target_order)
    return stats.sort_values(["_method_order", "_target_order"])[
        [
            "comparator",
            "method",
            "method_k",
            "target_label",
            "protection_target_x",
            "sampled_n",
            "sampled_mean_marker",
            "sampled_median",
            "sampled_q025",
            "sampled_q975",
            "fixed_panel_value",
            "fixed_top5_coverage",
            "target_specific_optimized_at_comparator_k",
            "interval_definition",
        ]
    ]


def build_panel_c(root: Path) -> pd.DataFrame:
    curves = pd.read_csv(root / "results/panels/national_k1_20_coverage_curves.csv")
    comp = pd.read_csv(root / "results/complementarity/top5_top10_species_incremental_and_taxonomic_contributions.csv")
    cumulative = (
        curves[
            curves["universe"].eq("priority")
            & curves["method"].eq("data_driven")
            & curves["protection_target_x"].eq(0.95)
            & curves["k"].between(1, 5)
        ][["k", "expected_weighted_joint_coverage"]]
        .sort_values("k")
        .copy()
    )
    cumulative["coverage_previous"] = cumulative["expected_weighted_joint_coverage"].shift(fill_value=0.0)
    cumulative["incremental_gain"] = cumulative["expected_weighted_joint_coverage"] - cumulative["coverage_previous"]
    loo = comp[comp["method"].eq("data_driven") & comp["k"].eq(5) & comp["rank"].between(1, 5)].copy()
    out = loo.merge(cumulative[["k", "incremental_gain", "expected_weighted_joint_coverage"]], left_on="rank", right_on="k", how="left")
    out = out.drop(columns=["k_y"]).rename(columns={"k_x": "panel_k", "expected_weighted_joint_coverage": "cumulative_coverage"})
    out["species_label"] = out["latin_name"].map(short_species)
    return out.sort_values("rank")


def build_panel_d(root: Path) -> pd.DataFrame:
    seq = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    top5 = seq[
        seq["universe"].eq("priority")
        & seq["protection_target_x"].eq(0.95)
        & seq["method"].eq("data_driven")
        & seq["rank"].between(1, 5)
    ].sort_values("rank")

    z = np.load(root / "results/probability/species_protective_target_tail_probability_x95.npz", allow_pickle=True)
    p = z["p"]
    direct_n = z["direct_n"]
    species = np.asarray(z["species"], dtype=object)
    chemicals = np.asarray(z["chemicals"], dtype=object)
    mechanisms = np.asarray(z["mechanisms"], dtype=object)
    weights = pd.read_csv(root / "results/weights/national_priority_chemicals.csv")[["DTXSID", "national_weight"]]
    weights["DTXSID"] = weights["DTXSID"].astype(str)
    weight_map = weights.set_index("DTXSID")["national_weight"].to_dict()

    target_meta = pd.DataFrame(
        {
            "target_index": np.arange(len(chemicals)),
            "chemical": chemicals,
            "domain_raw": mechanisms,
            "domain": [chemical_domain_label(str(x)) for x in mechanisms],
            "weight": [float(weight_map.get(str(c), 0.0)) for c in chemicals],
        }
    )
    if target_meta["weight"].sum() <= 0:
        target_meta["weight"] = 1.0
    domains = (
        target_meta.groupby("domain", as_index=False)
        .agg(domain_weight=("weight", "sum"), n_targets=("target_index", "size"), n_chemicals=("chemical", "nunique"))
        .sort_values("domain_weight", ascending=False)
        .head(8)["domain"]
        .tolist()
    )
    species_index = {str(s): i for i, s in enumerate(species)}
    rows: list[dict[str, object]] = []
    for top_row in top5.itertuples(index=False):
        sidx = species_index.get(str(top_row.latin_name))
        if sidx is None:
            continue
        for domain in domains:
            mask = target_meta["domain"].eq(domain).to_numpy()
            w = target_meta.loc[mask, "weight"].to_numpy(dtype=float)
            if w.sum() <= 0:
                w = np.ones(mask.sum(), dtype=float)
            probs = p[sidx, mask].astype(float)
            dirs = direct_n[sidx, mask].astype(float)
            rows.append(
                {
                    "rank": int(top_row.rank),
                    "latin_name": str(top_row.latin_name),
                    "species_label": short_species(str(top_row.latin_name)),
                    "domain": domain,
                    "domain_principle": "explicit MIE/form label; distinct inhibition targets are not collapsed",
                    "weighted_mean_tail_probability": float(np.average(probs, weights=w)),
                    "direct_support_targets": int((dirs > 0).sum()),
                    "weighted_direct_support_fraction": float(np.average((dirs > 0).astype(float), weights=w)),
                    "n_targets": int(mask.sum()),
                    "n_chemicals": int(target_meta.loc[mask, "chemical"].nunique()),
                    "domain_weight": float(w.sum()),
                }
            )
    return pd.DataFrame(rows)


def draw(root: Path, out_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    apply_style(font_size=8.6)
    panel_a = build_panel_a(root)
    panel_b, panel_b_samples = build_panel_b(root)
    panel_c = build_panel_c(root)
    panel_d = build_panel_d(root)

    fig = plt.figure(figsize=(8.15, 5.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], wspace=0.36, hspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    labels = {0.80: "lower 20%", 0.90: "lower 10%", 0.95: "lower 5%"}
    marker_styles = {0.80: "o", 0.90: "s", 0.95: "D"}
    for target, group in panel_a.groupby("protection_target_x", sort=True):
        group = group.sort_values("k")
        ax_a.plot(
            group["k"],
            group["expected_weighted_joint_coverage"],
            marker=marker_styles.get(float(target), "o"),
            ms=6.4,
            lw=1.35,
            ls="-",
            color=TARGET_COLORS.get(float(target), PALETTE["blue"]),
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=labels.get(float(target), f"x={target:g}"),
        )
        k5 = group[group["k"].eq(5)].iloc[0]
        ax_a.text(5.35, k5["expected_weighted_joint_coverage"], f"{k5['expected_weighted_joint_coverage']:.3f}", fontsize=7.2, va="center")
        if math.isclose(float(target), 0.95):
            k10 = group[group["k"].eq(10)].iloc[0]
            ax_a.text(
                10.35,
                k10["expected_weighted_joint_coverage"],
                f"{k10['expected_weighted_joint_coverage']:.3f}",
                fontsize=7.2,
                va="center",
                color=TARGET_COLORS.get(float(target), PALETTE["blue"]),
            )
    ax_a.axvline(5, ls="--", lw=0.9, color=PALETTE["neutral"], alpha=0.78)
    ax_a.axvline(10, ls="--", lw=0.8, color=PALETTE["neutral"], alpha=0.42)
    marker_box = dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.55)
    ax_a.text(4.55, 0.035, "Fixed national\nTop-5", ha="center", va="bottom", fontsize=6.8, color=PALETTE["neutral_dark"], linespacing=0.95, bbox=marker_box)
    ax_a.text(10.75, 0.035, "Extended national\nTop-10", ha="center", va="bottom", fontsize=6.8, color=PALETTE["neutral_dark"], alpha=0.86, linespacing=0.95, bbox=marker_box)
    ax_a.set_xlim(1, 20)
    ax_a.set_ylim(0, 1.03)
    ax_a.set_xticks([1, 5, 10, 15, 20])
    ax_a.set_xlabel("Panel size (k)")
    ax_a.set_ylabel("National chemical-priority-\nweighted expected capture")
    ax_a.set_title("Expected capture by panel size")
    ax_a.legend(fontsize=7.3, loc="lower right")
    clean_axis(ax_a, grid=True)
    aligned_panel_label(ax_a, "a")

    panel_b_target_labels = {0.80: "lower-20% tail", 0.90: "lower-10% tail", 0.95: "lower-5% tail"}
    method_order_b = [
        "random_all_species_probability",
        "taxonomy_diversity_baseline",
        "Canada_Type_A_composition",
        "EPA_WQC_taxonomic_requirements",
        "US_EPA_WET_species",
    ]
    x_positions = {method: i for i, method in enumerate(method_order_b)}
    target_offsets = {0.80: -0.155, 0.90: 0.0, 0.95: 0.155}
    bar_width = 0.12
    jitter_rng = np.random.default_rng(20260701 + 11)
    for target in [0.80, 0.90, 0.95]:
        top5 = float(panel_b[panel_b["protection_target_x"].eq(target)]["fixed_top5_coverage"].iloc[0])
        color = TARGET_COLORS[target]
        ax_b.axhline(top5, color=color, lw=1.55, ls=(0, (4, 2)), zorder=1)
        label_offset = {0.80: 0.024, 0.90: 0.034, 0.95: 0.038}[target]
        ax_b.text(
            len(method_order_b) - 0.18,
            min(top5 + label_offset, 1.0),
            f"{panel_b_target_labels[target]} Top-5 panel {top5:.3f}",
            ha="right",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.55),
        )
    for method in method_order_b:
        x0 = x_positions[method]
        for target in [0.80, 0.90, 0.95]:
            row = panel_b[panel_b["method"].eq(method) & panel_b["protection_target_x"].eq(target)].iloc[0]
            x_pos = x0 + target_offsets[target]
            color = TARGET_COLORS[target]
            sampled_n = int(row["sampled_n"]) if pd.notna(row["sampled_n"]) else 0
            fixed_panel_value = (
                float(row["deterministic_proxy_coverage"])
                if pd.notna(row["deterministic_proxy_coverage"])
                else np.nan
            )
            if sampled_n > 0:
                values = panel_b_samples[
                    panel_b_samples["method"].eq(method)
                    & panel_b_samples["protection_target_x"].eq(target)
                ]["coverage"].to_numpy(dtype=float)
                jitter = jitter_rng.normal(0, 0.020, size=len(values))
                ax_b.scatter(
                    np.full(len(values), x_pos) + jitter,
                    values,
                    s=1.6,
                    alpha=0.034,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                    zorder=2,
                )
                bar_height = float(row["sampled_median"])
                ax_b.bar(
                    x_pos,
                    bar_height,
                    width=bar_width,
                    color=color,
                    alpha=0.26,
                    edgecolor=color,
                    linewidth=0.6,
                    zorder=3,
                )
                interval_low = float(row["sampled_q025"])
                interval_high = float(row["sampled_q975"])
                ax_b.vlines(
                    x_pos,
                    interval_low,
                    interval_high,
                    color=PALETTE["neutral_dark"],
                    lw=0.9,
                    zorder=10,
                )
                ax_b.hlines(
                    [interval_low, interval_high],
                    x_pos - bar_width * 0.36,
                    x_pos + bar_width * 0.36,
                    color=PALETTE["neutral_dark"],
                    lw=0.9,
                    zorder=10,
                )
                ax_b.hlines(bar_height, x_pos - bar_width / 2, x_pos + bar_width / 2, color=color, lw=1.0, zorder=8)
                sampled_mean = float(row["sampled_mean"])
                ax_b.scatter(
                    [x_pos],
                    [sampled_mean],
                    marker="o",
                    s=18,
                    color="white",
                    edgecolor=PALETTE["neutral_dark"],
                    linewidth=0.7,
                    zorder=11,
                )
            else:
                ax_b.bar(
                    x_pos,
                    fixed_panel_value,
                    width=bar_width,
                    color=color,
                    alpha=0.24,
                    edgecolor=color,
                    linewidth=0.6,
                    zorder=3,
                )
    ax_b.set_xticks([x_positions[m] for m in method_order_b])
    ax_b.set_xticklabels(
        ["Random\nk=5", "Taxonomic\ndiversity\nk=5", "CCME\nType A\nproxy\nk=7", "EPA WQC\nproxy\nk=8", "EPA WET\npanel\nk=8"],
        fontsize=7.2,
        linespacing=0.95,
    )
    ax_b.tick_params(axis="x", pad=3.0)
    ax_b.set_xlim(-0.48, len(method_order_b) - 0.24)
    ax_b.set_ylim(0, 1.03)
    ax_b.set_ylabel("Expected capture", fontsize=7.4)
    ax_b.set_title("Comparator performance across tail targets")
    ax_b.grid(axis="y", color="#E6E6E6", lw=0.45, zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    handles_b = [
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.80], edgecolor=TARGET_COLORS[0.80], alpha=0.28, label="lower-20%"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.90], edgecolor=TARGET_COLORS[0.90], alpha=0.28, label="lower-10%"),
        mpl.patches.Patch(facecolor=TARGET_COLORS[0.95], edgecolor=TARGET_COLORS[0.95], alpha=0.28, label="lower-5%"),
        mpl.lines.Line2D([0], [0], color=PALETTE["neutral_dark"], lw=1.15, label="95% sampled interval"),
        mpl.lines.Line2D([0], [0], marker="o", color=PALETTE["neutral_dark"], markerfacecolor="white", markersize=4.5, lw=0, label="sampled mean"),
    ]
    ax_b.legend(
        handles=handles_b,
        loc="center left",
        bbox_to_anchor=(1.02, 0.56),
        ncol=1,
        fontsize=7.3,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.52,
        borderaxespad=0.0,
        labelspacing=0.45,
    )
    aligned_panel_label(ax_b, "b")

    x = np.arange(len(panel_c))
    ax_c.set_axisbelow(True)
    ax_c.bar(x, panel_c["incremental_gain"], color=PALETTE["blue_light"], edgecolor=PALETTE["blue"], lw=0.5, label="Incremental gain", zorder=2)
    ax_c.plot(x, panel_c["leave_one_out_loss"], color=PALETTE["red"], marker="o", lw=1.25, ms=3, label="Leave-one-out loss", zorder=3)
    for i, row in enumerate(panel_c.itertuples(index=False)):
        if i == len(panel_c) - 1:
            x_offset, ha = 0.12, "left"
        elif i == len(panel_c) - 2:
            x_offset, ha = -0.10, "right"
        else:
            x_offset, ha = 0.07, "left"
        ax_c.text(
            i + x_offset,
            row.leave_one_out_loss + 0.0025,
            f"{row.leave_one_out_loss:.3f}",
            ha=ha,
            va="bottom",
            fontsize=6.8,
            color=PALETTE["red"],
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(panel_c["species_label"], rotation=30, ha="right", fontsize=7.2)
    ax_c.set_ylim(0, max(panel_c["incremental_gain"].max(), panel_c["leave_one_out_loss"].max()) * 1.35)
    ax_c.set_ylabel("Change in lower-5% expected capture")
    ax_c.set_title("Non-redundant Top-5 contributions")
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    label_to_handle = dict(zip(labels_c, handles_c))
    ax_c.legend(
        [label_to_handle["Incremental gain"], label_to_handle["Leave-one-out loss"]],
        ["Incremental gain", "Leave-one-out loss"],
        fontsize=7.0,
        loc="upper right",
    )
    clean_axis(ax_c, grid=False)
    ax_c.yaxis.grid(True, color="#EAEAEA", lw=0.45, zorder=0)
    aligned_panel_label(ax_c, "c")

    species_order = panel_d.sort_values("rank")["species_label"].drop_duplicates().tolist()
    domain_order = panel_d.groupby("domain")["domain_weight"].max().sort_values(ascending=False).index.tolist()
    grid = panel_d.set_index(["species_label", "domain"])
    prob_min = float(panel_d["weighted_mean_tail_probability"].min())
    prob_max = float(panel_d["weighted_mean_tail_probability"].max())
    prob_pad = max((prob_max - prob_min) * 0.08, 0.005)
    norm_d = mpl.colors.PowerNorm(gamma=0.65, vmin=max(0.0, prob_min - prob_pad), vmax=prob_max + prob_pad)
    cmap_d = continuous_cmap(
        "bhbt_domain_probability",
        ["#FCFBFD", "#DADAEB", "#9E9AC8", "#6A51A3", "#3F007D"],
    )
    for i, species in enumerate(species_order):
        for j, domain in enumerate(domain_order):
            if (species, domain) not in grid.index:
                continue
            row = grid.loc[(species, domain)]
            prob = float(row["weighted_mean_tail_probability"])
            size = support_marker_size(float(row["direct_support_targets"]))
            ax_d.scatter(
                j,
                i,
                s=size,
                color=cmap_d(norm_d(prob)),
                edgecolor="white",
                linewidth=0.35,
                alpha=0.94,
            )
    ax_d.set_xticks(np.arange(len(domain_order)))
    ax_d.set_xticklabels(domain_order, rotation=45, ha="right", fontsize=7.2)
    ax_d.set_yticks(np.arange(len(species_order)))
    ax_d.set_yticklabels(species_order, fontsize=7.2)
    ax_d.set_title("Species-by-chemical-domain complementarity")
    ax_d.set_xlabel("Priority chemical domain")
    ax_d.set_ylabel("Fixed national Top-5 species")
    ax_d.set_xlim(-0.6, len(domain_order) - 0.4)
    ax_d.set_ylim(len(species_order) - 0.4, -0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_d, fraction=0.046, pad=0.02)
    cbar.set_label("Mean lower-tail probability\n(higher = darker)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.4)
    legend_values = size_legend_values(int(panel_d["direct_support_targets"].max()))
    size_handles = [
        ax_d.scatter(
            [],
            [],
            s=support_marker_size(value),
            facecolor="#C7C7C7",
            edgecolor=PALETTE["neutral_dark"],
            linewidth=0.35,
        )
        for value in legend_values
    ]
    ax_d.legend(
        size_handles,
        [str(value) for value in legend_values],
        title="Measured target cells",
        fontsize=6.4,
        title_fontsize=6.8,
        loc="upper left",
        bbox_to_anchor=(1.27, 1.01),
        borderaxespad=0.0,
        labelspacing=0.75,
        handletextpad=0.8,
        frameon=False,
    )
    aligned_panel_label(ax_d, "d")

    outputs = export_figure(fig, out_dir / "figure_02_top5_performance_complementarity")
    return outputs, {
        "figure_02a_top_k_curves": panel_a,
        "figure_02b_feasible_combination_summary": panel_b,
        "figure_02b_mean_interval_stats": build_panel_b_stats(panel_b),
        "figure_02b_feasible_combination_samples": panel_b_samples,
        "figure_02c_gain_loss": panel_c,
        "figure_02d_species_domain_complementarity": panel_d,
    }


def main() -> None:
    root = project_root_from_script(SCRIPT)
    out_dir = figure_dir_from_script(SCRIPT)
    outputs, tables = draw(root, out_dir)
    source_tables = []
    for name, table in tables.items():
        source_tables.append(write_csv(table, out_dir / f"{name}.csv"))
    source_tables.append(write_combined_source(tables, out_dir / "source_data.csv"))
    save_contract_note(
        out_dir,
        [
            f"# {FIGURE_ID} contract",
            "Core conclusion: The fixed national Top-5 improves national chemical-priority-weighted expected capture of working measured-tail targets over random, taxonomic-diversity, composition-slot and EPA standard-species comparators, with gains carried by complementary species-domain contributions.",
            "Archetype: quantitative grid.",
    "Weighting note: Weighted means national chemical-priority weights from results/weights/national_priority_chemicals.csv; it is not species, sample-size or regulatory-protection weighting.",
            "Comparator note: EPA WQC and CCME Type A frameworks define taxonomic/composition requirements, not a unique species list here; sampled means summarize seeded feasible combinations under the study proxy rules. EPA WET is treated as a fixed standard/method-species comparator with no sampled mean or interval.",
            "Panel map: a fixed x=0.95 national-sequence prefixes evaluated across three lower-tail targets; b comparator performance across lower-tail targets with fixed Top-5 reference lines; c incremental gain plus leave-one-out loss; d species-by-chemical-domain complementarity.",
            "Reviewer risk: Expected capture is not a protection guarantee or an HC5 estimate; Panel B sampled intervals describe feasible-combination distributions, not statistical confidence intervals or agency-endorsed panels; Panel D keeps explicit MIE/form domain labels rather than broad verb-only groups.",
            f"Manuscript caption: {MANUSCRIPT_CAPTION}",
        ],
    )
    write_manifest(
        out_dir,
        figure_id=FIGURE_ID,
        title=TITLE,
        manuscript_caption=MANUSCRIPT_CAPTION,
        core_conclusion="The fixed national Top-5 improves national chemical-priority-weighted expected capture of working measured-tail targets over random, taxonomic-diversity, composition-slot and EPA standard-species comparators, with gains carried by complementary species-domain contributions.",
        archetype="quantitative grid",
        panel_map={
            "a": "Prefixes of the x=0.95 national sequence evaluated across k=1-20 for lower 20%, lower 10% and lower 5% measured-tail targets.",
            "b": "Comparator performance across lower-tail targets for random, taxonomic-diversity, CCME Type A proxy, EPA WQC proxy and fixed EPA WET panel comparators, with sampled means, sampled feasible-combination intervals and fixed Top-5 reference lines.",
            "c": "Incremental greedy gain and leave-one-out loss for each fixed national Top-5 species on a shared expected-capture axis.",
            "d": "Fixed national Top-5 species by priority chemical-domain matrix using explicit MIE/form labels; color is mean tail probability and point size is the count of measured target cells.",
        },
        source_tables=source_tables,
        input_dependencies=[
            "results/panels/national_k1_20_coverage_curves.csv",
            "results/panels/fixed_x95_sequence_cross_target_curves.csv",
            "results/panels/fixed_national_top5_cross_target_audit.csv",
            "results/probability/species_chemical_tail_probability_x80.npz",
            "results/probability/species_chemical_tail_probability_x90.npz",
            "results/probability/species_chemical_tail_probability_x95.npz",
            "results/probability/species_probability_summary_x80.csv",
            "results/panels/national_panel_sequences.csv",
            "results/complementarity/top5_top10_species_incremental_and_taxonomic_contributions.csv",
            "results/probability/species_protective_target_tail_probability_x95.npz",
            "results/weights/national_priority_chemicals.csv",
        ],
        outputs=outputs,
        reviewer_risk="Expected capture is probabilistic measured-tail coverage, not direct ecosystem protection or HC5 replacement.",
        notes=[
            f"Panel B sampled rows are figure-derived from {PANEL_B_N_SAMPLES} seeded feasible-panel draws per sampled method x target, using the same candidate species universe and target-specific probability matrices as the fixed-sequence cross-target curves.",
            "Panels A and B lock membership to the first five species of the x=0.95 national sequence before evaluation at x=0.95, x=0.90 and x=0.80.",
            "Panel B sampled means are arithmetic means across the same seeded feasible-panel draws that define the sampled intervals.",
            "Panel B EPA WQC and CCME Type A proxy distributions are study operationalizations of slot/composition rules; they are not agency-endorsed species lists.",
            "Panel B US EPA Whole Effluent Toxicity (WET) is a fixed standard/method-species comparator, so no sampled mean marker or sampled interval is shown.",
            "Panel C uses one y-axis to avoid dual-axis visual inflation.",
            "Panel D encodes mean tail probability by color and measured species-target cell count by point area; a measured cell means direct measured toxicity evidence for that species in a chemical x protective-effect target.",
            "Panel D domains keep specific MIE/form labels; distinct inhibition mechanisms are not collapsed into a generic inhibition column.",
        ],
    )
    print({"figure": FIGURE_ID, "outputs": outputs, "source_tables": source_tables})


if __name__ == "__main__":
    main()
