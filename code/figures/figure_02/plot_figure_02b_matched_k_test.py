#!/usr/bin/env python3
"""Exploratory matched-k replacement candidate for Figure 2 panel b.

This script is intentionally separate from plot_figure_02.py. It does not
modify the manuscript figure; it renders a review-only panel b candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import PALETTE, apply_style, export_figure, figure_dir_from_script, project_root_from_script, write_csv  # noqa: E402


TARGETS = [0.80, 0.90, 0.95]
TARGET_LABELS = {0.80: "HC20-tail", 0.90: "HC10-tail", 0.95: "HC5-tail"}


def continuous_cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=256)


def text_color_for_value(value: float, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.46 else PALETTE["neutral_dark"]


def lookup_coverage(curves: pd.DataFrame, method: str, k: int, target: float) -> float:
    hit = curves[
        curves["universe"].eq("priority")
        & curves["method"].eq(method)
        & curves["k"].eq(k)
        & curves["protection_target_x"].eq(target)
    ]
    if hit.empty:
        raise ValueError(f"Missing coverage for {method}, k={k}, target={target}")
    return float(hit["expected_weighted_joint_coverage"].iloc[0])


def build_matched_k_table(root: Path) -> pd.DataFrame:
    curves = pd.read_csv(root / "results/panels/national_k1_20_coverage_curves.csv")
    random_top5 = pd.read_csv(root / "manuscript_plot/figure_02/figure_02b_top5_baselines.csv")
    random_top5 = random_top5[random_top5["method"].eq("random_all_species_probability")].copy()

    row_specs = [
        {
            "group": "k=5 size-matched baselines",
            "row_label": "Optimized Top-5",
            "method": "data_driven",
            "method_k": 5,
            "reference_method": "data_driven",
            "reference_k": 5,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "k=5 size-matched baselines",
            "row_label": "Taxonomic class-diversity k=5",
            "method": "taxonomy_diversity_baseline",
            "method_k": 5,
            "reference_method": "data_driven",
            "reference_k": 5,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "k=5 size-matched baselines",
            "row_label": "Random 5-species k=5",
            "method": "random_all_species_probability",
            "method_k": 5,
            "reference_method": "data_driven",
            "reference_k": 5,
            "source": "figure_derived_random_panel_interval",
        },
        {
            "group": "CCME Type A minimum",
            "row_label": "Optimized Top-7",
            "method": "data_driven",
            "method_k": 7,
            "reference_method": "data_driven",
            "reference_k": 7,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "CCME Type A minimum",
            "row_label": "CCME Type A proxy k=7",
            "method": "Canada_Type_A_composition",
            "method_k": 7,
            "reference_method": "data_driven",
            "reference_k": 7,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "EPA 8-species rules",
            "row_label": "Optimized Top-8",
            "method": "data_driven",
            "method_k": 8,
            "reference_method": "data_driven",
            "reference_k": 8,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "EPA 8-species rules",
            "row_label": "EPA WQC proxy k=8",
            "method": "EPA_WQC_taxonomic_requirements",
            "method_k": 8,
            "reference_method": "data_driven",
            "reference_k": 8,
            "source": "national_k1_20_coverage_curves",
        },
        {
            "group": "EPA 8-species rules",
            "row_label": "EPA WET fixed k=8",
            "method": "EPA_WET_fixed_method_species",
            "method_k": 8,
            "reference_method": "data_driven",
            "reference_k": 8,
            "source": "national_k1_20_coverage_curves",
        },
    ]

    rows: list[dict[str, object]] = []
    for order, spec in enumerate(row_specs):
        for target in TARGETS:
            if spec["method"] == "random_all_species_probability":
                hit = random_top5[random_top5["protection_target_x"].eq(target)]
                if hit.empty:
                    raise ValueError(f"Missing random k=5 coverage for target={target}")
                coverage = float(hit["coverage"].iloc[0])
            else:
                coverage = lookup_coverage(curves, str(spec["method"]), int(spec["method_k"]), target)
            reference = lookup_coverage(curves, str(spec["reference_method"]), int(spec["reference_k"]), target)
            rows.append(
                {
                    "row_order": order,
                    "group": spec["group"],
                    "row_label": spec["row_label"],
                    "method": spec["method"],
                    "method_k": spec["method_k"],
                    "reference_method": spec["reference_method"],
                    "reference_k": spec["reference_k"],
                    "protection_target_x": target,
                    "coverage": coverage,
                    "matched_optimized_coverage": reference,
                    "delta_vs_matched_optimized": coverage - reference,
                    "source": spec["source"],
                }
            )
    return pd.DataFrame(rows)


def render_panel(table: pd.DataFrame, out_dir: Path) -> list[str]:
    apply_style(font_size=7.0)
    row_order = table[["row_order", "row_label", "group"]].drop_duplicates().sort_values("row_order")
    rows = row_order["row_label"].tolist()
    values = table.pivot(index="row_label", columns="protection_target_x", values="coverage").reindex(rows)[TARGETS].to_numpy(float)
    deltas = (
        table.pivot(index="row_label", columns="protection_target_x", values="delta_vs_matched_optimized")
        .reindex(rows)[TARGETS]
        .to_numpy(float)
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    cmap = continuous_cmap("bhbt_matched_k_blues", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], rotation=28, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.0)
    ax.set_title("Matched-k baseline comparison", pad=8)

    for i, row_label in enumerate(rows):
        is_reference = row_label.startswith("Optimized")
        for j, target in enumerate(TARGETS):
            color = text_color_for_value(values[i, j], cmap, norm)
            if is_reference:
                label = f"{values[i, j]:.3f}"
            else:
                label = f"{values[i, j]:.3f}\n$\\Delta$ {deltas[i, j]:+.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.2, color=color, linespacing=1.15)

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
        -0.20,
        r"$\Delta$ = row coverage minus matched optimized Top-k",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=PALETTE["neutral_dark"],
    )
    fig.tight_layout(pad=0.6)
    return export_figure(fig, out_dir / "figure_02b_matched_k_test")


def main() -> None:
    root = project_root_from_script(SCRIPT)
    out_dir = figure_dir_from_script(SCRIPT)
    table = build_matched_k_table(root)
    source_name = write_csv(table, out_dir / "figure_02b_matched_k_test.csv")
    outputs = render_panel(table, out_dir)
    print({"source_table": source_name, "outputs": outputs})


if __name__ == "__main__":
    main()
