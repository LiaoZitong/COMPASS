#!/usr/bin/env python3
"""Render R1 SI Figure S4 from the detailed AP03 sensitivity audits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[3]
ANALYSIS_ROOT = PACKAGE_ROOT / "results" / "revision_r1"
BHBT_ROOT = PACKAGE_ROOT
AP03A_DIR = ANALYSIS_ROOT / "04_ap03_module_a"
AP03B_DIR = ANALYSIS_ROOT / "05_ap03_module_b"
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "si_figure_s4"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, apply_style, clean_axis, export_figure, short_species  # noqa: E402


FIGURE_BASENAME = "figure_s4_r1_available_member_followup_sensitivity"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_panel_a() -> pd.DataFrame:
    frame = pd.read_csv(AP03A_DIR / "ap03a_member_count_concordance.csv")
    frame = frame[frame["reference"].eq("reference_hc5_typical_log10")].copy()
    order = ["all_eligible_rows", "exactly_1_measured", "exactly_2_measured", "exactly_3_measured"]
    labels = {
        "all_eligible_rows": "All eligible",
        "exactly_1_measured": "Exactly 1 measured",
        "exactly_2_measured": "Exactly 2 measured",
        "exactly_3_measured": "Exactly 3 measured",
    }
    frame = frame.set_index("stratum").loc[order].reset_index()
    frame["stratum_label"] = frame["stratum"].map(labels) + "\n(n=" + frame["n_rows"].astype(str) + ")"
    return frame


def build_panel_b() -> pd.DataFrame:
    frame = pd.read_csv(AP03A_DIR / "ap03a_member_ablation_matched.csv")
    frame = frame[frame["reference"].eq("reference_hc5_typical_log10")].copy()
    frame["species_label"] = frame["removed_member"].map(short_species)
    return frame.sort_values("delta_spearman_ablated_minus_full")


def build_panel_c() -> pd.DataFrame:
    frame = pd.read_csv(AP03B_DIR / "ap03b_lower20_hazard_recall.csv")
    frame = frame[
        frame["scope"].eq("full_scoreable")
        & frame["variant"].isin(["r1_primary_max", "r1_balanced", "r1_mortality_excluded"])
        & frame["budget"].eq("top_20pct")
    ].copy()
    label_map = {
        "r1_primary_max": "Primary max",
        "r1_balanced": "Balanced",
        "r1_mortality_excluded": "Mortality-excluded",
    }
    frame["variant_label"] = frame["variant"].map(label_map)
    frame["chemical_scoreability_percent"] = frame["n_scoreable_chemicals"] / 639.0 * 100.0
    frame["labeled_hazard_scoreability_percent"] = frame["scoreable_hazard_coverage"] * 100.0
    frame["n_unscoreable_chemicals"] = 639 - frame["n_scoreable_chemicals"]
    return frame


def build_panel_d() -> pd.DataFrame:
    frame = pd.read_csv(AP03B_DIR / "ap03b_pairwise_ranking_robustness.csv")
    frame = frame[frame["scope"].eq("matched_support")].copy()
    order = [
        "R0 primary vs R1 strict-LOO primary",
        "R1 primary max vs balanced",
        "R1 primary max vs mortality-excluded balanced",
        "R1 balanced vs mortality-excluded balanced",
    ]
    labels = {
        order[0]: "R0 primary vs\nR1 primary",
        order[1]: "R1 primary vs\nbalanced",
        order[2]: "R1 primary vs\nmortality-excluded",
        order[3]: "Balanced vs\nmortality-excluded",
    }
    frame = frame.set_index("comparison").loc[order].reset_index()
    frame["comparison_label"] = frame["comparison"].map(labels)
    return frame


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-25, 15), textcoords="offset points", ha="left", va="top", fontweight="bold", fontsize=11, clip_on=False)


def draw(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, panel_d: pd.DataFrame) -> list[str]:
    apply_style(font_size=7.1)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), gridspec_kw={"wspace": 0.32, "hspace": 0.56})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    y = np.arange(len(panel_a))
    for offset, metric, low, high, color, marker, label in [
        (-0.12, "spearman_rho", "spearman_rho_ci95_lower", "spearman_rho_ci95_upper", PALETTE["blue2"], "o", r"Spearman $\rho$"),
        (0.12, "kendall_tau", "kendall_tau_ci95_lower", "kendall_tau_ci95_upper", PALETTE["gold"], "D", r"Kendall $\tau$"),
    ]:
        values = panel_a[metric].to_numpy(float)
        lower = panel_a[low].to_numpy(float)
        upper = panel_a[high].to_numpy(float)
        ax_a.errorbar(values, y + offset, xerr=np.vstack([values - lower, upper - values]), fmt=marker, color=color, ms=4.5, lw=0.9, capsize=1.8, label=label)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(panel_a["stratum_label"], fontsize=6.3)
    ax_a.set_xlim(0.42, 1.01)
    ax_a.set_ylim(len(panel_a) - 0.55, -0.55)
    ax_a.set_xlabel("Rank association with full-data HC5")
    ax_a.set_title("Concordance by measured-member count", loc="left", fontsize=8.2, pad=25)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=6.0, ncol=2, borderaxespad=0.0, columnspacing=0.8, handletextpad=0.4)
    clean_axis(ax_a, grid=False)
    ax_a.xaxis.grid(True, color="#E6E6E6", lw=0.45, zorder=0)
    aligned_panel_label(ax_a, "a")

    y = np.arange(len(panel_b))
    values = panel_b["delta_spearman_ablated_minus_full"].to_numpy(float)
    lower = panel_b["delta_spearman_ci95_lower"].to_numpy(float)
    upper = panel_b["delta_spearman_ci95_upper"].to_numpy(float)
    colors = [PALETTE["red"] if high < 0 else PALETTE["blue2"] for high in upper]
    for idx, (value, low, high, color) in enumerate(zip(values, lower, upper, colors)):
        ax_b.errorbar(value, idx, xerr=np.asarray([[value - low], [high - value]]), fmt="o", color=color, ms=4.7, lw=0.95, capsize=2)
    ax_b.axvline(0, color=PALETTE["neutral_dark"], lw=0.7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(
        [f"{label}\nmatched n={n}; dropped={d}" for label, n, d in zip(panel_b["species_label"], panel_b["n_matched_rows"], panel_b["n_rows_dropped_no_remaining_member"])],
        fontsize=5.8,
        fontstyle="italic",
    )
    ax_b.set_xlim(min(-0.105, float(lower.min()) - 0.01), max(0.085, float(upper.max()) + 0.01))
    ax_b.set_ylim(len(panel_b) - 0.55, -0.55)
    ax_b.set_xlabel(r"Change in Spearman $\rho$ after member removal")
    ax_b.set_title("Matched member-ablation sensitivity", loc="left", fontsize=8.2)
    clean_axis(ax_b, grid=False)
    ax_b.xaxis.grid(True, color="#E6E6E6", lw=0.45, zorder=0)
    aligned_panel_label(ax_b, "b")

    x = np.arange(len(panel_c))
    width = 0.34
    ax_c.bar(x - width / 2, panel_c["chemical_scoreability_percent"], width, color=PALETTE["blue2"], label="Chemicals scoreable")
    ax_c.bar(x + width / 2, panel_c["labeled_hazard_scoreability_percent"], width, color=PALETTE["teal"], label="Lower-20% HC5 hazards scoreable")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(panel_c["variant_label"], rotation=22, ha="right", fontsize=6.2)
    ax_c.set_ylim(0, 125)
    ax_c.set_ylabel("Scoreable fraction (%)")
    ax_c.set_title("Scoreability by aggregation rule", loc="left", fontsize=8.2)
    for idx, row in enumerate(panel_c.itertuples(index=False)):
        ax_c.text(idx - width / 2, row.chemical_scoreability_percent + 2, f"{int(row.n_scoreable_chemicals)}/639", ha="center", fontsize=5.4)
        ax_c.text(idx + width / 2, row.labeled_hazard_scoreability_percent + 2, f"{int(row.n_lower20_hazard_scoreable)}/23", ha="center", fontsize=5.4)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=5.8, ncol=2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    clean_axis(ax_c, grid=False)
    ax_c.yaxis.grid(True, color="#E6E6E6", lw=0.45, zorder=0)
    aligned_panel_label(ax_c, "c")

    columns = ["spearman_rho", "top_5pct_jaccard", "top_10pct_jaccard", "top_20pct_jaccard", "top100_overlap_fraction"]
    labels = [r"Spearman $\rho$", "Top-5%\nJaccard", "Top-10%\nJaccard", "Top-20%\nJaccard", "Top-100\noverlap"]
    matrix = panel_d[columns].to_numpy(float)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("agreement", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    image = ax_d.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_d.set_xticks(np.arange(len(labels)))
    ax_d.set_xticklabels(labels, fontsize=5.7)
    ax_d.set_yticks(np.arange(len(panel_d)))
    ax_d.set_yticklabels(panel_d["comparison_label"], fontsize=5.7)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax_d.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value >= 0.55 else PALETTE["neutral_dark"])
    ax_d.set_title("Agreement on matched support (n = 136)", loc="left", fontsize=8.2)
    ax_d.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(panel_d), 1), minor=True)
    ax_d.grid(which="minor", color="white", lw=0.5)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.045, pad=0.025)
    colorbar.set_label("Agreement", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.4)
    aligned_panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.10, right=0.975, top=0.89, bottom=0.10)
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(outputs: list[str], panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, panel_d: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "figure_s4a_member_count_concordance.csv": panel_a,
        "figure_s4b_member_ablation.csv": panel_b,
        "figure_s4c_scoreability.csv": panel_c,
        "figure_s4d_matched_support_agreement.csv": panel_d,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False)
    caption = (
        "Figure S4. Available-member and follow-up-ranking sensitivity diagnostics. "
        "a, Spearman and Kendall rank association between the available-member Top5-apical threshold and the typical full-data HC5, stratified by measured Top-5 member count; horizontal lines are chemical-bootstrap 95% intervals. The four rows with exactly four measured members are retained in the source table but omitted from the plot because their bootstrap intervals span the full correlation range. "
        "b, Change in Spearman correlation after removing each Top-5 member, evaluated on rows that remain scoreable after removal; horizontal lines are bootstrap 95% intervals. Daphnia magna removal leaves 61 matched rows and makes 54 rows unscoreable. "
        "c, Chemical and labeled lower-20% HC5-hazard scoreability for the three R1 follow-up-ranking variants. "
        "d, Matched-support agreement across 136 chemicals, showing rank correlation and overlap at four follow-up-list sizes. All panels retain observed support and NOT SCOREABLE outcomes without imputing missing non-mortality evidence."
    )
    (OUT_DIR / "figure_contract.md").write_text(
        "# R1 Figure S4 contract\n\n"
        "Core conclusion: the available-member result is robust across the well-supported count strata, whereas member removal and effect-family aggregation expose specific dependence and scoreability constraints that must remain visible in interpretation.\n\n"
        "Legends are separated from plotted data and titles by dedicated information bands; inter-panel spacing is tightened and typography is Times New Roman.\n\n"
        f"Manuscript caption: {caption}\n",
        encoding="utf-8",
    )
    files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure S4",
        "version": "R1_AP03A_AP03B",
        "manuscript_caption": caption,
        "input_dependencies": [
            str(AP03A_DIR / "ap03a_member_count_concordance.csv"),
            str(AP03A_DIR / "ap03a_member_ablation_matched.csv"),
            str(AP03B_DIR / "ap03b_lower20_hazard_recall.csv"),
            str(AP03B_DIR / "ap03b_pairwise_ranking_robustness.csv"),
        ],
        "outputs": files,
        "qa": {"n_matched_support": 136, "n_labeled_hazards": 23, "editable_svg_text": True, "png_dpi": 300, "tiff_dpi": 600},
    }
    manifest["sha256"] = {name: sha256_file(OUT_DIR / name) for name in files if (OUT_DIR / name).exists()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a = build_panel_a()
    panel_b = build_panel_b()
    panel_c = build_panel_c()
    panel_d = build_panel_d()
    if set(panel_d["n_chemicals"].astype(int)) != {136} or int(panel_c["n_lower20_hazard_all"].max()) != 23:
        raise AssertionError("AP03 SI diagnostic denominator mismatch")
    outputs = draw(panel_a, panel_b, panel_c, panel_d)
    write_outputs(outputs, panel_a, panel_b, panel_c, panel_d)
    print(json.dumps({"status": "PASS", "output_dir": str(OUT_DIR), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
