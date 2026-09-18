#!/usr/bin/env python3
"""Render R1 Figure 3 from approved AP03A/AP03B outputs."""

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
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "main_figure_03"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, apply_style, clean_axis, export_figure, short_species  # noqa: E402


FIGURE_BASENAME = "figure_03_r1_measured_validation_followup_robustness"
TOP5 = [
    "Gastrophryne carolinensis",
    "Neocloeon triangulifer",
    "Hyalella azteca",
    "Daphnia ambigua",
    "Daphnia magna",
]
VARIANT_ORDER = ["r1_primary_max", "r1_balanced", "r1_mortality_excluded"]
VARIANT_LABELS = {
    "r1_primary_max": "Primary max\n(n = 639)",
    "r1_balanced": "Balanced\n(n = 639)",
    "r1_mortality_excluded": "Mortality-excluded\n(n = 145)",
}
VARIANT_COLORS = {
    "r1_primary_max": PALETTE["blue"],
    "r1_balanced": PALETTE["teal"],
    "r1_mortality_excluded": PALETTE["red"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sensitivity_percentile(values: pd.Series) -> pd.Series:
    """Percentile rank where a lower log10 concentration is more sensitive."""

    return (-values.astype(float)).rank(method="average", pct=True) * 100.0


def build_panel_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(AP03A_DIR / "ap03a_top5_availability_and_contribution_ledger.csv")
    ledger = ledger.dropna(subset=["available_member_minimum_log10", "reference_hc5_typical_log10"]).copy()
    ledger["top5_apical_sensitivity_percentile"] = sensitivity_percentile(ledger["available_member_minimum_log10"])
    ledger["reference_hc5_sensitivity_percentile"] = sensitivity_percentile(ledger["reference_hc5_typical_log10"])
    ledger["measured_member_label"] = ledger["n_top5_species_measured"].astype(int).map(lambda value: f"{value} measured member{'s' if value != 1 else ''}")
    ledger["top5_apical_definition"] = "minimum measured protective apical threshold among available members of the fixed R1 national Top-5"

    ledger["decile"] = pd.qcut(
        ledger["top5_apical_sensitivity_percentile"], q=10, labels=False, duplicates="drop"
    ).astype(int) + 1
    binned = (
        ledger.groupby("decile", as_index=False)
        .agg(
            n_chemicals=("dtxsid", "size"),
            top5_apical_sensitivity_percentile=("top5_apical_sensitivity_percentile", "median"),
            reference_hc5_sensitivity_percentile=("reference_hc5_sensitivity_percentile", "median"),
        )
        .sort_values("decile")
    )
    return ledger, binned


def build_panel_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    contribution = pd.read_csv(AP03A_DIR / "ap03a_member_contribution_summary.csv")
    contribution["species_label"] = contribution["latin_name"].map(short_species)
    contribution["measured_percent"] = contribution["fraction_rows_measured"] * 100.0
    contribution["minimum_contributor_percent"] = contribution["fractional_contribution_fraction"] * 100.0
    contribution["minimum_contributor_ci95_lower_percent"] = contribution["fractional_contribution_fraction_ci95_lower"] * 100.0
    contribution["minimum_contributor_ci95_upper_percent"] = contribution["fractional_contribution_fraction_ci95_upper"] * 100.0
    contribution = contribution.set_index("latin_name").loc[TOP5].reset_index()

    ledger = pd.read_csv(AP03A_DIR / "ap03a_top5_availability_and_contribution_ledger.csv")
    counts = (
        ledger["n_top5_species_measured"].astype(int).value_counts().reindex(range(1, 6), fill_value=0).rename_axis("n_measured_members").reset_index(name="n_chemicals")
    )
    counts["percent_chemicals"] = counts["n_chemicals"] / len(ledger) * 100.0
    return contribution, counts


def build_panel_c() -> pd.DataFrame:
    recall = pd.read_csv(AP03B_DIR / "ap03b_lower20_hazard_recall.csv")
    recall = recall[
        recall["scope"].eq("full_scoreable")
        & recall["variant"].isin(VARIANT_ORDER)
        & recall["budget"].isin(["top_5pct", "top_10pct", "top_20pct"])
    ].copy()
    recall["budget_percent"] = recall["budget"].str.extract(r"(\d+)").astype(int)
    recall["recall_percent"] = recall["recall_of_all_23"] * 100.0
    recall["ci95_lower_percent"] = recall["recall_of_all_23_ci95_lower"] * 100.0
    recall["ci95_upper_percent"] = recall["recall_of_all_23_ci95_upper"] * 100.0
    recall["hazard_denominator"] = 23
    return recall.sort_values(["variant", "budget_percent"])


def build_panel_d() -> pd.DataFrame:
    robust = pd.read_csv(AP03B_DIR / "ap03b_pairwise_ranking_robustness.csv")
    keep = [
        "R1 primary max vs balanced",
        "R1 primary max vs mortality-excluded balanced",
    ]
    robust = robust[robust["scope"].eq("matched_support") & robust["comparison"].isin(keep)].copy()
    label_map = {
        "R1 primary max vs balanced": "Primary vs\nbalanced",
        "R1 primary max vs mortality-excluded balanced": "Primary vs\nmortality-excluded",
    }
    robust["comparison_label"] = robust["comparison"].map(label_map)
    return robust.set_index("comparison").loc[keep].reset_index()


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


def draw(
    panel_a: pd.DataFrame,
    panel_a_bins: pd.DataFrame,
    panel_b: pd.DataFrame,
    member_counts: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> list[str]:
    apply_style(font_size=7.3)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.25, 5.10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], wspace=0.27, hspace=0.44)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    member_colors = {1: "#D9E2F3", 2: "#9DC3E6", 3: "#5B9BD5", 4: PALETTE["blue"]}
    for n_members in sorted(panel_a["n_top5_species_measured"].unique()):
        group = panel_a[panel_a["n_top5_species_measured"].eq(n_members)]
        ax_a.scatter(
            group["top5_apical_sensitivity_percentile"],
            group["reference_hc5_sensitivity_percentile"],
            s=20,
            facecolor=member_colors[int(n_members)],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            label=f"{int(n_members)} member{'s' if int(n_members) != 1 else ''} (n = {len(group)})",
        )
    ax_a.plot([0, 100], [0, 100], color=PALETTE["neutral"], lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax_a.plot(
        panel_a_bins["top5_apical_sensitivity_percentile"],
        panel_a_bins["reference_hc5_sensitivity_percentile"],
        color=PALETTE["blue"],
        lw=1.4,
        marker="o",
        ms=3.0,
        label="Decile medians",
    )
    ax_a.text(
        3,
        95,
        r"Spearman $\rho$ = 0.847" + "\n95% CI 0.756-0.904",
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.4),
    )
    ax_a.set(xlim=(0, 102), ylim=(0, 102), xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100])
    ax_a.set_xlabel("Available-member Top5-apical\nsensitivity percentile")
    ax_a.set_ylabel("Full-data HC5 sensitivity percentile")
    ax_a.set_title("Measured available-member concordance", loc="left", fontsize=8.2)
    ax_a.legend(loc="lower right", fontsize=5.7, handletextpad=0.4, labelspacing=0.35)
    clean_axis(ax_a, grid=True)
    aligned_panel_label(ax_a, "a")

    x = np.arange(len(panel_b))
    width = 0.35
    ax_b.bar(x - width / 2, panel_b["measured_percent"], width, color=PALETTE["blue2"], label="Measured in eligible rows")
    lower = panel_b["minimum_contributor_percent"] - panel_b["minimum_contributor_ci95_lower_percent"]
    upper = panel_b["minimum_contributor_ci95_upper_percent"] - panel_b["minimum_contributor_percent"]
    ax_b.bar(
        x + width / 2,
        panel_b["minimum_contributor_percent"],
        width,
        color=PALETTE["gold"],
        yerr=np.vstack([lower, upper]),
        capsize=1.7,
        error_kw={"lw": 0.7},
        label="Minimum-contributor credit",
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(panel_b["species_label"], rotation=34, ha="right", fontsize=6.2, fontstyle="italic")
    # Reserve a clean information band above 100% so the member-count note and
    # the legend never obscure either one another or the bars.
    ax_b.set_ylim(0, 132)
    ax_b.set_ylabel("Eligible chemical-medium rows (%)")
    ax_b.set_title("Availability and minimum contribution", loc="left", fontsize=8.2)
    count_text = ", ".join(f"{int(row.n_measured_members)}={int(row.n_chemicals)}" for row in member_counts.itertuples(index=False))
    ax_b.text(
        0.01,
        0.985,
        "Measured-member count (n = 115):\n" + count_text,
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
    )
    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.86),
        fontsize=5.6,
        ncol=2,
        columnspacing=0.7,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.30,
        borderaxespad=0.0,
    )
    clean_axis(ax_b, grid=False)
    ax_b.yaxis.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    aligned_panel_label(ax_b, "b")

    marker_map = {"r1_primary_max": "o", "r1_balanced": "s", "r1_mortality_excluded": "D"}
    for variant in VARIANT_ORDER:
        group = panel_c[panel_c["variant"].eq(variant)].sort_values("budget_percent")
        values = group["recall_percent"].to_numpy(float)
        errors = np.vstack(
            [
                values - group["ci95_lower_percent"].to_numpy(float),
                group["ci95_upper_percent"].to_numpy(float) - values,
            ]
        )
        ax_c.errorbar(
            group["budget_percent"],
            values,
            yerr=errors,
            color=VARIANT_COLORS[variant],
            marker=marker_map[variant],
            ms=4.5,
            lw=1.25,
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=VARIANT_LABELS[variant],
        )
    ax_c.plot([5, 20], [5, 20], color=PALETTE["neutral"], ls=(0, (3, 2)), lw=0.9, label="Random expectation\n(full scoreable universe)")
    ax_c.set(xlim=(3.8, 21.2), ylim=(0, 86), xticks=[5, 10, 15, 20])
    ax_c.set_xlabel("Chemicals selected for follow-up (%)")
    ax_c.set_ylabel("Lower-20% HC5 chemicals recovered (%)")
    ax_c.set_title("Evidence-conditioned follow-up recall", loc="left", fontsize=8.2)
    ax_c.legend(loc="upper left", fontsize=5.8, labelspacing=0.45, handlelength=1.7)
    clean_axis(ax_c, grid=True)
    aligned_panel_label(ax_c, "c")

    y = np.arange(len(panel_d))
    rho = panel_d["spearman_rho"].to_numpy(float)
    rho_low = panel_d["spearman_rho_ci95_lower"].to_numpy(float)
    rho_high = panel_d["spearman_rho_ci95_upper"].to_numpy(float)
    jac = panel_d["top_20pct_jaccard"].to_numpy(float)
    jac_low = panel_d["top_20pct_jaccard_ci95_lower"].to_numpy(float)
    jac_high = panel_d["top_20pct_jaccard_ci95_upper"].to_numpy(float)
    ax_d.errorbar(
        rho,
        y - 0.11,
        xerr=np.vstack([rho - rho_low, rho_high - rho]),
        fmt="o",
        color=PALETTE["blue2"],
        ms=5,
        lw=1.0,
        capsize=2,
        label=r"Spearman $\rho$ (95% CI)",
    )
    ax_d.errorbar(
        jac,
        y + 0.11,
        xerr=np.vstack([jac - jac_low, jac_high - jac]),
        fmt="D",
        color=PALETTE["gold"],
        ms=4.5,
        lw=1.0,
        capsize=2,
        label="Top-20% Jaccard (95% CI)",
    )
    ax_d.axvline(0, color=PALETTE["neutral_dark"], lw=0.65)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(panel_d["comparison_label"], fontsize=6.7)
    ax_d.set_xlim(-0.02, 0.68)
    ax_d.set_ylim(len(panel_d) - 0.55, -0.55)
    ax_d.set_xlabel("Matched-support agreement")
    ax_d.set_title("Ranking robustness on common support", loc="left", fontsize=8.2)
    ax_d.text(0.98, 0.04, "n = 136 chemicals", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=6.2, color=PALETTE["neutral_dark"])
    ax_d.legend(loc="upper right", fontsize=6.0, handlelength=1.5)
    clean_axis(ax_d, grid=False)
    ax_d.xaxis.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    aligned_panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.09)
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(
    outputs: list[str],
    panel_a: pd.DataFrame,
    panel_a_bins: pd.DataFrame,
    panel_b: pd.DataFrame,
    member_counts: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "figure_03a_available_member_concordance.csv": panel_a,
        "figure_03a_decile_medians.csv": panel_a_bins,
        "figure_03b_member_availability_contribution.csv": panel_b,
        "figure_03b_measured_member_count_distribution.csv": member_counts,
        "figure_03c_lower20_hazard_recall.csv": panel_c,
        "figure_03d_matched_support_robustness.csv": panel_d,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False)

    caption = (
        "Figure 3. Measured available-member thresholds retain substantial rank information while follow-up rankings depend on the effect-family aggregation rule. "
        "a, Sensitivity-rank concordance between the minimum measured protective apical threshold among available members of the fixed national Top-5 (Top5-apical) and the full-data EPA-framework HC5 across 115 eligible chemical-medium rows. Points are colored by the number of measured Top-5 members, the dashed line is 1:1 agreement, and the blue line joins decile medians. The interval for Spearman's rho was obtained by chemical bootstrap. "
        "b, Frequency with which each Top-5 species was measured and its fractional credit as the minimum-concentration contributor; error bars are chemical-bootstrap 95% intervals. The upper annotation gives the distribution of measured-member counts. "
        "c, Recovery of the 23 lower-20% HC5 chemicals at 5%, 10%, and 20% follow-up budgets for the strict-LOO primary maximum, balanced effect-family, and mortality-excluded rankings. Error bars are bootstrap 95% intervals. The mortality-excluded analysis scores 145 chemicals and covers 19 of the 23 labeled hazards; unscoreable chemicals remain outside that conditional ranking. "
        "d, Matched-support agreement across 136 chemicals with at least two protective effect families and at least one non-mortality family. Points show Spearman rank correlation and top-20% Jaccard overlap with bootstrap 95% intervals. These metrics characterize evidence-conditioned prioritization for follow-up testing."
    )
    contract = "\n".join(
        [
            "# R1 Figure 3 contract",
            "",
            "Core conclusion: available-member Top5-apical retains substantial HC5 rank information, while effect-family choices materially change follow-up ordering and therefore require explicit scoreability and matched-support reporting.",
            "",
            "- Panel a uses measured thresholds only; no missing Top-5 member is imputed.",
            "- Panel b separates measurement availability from minimum-contributor credit.",
            "- Panel c reports recall of all 23 labeled lower-20% HC5 chemicals and retains the mortality-excluded scoreability denominator.",
            "- Panel d compares aggregation rules only on the common-support set.",
            "- Detailed member-ablation and bootstrap diagnostics are routed to SI Figure S4.",
            "- Layout: the R1 four-panel evidence structure is retained, horizontal and vertical gaps are reduced, the panel-b legend sits in a dedicated band below the member-count note, and typography is Times New Roman.",
            "",
            f"Manuscript caption: {caption}",
            "",
        ]
    )
    (OUT_DIR / "figure_contract.md").write_text(contract, encoding="utf-8")

    files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure 3",
        "version": "R1_AP03A_AP03B",
        "title": "Measured validation and evidence-conditioned follow-up robustness",
        "core_conclusion": "Available-member thresholds retain HC5 rank information, whereas effect-family aggregation materially affects follow-up ordering.",
        "manuscript_caption": caption,
        "panel_map": {
            "a": "Available-member Top5-apical versus full-data HC5 sensitivity rank",
            "b": "Species availability and fractional minimum-contributor credit",
            "c": "Lower-20% HC5 recall under three R1 follow-up-ranking variants",
            "d": "Matched-support rank and top-20% overlap robustness",
        },
        "input_dependencies": [
            str(AP03A_DIR / "ap03a_top5_availability_and_contribution_ledger.csv"),
            str(AP03A_DIR / "ap03a_member_count_concordance.csv"),
            str(AP03A_DIR / "ap03a_member_contribution_summary.csv"),
            str(AP03B_DIR / "ap03b_lower20_hazard_recall.csv"),
            str(AP03B_DIR / "ap03b_pairwise_ranking_robustness.csv"),
        ],
        "outputs": files,
        "qa": {
            "n_available_member_rows": int(len(panel_a)),
            "spearman_rho_assertion": 0.8472895131381678,
            "n_lower20_hazards": 23,
            "n_matched_support": 136,
            "editable_svg_text": True,
            "pdf_truetype_text": True,
            "png_dpi": 300,
            "tiff_dpi": 600,
        },
    }
    manifest["sha256"] = {name: sha256_file(OUT_DIR / name) for name in files if (OUT_DIR / name).exists()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a, panel_a_bins = build_panel_a()
    panel_b, member_counts = build_panel_b()
    panel_c = build_panel_c()
    panel_d = build_panel_d()

    all_metric = pd.read_csv(AP03A_DIR / "ap03a_member_count_concordance.csv")
    rho = float(
        all_metric[
            all_metric["stratum"].eq("all_eligible_rows")
            & all_metric["reference"].eq("reference_hc5_typical_log10")
        ]["spearman_rho"].iloc[0]
    )
    if len(panel_a) != 115 or not np.isclose(rho, 0.8472895131381678, atol=1e-12):
        raise AssertionError(f"AP03A gate mismatch: n={len(panel_a)}, rho={rho}")
    if int(panel_c["n_lower20_hazard_all"].max()) != 23:
        raise AssertionError("AP03B lower-20% hazard denominator is not 23")
    if set(panel_d["n_chemicals"].astype(int)) != {136}:
        raise AssertionError("AP03B matched-support denominator is not 136")

    outputs = draw(panel_a, panel_a_bins, panel_b, member_counts, panel_c, panel_d)
    write_outputs(outputs, panel_a, panel_a_bins, panel_b, member_counts, panel_c, panel_d)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(OUT_DIR),
                "n_available_member_rows": len(panel_a),
                "spearman_rho": rho,
                "n_matched_support": int(panel_d["n_chemicals"].iloc[0]),
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
