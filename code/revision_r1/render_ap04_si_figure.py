#!/usr/bin/env python3
"""Render publication-ready SI Figure S8 from the frozen AP04 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BLUE = "#2C7FB8"
TEAL = "#41B6C4"
RED = "#B24A48"
GREY = "#8F9499"
DARK = "#2E3436"
GRID = "#D7DCE0"

WEIGHT_ORDER = [
    "baseline_45_35_10_10",
    "equal_25_25_25_25",
    "detection_heavy_60_20_10_10",
    "concentration_heavy_20_60_10_10",
    "no_censoring_renormalized",
    "detection_plus_0p10_renormalized",
    "detection_minus_0p10_renormalized",
    "concentration_plus_0p10_renormalized",
    "concentration_minus_0p10_renormalized",
    "site_plus_0p10_renormalized",
    "site_minus_0p10_renormalized",
    "censoring_plus_0p10_renormalized",
    "censoring_minus_0p10_renormalized",
]

WEIGHT_LABELS = {
    "baseline_45_35_10_10": "Baseline",
    "equal_25_25_25_25": "Equal weights",
    "detection_heavy_60_20_10_10": "Detection-heavy",
    "concentration_heavy_20_60_10_10": "Concentration-heavy",
    "no_censoring_renormalized": "No censoring term",
    "detection_plus_0p10_renormalized": "Detection +0.10",
    "detection_minus_0p10_renormalized": "Detection -0.10",
    "concentration_plus_0p10_renormalized": "Concentration +0.10",
    "concentration_minus_0p10_renormalized": "Concentration -0.10",
    "site_plus_0p10_renormalized": "Sites +0.10",
    "site_minus_0p10_renormalized": "Sites -0.10",
    "censoring_plus_0p10_renormalized": "Censoring +0.10",
    "censoring_minus_0p10_renormalized": "Censoring -0.10",
}

POLICY_ORDER = [
    "soft_baseline",
    "hard_exclude_current_or_unresolved_official_nas",
    "state_confirmed_only",
]

POLICY_LABELS = {
    "soft_baseline": "Soft relevance",
    "hard_exclude_current_or_unresolved_official_nas": "Exclude current or\nunresolved NAS",
    "state_confirmed_only": "State-confirmed only",
}

TIER_ORDER = [
    "soft_all",
    "regional_breadth_or_better",
    "neighbor_or_state",
    "state_confirmed",
]

TIER_LABELS = {
    "soft_all": "Soft relevance",
    "regional_breadth_or_better": "Regional or stronger",
    "neighbor_or_state": "Neighbor or state",
    "state_confirmed": "State-confirmed",
}

CAPTION = [
    "Figure S8. Sensitivity of localized Top-5 panels to weighting and state-level evidence support.",
    (
        "a, Percentage of the 32 complete-data localizable states retaining the identical Top-5 member set under "
        "the prespecified chemical-weight scenarios. The baseline coefficients for detection frequency, 90th-"
        "percentile concentration, monitoring-site count and censoring were 0.45, 0.35, 0.10 and 0.10, respectively. "
        "b, Member-set retention under alternative species-evidence policies. Labels give the number of states with "
        "a changed member set and the largest state-level change in full-universe expected coverage. c, Mean Top-5 "
        "Jaccard similarity and the number of additional national-panel fallbacks (FB) when chemical monitoring "
        "support was retained at 100%, 75%, 50% or 25% and species-evidence eligibility was progressively restricted. "
        "d, State-level change in full-universe expected coverage under chemical-record reduction with the baseline "
        "soft species-relevance policy. Top labels give the number of states that remained localizable out of 32, and "
        "crosses mark states that fell below the three-chemical localization threshold; "
        "AL and MT became additional national-panel fallbacks at 50% and 25% retention. Panels selected on reduced "
        "inputs were evaluated on the complete baseline state universe. Values are deterministic sensitivity "
        "diagnostics rather than sampling-based confidence intervals."
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_header(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.02,
        1.055,
        letter,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.055,
        1.055,
        title,
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left",
        va="bottom",
    )


def format_percent(value: float) -> str:
    return "100" if np.isclose(value, 100.0) else f"{value:.1f}"


def build_source_data(
    weights: pd.DataFrame,
    policies: pd.DataFrame,
    reduction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    w = weights.loc[weights["baseline_localized"].astype(bool)].copy()
    weight_summary = (
        w.groupby("weight_scenario", sort=False)
        .agg(
            n_states=("state_code", "size"),
            unchanged_states=("membership_jaccard_vs_baseline", lambda x: int(np.isclose(x, 1.0).sum())),
            mean_jaccard=("membership_jaccard_vs_baseline", "mean"),
            minimum_jaccard=("membership_jaccard_vs_baseline", "min"),
            maximum_absolute_coverage_change_pp=(
                "full_universe_coverage_delta_pp",
                lambda x: float(np.abs(x).max()),
            ),
        )
        .reset_index()
    )
    weight_summary["unchanged_percent"] = 100.0 * weight_summary["unchanged_states"] / weight_summary["n_states"]
    weight_summary["display_label"] = weight_summary["weight_scenario"].map(WEIGHT_LABELS)
    weight_summary["display_order"] = weight_summary["weight_scenario"].map(
        {name: i for i, name in enumerate(WEIGHT_ORDER)}
    )
    weight_summary = weight_summary.sort_values("display_order").reset_index(drop=True)

    p = policies.loc[policies["baseline_localized"].astype(bool)].copy()
    policy_summary = (
        p.groupby("species_policy", sort=False)
        .agg(
            n_states=("state_code", "size"),
            unchanged_states=("membership_jaccard_vs_baseline", lambda x: int(np.isclose(x, 1.0).sum())),
            mean_jaccard=("membership_jaccard_vs_baseline", "mean"),
            worst_coverage_change_pp=("full_universe_coverage_delta_pp", "min"),
        )
        .reset_index()
    )
    policy_summary["changed_states"] = policy_summary["n_states"] - policy_summary["unchanged_states"]
    policy_summary["unchanged_percent"] = 100.0 * policy_summary["unchanged_states"] / policy_summary["n_states"]
    policy_summary["display_label"] = policy_summary["species_policy"].map(POLICY_LABELS)
    policy_summary["display_order"] = policy_summary["species_policy"].map(
        {name: i for i, name in enumerate(POLICY_ORDER)}
    )
    policy_summary = policy_summary.sort_values("display_order").reset_index(drop=True)

    d = reduction.loc[reduction["baseline_localized"].astype(bool)].copy()
    fractions = [1.0, 0.75, 0.50, 0.25]
    cells: list[dict[str, object]] = []
    for tier_index, tier in enumerate(TIER_ORDER):
        for fraction_index, fraction in enumerate(fractions):
            cell = d.loc[
                d["species_evidence_tier"].eq(tier)
                & np.isclose(d["chemical_retention_fraction"], fraction)
            ]
            if len(cell) != 32:
                raise RuntimeError(
                    f"Expected 32 baseline-localizable states for {tier} at {fraction}, found {len(cell)}"
                )
            cells.append(
                {
                    "species_evidence_tier": tier,
                    "display_label": TIER_LABELS[tier],
                    "tier_order": tier_index,
                    "chemical_retention_fraction": fraction,
                    "fraction_order": fraction_index,
                    "mean_jaccard": float(cell["membership_jaccard_vs_baseline"].mean()),
                    "additional_fallback_states": int((~cell["localized"].astype(bool)).sum()),
                    "mean_absolute_coverage_change_pp": float(
                        cell["full_universe_coverage_delta_pp"].abs().mean()
                    ),
                }
            )
    nested_summary = pd.DataFrame(cells)

    state_change = d.loc[d["species_evidence_tier"].eq("soft_all")].copy()
    state_change["retention_percent"] = 100.0 * state_change["chemical_retention_fraction"]
    state_change["additional_fallback"] = ~state_change["localized"].astype(bool)
    state_change = state_change.sort_values(
        ["chemical_retention_fraction", "state_code"], ascending=[False, True]
    ).reset_index(drop=True)
    return weight_summary, policy_summary, nested_summary, state_change


def render(
    weight_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    nested_summary: pd.DataFrame,
    state_change: pd.DataFrame,
    out_stem: Path,
) -> list[Path]:
    configure_style()
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.16, 1.0],
        height_ratios=[1.08, 1.0],
        left=0.205,
        right=0.975,
        bottom=0.105,
        top=0.94,
        wspace=0.54,
        hspace=0.52,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a: exact member-set retention under chemical-weight scenarios.
    y = np.arange(len(weight_summary))[::-1]
    values = weight_summary["unchanged_percent"].to_numpy()
    colors = [GREY if name == "baseline_45_35_10_10" else BLUE for name in weight_summary["weight_scenario"]]
    ax_a.hlines(y, 80.0, values, color="#D9E6EF", linewidth=1.2, zorder=1)
    ax_a.scatter(values, y, c=colors, s=22, zorder=3, edgecolors="white", linewidths=0.45)
    ax_a.set_yticks(y, weight_summary["display_label"])
    ax_a.set_xlim(80.0, 101.5)
    ax_a.set_xticks([80, 90, 100])
    ax_a.set_xlabel("States with unchanged Top-5 membership (%)")
    ax_a.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.8)
    for yi, value in zip(y, values):
        ax_a.text(value + 0.55, yi, format_percent(value), ha="left", va="center", fontsize=5.9, color=DARK)
    ax_a.tick_params(axis="y", length=0, pad=2)
    panel_header(ax_a, "a", "Chemical-weight perturbations")

    # b: alternative species-evidence policies.
    yb = np.arange(len(policy_summary))[::-1]
    policy_values = policy_summary["unchanged_percent"].to_numpy()
    policy_colors = [GREY, TEAL, BLUE]
    ax_b.hlines(yb, 80.0, policy_values, color="#D9E6EF", linewidth=1.4, zorder=1)
    ax_b.scatter(policy_values, yb, c=policy_colors, s=30, zorder=3, edgecolors="white", linewidths=0.45)
    ax_b.set_yticks(yb, policy_summary["display_label"])
    ax_b.set_xlim(80.0, 101.5)
    ax_b.set_ylim(-0.48, 2.20)
    ax_b.set_xticks([80, 90, 100])
    ax_b.set_xlabel("States with unchanged Top-5 membership (%)")
    ax_b.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.8)
    for yi, row in zip(yb, policy_summary.itertuples(index=False)):
        if row.changed_states == 0:
            annotation = "0 changed"
        else:
            annotation = f"{row.changed_states} changed; worst {row.worst_coverage_change_pp:.2f} pp"
        ax_b.text(80.3, yi - 0.22, annotation, fontsize=5.6, color=DARK, ha="left", va="top")
        ax_b.text(row.unchanged_percent + 0.55, yi, format_percent(row.unchanged_percent), fontsize=5.9, color=DARK, ha="left", va="center")
    ax_b.tick_params(axis="y", length=0, pad=2)
    panel_header(ax_b, "b", "Species-evidence policies")

    # c: factorial reduction in chemical and species evidence.
    fractions = [1.0, 0.75, 0.50, 0.25]
    mean_matrix = np.full((len(TIER_ORDER), len(fractions)), np.nan)
    fallback_matrix = np.zeros_like(mean_matrix, dtype=int)
    for row in nested_summary.itertuples(index=False):
        mean_matrix[int(row.tier_order), int(row.fraction_order)] = float(row.mean_jaccard)
        fallback_matrix[int(row.tier_order), int(row.fraction_order)] = int(row.additional_fallback_states)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "si_blue",
        ["#F6FAFD", "#CFE3F0", "#86B8D8", BLUE],
    )
    image = ax_c.imshow(mean_matrix, cmap=cmap, vmin=0.75, vmax=1.0, aspect="auto")
    ax_c.set_xticks(np.arange(len(fractions)), ["100", "75", "50", "25"])
    ax_c.set_yticks(np.arange(len(TIER_ORDER)), [TIER_LABELS[tier] for tier in TIER_ORDER])
    ax_c.set_xlabel("State chemical records retained (%)")
    ax_c.tick_params(axis="both", length=0)
    ax_c.set_xticks(np.arange(-0.5, len(fractions), 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(TIER_ORDER), 1), minor=True)
    ax_c.grid(which="minor", color="white", linewidth=0.7)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            text_color = "white" if mean_matrix[i, j] >= 0.975 else DARK
            ax_c.text(
                j,
                i,
                f"J={mean_matrix[i, j]:.3f}\nFB={fallback_matrix[i, j]}",
                ha="center",
                va="center",
                fontsize=5.9,
                color=text_color,
                linespacing=1.12,
            )
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.040, pad=0.018, aspect=28)
    cbar.set_label("Mean Top-5 Jaccard", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.0, length=2)
    panel_header(ax_c, "c", "Nested state-support reduction")

    # d: state-level full-universe coverage change under chemical reduction.
    frac_order = [1.0, 0.75, 0.50, 0.25]
    for x_index, fraction in enumerate(frac_order):
        group = state_change.loc[np.isclose(state_change["chemical_retention_fraction"], fraction)].sort_values("state_code")
        offsets = np.linspace(-0.16, 0.16, len(group))
        localized = group["localized"].astype(bool).to_numpy()
        y_values = group["full_universe_coverage_delta_pp"].to_numpy()
        ax_d.scatter(
            x_index + offsets[localized],
            y_values[localized],
            color=BLUE,
            s=14,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        if (~localized).any():
            ax_d.scatter(
                x_index + offsets[~localized],
                y_values[~localized],
                color=RED,
                marker="x",
                s=28,
                linewidths=1.1,
                zorder=4,
            )
            for offset, (_, row) in zip(offsets[~localized], group.loc[~localized].iterrows()):
                state_code = str(row["state_code"])
                if state_code == "AL":
                    label_offset = (-3, 5)
                    label_ha = "right"
                    label_va = "bottom"
                else:
                    label_offset = (3, -1)
                    label_ha = "left"
                    label_va = "center"
                ax_d.annotate(
                    state_code,
                    (x_index + offset, float(row["full_universe_coverage_delta_pp"])),
                    xytext=label_offset,
                    textcoords="offset points",
                    fontsize=5.8,
                    color=RED,
                    ha=label_ha,
                    va=label_va,
                )
        ax_d.text(
            x_index,
            0.15,
            f"{int(localized.sum())}/32",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=DARK,
        )
    ax_d.axhline(0, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax_d.set_xticks(np.arange(len(frac_order)), ["100", "75", "50", "25"])
    ax_d.set_xlabel("State chemical records retained (%)")
    ax_d.set_ylabel("Full-universe expected-coverage change (pp)")
    ax_d.set_ylim(-7.15, 0.58)
    ax_d.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=5, label="Localized"),
            Line2D([0], [0], marker="x", color=RED, linestyle="none", markersize=5, label="National fallback"),
        ],
        loc="lower left",
        borderaxespad=0.2,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    panel_header(ax_d, "d", "Coverage under monitoring reduction")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    export_settings = {
        ".svg": {},
        ".pdf": {},
        ".tiff": {"dpi": 600},
        ".png": {"dpi": 300},
    }
    for suffix, kwargs in export_settings.items():
        path = out_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "weight_detail": input_dir / "ap04_weight_sensitivity_state_detail.csv",
        "species_policy_detail": input_dir / "ap04_species_policy_sensitivity_state_detail.csv",
        "data_reduction_detail": input_dir / "ap04_data_reduction_state_detail.csv.gz",
        "state_ledger": input_dir / "ap04_state_support_and_fallback_ledger.csv",
        "frozen_config": input_dir / "ap04_frozen_scenario_config.json",
        "gate": input_dir / "gate_ap04.json",
        "figure_contract": input_dir / "FIGURE_S8_LOCALIZATION_SENSITIVITY_CONTRACT.md",
        "script": Path(__file__).resolve(),
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    gate = json.loads(source_paths["gate"].read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("AP04 gate must be PASS before rendering Figure S8")

    weights = pd.read_csv(source_paths["weight_detail"])
    policies = pd.read_csv(source_paths["species_policy_detail"])
    reduction = pd.read_csv(source_paths["data_reduction_detail"])
    weight_summary, policy_summary, nested_summary, state_change = build_source_data(
        weights,
        policies,
        reduction,
    )

    if len(weight_summary) != 13 or len(policy_summary) != 3 or len(nested_summary) != 16:
        raise RuntimeError("Unexpected Figure S8 summary dimensions")
    if int(state_change["state_code"].nunique()) != 32:
        raise RuntimeError("Figure S8 requires 32 complete-data localizable states")

    source_frames = []
    for panel, frame in [
        ("a", weight_summary),
        ("b", policy_summary),
        ("c", nested_summary),
        ("d", state_change),
    ]:
        part = frame.copy()
        part.insert(0, "panel", panel)
        source_frames.append(part)
    source_data = pd.concat(source_frames, ignore_index=True, sort=False)
    source_data_path = output_dir / "figure_s8_localization_sensitivity_source_data.csv"
    source_data.to_csv(source_data_path, index=False)

    out_stem = output_dir / "figure_s8_localization_sensitivity"
    outputs = render(weight_summary, policy_summary, nested_summary, state_change, out_stem)

    svg_text = out_stem.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text is not editable")

    provenance = {
        "schema_version": 1,
        "figure_id": "figure_s8",
        "placement": "Supporting Information immediately after Figure S7",
        "backend": "Python/Matplotlib",
        "core_conclusion": (
            "Localized Top-5 membership is stable across prespecified chemical-weight and species-evidence choices "
            "under complete support, while monitoring depletion produces threshold-driven fallbacks and concentrated "
            "coverage loss in low-support states."
        ),
        "filters_and_denominators": {
            "complete_data_localizable_states": 32,
            "baseline_fallback_states": ["PA", "TN", "WV"],
            "chemical_weight_scenarios": 13,
            "species_evidence_policies": 3,
            "nested_reduction_scenarios": 16,
            "coverage_evaluation": "all reduced-input panels evaluated on the complete baseline state universe",
        },
        "caption_paragraphs": CAPTION,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in source_paths.items()
        },
        "source_data": {
            "path": str(source_data_path),
            "sha256": sha256(source_data_path),
            "size_bytes": source_data_path.stat().st_size,
        },
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    provenance_path = output_dir / "figure_s8_localization_sensitivity_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "figure_id": "figure_s8",
        "status": "PASS",
        "manuscript_caption_paragraphs": CAPTION,
        "source_data": source_data_path.name,
        "provenance": provenance_path.name,
        "outputs": [path.name for path in outputs],
        "qa_assertions": {
            "ap04_gate_pass": True,
            "python_backend_exclusive": True,
            "editable_svg_text": True,
            "complete_data_localizable_states": 32,
            "baseline_fallback_states": ["PA", "TN", "WV"],
            "new_fallback_states_at_50_and_25_percent": ["AL", "MT"],
        },
    }
    manifest_path = output_dir / "figure_s8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
