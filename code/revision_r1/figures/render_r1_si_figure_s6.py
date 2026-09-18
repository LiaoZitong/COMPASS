#!/usr/bin/env python3
"""Render SI Figure S6 from the current strict-LOO robustness analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, clean_axis, export_figure  # noqa: E402


FIGURE_BASENAME = "figure_s6_strict_loo_profile_mnar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory produced by run_strict_loo_robustness.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-22, 13),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        clip_on=False,
    )


def load_inputs(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    scenarios = pd.read_csv(analysis_dir / "strict_loo_mnar_sensitivity_scenarios.csv")
    profile = pd.read_csv(analysis_dir / "strict_loo_framework_profile_metrics.csv")
    manifest = json.loads(
        (analysis_dir / "strict_loo_robustness_manifest.json").read_text(encoding="utf-8")
    )
    expected_or = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    observed_or = scenarios.odds_ratio_tail_positive_testing.to_numpy(float)
    if not np.allclose(observed_or, expected_or):
        raise RuntimeError(f"Unexpected MNAR odds-ratio grid: {observed_or.tolist()}")
    if len(profile) != 1 or int(profile.iloc[0].n_contexts) < 1:
        raise RuntimeError("Profile summary must contain one non-empty strict-LOO method row")
    if not scenarios.top5_jaccard_vs_strict_loo_core.eq(1.0).all():
        raise RuntimeError("Top-5 membership changed; revise the Figure S6 annotation")
    return scenarios, profile, manifest


def make_profile_source(profile: pd.DataFrame) -> pd.DataFrame:
    row = profile.iloc[0]
    definitions = [
        ("Median subset/reference HC5 ratio", "median_ratio"),
        ("Spearman rank correlation", "spearman_r"),
        ("Within a factor of 2", "within_factor2"),
        ("False-safe by a factor of 2", "false_safe_factor2_rate"),
    ]
    records: list[dict[str, object]] = []
    for label, prefix in definitions:
        records.append(
            {
                "metric": label,
                "q025": float(row[f"{prefix}_q025"]),
                "q50": float(row[f"{prefix}_q50"]),
                "q975": float(row[f"{prefix}_q975"]),
                "n_contexts": int(row.n_contexts),
                "method": str(row.method),
            }
        )
    return pd.DataFrame(records)


def draw(scenarios: pd.DataFrame, profile_source: pd.DataFrame, output_dir: Path) -> list[str]:
    apply_figure_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.20, 3.20),
        gridspec_kw={"width_ratios": [1.02, 1.18], "wspace": 0.34},
    )

    odds = scenarios.odds_ratio_tail_positive_testing.to_numpy(float)
    capture = scenarios.coverage_copula.to_numpy(float)
    core = float(scenarios.strict_loo_core_coverage.iloc[0])
    ax_a.plot(
        odds,
        capture,
        color=PALETTE["blue2"],
        marker="o",
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        linewidth=1.5,
        zorder=3,
    )
    ax_a.axhline(core, color=PALETTE["neutral"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax_a.text(
        0.98,
        core + 0.012,
        f"Primary strict-LOO panel = {core:.3f}",
        transform=ax_a.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=PALETTE["neutral_dark"],
    )
    for current_or, value in zip(odds, capture):
        offset_y = 5 if current_or not in {0.5, 1.0} else (-11 if current_or == 0.5 else 5)
        ax_a.annotate(
            f"{value:.3f}",
            (current_or, value),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset_y > 0 else "top",
            fontsize=6.5,
            color=PALETTE["neutral_dark"],
        )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(odds)
    ax_a.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax_a.set_ylim(0, max(0.49, capture.max() + 0.055))
    ax_a.set_xlabel("Tail-positive testing odds ratio")
    ax_a.set_ylabel("Expected capture")
    ax_a.set_title("Testing-selection sensitivity", loc="left", fontsize=8.7)
    ax_a.text(
        0.03,
        0.05,
        "Top-5 membership unchanged\nacross the tested range",
        transform=ax_a.transAxes,
        fontsize=6.8,
        color=PALETTE["neutral_dark"],
        ha="left",
        va="bottom",
    )
    clean_axis(ax_a, grid=True)
    ax_a.grid(axis="x", visible=False)
    panel_label(ax_a, "a")

    labels = [
        "Median subset/reference\nHC5 ratio",
        "Spearman rank\ncorrelation",
        "Within a factor of 2",
        "False-safe by a\nfactor of 2",
    ]
    y = np.arange(len(profile_source))[::-1]
    q50 = profile_source.q50.to_numpy(float)
    q025 = profile_source.q025.to_numpy(float)
    q975 = profile_source.q975.to_numpy(float)
    ax_b.errorbar(
        q50,
        y,
        xerr=np.vstack([q50 - q025, q975 - q50]),
        fmt="o",
        color=PALETTE["red"],
        ecolor=PALETTE["red"],
        elinewidth=1.35,
        capsize=3.0,
        markersize=5.2,
        zorder=3,
    )
    for value, ypos in zip(q50, y):
        ax_b.text(
            min(value + 0.035, 1.005),
            ypos,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=PALETTE["neutral_dark"],
        )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.set_xlim(0, 1.06)
    ax_b.set_xlabel("Metric value (median and 95% interval)")
    ax_b.set_title("Profile-likelihood propagation", loc="left", fontsize=8.7)
    clean_axis(ax_b, grid=True)
    ax_b.grid(axis="y", visible=False)
    panel_label(ax_b, "b")

    fig.subplots_adjust(left=0.10, right=0.975, top=0.88, bottom=0.18)
    return export_figure(fig, output_dir / FIGURE_BASENAME)


def write_outputs(
    analysis_dir: Path,
    output_dir: Path,
    outputs: list[str],
    scenarios: pd.DataFrame,
    profile_source: pd.DataFrame,
    analysis_manifest: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_name = "figure_s6a_mnar_sensitivity_source_data.csv"
    profile_name = "figure_s6b_profile_likelihood_source_data.csv"
    scenarios.to_csv(output_dir / scenario_name, index=False)
    profile_source.to_csv(output_dir / profile_name, index=False)
    caption = (
        "Figure S6. Profile-likelihood propagation and testing-selection sensitivity for the strict focal-species leave-one-out analysis. "
        "a, Expected lower-5% capture after inverse-probability-weighted recalibration and explicit MNAR odds-ratio shifts. "
        "The dashed line is the primary strict-LOO Top-5 estimate; the same five species were selected throughout the tested odds-ratio range, although their rank order changed. "
        "b, Median and 95% simulation intervals across 413 measured protective contexts for the sentinel-subset/reference HC5 ratio, Spearman rank correlation, the proportion within a factor of 2, and the false-safe-by-a-factor-of-2 rate. "
        "The profile intervals propagate censored-record likelihood uncertainty within measured contexts, whereas the MNAR scenarios bound sensitivity to unidentifiable testing-selection assumptions; neither analysis changes the primary probability matrix."
    )
    contract = (
        "# Figure S6 contract\n\n"
        "Core conclusion: the strict-LOO Top-5 membership is stable across the prespecified testing-selection scenarios, while profile propagation quantifies the magnitude and ranking uncertainty within measured protective contexts.\n\n"
        "Archetype: two-panel quantitative robustness summary.\n\n"
        "Panel a evidence: strict-LOO MNAR scenario coverage, the primary strict-LOO reference line, and Top-5 membership stability.\n\n"
        "Panel b evidence: likelihood-propagated medians and 95% intervals for four prespecified framework metrics across 413 contexts.\n\n"
        f"Manuscript caption: {caption}\n"
    )
    (output_dir / "figure_contract.md").write_text(contract, encoding="utf-8")
    files = [*outputs, scenario_name, profile_name, "figure_contract.md"]
    manifest = {
        "figure_id": "Figure S6",
        "version": "STRICT_LOO_PROFILE_MNAR_20260918",
        "backend": "Python/Matplotlib",
        "font": "Times New Roman",
        "manuscript_caption": caption,
        "core_conclusion": "Strict-LOO Top-5 membership remained stable over the tested selection assumptions, and profile propagation retained high rank agreement while quantifying magnitude uncertainty.",
        "archetype": "two-panel quantitative robustness summary",
        "input_dependencies": [
            str(analysis_dir / "strict_loo_mnar_sensitivity_scenarios.csv"),
            str(analysis_dir / "strict_loo_framework_profile_metrics.csv"),
            str(analysis_dir / "strict_loo_robustness_manifest.json"),
        ],
        "analysis_manifest_summary": {
            "profile": analysis_manifest["profile"],
            "mnar": analysis_manifest["mnar"],
        },
        "outputs": files,
        "qa": {
            "editable_svg_text": True,
            "pdf_truetype_text": True,
            "png_dpi": 300,
            "tiff_dpi": 600,
            "source_data_in_figure_folder": True,
            "red_green_not_sole_encoding": True,
        },
    }
    manifest["sha256"] = {
        name: sha256_file(output_dir / name)
        for name in files
        if (output_dir / name).is_file()
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios, profile, manifest = load_inputs(analysis_dir)
    profile_source = make_profile_source(profile)
    outputs = draw(scenarios, profile_source, output_dir)
    write_outputs(analysis_dir, output_dir, outputs, scenarios, profile_source, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "outputs": outputs,
                "profile_contexts": int(profile.iloc[0].n_contexts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
