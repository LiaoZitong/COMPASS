#!/usr/bin/env python3
"""Render R1 SI Figure S7 from AP04 support and strict-LOO state testing priorities."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[3]
ANALYSIS_ROOT = PACKAGE_ROOT / "results" / "revision_r1"
BHBT_ROOT = PACKAGE_ROOT
AP02_DIR = ANALYSIS_ROOT / "02_ap02_strict_loo"
AP04_DIR = ANALYSIS_ROOT / "06_ap04_localization_sensitivity"
FIG5_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "main_figure_05"
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "si_figure_s7"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, apply_style, clean_axis, export_figure  # noqa: E402


FIGURE_BASENAME = "figure_s7_r1_state_support_testing"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readable_moa_term(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return "Unassigned"
    value = value.strip()
    if value.upper() in {"MOA_UNRESOLVED", "UNRESOLVED", "NONE", "NAN"}:
        return "Unassigned"
    rest = value.split("::", 1)[1] if "::" in value else value
    rest = rest.replace("_", " ").strip()
    lower = rest.lower()
    if any(token in lower for token in ("acetylcholinesterase", "sodium channel", "ionotrop", "glutamate", "neuro")):
        return "neurotransmission"
    if any(token in lower for token in ("thyro", "deiodinase", "symporter")):
        return "thyroid axis"
    if any(token in lower for token in ("deposition of energy", "mitochond", "oxidative phosphorylation", "nadh")):
        return "energy metabolism"
    if any(token in lower for token in ("photosystem", "photosynthesis")):
        return "photosynthesis"
    if any(token in lower for token in ("ahr", "androgen", "estrogen", "ppar", "receptor", "cyp")):
        return "receptor/xenobiotic signaling"
    if "histone deacetylase" in lower:
        return "histone deacetylase inhibition"
    if "vegfr2" in lower:
        return "VEGFR2 inhibition"
    if "calcineurin" in lower:
        return "calcineurin activity inhibition"
    if any(token in lower for token in ("alkylation", "reactive", "oxid", "strand breaks")):
        return "reactive chemistry"
    if "," in rest:
        parts = [part.strip() for part in rest.split(",") if part.strip()]
        if len(parts) >= 2:
            rest = f"{parts[1]} {parts[0]}".strip()
        elif parts:
            rest = parts[0]
    return rest if rest else "Unassigned"


def build_panel_a() -> pd.DataFrame:
    ledger = pd.read_csv(AP04_DIR / "ap04_state_support_and_fallback_ledger.csv")
    ledger["localized_capture_percent"] = ledger["baseline_expected_weighted_joint_coverage_x95"] * 100.0
    ledger["national_capture_percent"] = ledger["national_top5_expected_weighted_joint_coverage_x95"] * 100.0
    ledger["gain_percent_points"] = ledger["localization_gain_over_national_top5_pp"]
    ledger["status_label"] = np.where(ledger["localized"].astype(bool), "Localized Top-5", "National fallback")
    return ledger


def build_state_moa_priorities() -> tuple[pd.DataFrame, pd.DataFrame]:
    z = np.load(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz", allow_pickle=True)
    probability = np.asarray(z["p"], dtype=float)
    reliability = np.asarray(z["reliability"], dtype=float)
    direct_n = np.asarray(z["direct_n"])
    species = np.asarray(z["species"], dtype=object).astype(str)
    chemicals = np.asarray(z["chemicals"], dtype=object).astype(str)
    chemical_index = {name: idx for idx, name in enumerate(chemicals)}

    state_weights = pd.read_csv(BHBT_ROOT / "results" / "weights" / "state_priority_chemical_weights.csv.gz", low_memory=False)
    state_weights["DTXSID"] = state_weights["DTXSID"].astype(str)
    soft = pd.read_csv(AP04_DIR / "ap04_reconstructed_soft_local_species_weights.csv.gz", low_memory=False)
    soft["latin_name"] = soft["latin_name"].astype(str)
    chemistry = pd.read_csv(BHBT_ROOT / "results" / "chemistry" / "chemical_moa_form_master.csv.gz", low_memory=False)
    chemistry["DTXSID"] = chemistry["DTXSID"].astype(str)
    moa_map = chemistry.set_index("DTXSID")["primary_moa"].to_dict()

    national_matrix = pd.read_csv(FIG5_DIR / "figure_05c_strict_loo_national_species_moa_matrix.csv")
    top_moa = (
        national_matrix.groupby("candidate_moa_term")["national_testing_priority_score"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    if len(top_moa) != 8:
        raise AssertionError(f"Expected eight national candidate-MOA terms, found {len(top_moa)}")

    detailed_rows: list[dict[str, object]] = []
    for state_code, group in state_weights.groupby("state_code"):
        state_code = str(state_code)
        state_chemicals = [name for name in group["DTXSID"].astype(str) if name in chemical_index]
        if not state_chemicals:
            continue
        indices = np.asarray([chemical_index[name] for name in state_chemicals], dtype=int)
        weights = group.set_index("DTXSID")["state_weight"].reindex(state_chemicals).fillna(0).clip(lower=0).to_numpy(float)
        weights = weights / weights.sum() if np.isfinite(weights).all() and weights.sum() > 0 else np.repeat(1 / len(weights), len(weights))
        species_weight = (
            soft[soft["state_code"].astype(str).eq(state_code)]
            .set_index("latin_name")["soft_local_species_weight"]
            .reindex(species)
            .fillna(0.03)
            .to_numpy(float)
        )
        for local_col, chemical in enumerate(state_chemicals):
            col = indices[local_col]
            moa_term = readable_moa_term(moa_map.get(chemical, "MOA_UNRESOLVED"))
            if moa_term == "Unassigned":
                continue
            missing = direct_n[:, col] == 0
            component = species_weight * probability[:, col] * (1.0 - probability[:, col]) * (1.0 - reliability[:, col]) * missing
            priority_value = float(weights[local_col] * component.sum())
            if priority_value <= 0:
                continue
            denominator = float((species_weight * missing).sum())
            detailed_rows.append(
                {
                    "state_code": state_code,
                    "DTXSID": chemical,
                    "candidate_moa_term": moa_term,
                    "state_weight": float(weights[local_col]),
                    "state_moa_testing_priority": priority_value,
                    "n_missing_species_pairs": int(missing.sum()),
                    "soft_weighted_missing_species_sum": denominator,
                    "mean_tail_probability_missing": float(np.average(probability[:, col], weights=species_weight * missing)) if denominator > 0 else 0.0,
                    "mean_reliability_missing": float(np.average(reliability[:, col], weights=species_weight * missing)) if denominator > 0 else 0.0,
                    "probability_source": "AP02_STRICT_LOO",
                    "testing_scope": "state-level",
                }
            )
    detailed = pd.DataFrame(detailed_rows)
    aggregated = (
        detailed[detailed["candidate_moa_term"].isin(top_moa)]
        .groupby(["state_code", "candidate_moa_term"], as_index=False)
        .agg(
            state_moa_testing_priority=("state_moa_testing_priority", "sum"),
            n_priority_chemicals=("DTXSID", "nunique"),
            n_missing_species_pairs=("n_missing_species_pairs", "sum"),
        )
    )
    states = sorted(state_weights["state_code"].astype(str).unique())
    full = pd.MultiIndex.from_product([states, top_moa], names=["state_code", "candidate_moa_term"]).to_frame(index=False)
    aggregated = full.merge(aggregated, on=["state_code", "candidate_moa_term"], how="left")
    for column in ["state_moa_testing_priority", "n_priority_chemicals", "n_missing_species_pairs"]:
        aggregated[column] = pd.to_numeric(aggregated[column], errors="coerce").fillna(0.0)
    aggregated["probability_source"] = "AP02_STRICT_LOO"
    aggregated["testing_scope"] = "state-level"
    if len(states) != 35 or len(aggregated) != 35 * 8:
        raise AssertionError(f"State-MOA matrix mismatch: states={len(states)}, rows={len(aggregated)}")
    return aggregated, detailed


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-26, 16), textcoords="offset points", ha="left", va="top", fontweight="bold", fontsize=11, clip_on=False)


def draw(panel_a: pd.DataFrame, state_moa: pd.DataFrame) -> list[str]:
    apply_style(font_size=8.0)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.25, 4.65), gridspec_kw={"width_ratios": [0.90, 1.40], "wspace": 0.24})

    localized = panel_a[panel_a["localized"].astype(bool)]
    fallback = panel_a[~panel_a["localized"].astype(bool)]
    ax_a.scatter(
        localized["n_priority_chemicals"],
        localized["localized_capture_percent"],
        s=31,
        color=PALETTE["blue2"],
        edgecolor="white",
        linewidth=0.45,
        label="Localized Top-5 (n = 32)",
    )
    ax_a.scatter(
        fallback["n_priority_chemicals"],
        fallback["localized_capture_percent"],
        s=36,
        facecolor="white",
        edgecolor=PALETTE["red"],
        marker="D",
        linewidth=1.1,
        label="National fallback (n = 3)",
    )
    label_states = set(localized.nlargest(2, "localized_capture_percent")["state_code"].astype(str))
    label_states |= set(localized.nsmallest(2, "localized_capture_percent")["state_code"].astype(str))
    label_states |= set(fallback["state_code"].astype(str))
    for row in panel_a[panel_a["state_code"].astype(str).isin(label_states)].itertuples(index=False):
        if str(row.state_code) in {"PA", "WV"}:
            continue
        ax_a.annotate(str(row.state_code), (row.n_priority_chemicals, row.localized_capture_percent), xytext=(4, 4), textcoords="offset points", fontsize=6.4)
    pa_wv = fallback[fallback["state_code"].astype(str).isin(["PA", "WV"])].iloc[0]
    ax_a.annotate("PA, WV", (pa_wv.n_priority_chemicals, pa_wv.localized_capture_percent), xytext=(6, 5), textcoords="offset points", fontsize=6.4)
    ax_a.set_xlabel("State-priority chemicals")
    ax_a.set_ylabel("Localized/fallback lower-5%\nexpected capture (%)")
    ax_a.set_title("State chemical support and expected capture", loc="left", fontsize=8.5)
    ax_a.legend(loc="upper right", fontsize=6.2)
    clean_axis(ax_a, grid=True)
    aligned_panel_label(ax_a, "a")

    totals = state_moa.groupby("state_code")["state_moa_testing_priority"].sum().sort_values(ascending=False)
    state_order = totals.index.astype(str).tolist()
    moa_order = (
        state_moa.groupby("candidate_moa_term")["state_moa_testing_priority"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    matrix = state_moa.pivot(index="state_code", columns="candidate_moa_term", values="state_moa_testing_priority").reindex(index=state_order, columns=moa_order).fillna(0.0)
    display = matrix.to_numpy(float) * 1000.0
    cmap = mpl.colors.LinearSegmentedColormap.from_list("state_priority", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0, vmax=max(float(display.max()), 1e-9))
    image = ax_b.imshow(display, aspect="auto", cmap=cmap, norm=norm)
    fallback_states = set(panel_a.loc[~panel_a["localized"].astype(bool), "state_code"].astype(str))
    ax_b.set_yticks(np.arange(len(state_order)))
    ax_b.set_yticklabels([f"{state}*" if state in fallback_states else state for state in state_order], fontsize=6.0)
    ax_b.set_xticks(np.arange(len(moa_order)))
    ax_b.set_xticklabels([textwrap.fill(value, width=15, break_long_words=False) for value in moa_order], rotation=43, ha="right", fontsize=6.0)
    ax_b.set_xlabel("Candidate mode-of-action group")
    ax_b.set_ylabel("State-level application")
    ax_b.set_title("Strict-LOO state testing priorities", loc="left", fontsize=8.5)
    ax_b.set_xticks(np.arange(-0.5, len(moa_order), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(state_order), 1), minor=True)
    ax_b.grid(which="minor", color="white", lw=0.42)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.042, pad=0.02)
    colorbar.set_label(r"Testing-priority score ($\times 10^3$)", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.8)
    ax_b.text(0.0, -0.20, "* national Top-5 fallback because fewer than three state-priority chemicals were available", transform=ax_b.transAxes, ha="left", va="top", fontsize=5.8, color=PALETTE["neutral_dark"])
    aligned_panel_label(ax_b, "b")

    fig.subplots_adjust(left=0.095, right=0.965, top=0.93, bottom=0.24)
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(outputs: list[str], panel_a: pd.DataFrame, state_moa: pd.DataFrame, detailed: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "figure_s7a_state_support_capture.csv": panel_a,
        "figure_s7b_state_moa_testing_matrix.csv": state_moa,
        "figure_s7b_state_chemical_moa_detail.csv.gz": detailed,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, compression="gzip" if name.endswith(".gz") else None)
    caption = (
        "Figure S7. State-level support for localized Top-5 panels and candidate-MOA testing priorities. "
        "a, Number of state-priority chemicals versus lower-5% expected capture for 32 localized panels and three national Top-5 fallbacks. Labels identify the two highest- and two lowest-capture localized applications and all fallback states. "
        "b, State-level candidate-MOA testing priorities recalculated from strict focal-species leave-one-out probability, reliability, and direct-support matrices. Each score integrates state chemical weights, soft species-relevance weights, decision uncertainty, evidence-support uncertainty, and missing direct species-chemical evidence, then sums across chemicals in the candidate-MOA group. Rows are ordered by total testing-priority score; asterisks identify the three prespecified fallback states. The common color scale uses a square-root transform for display. These priorities identify evidence-acquisition opportunities within the state-weighted screening context."
    )
    (OUT_DIR / "figure_contract.md").write_text(
        "# R1 Figure S7 contract\n\n"
        "Core conclusion: state support varies widely, and strict-LOO testing-priority patterns identify where additional species-chemical evidence would most improve the state-weighted screening evidence base.\n\n"
        "Figure S7 reports the baseline state support and testing surface. Scenario robustness and monitoring/species-support reduction are reported separately in Figure S8 and Table S7.\n\n"
        "The two panels use a narrower gap and larger Times New Roman labels to preserve readability at SI display width.\n\n"
        f"Manuscript caption: {caption}\n",
        encoding="utf-8",
    )
    files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure S7",
        "version": "R1_AP04_AP02_STRICT_LOO",
        "manuscript_caption": caption,
        "input_dependencies": [
            str(AP04_DIR / "ap04_state_support_and_fallback_ledger.csv"),
            str(AP04_DIR / "ap04_reconstructed_soft_local_species_weights.csv.gz"),
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz"),
            str(BHBT_ROOT / "results" / "weights" / "state_priority_chemical_weights.csv.gz"),
            str(BHBT_ROOT / "results" / "chemistry" / "chemical_moa_form_master.csv.gz"),
            str(FIG5_DIR / "figure_05c_strict_loo_national_species_moa_matrix.csv"),
        ],
        "outputs": files,
        "qa": {"states": 35, "localized_states": 32, "fallback_states": ["PA", "TN", "WV"], "candidate_moa_groups": 8, "editable_svg_text": True, "png_dpi": 300, "tiff_dpi": 600},
    }
    manifest["sha256"] = {name: sha256_file(OUT_DIR / name) for name in files if (OUT_DIR / name).exists()}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a = build_panel_a()
    state_moa, detailed = build_state_moa_priorities()
    outputs = draw(panel_a, state_moa)
    write_outputs(outputs, panel_a, state_moa, detailed)
    print(json.dumps({"status": "PASS", "output_dir": str(OUT_DIR), "state_moa_rows": len(state_moa), "detail_rows": len(detailed), "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
