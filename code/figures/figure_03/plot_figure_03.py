#!/usr/bin/env python3
"""Figure 3: measured Top5-apical comparison and panel-score follow-up ranking."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import (  # noqa: E402
    PALETTE,
    apply_style,
    clean_axis,
    export_figure,
    figure_dir_from_script,
    percentile_rank_high,
    project_root_from_script,
    q_panel,
    save_contract_note,
    short_method,
    write_combined_source,
    write_csv,
    write_manifest,
)


FIGURE_ID = "figure_03"
TITLE = "Measured Top5-apical ranks track full-data HC5 while Top-5 scores prioritize follow-up"
MANUSCRIPT_CAPTION = (
    "Figure 3. Top5-apical preserves HC5 rank information, while the fixed Top-5 score prioritizes chemical follow-up. "
    "a, Sensitivity-rank agreement between Top5-apical and EPA-framework full-data HC5 across 115 chemicals. "
    "EPA-framework eligibility was operationalized from the EPA 1985 freshwater taxonomic-slot structure, after which "
    "the reference HC5 was fitted using all measured protective species in each eligible context. Grey points show "
    "paired rows, the dashed line shows 1:1 rank agreement, and the blue line shows binned medians. b, Recall of 23 "
    "lower-20% EPA-HC5 positives as the follow-up list expands across 639 chemicals ranked by the fixed Top-5 "
    "measured-tail probability score. Random ranking, "
    "the EPA WQC proxy (matched k = 5), and the EPA WET subset (matched k = 5) are shown as comparators. "
    "c, Mean high-hazard recall gain across the lower-5%, lower-10%, lower-20% and lower-30% EPA-HC5 definitions, "
    "calculated as 100 times the difference between the area-normalized recall of the fixed Top-5 measured-tail "
    "probability score and that of each comparator over the 5–30% follow-up-list range. d, Tertile concordance between "
    "Top5-apical concentration sensitivity rank and EPA-framework full-data HC5 sensitivity rank among the 115 "
    "chemicals. Panels a and d evaluate the concentration-based Top5-apical signal, whereas panels b and c evaluate "
    "chemical follow-up ranking using the fixed Top-5 measured-tail probability score."
)
HIGH_HAZARD_QUANTILE = 0.20
HIGH_HAZARD_ROBUSTNESS_QUANTILES = [0.05, 0.10, 0.20, 0.30]
EARLY_RETRIEVAL_FRACTIONS = [round(float(x), 2) for x in np.arange(0.05, 0.3001, 0.05)]
EPA_WQC_LABEL = "EPA WQC proxy (matched k = 5)"
EPA_WET_SUBSET_LABEL = "EPA WET subset (matched k = 5)"


def figure3_method_label(method: str) -> str:
    if method == "data_driven":
        return "Top-5 panel score"
    if method == "EPA_WQC_taxonomic_requirements":
        return EPA_WQC_LABEL
    if method == "EPA_WET_fixed_method_species":
        return EPA_WET_SUBSET_LABEL
    if method == "random_panel_eligible_candidate_species":
        return "Random 5-species"
    return short_method(method)


def figure3_display_label(label: str) -> str:
    if label == "Top-5 panel score":
        return "Fixed Top-5 measured-tail\nprobability score"
    if label == EPA_WQC_LABEL:
        return "EPA WQC proxy\n(matched k = 5)"
    if label == EPA_WET_SUBSET_LABEL:
        return "EPA WET subset\n(matched k = 5)"
    return label


def high_hazard_cutoff_label(quantile: float) -> str:
    return f"Lower {quantile * 100:.0f}%"


def high_hazard_definition(quantile: float) -> str:
    return (
        f"lower {quantile * 100:.0f}% EPA-framework reference_hc5_typical_log10 among the "
        "115 EPA-labeled comparison rows, embedded in the 639-chemical fixed Top-5 probability-score ranking universe"
    )


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-28, 28),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12.0,
        color="black",
        clip_on=False,
    )


def build_panel_a(root: Path) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    bridge = pd.read_csv(root / "results/warning_hc5_bridge/top5_panel_apical_threshold_vs_reference_hc5.csv").dropna(
        subset=["reference_hc5_typical_log10", "panel_min_apical_log10"]
    )
    out = bridge.copy()
    out["full_hazard_percentile"] = percentile_rank_high(out["reference_hc5_typical_log10"])
    out["top5_trigger_percentile"] = percentile_rank_high(out["panel_min_apical_log10"])
    out["log10_ratio_top5_to_hc5"] = out["panel_min_apical_log10"] - out["reference_hc5_typical_log10"]
    out["hazard_percentile_algorithm"] = (
        "x is the percentile rank of the Top5-apical panel minimum: rank(-panel_min_apical_log10, pct=True) * 100; "
        "y is the percentile rank of the EPA-framework full-data typical HC5: rank(-reference_hc5_typical_log10, pct=True) * 100; "
        "computed among all paired Top5-apical panel-min/EPA-framework HC5 comparison rows"
    )
    binned = out.dropna(subset=["full_hazard_percentile", "top5_trigger_percentile"]).copy()
    binned["top5_trigger_decile"] = pd.qcut(
        binned["top5_trigger_percentile"],
        q=10,
        labels=False,
        duplicates="drop",
    ).astype(int) + 1
    binned_summary = (
        binned.groupby("top5_trigger_decile", as_index=False)
        .agg(
            n_chemicals=("dtxsid", "size"),
            top5_trigger_percentile_min=("top5_trigger_percentile", "min"),
            top5_trigger_percentile_max=("top5_trigger_percentile", "max"),
            top5_trigger_percentile_median=("top5_trigger_percentile", "median"),
            full_hazard_percentile_median=("full_hazard_percentile", "median"),
        )
        .sort_values("top5_trigger_decile")
    )
    binned_summary["summary_algorithm"] = "10 equal-count bins of Top5-apical panel-min sensitivity rank; y is the median EPA-framework full-data HC5 sensitivity rank within each bin"
    metrics = pd.read_csv(root / "results/warning_hc5_bridge/top5_panel_apical_hc5_positive_control_metrics.csv")
    rho = float(metrics.loc[metrics["reference"].eq("reference_hc5_typical_log10"), "spearman_r"].iloc[0])
    return out, rho, binned_summary


def aggregate_chemical_scores(target_scores: np.ndarray, chemicals: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"dtxsid": chemicals.astype(str), "score": target_scores.astype(float)})
    return frame.groupby("dtxsid")["score"].max()


def recall_curve(scores: pd.Series, high_hazard: set[str], fractions: list[float], method: str, interval: tuple[float, float] | None = None) -> list[dict[str, object]]:
    scores = scores.dropna().sort_values(ascending=False)
    universe = set(scores.index.astype(str))
    high = set(high_hazard) & universe
    denom = max(len(high), 1)
    rows: list[dict[str, object]] = []
    for frac in fractions:
        n_select = max(1, int(math.ceil(len(scores) * frac)))
        selected = set(scores.head(n_select).index.astype(str))
        recall = len(selected & high) / denom
        rows.append(
            {
                "method": method,
                "method_label": figure3_method_label(method),
                "top_fraction": frac,
                "top_fraction_percent": frac * 100.0,
                "n_selected_chemicals": n_select,
                "n_scoreable_chemicals": len(scores),
                "n_high_hazard_total": len(high),
                "high_hazard_recall": recall,
                "interval_low": np.nan if interval is None else interval[0],
                "interval_high": np.nan if interval is None else interval[1],
            }
        )
    return rows


def build_retrieval_table(root: Path, high_hazard_quantile: float, fractions: list[float], n_random: int = 500, seed: int = 20260622) -> pd.DataFrame:
    z = np.load(root / "results/probability/species_protective_target_tail_probability_x95.npz", allow_pickle=True)
    p = z["p"]
    species = np.asarray(z["species"], dtype=object).astype(str)
    chemicals = np.asarray(z["chemicals"], dtype=object).astype(str)
    species_index = {name: idx for idx, name in enumerate(species)}
    scoreable_chemicals = set(pd.Series(chemicals.astype(str)).dropna().astype(str).unique())

    ref = pd.read_csv(root / "results/warning_hc5_bridge/top5_panel_apical_threshold_vs_reference_hc5.csv").dropna(
        subset=["reference_hc5_typical_log10"]
    )
    ref = ref[ref["dtxsid"].astype(str).isin(scoreable_chemicals)].copy()
    threshold = ref["reference_hc5_typical_log10"].quantile(high_hazard_quantile)
    high_hazard = set(ref.loc[ref["reference_hc5_typical_log10"].le(threshold), "dtxsid"].astype(str))

    candidate = pd.read_csv(root / "results/probability/candidate_universe_locked.csv")
    if "panel_eligible" in candidate.columns:
        candidate_pool = set(candidate.loc[candidate["panel_eligible"].fillna(False).astype(bool), "latin_name"].astype(str))
    else:
        candidate_pool = set(candidate["latin_name"].astype(str))
    random_candidate_idx = np.asarray([idx for idx, name in enumerate(species) if name in candidate_pool], dtype=int)
    if len(random_candidate_idx) < 5:
        random_candidate_idx = np.arange(len(species), dtype=int)
    random_candidate_names = set(species[random_candidate_idx])

    seq = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    panel_methods = ["data_driven", "EPA_WQC_taxonomic_requirements", "EPA_WET_fixed_method_species"]
    rows: list[dict[str, object]] = []
    for method in panel_methods:
        names = (
            seq[
                seq["universe"].eq("priority")
                & seq["protection_target_x"].eq(0.95)
                & seq["method"].eq(method)
                & seq["rank"].between(1, 5)
            ]
            .sort_values("rank")["latin_name"]
            .astype(str)
            .tolist()
        )
        idx = [species_index[name] for name in names if name in species_index]
        scores = aggregate_chemical_scores(q_panel(p, idx), chemicals)
        for row in recall_curve(scores, high_hazard, fractions, method):
            row["panel_species"] = " | ".join(names)
            row["high_hazard_definition"] = high_hazard_definition(high_hazard_quantile)
            row["high_hazard_tail_percent"] = float(high_hazard_quantile * 100.0)
            row["high_hazard_cutoff_label"] = high_hazard_cutoff_label(high_hazard_quantile)
            row["species_candidate_pool"] = "priority Top-5 panel-eligible candidate universe"
            row["species_candidate_pool_n"] = int(len(random_candidate_idx))
            row["curve_interpretation"] = (
                "Chemicals are ranked by each five-species panel score; x is the fraction selected for follow-up, "
                "and y is the share of EPA-framework lower-HC5 chemicals recovered. Higher y at the same x means better early follow-up ranking."
            )
            rows.append(row)

    rng = np.random.default_rng(seed)
    random_recalls: dict[float, list[float]] = {f: [] for f in fractions}
    random_scoreable_n: dict[float, int] = {f: 0 for f in fractions}
    for _ in range(n_random):
        idx = rng.choice(random_candidate_idx, size=5, replace=False)
        scores = aggregate_chemical_scores(q_panel(p, idx), chemicals)
        curve = recall_curve(scores, high_hazard, fractions, "random_panel_eligible_candidate_species")
        for row in curve:
            random_recalls[float(row["top_fraction"])].append(float(row["high_hazard_recall"]))
            random_scoreable_n[float(row["top_fraction"])] = int(row["n_scoreable_chemicals"])
    for frac, values in random_recalls.items():
        arr = np.asarray(values, dtype=float)
        n_scoreable = random_scoreable_n.get(float(frac), np.nan)
        n_selected = int(math.ceil(n_scoreable * frac)) if pd.notna(n_scoreable) else np.nan
        rows.append(
            {
                "method": "random_panel_eligible_candidate_species",
                "method_label": figure3_method_label("random_panel_eligible_candidate_species"),
                "top_fraction": frac,
                "top_fraction_percent": frac * 100.0,
                "n_selected_chemicals": n_selected,
                "n_scoreable_chemicals": n_scoreable,
                "n_high_hazard_total": len(high_hazard),
                "high_hazard_recall": float(arr.mean()),
                "interval_low": float(np.quantile(arr, 0.025)),
                "interval_high": float(np.quantile(arr, 0.975)),
                "panel_species": f"{n_random} random 5-species panels from {len(random_candidate_names)} panel-eligible candidate species",
                "high_hazard_definition": high_hazard_definition(high_hazard_quantile),
                "high_hazard_tail_percent": float(high_hazard_quantile * 100.0),
                "high_hazard_cutoff_label": high_hazard_cutoff_label(high_hazard_quantile),
                "species_candidate_pool": "priority Top-5 panel-eligible candidate universe",
                "species_candidate_pool_n": int(len(random_candidate_idx)),
                "curve_interpretation": (
                    "Chemicals are ranked by each five-species panel score; x is the fraction selected for follow-up, "
                    "and y is the share of EPA-framework lower-HC5 chemicals recovered. Higher y at the same x means better early follow-up ranking."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_panel_b(root: Path, n_random: int = 500, seed: int = 20260622) -> pd.DataFrame:
    fractions = [round(float(x), 2) for x in np.arange(0.05, 1.0001, 0.05)]
    return build_retrieval_table(root, HIGH_HAZARD_QUANTILE, fractions, n_random=n_random, seed=seed)


def normalized_early_auc(group: pd.DataFrame, fractions: list[float]) -> float:
    g = group[group["top_fraction"].isin(fractions)].sort_values("top_fraction")
    if len(g) < 2:
        return float("nan")
    x = g["top_fraction"].astype(float).to_numpy()
    y = g["high_hazard_recall"].astype(float).to_numpy()
    return float(np.trapezoid(y, x) / (x.max() - x.min()))


def build_panel_c(root: Path, n_random: int = 500, seed: int = 20260622) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quantile in HIGH_HAZARD_ROBUSTNESS_QUANTILES:
        table = build_retrieval_table(root, quantile, EARLY_RETRIEVAL_FRACTIONS, n_random=n_random, seed=seed)
        auc_rows = []
        for (method, method_label), group in table.groupby(["method", "method_label"], sort=False):
            auc_rows.append(
                {
                    "method": method,
                    "method_label": method_label,
                    "early_retrieval_auc": normalized_early_auc(group, EARLY_RETRIEVAL_FRACTIONS),
                }
            )
        auc = pd.DataFrame(auc_rows)
        top_auc = float(auc.loc[auc["method"].eq("data_driven"), "early_retrieval_auc"].iloc[0])
        n_high = int(table["n_high_hazard_total"].dropna().iloc[0])
        for row in auc.itertuples(index=False):
            if row.method == "data_driven":
                continue
            rows.append(
                {
                    "high_hazard_tail_percent": float(quantile * 100.0),
                    "high_hazard_cutoff_label": high_hazard_cutoff_label(quantile),
                    "high_hazard_definition": high_hazard_definition(quantile),
                    "n_high_hazard_total": n_high,
                    "comparator_method": row.method,
                    "comparator_label": row.method_label,
                    "optimized_early_retrieval_auc": top_auc,
                    "comparator_early_retrieval_auc": float(row.early_retrieval_auc),
                    "early_retrieval_gain": top_auc - float(row.early_retrieval_auc),
                    "mean_high_hazard_recall_gain_percent": (top_auc - float(row.early_retrieval_auc)) * 100.0,
                    "early_retrieval_gain_definition": "Mean high-hazard recall gain (%) equals 100 times fixed Top-5 panel-score area-normalized early-retrieval recall minus comparator area-normalized early-retrieval recall.",
                    "early_retrieval_window": "area-normalized recall AUC across follow-up fractions 5%, 10%, 15%, 20%, 25% and 30%",
                    "interpretation": "Positive values mean the fixed Top-5 panel score recovers more lower-HC5 chemicals than the comparator at the same early follow-up budget.",
                }
            )
    return pd.DataFrame(rows)


def build_panel_d(panel_a: pd.DataFrame) -> pd.DataFrame:
    d = panel_a.dropna(subset=["reference_hc5_typical_log10", "panel_min_apical_log10"]).copy()
    class_definition = "equal-frequency tertiles of sensitivity rank among paired comparison rows; Top5-apical uses panel_min_apical_log10 and EPA-framework full-data HC5 uses reference_hc5_typical_log10"
    d["full_hazard_class"] = pd.qcut(
        -d["reference_hc5_typical_log10"],
        q=3,
        labels=["Low", "Moderate", "High"],
        duplicates="drop",
    ).astype(str)
    d["top5_trigger_class"] = pd.qcut(
        -d["panel_min_apical_log10"],
        q=3,
        labels=["Low", "Moderate", "High"],
        duplicates="drop",
    ).astype(str)
    tab = pd.crosstab(d["full_hazard_class"], d["top5_trigger_class"]).reindex(index=["High", "Moderate", "Low"], columns=["Low", "Moderate", "High"], fill_value=0)
    row_pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    rows = []
    for full_class in tab.index:
        for top_class in tab.columns:
            rows.append(
                {
                    "full_hazard_class": full_class,
                    "top5_trigger_class": top_class,
                    "n_chemicals": int(tab.loc[full_class, top_class]),
                    "row_percent": float(row_pct.loc[full_class, top_class]),
                    "risk_class_definition": class_definition,
                }
            )
    return pd.DataFrame(rows)


def draw(root: Path, out_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    apply_style()
    panel_a, rho, panel_a_bins = build_panel_a(root)
    panel_b = build_panel_b(root)
    panel_c = build_panel_c(root)
    panel_d = build_panel_d(panel_a)

    fig = plt.figure(figsize=(7.45, 6.00))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.scatter(
        panel_a["top5_trigger_percentile"],
        panel_a["full_hazard_percentile"],
        s=11,
        color=PALETTE["neutral_light"],
        alpha=0.62,
        edgecolor="white",
        linewidth=0.18,
        label="Paired chemicals",
        zorder=2,
    )
    ax_a.plot([0, 100], [0, 100], ls="--", color=PALETTE["neutral"], lw=0.85, label="1:1 rank agreement", zorder=1)
    ax_a.plot(
        panel_a_bins["top5_trigger_percentile_median"],
        panel_a_bins["full_hazard_percentile_median"],
        marker="o",
        ms=3.1,
        lw=1.55,
        color=PALETTE["blue"],
        label="Binned median",
        zorder=4,
    )
    ax_a.text(
        4,
        96,
        f"EPA-framework paired rows\nn = {len(panel_a):,}; Spearman ρ = {rho:.3f}",
        fontsize=5.7,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.8),
    )
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("Top5-apical sensitivity rank (%)")
    ax_a.set_ylabel("EPA-framework full-data HC5 sensitivity rank (%)")
    ax_a.set_title("Sensitivity-rank agreement with EPA-framework HC5")
    ax_a.legend(fontsize=5.0, loc="lower right", handlelength=1.7)
    clean_axis(ax_a, grid=True)
    aligned_panel_label(ax_a, "a")

    method_colors = {
        "Top-5 panel score": PALETTE["blue"],
        EPA_WQC_LABEL: PALETTE["teal"],
        EPA_WET_SUBSET_LABEL: PALETTE["gold"],
        "Random 5-species": PALETTE["neutral"],
    }
    for label, group in panel_b.groupby("method_label", sort=False):
        group = group.sort_values("top_fraction_percent")
        if label == "Random 5-species":
            ax_b.fill_between(
                group["top_fraction_percent"],
                group["interval_low"],
                group["interval_high"],
                color="#B9B9B9",
                alpha=0.30,
                lw=0,
                label="Random 95% interval",
                zorder=1,
            )
            ax_b.plot(
                group["top_fraction_percent"],
                group["high_hazard_recall"],
                color=PALETTE["neutral_dark"],
                lw=1.05,
                ls="--",
                marker="o",
                ms=2.2,
                markerfacecolor="white",
                markeredgewidth=0.6,
                label="Random mean",
                zorder=2,
            )
        else:
            ax_b.plot(
                group["top_fraction_percent"],
                group["high_hazard_recall"],
                marker="o",
                ms=2.2,
                lw=1.25,
                color=method_colors.get(label, PALETTE["blue"]),
                label=figure3_display_label(label),
            )
    ax_b.set_xlim(4, 101)
    ax_b.set_ylim(0, 1.03)
    ax_b.set_xticks([5, 10, 20, 40, 60, 80, 100])
    ax_b.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_b.set_xlabel("Follow-up list size (% chemicals)")
    ax_b.set_ylabel("High-hazard recall")
    ax_b.set_title("Lower-20% EPA-HC5 retrieval")
    handles, labels = ax_b.get_legend_handles_labels()
    legend_order = [
        figure3_display_label("Top-5 panel score"),
        figure3_display_label(EPA_WQC_LABEL),
        figure3_display_label(EPA_WET_SUBSET_LABEL),
        "Random mean",
        "Random 95% interval",
    ]
    ordered = [(h, l) for target in legend_order for h, l in zip(handles, labels) if l == target]
    ax_b.legend([h for h, _ in ordered], [l for _, l in ordered], fontsize=4.8, loc="lower right")
    clean_axis(ax_b, grid=True)
    aligned_panel_label(ax_b, "b")

    comparator_order = [EPA_WQC_LABEL, EPA_WET_SUBSET_LABEL, "Random 5-species"]
    comparator_colors = {
        EPA_WQC_LABEL: PALETTE["teal"],
        EPA_WET_SUBSET_LABEL: PALETTE["gold"],
        "Random 5-species": PALETTE["neutral_dark"],
    }
    comparator_styles = {
        EPA_WQC_LABEL: "-",
        EPA_WET_SUBSET_LABEL: "-",
        "Random 5-species": "--",
    }
    direct_labels = {
        EPA_WQC_LABEL: "vs EPA WQC proxy",
        EPA_WET_SUBSET_LABEL: "vs EPA WET subset",
        "Random 5-species": "vs Random",
    }
    for label in comparator_order:
        group = panel_c[panel_c["comparator_label"].eq(label)].sort_values("high_hazard_tail_percent")
        ax_c.plot(
            group["high_hazard_tail_percent"],
            group["mean_high_hazard_recall_gain_percent"],
            marker="o",
            ms=3.2,
            lw=1.25,
            ls=comparator_styles[label],
            color=comparator_colors[label],
        )
        last = group.iloc[-1]
        ax_c.text(
            float(last["high_hazard_tail_percent"]) + 0.55,
            float(last["mean_high_hazard_recall_gain_percent"]),
            direct_labels[label],
            fontsize=5.2,
            color=comparator_colors[label],
            va="center",
            ha="left",
        )
    ax_c.axhline(0, color=PALETTE["neutral"], lw=0.8, ls="--", zorder=0)
    tick_frame = panel_c.drop_duplicates("high_hazard_tail_percent").sort_values("high_hazard_tail_percent")
    ax_c.set_xlim(3.8, 35.8)
    ax_c.set_xticks(tick_frame["high_hazard_tail_percent"].to_numpy())
    cutoff_tick_labels = [
        f"{row.high_hazard_tail_percent:.0f}%\n(n = {int(row.n_high_hazard_total)})"
        for row in tick_frame.itertuples(index=False)
    ]
    ax_c.set_xticklabels(cutoff_tick_labels)
    ax_c.set_xlabel("EPA-HC5 high-hazard cutoff")
    ax_c.set_ylabel("Mean high-hazard recall gain (%)")
    ax_c.set_title("High-hazard cutoff robustness")
    clean_axis(ax_c, grid=True)
    aligned_panel_label(ax_c, "c")

    matrix = panel_d.pivot(index="full_hazard_class", columns="top5_trigger_class", values="row_percent").reindex(index=["High", "Moderate", "Low"], columns=["Low", "Moderate", "High"])
    counts = panel_d.pivot(index="full_hazard_class", columns="top5_trigger_class", values="n_chemicals").reindex(index=["High", "Moderate", "Low"], columns=["Low", "Moderate", "High"])
    im = ax_d.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax_d.set_xticks(np.arange(3))
    ax_d.set_xticklabels(["Low\n0-33%", "Moderate\n33-67%", "High\n67-100%"])
    ax_d.set_yticks(np.arange(3))
    ax_d.set_yticklabels(["High\n67-100%", "Moderate\n33-67%", "Low\n0-33%"])
    for i in range(3):
        for j in range(3):
            value = matrix.iloc[i, j]
            n = counts.iloc[i, j]
            ax_d.text(j, i, f"{value:.0f}%\n(n={int(n)})", ha="center", va="center", fontsize=5.6, color="white" if value > 48 else PALETTE["neutral_dark"])
    ax_d.set_xlabel("Top5-apical sensitivity-rank tertile")
    ax_d.set_ylabel("EPA 1985-slot HC5 tertile")
    ax_d.set_title("Decision concordance")
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.025)
    cbar.set_label("Row %", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)
    aligned_panel_label(ax_d, "d")

    outputs = export_figure(fig, out_dir / "figure_03_top5_full_ssd_warning")
    return outputs, {
        "figure_03a_rank_percentile": panel_a,
        "figure_03a_rank_percentile_decile_medians": panel_a_bins,
        "figure_03b_capture_curve": panel_b,
        "figure_03c_cutoff_robustness": panel_c,
        "figure_03d_concordance_heatmap": panel_d,
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
            "Core conclusion: Measured Top5-apical ranks track EPA-framework full-data HC5, while fixed Top-5 probability scores enrich lower-HC5 chemicals early in follow-up lists.",
            "Archetype: quantitative grid.",
            "Panel map: a all paired Top5-apical panel-min/EPA-framework full-data HC5 sensitivity ranks with binned median synchrony; b follow-up-list curves showing recovery of EPA-HC5 lower-20% chemicals from fixed Top-5 measured-tail probability rankings; c cutoff-robustness early-retrieval gain across lower-5%, lower-10%, lower-20% and lower-30% EPA-HC5 definitions; d sensitivity-rank tertile decision-concordance heatmap.",
            "Reviewer risk: Panels a and d use 115 paired measured-threshold rows; panels b and c use fixed Top-5 measured-tail probability scores across 639 chemicals. Follow-up list size denotes the number of chemicals selected.",
            f"Manuscript caption: {MANUSCRIPT_CAPTION}",
        ],
    )
    write_manifest(
        out_dir,
        figure_id=FIGURE_ID,
        title=TITLE,
        manuscript_caption=MANUSCRIPT_CAPTION,
        core_conclusion="Measured Top5-apical ranks track EPA-framework full-data HC5, while fixed Top-5 probability scores enrich lower-HC5 chemicals early in follow-up lists.",
        archetype="quantitative grid",
        panel_map={
            "a": "Sensitivity-rank synchrony between Top5-apical panel minimum and EPA-framework full-data typical HC5, showing all paired comparison rows and 10 equal-count binned medians.",
            "b": "Follow-up-list curves for the fixed Top-5 panel score, EPA WQC proxy (matched k = 5), EPA WET subset (matched k = 5) and seeded random 5-species panels; x is follow-up list size and y is recall of EPA-framework lower-20% HC5 chemicals, so higher curves indicate better early retrieval under the same follow-up budget.",
            "c": "High-hazard cutoff robustness: mean high-hazard recall gain (%) over each comparator, calculated as 100 times fixed Top-5 panel-score area-normalized recall minus comparator area-normalized recall across 5-30% follow-up for EPA-HC5 lower-5%, lower-10%, lower-20% and lower-30% definitions.",
            "d": "Decision-concordance heatmap using equal-frequency sensitivity-rank tertiles for Top5-apical panel minimum and EPA-framework full-data typical HC5.",
        },
        source_tables=source_tables,
        input_dependencies=[
            "results/warning_hc5_bridge/top5_panel_apical_threshold_vs_reference_hc5.csv",
            "results/warning_hc5_bridge/top5_panel_apical_hc5_positive_control_metrics.csv",
            "results/probability/species_protective_target_tail_probability_x95.npz",
            "results/panels/national_panel_sequences.csv",
        ],
        outputs=outputs,
        reviewer_risk="Panels a and d use all eligible paired Top5-apical panel-min/EPA-framework HC5 comparison rows. Panels b and c use fixed Top-5 probability scores across 639 chemicals, and follow-up list size is the fraction of chemicals selected.",
        notes=[
            "Stage 16 now requires EPA_1985_8_slots context feasibility before fitting a reference HC5: salmonid, non-salmonid fish, chordate, planktonic crustacean, benthic crustacean, insect, other phylum and one additional distinct-family slot.",
            "The EPA-framework reference HC5 remains fitted from all measured protective species within each eligible context, not from the eight slot representatives only.",
            "After applying the EPA-framework context gate, stage 16 creates 201 eligible reference HC5 contexts, 115 chemical-medium reference rows and 115 paired Top5-apical comparison rows.",
            "Panel A uses Top5-apical panel-min sensitivity rank on x and EPA-framework full-data typical HC5 sensitivity rank on y; both are rank(-log10 concentration, pct=True) * 100 within the paired comparison rows.",
            "Panel B ranks chemicals with the fixed Top-5 measured-tail probability score and measures how quickly that ranking recovers EPA-framework lower-20% HC5 chemicals.",
            "Panel B defines high-hazard chemicals as the lower 20% of EPA-framework reference_hc5_typical_log10 among the 115 EPA-labeled comparison rows, yielding 23 positives embedded in the 639-chemical fixed Top-5 probability-score ranking universe; unlabeled scoreable chemicals can occupy follow-up-list positions but do not define positives.",
            "Panel C recomputes mean high-hazard recall gain (%) across lower-5%, lower-10%, lower-20% and lower-30% EPA-HC5 definitions; gain is 100 times fixed Top-5 panel-score area-normalized recall minus comparator area-normalized recall over 5-30% follow-up.",
            "Panel D high, moderate and low classes are equal-frequency tertiles of the corresponding sensitivity ranks.",
            "The EPA WET subset (matched k = 5) line is sourced from the first five species in the EPA_WET_fixed_method_species sequence; all five displayed species are flagged official_or_common_method_species in the candidate universe, matching the Figure 1A comparator wording.",
            "Random capture curve uses 500 random 5-species panels drawn from the same panel-eligible candidate species universe used by the priority Top-5 sequence, with seed 20260622.",
        ],
    )
    print({"figure": FIGURE_ID, "outputs": outputs, "source_tables": source_tables})


if __name__ == "__main__":
    main()
