#!/usr/bin/env python3
"""Render R1 Figure 5 from AP04 localization and strict-LOO testing priorities."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import shapefile


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[3]
ANALYSIS_ROOT = PACKAGE_ROOT / "results" / "revision_r1"
BHBT_ROOT = PACKAGE_ROOT
AP02_DIR = ANALYSIS_ROOT / "02_ap02_strict_loo"
AP04_DIR = ANALYSIS_ROOT / "06_ap04_localization_sensitivity"
OUT_DIR = ANALYSIS_ROOT / "08_r1_exhibits" / "main_figure_05"
BOUNDARY_ZIP = BHBT_ROOT / "data" / "external" / "cb_2024_us_state_500k.zip"

sys.path.insert(0, str(PACKAGE_ROOT / "code" / "figures"))
from plot_common import PALETTE, apply_style, clean_axis, export_figure, short_species  # noqa: E402


FIGURE_BASENAME = "figure_05_r1_localization_testing_priorities"
CONTIGUOUS_XLIM = (-125.0, -66.6)
CONTIGUOUS_YLIM = (24.3, 49.5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_positive(values: pd.Series | np.ndarray, floor: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, floor, None)
    maximum = float(array.max()) if array.size else 0.0
    return array / maximum if maximum > 0 else np.zeros_like(array)


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


def load_ap04_state_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(AP04_DIR / "ap04_state_support_and_fallback_ledger.csv")
    localized = ledger[ledger["localized"].astype(bool)].copy()
    fallback = ledger[~ledger["localized"].astype(bool)].copy()
    localized["localized_capture_percent"] = localized["baseline_expected_weighted_joint_coverage_x95"] * 100.0
    localized["national_capture_percent"] = localized["national_top5_expected_weighted_joint_coverage_x95"] * 100.0
    localized["gain_percent_points"] = localized["localization_gain_over_national_top5_pp"]
    localized["panel_species"] = localized["baseline_panel_species"].str.split("; ")
    rows: list[dict[str, object]] = []
    for row in localized.itertuples(index=False):
        for rank, latin_name in enumerate(row.panel_species, start=1):
            rows.append(
                {
                    "state_code": str(row.state_code),
                    "rank": rank,
                    "latin_name": latin_name,
                    "species_label": short_species(latin_name),
                    "gain_percent_points": float(row.gain_percent_points),
                }
            )
    sequences = pd.DataFrame(rows)
    if len(localized) != 32 or len(sequences) != 160 or len(fallback) != 3:
        raise AssertionError(
            f"AP04 state contract mismatch: localized={len(localized)}, sequence rows={len(sequences)}, fallback={len(fallback)}"
        )
    return localized, fallback, sequences


def build_strict_loo_testing_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = np.load(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz", allow_pickle=True)
    probability = np.asarray(z["p"], dtype=float)
    reliability = np.asarray(z["reliability"], dtype=float)
    direct_n = np.asarray(z["direct_n"])
    species = np.asarray(z["species"], dtype=object).astype(str)
    chemicals = np.asarray(z["chemicals"], dtype=object).astype(str)
    chemical_index = {name: idx for idx, name in enumerate(chemicals)}

    priority = pd.read_csv(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv")
    priority["DTXSID"] = priority["DTXSID"].astype(str)
    pchems = [name for name in priority["DTXSID"] if name in chemical_index]
    pidx = np.asarray([chemical_index[name] for name in pchems], dtype=int)
    weights = priority.set_index("DTXSID")["national_weight"].reindex(pchems).fillna(0).to_numpy(float)
    weights /= weights.sum()
    pp = probability[:, pidx]
    rr = reliability[:, pidx]
    nn = direct_n[:, pidx]

    candidate = pd.read_csv(BHBT_ROOT / "results" / "probability" / "candidate_universe_locked.csv")
    cmeta = candidate.set_index("latin_name").reindex(species)
    occurrence_breadth = cmeta["eligible_states"].fillna(0).to_numpy(float)
    occurrence_breadth /= max(float(np.nanmax(occurrence_breadth)), 1.0)
    uncertainty = (pp * (1.0 - pp)) @ weights
    under_tested = 1.0 / np.sqrt(1.0 + (nn > 0).sum(axis=1))
    mechanism_gap = ((nn == 0) * pp).mean(axis=1)
    species_score = uncertainty * under_tested * (0.25 + 0.75 * occurrence_breadth) * (0.5 + 0.5 * mechanism_gap)
    species_priority = pd.DataFrame(
        {
            "latin_name": species,
            "species_testing_priority": species_score,
            "weighted_probability_uncertainty": uncertainty,
            "under_tested_score": under_tested,
            "occurrence_breadth_score": occurrence_breadth,
            "moa_probability_gap": mechanism_gap,
            "n_direct_priority_chemicals": (nn > 0).sum(axis=1),
            "n_observed_contexts": cmeta["n_contexts"].to_numpy(),
            "n_observed_chemicals": cmeta["n_chemicals"].to_numpy(),
        }
    ).sort_values("species_testing_priority", ascending=False)

    chemistry = pd.read_csv(BHBT_ROOT / "results" / "chemistry" / "chemical_moa_form_master.csv.gz", low_memory=False)
    chemistry["DTXSID"] = chemistry["DTXSID"].astype(str)
    chem = chemistry.set_index("DTXSID").reindex(pchems)
    priority_meta = priority.set_index("DTXSID").reindex(pchems)
    n_species_direct = (nn > 0).sum(axis=0)
    species_gap = 1.0 / np.sqrt(1.0 + n_species_direct)
    mean_uncertainty = (pp * (1.0 - pp)).mean(axis=0)
    primary_moa = chem["primary_moa"].fillna("MOA_UNRESOLVED").astype(str)
    moa_terms = primary_moa.map(readable_moa_term)
    moa_support = pd.Series(n_species_direct, index=pchems).groupby(moa_terms.values).transform("sum").to_numpy()
    moa_gap = 1.0 / np.sqrt(1.0 + moa_support)
    chemical_form = chem["chemical_form_class"].fillna("unresolved chemical form").astype(str)
    form_support = pd.Series(n_species_direct, index=pchems).groupby(chemical_form.values).transform("sum").to_numpy()
    form_gap = 1.0 / np.sqrt(1.0 + form_support)
    gap_multiplier = 0.5 + 0.25 * normalized_positive(moa_gap) + 0.25 * normalized_positive(form_gap)
    chemical_score = weights * species_gap * (0.5 + 0.5 * mean_uncertainty / 0.25) * gap_multiplier
    chemical_priority = pd.DataFrame(
        {
            "DTXSID": pchems,
            "preferred_name": priority_meta["PREFERRED_NAME"].to_numpy(),
            "national_weight": weights,
            "chemical_testing_priority": chemical_score,
            "n_candidate_species_with_direct_data": n_species_direct,
            "species_data_gap": species_gap,
            "mean_probability_uncertainty": mean_uncertainty,
            "primary_moa": primary_moa.to_numpy(),
            "moa_term": moa_terms.to_numpy(),
            "chemical_form_class": chemical_form.to_numpy(),
            "moa_group_gap": moa_gap,
            "chemical_form_gap": form_gap,
            "mechanism_form_gap_multiplier": gap_multiplier,
        }
    ).sort_values("chemical_testing_priority", ascending=False)
    chemical_priority["chemical_testing_priority_norm"] = (
        chemical_priority["chemical_testing_priority"] / chemical_priority["chemical_testing_priority"].max()
    )

    name_map = chemistry.set_index("DTXSID")["PREFERRED_NAME"].to_dict()
    moa_map = chemistry.set_index("DTXSID")["primary_moa"].to_dict()
    weight_map = dict(zip(pchems, weights))
    candidate_moa_map = dict(zip(chemical_priority["DTXSID"], chemical_priority["moa_term"]))
    form_map = dict(zip(chemical_priority["DTXSID"], chemical_priority["chemical_form_class"]))
    gap_map = dict(zip(chemical_priority["DTXSID"], chemical_priority["mechanism_form_gap_multiplier"]))
    rows: list[dict[str, object]] = []
    for chemical in chemical_priority.head(200)["DTXSID"].astype(str):
        col = chemical_index[chemical]
        missing = np.where(direct_n[:, col] == 0)[0]
        for row_idx in missing:
            value = (
                weight_map.get(chemical, 0.0)
                * probability[row_idx, col]
                * (1.0 - probability[row_idx, col])
                * (1.0 - reliability[row_idx, col])
                * gap_map.get(chemical, 0.5)
            )
            rows.append(
                {
                    "latin_name": species[row_idx],
                    "DTXSID": chemical,
                    "preferred_name": name_map.get(chemical, ""),
                    "primary_moa": moa_map.get(chemical, "MOA_UNRESOLVED"),
                    "moa_term": candidate_moa_map.get(chemical, "Unassigned"),
                    "chemical_form_class": form_map.get(chemical, "unresolved chemical form"),
                    "mechanism_form_gap_multiplier": gap_map.get(chemical, 0.5),
                    "pair_testing_priority": float(value),
                    "tail_probability": float(probability[row_idx, col]),
                    "reliability": float(reliability[row_idx, col]),
                    "testing_scope": "national",
                    "probability_source": "AP02_STRICT_LOO",
                }
            )
    pairs = pd.DataFrame(rows)
    pairs = pairs[~pairs["moa_term"].str.contains("Unassigned|UNRESOLVED", case=False, na=False)].copy()
    pairs = pairs.sort_values("pair_testing_priority", ascending=False)

    species_totals = (
        pairs.groupby("latin_name", as_index=False)
        .agg(species_total_testing_priority=("pair_testing_priority", "sum"), species_total_missing_pairs=("DTXSID", "size"))
        .sort_values(["species_total_testing_priority", "species_total_missing_pairs", "latin_name"], ascending=[False, False, True])
    )
    moa_totals = (
        pairs.groupby("moa_term", as_index=False)
        .agg(candidate_moa_total_testing_priority=("pair_testing_priority", "sum"), candidate_moa_total_missing_pairs=("DTXSID", "size"))
        .sort_values(["candidate_moa_total_testing_priority", "candidate_moa_total_missing_pairs", "moa_term"], ascending=[False, False, True])
    )
    top_species = species_totals.head(10)["latin_name"].astype(str).tolist()
    top_moa = moa_totals.head(8)["moa_term"].astype(str).tolist()
    chemical_counts = chemical_priority.groupby("moa_term")["DTXSID"].nunique().to_dict()
    selected = pairs[pairs["latin_name"].isin(top_species) & pairs["moa_term"].isin(top_moa)].copy()
    summary = (
        selected.groupby(["latin_name", "moa_term"], as_index=False)
        .agg(
            n_missing_pairs=("DTXSID", "size"),
            national_testing_priority_score=("pair_testing_priority", "sum"),
            mean_tail_probability=("tail_probability", "mean"),
            mean_reliability=("reliability", "mean"),
        )
    )
    full = pd.MultiIndex.from_product([top_species, top_moa], names=["latin_name", "moa_term"]).to_frame(index=False)
    summary = full.merge(summary, on=["latin_name", "moa_term"], how="left")
    for column in ["n_missing_pairs", "national_testing_priority_score", "mean_tail_probability", "mean_reliability"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    summary["n_priority_chemicals_in_moa"] = summary["moa_term"].map(chemical_counts).fillna(0).astype(int)
    summary["missingness_density"] = summary["n_missing_pairs"] / summary["n_priority_chemicals_in_moa"].replace(0, np.nan)
    summary["missingness_density"] = summary["missingness_density"].fillna(0.0)
    summary = summary.merge(species_priority[["latin_name", "species_testing_priority"]], on="latin_name", how="left")
    summary = summary.merge(species_totals[["latin_name", "species_total_testing_priority"]], on="latin_name", how="left")
    summary = summary.merge(moa_totals[["moa_term", "candidate_moa_total_testing_priority"]], on="moa_term", how="left")
    summary["candidate_moa_term"] = summary["moa_term"]
    summary["testing_scope"] = "national"
    summary["score_aggregation"] = "sum of national species-chemical pair priorities across missing direct-evidence chemicals"
    summary["score_components"] = "national weight x probability uncertainty x evidence uncertainty x candidate-MOA/chemical-form gap"
    summary["probability_source"] = "AP02_STRICT_LOO"

    if len(summary) != 80 or summary["latin_name"].nunique() != 10 or summary["moa_term"].nunique() != 8:
        raise AssertionError("Strict-LOO national testing matrix is not a complete 10 x 8 grid")
    return summary, species_priority, chemical_priority, pairs.head(8000).copy()


def load_boundary_parts() -> list[dict[str, object]]:
    if not BOUNDARY_ZIP.exists():
        raise FileNotFoundError(BOUNDARY_ZIP)
    reader = shapefile.Reader(str(BOUNDARY_ZIP))
    parts: list[dict[str, object]] = []
    for shape_record in reader.shapeRecords():
        props = shape_record.record.as_dict()
        points = np.asarray(shape_record.shape.points, dtype=float)
        breaks = list(shape_record.shape.parts) + [len(points)]
        for start, stop in zip(breaks[:-1], breaks[1:]):
            coords = points[start:stop]
            if len(coords) >= 3:
                parts.append({"state_code": str(props.get("STUSPS")), "coords": coords})
    return parts


def polygon_area_centroid(coords: np.ndarray) -> tuple[float, float, float]:
    coords = np.asarray(coords, dtype=float)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    x = coords[:, 0]
    y = coords[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * float(cross.sum())
    if abs(signed_area) < 1e-9:
        return 0.0, float(np.nanmean(x[:-1])), float(np.nanmean(y[:-1]))
    cx = float(((x[:-1] + x[1:]) * cross).sum() / (6.0 * signed_area))
    cy = float(((y[:-1] + y[1:]) * cross).sum() / (6.0 * signed_area))
    return abs(signed_area), cx, cy


def state_centroids(parts: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    accumulator: dict[str, list[float]] = {}
    for part in parts:
        coords = np.asarray(part["coords"], dtype=float)
        if coords[:, 0].max() < CONTIGUOUS_XLIM[0] or coords[:, 0].min() > CONTIGUOUS_XLIM[1]:
            continue
        if coords[:, 1].max() < CONTIGUOUS_YLIM[0] or coords[:, 1].min() > CONTIGUOUS_YLIM[1]:
            continue
        area, cx, cy = polygon_area_centroid(coords)
        if area <= 0 or not np.isfinite(cx) or not np.isfinite(cy):
            continue
        values = accumulator.setdefault(str(part["state_code"]), [0.0, 0.0, 0.0])
        values[0] += area * cx
        values[1] += area * cy
        values[2] += area
    return {key: (sx / area, sy / area) for key, (sx, sy, area) in accumulator.items() if area > 0}


def build_leading_species_map_data(sequences: pd.DataFrame) -> pd.DataFrame:
    """Create the rank-1 fill/rank-2 point ledger used by the R0-style map."""

    rank1 = sequences.loc[sequences["rank"].eq(1), ["state_code", "latin_name"]].rename(
        columns={"latin_name": "rank1_latin_name"}
    )
    rank2 = sequences.loc[sequences["rank"].eq(2), ["state_code", "latin_name"]].rename(
        columns={"latin_name": "rank2_latin_name"}
    )
    out = rank1.merge(rank2, on="state_code", how="left")
    counts = pd.concat([out["rank1_latin_name"], out["rank2_latin_name"]]).dropna().value_counts()
    top_species = set(counts.head(9).index)
    for rank in (1, 2):
        latin = f"rank{rank}_latin_name"
        group = f"rank{rank}_species_group"
        label = f"rank{rank}_species_label"
        out[group] = np.where(out[latin].isin(top_species), out[latin], "Other")
        out[label] = out[group].map(lambda value: "Other" if value == "Other" else short_species(value))
    return out.sort_values("state_code").reset_index(drop=True)


def aligned_panel_label(ax: plt.Axes, label: str, x_offset: int = -27, y_offset: int = 16) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=11,
        clip_on=False,
    )


def draw_map(ax: plt.Axes, map_data: pd.DataFrame, fallback: pd.DataFrame, parts: list[dict[str, object]]) -> None:
    rank1_by_state = map_data.set_index("state_code")["rank1_species_label"].to_dict()
    rank1_counts = map_data["rank1_species_label"].value_counts()
    rank2_counts = map_data["rank2_species_label"].value_counts()
    species_labels = sorted(
        (set(rank1_counts.index) | set(rank2_counts.index)) - {"Other"},
        key=lambda label: (-int(rank1_counts.get(label, 0)), -int(rank1_counts.get(label, 0) + rank2_counts.get(label, 0)), label),
    )
    groups = [*species_labels]
    if "Other" in set(map_data["rank1_species_label"]) | set(map_data["rank2_species_label"]):
        groups.append("Other")
    categorical_colors = [
        "#D00000", "#46237A", "#1B998B", "#BA63FF", "#3185FC",
        "#FFBA08", "#FF7B9C", "#FF9B85", "#8FE388", "#CBFF8C",
    ]
    color_map = {group: categorical_colors[index % len(categorical_colors)] for index, group in enumerate(groups)}
    fallback_states = set(fallback["state_code"].astype(str))
    for part in parts:
        code = str(part["state_code"])
        coords = np.asarray(part["coords"], dtype=float)
        if coords[:, 0].max() < CONTIGUOUS_XLIM[0] or coords[:, 0].min() > CONTIGUOUS_XLIM[1]:
            continue
        if coords[:, 1].max() < CONTIGUOUS_YLIM[0] or coords[:, 1].min() > CONTIGUOUS_YLIM[1]:
            continue
        if code in rank1_by_state:
            face = color_map.get(rank1_by_state[code], color_map.get("Other", "#BDBDBD"))
            hatch = None
            edge = "white"
            linewidth = 0.75
        elif code in fallback_states:
            face = "#F2F2F2"
            hatch = "////"
            edge = PALETTE["neutral_dark"]
            linewidth = 0.55
        else:
            face = "#FAFAFA"
            hatch = None
            edge = "#BDBDBD"
            linewidth = 0.35
        ax.fill(coords[:, 0], coords[:, 1], facecolor=face, edgecolor=edge, linewidth=linewidth, hatch=hatch, zorder=1)

    centroids = state_centroids(parts)
    # Keep the R0 convention of separating the state code from the rank-2
    # point.  The denser central/eastern states use alternating offsets, while
    # the three smallest evaluated states are called out into the Atlantic
    # margin with straight leaders.
    offsets = {
        "AL": (-0.72, 0.72), "FL": (-0.20, 0.72), "GA": (0.76, -0.68),
        "IL": (-0.72, 0.76), "IN": (0.78, -0.72), "KY": (1.30, 0.48),
        "MS": (0.70, -0.64), "NC": (0.82, -0.76), "OH": (0.82, 0.72),
        "SC": (0.82, -0.66), "VA": (1.05, -0.12), "WV": (-0.52, 0.70),
    }
    leader_positions = {
        "CT": (-68.95, 42.45),
        "NJ": (-69.25, 40.35),
        "MD": (-69.85, 38.55),
    }

    map_lookup = map_data.set_index("state_code")
    for code in sorted(set(map_data["state_code"].astype(str)) | fallback_states):
        if code not in centroids:
            continue
        x, y = centroids[code]
        dx, dy = offsets.get(code, (0.0, 0.0))
        if code in map_lookup.index:
            row = map_lookup.loc[code]
            point_color = color_map.get(str(row["rank2_species_label"]), color_map.get("Other", "#BDBDBD"))
            ax.scatter(x, y, s=34, marker="o", color=point_color, edgecolor=point_color, linewidth=0.0, zorder=4)
        if code in leader_positions:
            label_x, label_y = leader_positions[code]
            ax.annotate(
                code,
                xy=(x, y),
                xytext=(label_x, label_y),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=5.2,
                fontweight="bold",
                color=PALETTE["neutral_dark"],
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#555555",
                    "linewidth": 0.55,
                    "shrinkA": 1.0,
                    "shrinkB": 2.0,
                    "connectionstyle": "arc3,rad=0",
                },
                zorder=6,
                clip_on=False,
            )
        else:
            label = ax.text(
                x + dx,
                y + dy + (0.66 if code not in offsets else 0.0),
                code,
                ha="center",
                va="center",
                fontsize=5.2,
                fontweight="bold",
                color="#171717",
                zorder=6,
            )
            label.set_path_effects([pe.withStroke(linewidth=1.15, foreground="white", alpha=0.96)])
    ax.set_xlim(*CONTIGUOUS_XLIM)
    ax.set_ylim(*CONTIGUOUS_YLIM)
    ax.set_aspect(1.0, adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Leading species in localized Top-5 panels", loc="left", fontsize=8.4, pad=2)
    handles = [mpl.patches.Patch(facecolor=color_map[label], edgecolor="white", label=label) for label in groups]
    handles.extend(
        [
            mpl.patches.Patch(facecolor="#F2F2F2", edgecolor=PALETTE["neutral_dark"], hatch="////", label="National fallback"),
            mpl.patches.Patch(facecolor="#FAFAFA", edgecolor="#BDBDBD", label="Not evaluated"),
        ]
    )
    legend = ax.legend(
        handles=handles,
        title="Fill = rank 1 species; point = rank 2 species",
        fontsize=5.2,
        title_fontsize=5.7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=6,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=0.75,
        labelspacing=0.38,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        if text.get_text() not in {"Other", "National fallback", "Not evaluated"}:
            text.set_fontstyle("italic")


def draw(
    localized: pd.DataFrame,
    fallback: pd.DataFrame,
    sequences: pd.DataFrame,
    testing: pd.DataFrame,
    parts: list[dict[str, object]],
) -> list[str]:
    apply_style(font_size=7.0)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.25, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 0.94], hspace=0.24)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.78, 0.72], wspace=0.12)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.96, 1.04], wspace=0.28)
    ax_map = fig.add_subplot(top[0, 0])
    ax_gain = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    map_data = build_leading_species_map_data(sequences)
    draw_map(ax_map, map_data, fallback, parts)

    ordered = localized.sort_values(
        ["gain_percent_points", "state_code"], ascending=[False, True]
    ).reset_index(drop=True)
    y_gain = np.arange(len(ordered))
    top_gain_states = set(ordered.head(3)["state_code"].astype(str))
    for y, row in zip(y_gain, ordered.itertuples(index=False)):
        highlighted = str(row.state_code) in top_gain_states
        ax_gain.plot(
            [row.national_capture_percent, row.localized_capture_percent],
            [y, y],
            color=PALETTE["blue"] if highlighted else "#9AB9D4",
            linewidth=1.0 if highlighted else 0.55,
            alpha=0.95 if highlighted else 0.72,
            zorder=2,
        )
    ax_gain.scatter(
        ordered["national_capture_percent"],
        y_gain,
        s=12,
        facecolor="white",
        edgecolor="#858585",
        linewidth=0.55,
        label="National Top-5",
        zorder=4,
    )
    ax_gain.scatter(
        ordered["localized_capture_percent"],
        y_gain,
        s=15,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.30,
        label="Localized Top-5",
        zorder=5,
    )
    ax_gain.set_yticks(y_gain)
    ax_gain.set_yticklabels(ordered["state_code"].astype(str), fontsize=5.1)
    for label in ax_gain.get_yticklabels():
        if label.get_text() in top_gain_states:
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue"])
    y_by_state = dict(zip(ordered["state_code"].astype(str), y_gain))
    for row in ordered.head(3).itertuples(index=False):
        ax_gain.text(
            float(row.localized_capture_percent) + 0.38,
            y_by_state[str(row.state_code)],
            f"{row.state_code} {row.gain_percent_points:+.2f} pp",
            ha="left",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=PALETTE["blue"],
            clip_on=False,
        )
    ax_gain.text(
        0.02,
        1.005,
        (
            f"Mean {ordered['national_capture_percent'].mean():.2f}% → "
            f"{ordered['localized_capture_percent'].mean():.2f}%  "
            f"(Δ {ordered['gain_percent_points'].mean():+.2f} pp)"
        ),
        transform=ax_gain.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=PALETTE["neutral_dark"],
    )
    ax_gain.set_xlabel("Expected capture of state-priority chemicals (%)", fontsize=6.0)
    ax_gain.set_title("Expected-capture gains from regional weighting", loc="left", fontsize=8.0, pad=11)
    ax_gain.set_xlim(0.0, max(24.0, float(ordered["localized_capture_percent"].max()) + 4.3))
    ax_gain.set_xticks([0, 5, 10, 15, 20])
    # The extra lower strip is reserved for the two-series key; no legend may
    # cover a state comparison point.
    ax_gain.set_ylim(len(ordered) + 2.55, -0.85)
    ax_gain.tick_params(axis="x", labelsize=5.3)
    ax_gain.tick_params(axis="y", length=0, pad=1.5)
    ax_gain.legend(
        loc="lower center",
        bbox_to_anchor=(0.54, 0.006),
        ncol=2,
        fontsize=5.0,
        handletextpad=0.35,
        labelspacing=0.25,
        columnspacing=0.75,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="none",
    )
    clean_axis(ax_gain, grid=False)
    ax_gain.xaxis.grid(True, color="#E8E8E8", lw=0.40, zorder=0)

    species_frequency = sequences.groupby("latin_name")["state_code"].nunique().sort_values(ascending=False)
    top_species = species_frequency.head(15).index.tolist()
    selected = sequences[sequences["latin_name"].isin(top_species)].copy()
    species_order = species_frequency.loc[top_species].sort_values(ascending=True).index.tolist()
    rank_colors = {1: "#08306B", 2: "#2171B5", 3: "#6BAED6", 4: "#9ECAE1", 5: "#C6DBEF"}
    max_count = max(int(selected.groupby(["latin_name", "rank"]).size().max()), 1)
    for y_idx, species_name in enumerate(species_order):
        group = selected[selected["latin_name"].eq(species_name)].groupby("rank").size()
        for rank, count in group.items():
            size = 18 + 95 * math.sqrt(int(count) / max_count)
            ax_c.scatter(rank, y_idx, s=size, color=rank_colors[int(rank)], edgecolor="white", linewidth=0.4)
            if count >= 3:
                ax_c.text(rank, y_idx, str(int(count)), ha="center", va="center", fontsize=4.9, color="white" if int(rank) <= 2 else PALETTE["neutral_dark"])
    ax_c.set_xticks(range(1, 6))
    ax_c.set_xlabel("Rank within localized Top-5")
    ax_c.set_yticks(np.arange(len(species_order)))
    ax_c.set_yticklabels([short_species(name) for name in species_order], fontsize=5.9, fontstyle="italic")
    ax_c.set_xlim(0.5, 5.5)
    # Reserve a two-line annotation/key band above the highest species row so
    # neither the scope note nor the size key can overlay a plotted point.
    ax_c.set_ylim(-0.6, len(species_order) + 2.2)
    ax_c.set_title("Selection frequency and rank", loc="left", fontsize=8.2)
    ax_c.text(
        0.0,
        0.986,
        "Top 15 of 35 species; complete panels in Table S6",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color=PALETTE["neutral_dark"],
        clip_on=True,
    )
    legend_counts = sorted(set([1, max(2, int(round(max_count / 2))), max_count]))
    handles = [ax_c.scatter([], [], s=18 + 95 * math.sqrt(value / max_count), color="#6BAED6", edgecolor="white", linewidth=0.4) for value in legend_counts]
    ax_c.legend(
        handles,
        [str(value) for value in legend_counts],
        title="States represented",
        fontsize=5.1,
        title_fontsize=5.3,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.925),
        ncol=3,
        handletextpad=0.18,
        columnspacing=0.52,
        labelspacing=0.20,
        borderaxespad=0.0,
        frameon=False,
    )
    clean_axis(ax_c, grid=False)
    ax_c.xaxis.grid(True, color="#E8E8E8", lw=0.45, zorder=0)
    aligned_panel_label(ax_c, "c")

    species_order_c = testing.groupby("latin_name")["national_testing_priority_score"].sum().sort_values(ascending=False).index.tolist()
    moa_order = testing.groupby("candidate_moa_term")["national_testing_priority_score"].sum().sort_values(ascending=False).index.tolist()
    matrix = testing.pivot(index="latin_name", columns="candidate_moa_term", values="national_testing_priority_score").reindex(index=species_order_c, columns=moa_order).fillna(0.0)
    display = matrix.to_numpy(float) * 1000.0
    cmap = mpl.colors.LinearSegmentedColormap.from_list("priority_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0, vmax=max(float(display.max()), 1e-9))
    image = ax_d.imshow(display, aspect="auto", cmap=cmap, norm=norm)
    ax_d.set_xticks(np.arange(len(moa_order)))
    ax_d.set_xticklabels([textwrap.fill(value, width=15, break_long_words=False) for value in moa_order], rotation=43, ha="right", fontsize=5.5)
    ax_d.set_yticks(np.arange(len(species_order_c)))
    ax_d.set_yticklabels([short_species(name) for name in species_order_c], fontsize=5.9, fontstyle="italic")
    ax_d.set_title("National evidence-acquisition priorities", loc="left", fontsize=8.2)
    ax_d.set_xlabel("Candidate mode-of-action group")
    ax_d.set_xticks(np.arange(-0.5, len(moa_order), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(species_order_c), 1), minor=True)
    ax_d.grid(which="minor", color="white", lw=0.45)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.025)
    colorbar.set_label(r"Priority score ($\times 10^3$)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.3)
    aligned_panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.075, right=0.97, top=0.955, bottom=0.08)
    # Use one figure-level baseline for the two top-row panel labels.  The map
    # keeps a fixed geographic aspect ratio, so axes-relative labels otherwise
    # land at different heights even though the subplot cells are aligned.
    top_label_y = 0.982
    for ax, label, x_pad in ((ax_map, "a", 0.048), (ax_gain, "b", 0.036)):
        position = ax.get_position()
        fig.text(
            max(0.008, position.x0 - x_pad),
            top_label_y,
            label,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )
    return export_figure(fig, OUT_DIR / FIGURE_BASENAME)


def write_outputs(
    outputs: list[str],
    localized: pd.DataFrame,
    fallback: pd.DataFrame,
    sequences: pd.DataFrame,
    testing: pd.DataFrame,
    species_priority: pd.DataFrame,
    chemical_priority: pd.DataFrame,
    top_pairs: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    map_data = build_leading_species_map_data(sequences)
    ordered_gain = localized.sort_values(
        ["gain_percent_points", "state_code"], ascending=[False, True]
    ).reset_index(drop=True)
    tables = {
        "figure_05a_state_localization_gain.csv": localized.drop(columns=["panel_species"], errors="ignore"),
        "figure_05a_fallback_states.csv": fallback,
        "figure_05a_leading_species.csv": map_data,
        "figure_05b_ordered_state_gain.csv": ordered_gain.drop(columns=["panel_species"], errors="ignore"),
        "figure_05c_localized_top5_sequences.csv": sequences,
        "figure_05d_strict_loo_national_species_moa_matrix.csv": testing,
        "figure_05d_strict_loo_species_priorities.csv": species_priority,
        "figure_05d_strict_loo_chemical_priorities.csv": chemical_priority,
        "figure_05d_strict_loo_top8000_pairs.csv.gz": top_pairs,
    }
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False, compression="gzip" if name.endswith(".gz") else None)

    caption = (
        "Figure 5. Regional weighting identifies localized sentinel panels and evidence-acquisition priorities. "
        "a, Geographic distribution of the leading species in localized Top-5 panels. State fill identifies the rank-1 species and the superimposed point identifies the rank-2 species; less frequent species are grouped as Other. Hatched states lacked the prespecified minimum state chemical support and retained the national fallback. State-level expected-capture gains are reported in Figure S7 and Table S6. "
        "b, Paired expected capture of the fixed national Top-5 and each localized Top-5 under the same state-priority chemical and soft species-relevance weights, ordered by localization gain. All 32 supported states are shown; labels identify the three largest gains. "
        "c, Selection frequency by within-panel rank for the 15 species most often included across the 32 localized panels. Numbers are state counts for cells with at least three selections; complete five-species sequences are reported in Table S6. "
        "d, National species-by-candidate-MOA testing-priority matrix recalculated from strict focal-species leave-one-out probabilities. The unnormalized score sums national chemical weight, probability uncertainty, evidence-support uncertainty, missing direct species-chemical evidence, and candidate-MOA/chemical-form gap components; the common color scale uses a square-root transform for display. Regional occurrence contributes a relevance weight to localization and is not treated as proof of native status."
    )
    contract = "\n".join(
        [
            "# R1 Figure 5 contract",
            "",
            "Core conclusion: state-level chemical and species-relevance weights alter the identity and geographic distribution of leading sentinel species across the 32 supported applications, while the strict-LOO evidence-gap model defines a separate national testing-priority surface.",
            "",
            "- Panel a restores the R0 rank-1-fill/rank-2-point map using AP04 strict-LOO baseline sequences and explicitly marks PA, TN, and WV as insufficient-support fallbacks.",
            "- Panel b restores the R0 paired expected-capture comparison using the updated AP04 strict-LOO values, while keeping it semantically separate from the species-distribution map.",
            "- Panel c summarizes the complete AP04 panel membership ledger without converting occurrence evidence into a hard deployability gate.",
            "- Panel d is recalculated from AP02 strict-LOO probability, reliability, and direct-support matrices; it is national, not state-conditioned.",
            "- Exact state panels and expected-capture gains are routed to Figure S7 and Table S6; robustness scenarios are routed to Figure S8 and Table S7.",
            "- Layout: the map remains the top-row hero panel; the restored gain comparison is panel b; lower panels are c and d; inter-panel spacing is compact and typography is Times New Roman.",
            "",
            f"Manuscript caption: {caption}",
            "",
        ]
    )
    (OUT_DIR / "figure_contract.md").write_text(contract, encoding="utf-8")

    files = [*outputs, *tables.keys(), "figure_contract.md"]
    manifest = {
        "figure_id": "Figure 5",
        "version": "R1_AP04_AP02_STRICT_LOO",
        "title": "Regional localization and national evidence-acquisition priorities",
        "core_conclusion": "Localized weighting changes the geographic distribution of leading sentinel species; national testing priorities identify evidence-acquisition targets.",
        "manuscript_caption": caption,
        "panel_map": {
            "a": "R0-style AP04 leading-species map: rank-1 fill and rank-2 point",
            "b": "Paired national-versus-localized expected capture for all 32 supported states, ordered by gain",
            "c": "Localized-panel selection frequency by rank",
            "d": "Strict-LOO national species-by-candidate-MOA testing-priority matrix",
        },
        "input_dependencies": [
            str(AP04_DIR / "ap04_state_support_and_fallback_ledger.csv"),
            str(AP02_DIR / "ap02_strict_loo_species_chemical_tail_probability_x95.npz"),
            str(BHBT_ROOT / "results" / "weights" / "national_priority_chemicals.csv"),
            str(BHBT_ROOT / "results" / "probability" / "candidate_universe_locked.csv"),
            str(BHBT_ROOT / "results" / "chemistry" / "chemical_moa_form_master.csv.gz"),
            str(BOUNDARY_ZIP),
        ],
        "outputs": files,
        "qa": {
            "localized_states": 32,
            "fallback_states": ["PA", "TN", "WV"],
            "gain_panel_states": 32,
            "gain_panel_top_labels": ordered_gain.head(3)["state_code"].astype(str).tolist(),
            "localized_sequence_rows": 160,
            "testing_matrix_dimensions": "10 x 8",
            "probability_source": "AP02_STRICT_LOO",
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
    localized, fallback, sequences = load_ap04_state_outputs()
    testing, species_priority, chemical_priority, top_pairs = build_strict_loo_testing_matrix()
    boundary_parts = load_boundary_parts()
    outputs = draw(localized, fallback, sequences, testing, boundary_parts)
    write_outputs(outputs, localized, fallback, sequences, testing, species_priority, chemical_priority, top_pairs)

    mean_gain = float(localized["gain_percent_points"].mean())
    median_gain = float(localized["gain_percent_points"].median())
    if not np.isclose(mean_gain, 6.751, atol=0.002) or not np.isclose(median_gain, 6.556, atol=0.002):
        raise AssertionError(f"AP04 gain summary mismatch: mean={mean_gain}, median={median_gain}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(OUT_DIR),
                "localized_states": len(localized),
                "fallback_states": sorted(fallback["state_code"].astype(str).tolist()),
                "mean_gain_pp": mean_gain,
                "median_gain_pp": median_gain,
                "testing_matrix_rows": len(testing),
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
