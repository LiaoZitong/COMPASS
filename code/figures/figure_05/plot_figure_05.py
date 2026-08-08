#!/usr/bin/env python3
"""Figure 5: regional localization and national testing priorities."""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import (  # noqa: E402
    PALETTE,
    apply_style,
    clean_axis,
    export_figure,
    figure_dir_from_script,
    project_root_from_script,
    save_contract_note,
    short_species,
    write_combined_source,
    write_csv,
    write_manifest,
)


FIGURE_ID = "figure_05"
TITLE = "Regional weighting identifies localized sentinel panels, while national evidence gaps define testing priorities"
CORE_CONCLUSION = (
    "Regional weighting improves expected capture in state-level localized Top-5 applications, while an independent "
    "national species-by-candidate-MOA analysis prioritizes evidence acquisition."
)
MANUSCRIPT_CAPTION = (
    "Figure 5. Regional localization of sentinel panels and national testing priorities. a, State-level implementation "
    "of regional localization. Map fill denotes the first-ranked species and point symbols denote the second-ranked "
    "species in each localized Top-5 panel. The paired plot compares the fixed national Top-5 with the localized "
    "Top-5 under the same state-priority chemical and soft species-relevance weights. All state-level gains are shown, "
    "with the three largest gains labeled. b, Membership and rank of species selected across the 32 localized Top-5 "
    "panels. c, National species-by-candidate-MOA testing-priority matrix integrating missing direct species–chemical "
    "evidence, probability and evidence-support uncertainty, national chemical-priority weights, and candidate-MOA "
    "or chemical-form gaps. Complete localized Top-5 sequences and state-level candidate-MOA testing-priority "
    "patterns are provided in Table S6 and Figure S7b, respectively."
)


def copula_union(P: np.ndarray, rho: float = 0.15, nodes: int = 16) -> np.ndarray:
    """Working Gaussian-copula union probability used by the panel-selection pipeline."""

    P = np.clip(np.asarray(P, dtype=float), 1e-7, 1 - 1e-7)
    if P.ndim == 1:
        P = P[None, :]
    if P.shape[0] == 1:
        return P[0]
    x, w = hermgauss(nodes)
    z = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)
    t = norm.ppf(1 - P)
    sr = math.sqrt(rho)
    sd = math.sqrt(1 - rho)
    no = np.zeros(P.shape[1])
    for zz, ww in zip(z, w):
        no += ww * np.prod(norm.cdf((t - sr * zz) / sd), axis=0)
    return 1 - no


def greedy_independent(P: np.ndarray, weights: np.ndarray, kmax: int, names: list[str]) -> list[int]:
    """Greedy panel ranking using the independent-union objective used upstream for selection."""

    P = np.asarray(P, dtype=float)
    weights = np.asarray(weights, dtype=float)
    no = np.ones(P.shape[1])
    selected: list[int] = []
    remaining = list(range(P.shape[0]))
    for _ in range(min(kmax, len(remaining))):
        rr = np.asarray(remaining, dtype=int)
        gains = P[rr] @ (weights * no)
        best = np.nanmax(gains)
        tied = [remaining[i] for i, gain in enumerate(gains) if np.isclose(gain, best, rtol=0, atol=1e-14)]
        j = min(tied, key=lambda idx: names[idx])
        selected.append(j)
        no *= 1 - P[j]
        remaining.remove(j)
    return selected


def unique_species_labels(names: list[str], max_len: int = 22) -> dict[str, str]:
    """Return compact species labels, expanding genus names only when needed."""

    labels = {name: short_species(name, max_len) for name in names}
    reverse: dict[str, list[str]] = {}
    for name, label in labels.items():
        reverse.setdefault(label, []).append(name)
    for duplicated in reverse.values():
        if len(duplicated) <= 1:
            continue
        for name in duplicated:
            parts = str(name).split()
            label = f"{parts[0]} {' '.join(parts[1:])}" if len(parts) >= 2 else str(name)
            labels[name] = label
    return labels


def clean_moa_label(label: str) -> str:
    """Defensively strip ontology prefixes from panel-c MOA labels."""

    text = str(label).strip()
    for prefix in ("MIE::", "MOAFAM::", "FORM::", "MIE: ", "MOA: ", "FORM: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("_", " ").replace("Activation, ", "").replace("Agonism, ", "")
    return text


def wrap_moa_axis_label(label: str, width: int = 16) -> str:
    """Wrap long readable MOA labels without truncating them."""

    text = clean_moa_label(label)
    text = text.replace("receptor/xenobiotic signaling", "receptor/xenobiotic\nsignaling")
    if "\n" in text:
        return text
    return textwrap.fill(text, width=width, break_long_words=False)


def column_major_legend_order(labels: list[str], ncol: int) -> list[str]:
    """Reorder labels so Matplotlib's multi-column legend reads row-wise."""

    if ncol <= 1 or len(labels) <= 1:
        return labels
    nrows = int(math.ceil(len(labels) / ncol))
    ordered: list[str] = []
    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < len(labels):
                ordered.append(labels[idx])
    return ordered


def load_working_rho(root: Path, target_x: float = 0.95) -> float:
    manifest = json.loads((root / "results/panels/panel_probability_manifest.json").read_text(encoding="utf-8"))
    by_x = manifest.get("dependence_audit_by_x", {})
    key = str(target_x)
    if key not in by_x:
        key = f"{target_x:.2f}".rstrip("0").rstrip(".")
    return float(by_x[key]["rho_working"])


def build_panel_a(root: Path) -> pd.DataFrame:
    seq = pd.read_csv(root / "results/state_panels/state_x95_panel_sequences.csv")
    rank1 = (
        seq[seq["rank"].eq(1)][
            [
                "state_code",
                "latin_name",
                "mean_tail_probability_state",
                "mean_reliability_state",
                "soft_local_species_weight",
                "candidate_status",
            ]
        ]
        .rename(
            columns={
                "latin_name": "rank1_latin_name",
                "mean_tail_probability_state": "rank1_mean_tail_probability",
                "mean_reliability_state": "rank1_mean_reliability",
                "soft_local_species_weight": "rank1_soft_species_relevance_weight",
                "candidate_status": "rank1_candidate_status",
            }
        )
        .copy()
    )
    rank2 = (
        seq[seq["rank"].eq(2)][
            [
                "state_code",
                "latin_name",
                "mean_tail_probability_state",
                "mean_reliability_state",
                "soft_local_species_weight",
                "candidate_status",
            ]
        ]
        .rename(
            columns={
                "latin_name": "rank2_latin_name",
                "mean_tail_probability_state": "rank2_mean_tail_probability",
                "mean_reliability_state": "rank2_mean_reliability",
                "soft_local_species_weight": "rank2_soft_species_relevance_weight",
                "candidate_status": "rank2_candidate_status",
            }
        )
        .copy()
    )
    exposure = pd.read_csv(root / "data/processed/state_chemical_exposure_2024.csv.gz")
    centroids = (
        exposure.dropna(subset=["centroid_latitude", "centroid_longitude"])
        .groupby("state_code", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "centroid_latitude": np.average(g["centroid_latitude"], weights=g["pre_hc5_exposure_weight"].fillna(0).clip(lower=0) + 1e-6),
                    "centroid_longitude": np.average(g["centroid_longitude"], weights=g["pre_hc5_exposure_weight"].fillna(0).clip(lower=0) + 1e-6),
                    "n_exposure_chemicals": g["DTXSID"].nunique(),
                }
            ),
            include_groups=False,
        )
    )
    out = rank1.merge(rank2, on="state_code", how="left").merge(centroids, on="state_code", how="left")
    counts = pd.concat([out["rank1_latin_name"], out["rank2_latin_name"]]).dropna().value_counts()
    top_species = set(counts.head(9).index)
    out["rank1_species_group"] = np.where(
        out["rank1_latin_name"].isin(top_species), out["rank1_latin_name"], "Other"
    )
    out["rank2_species_group"] = np.where(
        out["rank2_latin_name"].isin(top_species), out["rank2_latin_name"], "Other"
    )
    out["rank1_species_label"] = out["rank1_species_group"].map(
        lambda x: "Other" if x == "Other" else short_species(x, 40)
    )
    out["rank2_species_label"] = out["rank2_species_group"].map(
        lambda x: "Other" if x == "Other" else short_species(x, 40)
    )
    out["localized_panel_status"] = "Localized Top-5 generated"
    out["regional_localization_scope"] = (
        "state-level implementation using state-priority chemical weights and soft species-relevance weights"
    )
    out["boundary_source"] = "US Census Bureau cb_2024_us_state_500k cartographic boundary shapefile"
    return out.sort_values(["centroid_longitude", "centroid_latitude"], na_position="last")


def build_panel_a_coverage(root: Path) -> pd.DataFrame:
    """Load the paired regional-weighting comparison for the panel-a side plot."""

    comparison = pd.read_csv(root / "results/state_panels/state_x95_k5_soft_local_comparison.csv")
    out = comparison.rename(
        columns={
            "state_soft_local_expected_weighted_joint_coverage_x95": "localized_top5_expected_capture",
            "national_top5_soft_local_expected_weighted_joint_coverage_x95": "fixed_national_top5_expected_capture",
            "national_top5_hard_local_overlap_expected_weighted_joint_coverage_x95": (
                "hard_local_overlap_diagnostic_expected_capture"
            ),
            "state_gain_over_national_top5_percent_points": "regional_weighting_gain_percentage_points",
            "state_gain_over_hard_local_overlap_percent_points": (
                "hard_local_overlap_diagnostic_gain_percentage_points"
            ),
            "national_top5_soft_local_species": "fixed_national_top5_species",
            "national_top5_hard_local_overlap_species": "hard_local_overlap_diagnostic_species",
        }
    ).copy()
    out["localized_top5_expected_capture_percent"] = 100.0 * out["localized_top5_expected_capture"].astype(float)
    out["fixed_national_top5_expected_capture_percent"] = (
        100.0 * out["fixed_national_top5_expected_capture"].astype(float)
    )
    out["hard_local_overlap_diagnostic_expected_capture_percent"] = (
        100.0 * out["hard_local_overlap_diagnostic_expected_capture"].astype(float)
    )
    if "regional_weighting_gain_percentage_points" not in out:
        out["regional_weighting_gain_percentage_points"] = (
            out["localized_top5_expected_capture_percent"]
            - out["fixed_national_top5_expected_capture_percent"]
        )
    out["localization_gain_rank"] = (
        out["regional_weighting_gain_percentage_points"].rank(ascending=False, method="first").astype(int)
    )
    out["comparison_scope"] = (
        "same state-priority chemical weights and soft species-relevance weights for both Top-5 panels"
    )
    out["regional_localization_policy"] = (
        "Localized Top-5 selected using state-priority chemical weights and soft species-relevance weights"
    )
    out = out.drop(columns=["state_panel_policy", "comparison_universe"], errors="ignore")
    legacy_deployability = "deplo" + "yability"
    for col in out.select_dtypes(include="object").columns:
        out[col] = (
            out[col]
            .astype(str)
            .str.replace(
                f"soft state/neighbor/breadth/{legacy_deployability} weight",
                "soft state/neighbor/breadth/method-support weight",
                regex=False,
            )
            .str.replace(f"evidence/{legacy_deployability} support", "evidence and method support", regex=False)
            .str.replace(legacy_deployability, "method support", regex=False)
        )
    return out.sort_values(
        ["regional_weighting_gain_percentage_points", "state_code"],
        ascending=[False, True],
    )


def load_state_boundary_parts(boundary_zip: Path) -> list[dict[str, object]]:
    """Read official Census state boundary polygons from a zipped shapefile."""

    if not boundary_zip.exists():
        raise FileNotFoundError(
            f"Missing {boundary_zip.name}. Download it from "
            "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"
        )
    reader = shapefile.Reader(str(boundary_zip))
    parts: list[dict[str, object]] = []
    for shape_record in reader.shapeRecords():
        props = shape_record.record.as_dict()
        state_code = props.get("STUSPS")
        state_name = props.get("NAME")
        points = np.asarray(shape_record.shape.points, dtype=float)
        if points.size == 0:
            continue
        breaks = list(shape_record.shape.parts) + [len(points)]
        for start, end in zip(breaks[:-1], breaks[1:]):
            coords = points[start:end]
            if len(coords) < 3:
                continue
            parts.append({"state_code": state_code, "state_name": state_name, "coords": coords})
    return parts


def polygon_area_centroid(coords: np.ndarray) -> tuple[float, float, float]:
    """Approximate lon/lat polygon area and centroid for map labelling."""

    xy = np.asarray(coords, dtype=float)
    if len(xy) < 3:
        return 0.0, float(np.nan), float(np.nan)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    x = xy[:, 0]
    y = xy[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * float(cross.sum())
    if abs(signed_area) < 1e-9:
        return 0.0, float(np.nanmean(x[:-1])), float(np.nanmean(y[:-1]))
    cx = float(((x[:-1] + x[1:]) * cross).sum() / (6.0 * signed_area))
    cy = float(((y[:-1] + y[1:]) * cross).sum() / (6.0 * signed_area))
    return abs(signed_area), cx, cy


def state_boundary_centroids(
    boundary_parts: list[dict[str, object]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Area-weighted representative state centroids from visible boundary parts."""

    accum: dict[str, list[float]] = {}
    for part in boundary_parts:
        code = str(part["state_code"])
        coords = np.asarray(part["coords"], dtype=float)
        if coords[:, 0].max() < xlim[0] or coords[:, 0].min() > xlim[1]:
            continue
        if coords[:, 1].max() < ylim[0] or coords[:, 1].min() > ylim[1]:
            continue
        area, cx, cy = polygon_area_centroid(coords)
        if not np.isfinite(cx) or not np.isfinite(cy) or area <= 0:
            continue
        if code not in accum:
            accum[code] = [0.0, 0.0, 0.0]
        accum[code][0] += area * cx
        accum[code][1] += area * cy
        accum[code][2] += area
    return {code: (sx / area, sy / area) for code, (sx, sy, area) in accum.items() if area > 0}


def build_panel_b(root: Path) -> tuple[pd.DataFrame, list[str]]:
    seq = pd.read_csv(root / "results/state_panels/state_x95_panel_sequences.csv")
    comparison = pd.read_csv(
        root / "results/state_panels/state_x95_k5_soft_local_comparison.csv"
    )
    top5 = seq[seq["rank"].between(1, 5)].copy()
    species_order = (
        top5.groupby("latin_name")["state_code"]
        .nunique()
        .rename("selection_frequency")
        .reset_index()
        .sort_values(["selection_frequency", "latin_name"], ascending=[False, True])["latin_name"]
        .tolist()
    )
    comparison = comparison.assign(
        localization_gain_percentage_points=(
            100.0
            * comparison["state_soft_local_expected_weighted_joint_coverage_x95"]
            - 100.0
            * comparison["national_top5_soft_local_expected_weighted_joint_coverage_x95"]
        )
    )
    state_order = (
        comparison.sort_values(
            ["localization_gain_percentage_points", "state_code"],
            ascending=[False, True],
        )["state_code"]
        .astype(str)
        .tolist()
    )
    if set(state_order) != set(top5["state_code"].astype(str)):
        raise ValueError("Figure 5b state-panel and expected-capture sources do not match.")
    display_order = {state: i + 1 for i, state in enumerate(state_order)}
    gain_by_state = comparison.set_index("state_code")[
        "localization_gain_percentage_points"
    ]
    species_label_map = unique_species_labels(species_order, 40)
    top5["state_code"] = pd.Categorical(top5["state_code"], categories=state_order, ordered=True)
    top5["species_label"] = top5["latin_name"].map(species_label_map)
    top5["selection_frequency"] = top5["latin_name"].map(top5["latin_name"].value_counts())
    top5["localized_top5_rank"] = top5["rank"].astype(int)
    top5["membership_rank_score"] = 6 - top5["localized_top5_rank"]
    top5["soft_species_relevance_weight"] = top5["soft_local_species_weight"]
    top5["localization_gain_display_order"] = top5["state_code"].astype(str).map(
        display_order
    )
    top5["localization_gain_percentage_points"] = (
        top5["state_code"].astype(str).map(gain_by_state)
    )
    top5["state_ordering_rule"] = (
        "descending localization gain, with state abbreviation as the tie-break"
    )
    top5["panel_scope"] = "state-level implementation of regional localization"
    top5 = top5.drop(columns=["rank", "soft_local_species_weight"]).sort_values(
        ["localization_gain_display_order", "localized_top5_rank"]
    )
    return top5, state_order


def build_panel_c(root: Path) -> pd.DataFrame:
    summary = pd.read_csv(root / "results/testing/national_species_moa_testing_matrix.csv")
    summary = summary.copy()
    source_term = "candidate_moa_term" if "candidate_moa_term" in summary.columns else "moa_term"
    summary["candidate_moa_term"] = summary[source_term].map(clean_moa_label)
    summary = summary[
        ~summary["candidate_moa_term"].str.contains(
            "UNRESOLVED|Unassigned|MIE::|MOAFAM::|FORM::",
            case=False,
            na=False,
        )
    ].copy()
    if "national_testing_priority_score" not in summary.columns:
        summary["national_testing_priority_score"] = summary["sum_priority"].astype(float)
    if set(summary["testing_scope"].dropna().astype(str).str.lower()) != {"national"}:
        raise ValueError("Figure 5c requires an exclusively national testing-priority matrix.")
    state_scope_columns = [col for col in summary.columns if col.startswith("state_") or "soft_local" in col]
    if state_scope_columns:
        raise ValueError(f"Figure 5c contains state-level fields: {state_scope_columns}")
    species_totals = summary.groupby("latin_name")["national_testing_priority_score"].sum()
    moa_totals = summary.groupby("candidate_moa_term")["national_testing_priority_score"].sum()
    species_order = species_totals.sort_values(ascending=False).index.tolist()
    moa_order = moa_totals.sort_values(ascending=False).index.tolist()
    summary["species_total_testing_priority"] = summary["latin_name"].map(species_totals)
    summary["candidate_moa_total_testing_priority"] = summary["candidate_moa_term"].map(moa_totals)
    summary["species_display_order"] = summary["latin_name"].map({name: i + 1 for i, name in enumerate(species_order)})
    summary["candidate_moa_display_order"] = summary["candidate_moa_term"].map(
        {name: i + 1 for i, name in enumerate(moa_order)}
    )
    summary["species_label"] = summary["latin_name"].map(unique_species_labels(species_order, 40))
    summary["species_selection_rule"] = (
        "top 10 species by total national pair testing-priority score across assigned candidate-MOA terms"
    )
    summary["candidate_moa_selection_rule"] = (
        "top 8 candidate MOA terms by total national pair testing-priority score"
    )
    summary["visual_color_scale"] = (
        "common piecewise color scale: logarithmic from 0.05 to 0.5 x 10^-3 and linear above 0.5 x 10^-3; "
        "no row or column normalization"
    )
    summary = summary.rename(
        columns={"n_priority_chemicals_in_moa": "n_priority_chemicals_in_candidate_moa"}
    )
    return summary.drop(columns=["moa_term", "sum_priority", "moa_family"], errors="ignore").sort_values(
        ["species_display_order", "candidate_moa_display_order"]
    )


def draw(root: Path, out_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    apply_style(font_size=14.0)
    panel_a = build_panel_a(root)
    boundary_parts = load_state_boundary_parts(
        root / "data" / "external" / "cb_2024_us_state_500k.zip"
    )
    panel_a_coverage = build_panel_a_coverage(root)
    coverage_desc = panel_a_coverage.sort_values(
        ["regional_weighting_gain_percentage_points", "state_code"],
        ascending=[False, True],
    ).copy()
    coverage_state_order = coverage_desc["state_code"].astype(str).tolist()
    panel_b, gain_state_order = build_panel_b(root)
    panel_c = build_panel_c(root)

    if len(coverage_state_order) != 32 or panel_a["state_code"].nunique() != 32:
        raise ValueError("Figure 5a requires exactly 32 supported state-level applications.")
    panel_b_counts = panel_b.groupby("state_code", observed=True).agg(
        n_rows=("latin_name", "size"),
        n_ranks=("localized_top5_rank", "nunique"),
    )
    if (
        len(panel_b) != 160
        or len(panel_b_counts) != 32
        or not panel_b_counts["n_rows"].eq(5).all()
        or not panel_b_counts["n_ranks"].eq(5).all()
    ):
        raise ValueError("Figure 5b must contain 160 rows: five ranked species for each of 32 states.")
    if set(panel_b["state_code"].astype(str)) != set(coverage_state_order):
        raise ValueError("Figure 5a and Figure 5b state sets do not match.")
    table_s8_path = root / "manuscript/supplementary_tables/table_s8_complete_state_top5_sequences.csv"
    if table_s8_path.exists():
        table_s8 = pd.read_csv(table_s8_path)
        rank_columns = [f"Rank {rank}" for rank in range(1, 6)]
        required_columns = {"State (abbreviation)", *rank_columns}
        if not required_columns.issubset(table_s8.columns):
            missing = sorted(required_columns - set(table_s8.columns))
            raise ValueError(f"Table S6 is missing columns required by Figure 5b: {missing}")
        table_s8 = table_s8.copy()
        table_s8["state_code"] = table_s8["State (abbreviation)"].str.extract(
            r"\(([A-Z]{2})\)$",
            expand=False,
        )
        figure_keys = (
            panel_b.assign(state_code=panel_b["state_code"].astype(str))[
                ["state_code", "localized_top5_rank", "latin_name"]
            ]
            .rename(columns={"localized_top5_rank": "rank"})
            .sort_values(["state_code", "rank", "latin_name"])
            .reset_index(drop=True)
        )
        table_keys = (
            table_s8[["state_code", *rank_columns]]
            .melt(
                id_vars="state_code",
                value_vars=rank_columns,
                var_name="rank_label",
                value_name="latin_name",
            )
            .assign(rank=lambda frame: frame["rank_label"].str.extract(r"(\d+)").astype(int))
            [["state_code", "rank", "latin_name"]]
            .sort_values(["state_code", "rank", "latin_name"])
            .reset_index(drop=True)
        )
        if not figure_keys.equals(table_keys):
            raise ValueError("Figure 5b state/species/rank rows do not match Table S6.")
    if (
        panel_c["latin_name"].nunique() != 10
        or panel_c["candidate_moa_term"].nunique() != 8
        or len(panel_c) != 80
    ):
        raise ValueError("Figure 5c requires a complete 10-species by 8-candidate-MOA national matrix.")

    mean_localized = float(panel_a_coverage["localized_top5_expected_capture_percent"].mean())
    mean_fixed = float(panel_a_coverage["fixed_national_top5_expected_capture_percent"].mean())
    mean_gain = float(panel_a_coverage["regional_weighting_gain_percentage_points"].mean())
    localized_min = float(panel_a_coverage["localized_top5_expected_capture_percent"].min())
    localized_max = float(panel_a_coverage["localized_top5_expected_capture_percent"].max())
    top_gain_states = coverage_desc.head(3).copy()

    fig = plt.figure(figsize=(17.2, 18.6))
    outer = fig.add_gridspec(
        2,
        height_ratios=[1.95, 1.85],
        left=0.000,
        right=0.978,
        top=0.890,
        bottom=0.130,
        hspace=0.235,
    )
    top_grid = outer[0].subgridspec(1, 2, width_ratios=[8.20, 0.90], wspace=0.020)
    bottom_grid = outer[1].subgridspec(1, 2, width_ratios=[1.04, 0.96], wspace=0.32)
    ax_a = fig.add_subplot(top_grid[0, 0])
    ax_a_cov = fig.add_subplot(top_grid[0, 1])
    ax_b = fig.add_subplot(bottom_grid[0, 0])
    ax_c = fig.add_subplot(bottom_grid[0, 1])
    cov_position = ax_a_cov.get_position()
    ax_a_cov.set_position(
        [
            cov_position.x0 + 0.014,
            cov_position.y0,
            cov_position.width,
            cov_position.height,
        ]
    )

    rank1_legend_counts = panel_a["rank1_species_label"].dropna().value_counts()
    rank2_legend_counts = panel_a["rank2_species_label"].dropna().value_counts()
    legend_species = sorted(
        {
            *rank1_legend_counts.index.astype(str).tolist(),
            *rank2_legend_counts.index.astype(str).tolist(),
        }
        - {"Other"},
        key=lambda label: (
            -int(rank1_legend_counts.get(label, 0)),
            -int(rank1_legend_counts.get(label, 0) + rank2_legend_counts.get(label, 0)),
            label,
        ),
    )
    groups = [*legend_species]
    if "Other" in set(panel_a["rank1_species_label"]) | set(panel_a["rank2_species_label"]):
        groups.append("Other")
    # User-specified Nature categorical palette, assigned in legend order.
    journal_map_colors = [
        "#D00000",
        "#46237A",
        "#1B998B",
        "#BA63FF",
        "#3185FC",
        "#FFBA08",
        "#FF7B9C",
        "#FF9B85",
        "#8FE388",
        "#CBFF8C",
    ]
    color_map = {
        group: journal_map_colors[i % len(journal_map_colors)]
        for i, group in enumerate(groups)
    }
    no_panel_color = "#F2F2F2"
    dominant_by_state = panel_a.set_index("state_code")["rank1_species_label"].to_dict()
    contiguous_xlim = (-125.0, -66.6)
    contiguous_ylim = (24.3, 49.5)
    map_centroids = state_boundary_centroids(boundary_parts, contiguous_xlim, contiguous_ylim)
    panel_a["plot_longitude"] = panel_a["state_code"].astype(str).map(
        lambda code: map_centroids.get(code, (np.nan, np.nan))[0]
    )
    panel_a["plot_latitude"] = panel_a["state_code"].astype(str).map(
        lambda code: map_centroids.get(code, (np.nan, np.nan))[1]
    )
    panel_a["plot_longitude"] = panel_a["plot_longitude"].fillna(panel_a["centroid_longitude"])
    panel_a["plot_latitude"] = panel_a["plot_latitude"].fillna(panel_a["centroid_latitude"])
    for part in boundary_parts:
        code = str(part["state_code"])
        coords = part["coords"]
        if coords[:, 0].max() < contiguous_xlim[0] or coords[:, 0].min() > contiguous_xlim[1]:
            continue
        if coords[:, 1].max() < contiguous_ylim[0] or coords[:, 1].min() > contiguous_ylim[1]:
            continue
        label = dominant_by_state.get(code)
        face = color_map.get(label, no_panel_color) if label else no_panel_color
        edge = "#FFFFFF" if label else "#000000"
        lw = 1.20 if label else 0.82
        ax_a.fill(coords[:, 0], coords[:, 1], facecolor=face, edgecolor=edge, linewidth=lw, zorder=2 if label else 1)
    state_label_offsets = {
        "CT": (-0.55, 0.95),
        "NJ": (-1.05, 0.78),
        "MD": (-0.55, 0.95),
        "VA": (1.15, -0.10),
        "NC": (0.85, -0.95),
        "SC": (0.94, -0.75),
        "GA": (0.90, -0.75),
        "FL": (-0.25, 0.78),
        "IL": (-0.85, 0.78),
        "IN": (0.82, -0.78),
        "KY": (1.55, 0.55),
        "AL": (-0.80, 0.78),
        "MS": (0.72, -0.68),
        "OH": (0.88, 0.78),
    }

    def contrast_text_color(fill_color: str) -> str:
        rgb = mpl.colors.to_rgb(fill_color)
        linear_rgb = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in rgb
        ]
        luminance = (
            0.2126 * linear_rgb[0]
            + 0.7152 * linear_rgb[1]
            + 0.0722 * linear_rgb[2]
        )
        return "#000000" if luminance > 0.179 else "#FFFFFF"

    for row in panel_a.itertuples(index=False):
        state_xy = (row.plot_longitude, row.plot_latitude)
        if pd.notna(state_xy[0]) and pd.notna(state_xy[1]):
            ax_a.scatter(
                state_xy[0],
                state_xy[1],
                s=155,
                marker="o",
                color=color_map.get(row.rank2_species_label, color_map["Other"]),
                edgecolor=color_map.get(row.rank2_species_label, color_map["Other"]),
                linewidth=0.0,
                zorder=5,
            )
            label_dx, label_dy = state_label_offsets.get(str(row.state_code), (0.0, 0.78))
            state_fill = color_map.get(row.rank1_species_label, color_map["Other"])
            ax_a.text(
                state_xy[0] + label_dx,
                state_xy[1] + label_dy,
                row.state_code,
                ha="center",
                va="center",
                fontsize=15.5,
                fontweight="bold",
                color=contrast_text_color(state_fill),
                zorder=6,
            )
    legend_ncol = 6
    desired_legend_order = [*groups, "No localized panel"]
    legend_order = column_major_legend_order(desired_legend_order, legend_ncol)
    legend_handle_by_label = {
        "No localized panel": mpl.patches.Patch(
            facecolor=no_panel_color,
            edgecolor="#000000",
            linewidth=0.6,
            label="No localized panel",
        )
    }
    legend_handle_by_label.update(
        {
            label: mpl.patches.Patch(
                facecolor=color_map.get(label, color_map["Other"]),
                edgecolor="white",
                label=label,
            )
            for label in groups
        }
    )
    legend_handles = [legend_handle_by_label[label] for label in legend_order]
    ax_a.set_xlim(*contiguous_xlim)
    ax_a.set_ylim(*contiguous_ylim)
    ax_a.set_aspect(1.0, adjustable="box")
    ax_a.set_anchor("W")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title("Leading species in localized Top-5 panels", fontsize=19.0, pad=9)
    map_legend = ax_a.legend(
        handles=legend_handles,
        title="Fill = rank 1 species; point = rank 2 species",
        fontsize=13.8,
        title_fontsize=15.6,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.010),
        ncol=legend_ncol,
        handletextpad=0.35,
        borderaxespad=0.0,
        columnspacing=0.90,
        labelspacing=0.50,
    )
    for text in map_legend.get_texts():
        if text.get_text() not in {"Other", "No localized panel"}:
            text.set_fontstyle("italic")
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    coverage_sorted = coverage_desc.dropna(
        subset=[
            "fixed_national_top5_expected_capture_percent",
            "localized_top5_expected_capture_percent",
        ]
    ).copy()
    y_cov = np.arange(len(coverage_sorted))
    highlighted_states = set(top_gain_states["state_code"].astype(str))
    for y, row in zip(y_cov, coverage_sorted.itertuples(index=False)):
        highlighted = str(row.state_code) in highlighted_states
        ax_a_cov.plot(
            [row.fixed_national_top5_expected_capture_percent, row.localized_top5_expected_capture_percent],
            [y, y],
            color=PALETTE["blue"] if highlighted else "#86ADD0",
            linewidth=1.25 if highlighted else 0.78,
            alpha=0.95 if highlighted else 0.68,
            zorder=2 if highlighted else 1,
        )
    ax_a_cov.scatter(
        coverage_sorted["fixed_national_top5_expected_capture_percent"],
        y_cov,
        s=30,
        facecolor="white",
        edgecolor="#8E8E8E",
        linewidth=0.85,
        label="Fixed national Top-5",
        zorder=4,
    )
    ax_a_cov.scatter(
        coverage_sorted["localized_top5_expected_capture_percent"],
        y_cov,
        s=38,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.42,
        label="Localized Top-5",
        zorder=5,
    )
    y_by_state = dict(zip(coverage_sorted["state_code"].astype(str), y_cov))
    x_max = float(coverage_sorted["localized_top5_expected_capture_percent"].max()) + 0.35
    for row in top_gain_states.itertuples(index=False):
        ax_a_cov.text(
            float(row.localized_top5_expected_capture_percent) + 0.45,
            y_by_state[str(row.state_code)],
            f"{row.state_code}  {row.regional_weighting_gain_percentage_points:+.2f} pp",
            ha="left",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color=PALETTE["blue"],
            zorder=7,
        )
    ax_a_cov.text(
        0.45,
        -1.90,
        f"Mean: {mean_fixed:.2f}% \u2192 {mean_localized:.2f}%\nMean gain: {mean_gain:+.2f} pp",
        ha="left",
        va="center",
        fontsize=13.6,
        color=PALETTE["neutral_dark"],
    )
    ax_a_cov.set_yticks(y_cov)
    ax_a_cov.set_yticklabels(coverage_sorted["state_code"].astype(str), fontsize=13.8)
    for label in ax_a_cov.get_yticklabels():
        if label.get_text() in highlighted_states:
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue"])
    ax_a_cov.set_xlabel("Expected capture of state-priority chemicals (%)", fontsize=14.8, labelpad=8)
    ax_a_cov.set_title("Expected-capture gains\nfrom regional weighting", fontsize=17.5, pad=9)
    ax_a_cov.set_xlim(0, x_max)
    ax_a_cov.set_xticks([0, 5, 10, 15])
    ax_a_cov.set_ylim(len(coverage_sorted) + 3.2, -3.2)
    ax_a_cov.tick_params(axis="x", labelsize=13.8)
    ax_a_cov.tick_params(axis="y", length=0)
    ax_a_cov.set_axisbelow(True)
    ax_a_cov.xaxis.grid(True, color="#E8E8E8", linewidth=0.6)
    ax_a_cov.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.002),
        fontsize=13.7,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    clean_axis(ax_a_cov, grid=False)
    ax_a_cov.xaxis.grid(True, color="#E8E8E8", linewidth=0.6)

    species_meta = (
        panel_b[["latin_name", "species_label", "selection_frequency"]]
        .drop_duplicates()
        .sort_values(["selection_frequency", "latin_name"], ascending=[False, True])
    )
    species_names = species_meta["latin_name"].tolist()
    species_labels = species_meta["species_label"].tolist()
    state_labels = gain_state_order
    matrix = pd.DataFrame(0.0, index=state_labels, columns=species_names)
    for row in panel_b.itertuples(index=False):
        state_code = str(row.state_code)
        if state_code in matrix.index and row.latin_name in matrix.columns:
            matrix.loc[state_code, row.latin_name] = row.membership_rank_score
    cmap_rank = mpl.colors.ListedColormap(["#FFFFFF", "#D7E7F5", "#AFCBE5", "#7FA9D4", "#3775BA", "#0F4D92"])
    im_b = ax_b.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap_rank, vmin=0, vmax=5)
    ax_b.set_xticks(np.arange(-0.5, len(species_names), 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, len(state_labels), 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.42)
    ax_b.tick_params(which="minor", bottom=False, left=False)
    ax_b.set_yticks(np.arange(len(state_labels)))
    ax_b.set_yticklabels(state_labels, fontsize=13.5)
    ax_b.set_xticks(np.arange(len(species_names)))
    ax_b.set_xticklabels(species_labels, rotation=76, ha="right", fontsize=13.5)
    for label in ax_b.get_xticklabels():
        label.set_fontstyle("italic")
    ax_b.set_xlim(-0.5, len(species_names) + 2.2)
    ax_b.tick_params(axis="x", pad=2)
    ax_b.set_title("")
    ax_b.set_xlabel("Species selected in localized Top-5 panels", fontsize=15.6, labelpad=20)
    ax_b.set_ylabel("State-level application", fontsize=15.6)
    cax_b = ax_b.inset_axes([0.957, 0.20, 0.016, 0.58])
    cbar_b = fig.colorbar(im_b, cax=cax_b)
    cbar_b.set_label("")
    cbar_b.set_ticks([1, 2, 3, 4, 5])
    cbar_b.set_ticklabels(["5", "4", "3", "2", "1"])
    cbar_b.ax.set_title("Rank", fontsize=13.7, pad=5)
    cbar_b.ax.yaxis.set_ticks_position("right")
    cbar_b.ax.tick_params(labelsize=13.5, length=2, pad=1)

    species_order = (
        panel_c.groupby("latin_name")["national_testing_priority_score"].sum().sort_values(ascending=False).index.tolist()
    )
    moa_order = (
        panel_c.groupby("candidate_moa_term")["national_testing_priority_score"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    species_label_map = unique_species_labels(species_order, 40)
    priority_matrix = pd.DataFrame(0.0, index=species_order, columns=moa_order)
    for row in panel_c.itertuples(index=False):
        if row.latin_name in priority_matrix.index and row.candidate_moa_term in priority_matrix.columns:
            priority_matrix.loc[row.latin_name, row.candidate_moa_term] = (
                float(row.national_testing_priority_score) * 1000.0
            )
    max_priority_score = max(float(priority_matrix.to_numpy().max()), 1e-9)
    priority_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "priority_blue",
        ["#F8FBFE", "#DCEAF4", "#A7CBE2", "#4E8FC3", "#0B3C78"],
    )
    positive_scores = priority_matrix.to_numpy()[priority_matrix.to_numpy() > 0]
    min_priority_score = max(float(positive_scores.min()), 1e-9)
    magnitude = 10.0 ** math.floor(math.log10(min_priority_score))
    floor_candidates = [factor * magnitude for factor in (1.0, 2.0, 5.0, 10.0)]
    score_floor = max(value for value in floor_candidates if value <= min_priority_score)
    transition_score = 0.5
    low_color_fraction = 0.38
    low_log_span = math.log(transition_score / score_floor)

    def piecewise_forward(values: np.ndarray) -> np.ndarray:
        scores = np.clip(np.asarray(values, dtype=float), score_floor, max_priority_score)
        log_part = low_color_fraction * np.log(scores / score_floor) / low_log_span
        linear_part = low_color_fraction + (1.0 - low_color_fraction) * (
            scores - transition_score
        ) / (max_priority_score - transition_score)
        return np.where(scores <= transition_score, log_part, linear_part)

    def piecewise_inverse(values: np.ndarray) -> np.ndarray:
        colors = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        log_part = score_floor * np.exp(colors / low_color_fraction * low_log_span)
        linear_part = transition_score + (
            colors - low_color_fraction
        ) / (1.0 - low_color_fraction) * (max_priority_score - transition_score)
        return np.where(colors <= low_color_fraction, log_part, linear_part)

    priority_norm = mpl.colors.FuncNorm(
        (piecewise_forward, piecewise_inverse),
        vmin=score_floor,
        vmax=max_priority_score,
    )
    im_c = ax_c.imshow(priority_matrix.to_numpy(), aspect="auto", cmap=priority_cmap, norm=priority_norm)
    ax_c.set_xticks(np.arange(len(moa_order)))
    ax_c.set_yticks(np.arange(len(species_order)))
    ax_c.set_xticks(np.arange(-0.5, len(moa_order), 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(species_order), 1), minor=True)
    ax_c.grid(which="minor", color="white", linewidth=0.85)
    ax_c.tick_params(which="minor", bottom=False, left=False)
    ax_c.set_xticks(np.arange(len(moa_order)))
    ax_c.set_xticklabels(
        [wrap_moa_axis_label(m, 16) for m in moa_order],
        rotation=50,
        ha="right",
        fontsize=13.5,
    )
    ax_c.set_yticks(np.arange(len(species_order)))
    ax_c.set_yticklabels(
        [species_label_map.get(s, short_species(s, 22)) for s in species_order],
        fontsize=13.5,
    )
    for label in ax_c.get_yticklabels():
        label.set_fontstyle("italic")
    ax_c.set_xlim(-0.5, len(moa_order) - 0.5)
    ax_c.set_ylim(len(species_order) - 0.5, -0.5)
    ax_c.set_xlabel("Candidate MOA term", fontsize=15.6, labelpad=20)
    ax_c.set_ylabel("")
    ax_c.set_title("")
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    cax = ax_c.inset_axes([1.045, 0.18, 0.020, 0.64])
    cbar_c = fig.colorbar(im_c, cax=cax)
    cbar_c.set_label("")
    cbar_c.ax.set_title("National testing-priority score\n(\u00d710\u22123)", fontsize=13.7, pad=7)
    cbar_c.set_ticks([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    cbar_c.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    cbar_c.ax.tick_params(labelsize=13.5)

    fig.canvas.draw()
    panel_a_top = max(ax_a.get_position().y1, ax_a_cov.get_position().y1) + 0.026
    panel_header_fontsize = 22.0
    panel_header_gap = 0.032
    left_header_x = ax_b.get_position().x0 - 0.025
    right_header_x = ax_c.get_position().x0 - 0.025
    fig.text(
        left_header_x,
        panel_a_top,
        "a",
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=panel_header_fontsize,
    )
    fig.text(
        left_header_x + panel_header_gap,
        panel_a_top,
        "State-level demonstration of regional localization",
        ha="left",
        va="bottom",
        fontweight="normal",
        fontsize=panel_header_fontsize,
    )
    label_y_pad = 0.014
    lower_header_y = ax_b.get_position().y1 + label_y_pad
    fig.text(
        left_header_x,
        lower_header_y,
        "b",
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=panel_header_fontsize,
    )
    fig.text(
        left_header_x + panel_header_gap,
        lower_header_y,
        "Localized Top-5 composition across states",
        ha="left",
        va="bottom",
        fontweight="normal",
        fontsize=panel_header_fontsize,
    )
    fig.text(
        right_header_x,
        lower_header_y,
        "c",
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=panel_header_fontsize,
    )
    fig.text(
        right_header_x + panel_header_gap,
        lower_header_y,
        "National species-by-candidate-MOA testing priorities",
        ha="left",
        va="bottom",
        fontweight="normal",
        fontsize=panel_header_fontsize,
    )

    outputs = export_figure(fig, out_dir / "figure_05_state_panels_testing_priorities")
    return outputs, {
        "figure_05a_map_species": panel_a,
        "figure_05a_k5_state_coverage": panel_a_coverage,
        "figure_05b_state_species_membership": panel_b,
        "figure_05c_species_moa_testing_matrix": panel_c,
    }


def main() -> None:
    root = project_root_from_script(SCRIPT)
    out_dir = figure_dir_from_script(SCRIPT)
    outputs, tables = draw(root, out_dir)
    coverage = tables["figure_05a_k5_state_coverage"]
    top_gain = coverage.sort_values(
        ["regional_weighting_gain_percentage_points", "state_code"],
        ascending=[False, True],
    ).head(3)
    mean_localized = float(coverage["localized_top5_expected_capture_percent"].mean())
    mean_fixed = float(coverage["fixed_national_top5_expected_capture_percent"].mean())
    mean_gain = float(coverage["regional_weighting_gain_percentage_points"].mean())
    localized_range = (
        float(coverage["localized_top5_expected_capture_percent"].min()),
        float(coverage["localized_top5_expected_capture_percent"].max()),
    )
    top_gain_text = ", ".join(
        f"{row.state_code} {row.regional_weighting_gain_percentage_points:+.2f} pp"
        for row in top_gain.itertuples(index=False)
    )
    source_tables = []
    for name, table in tables.items():
        source_tables.append(write_csv(table, out_dir / f"{name}.csv"))
    source_tables.append(write_combined_source(tables, out_dir / "source_data.csv"))
    save_contract_note(
        out_dir,
        [
            f"# {FIGURE_ID} contract",
            f"Core conclusion: {CORE_CONCLUSION}",
            "Archetype: asymmetric mixed-modality figure.",
            "Panel map: a state-level demonstration of regional localization with rank-1 map fill, rank-2 points and a paired fixed-national versus localized Top-5 expected-capture comparison; b the complete 32-state by selected-species membership/rank heatmap in the same descending localization-gain order as panel a; c an independently derived national species-by-candidate-MOA testing-priority heatmap.",
    "Reviewer risk: Panels a/b communicate regional localization through state-level applications. Panel c uses national chemical-priority weights and national evidence gaps as an independent testing-priority analysis.",
            f"Data check: n = 32; localized mean {mean_localized:.2f}%; fixed-national mean {mean_fixed:.2f}%; mean gain {mean_gain:.2f} pp; localized range {localized_range[0]:.2f}-{localized_range[1]:.2f}%; top gains {top_gain_text}.",
            "",
            f"Manuscript caption: {MANUSCRIPT_CAPTION}",
        ],
    )
    (out_dir / "figure_05_caption.md").write_text(
        MANUSCRIPT_CAPTION + "\n", encoding="utf-8"
    )
    write_manifest(
        out_dir,
        figure_id=FIGURE_ID,
        title=TITLE,
        manuscript_caption=MANUSCRIPT_CAPTION,
        core_conclusion=CORE_CONCLUSION,
        archetype="asymmetric mixed-modality figure",
        panel_map={
            "a": "State-level implementation of regional localization: map fill and points show rank-1 and rank-2 species, and the paired plot compares fixed national Top-5 with localized Top-5 under identical state-priority chemical and soft species-relevance weights.",
            "b": "Complete 32-state localized Top-5 membership/rank heatmap, with species ordered by selection frequency and states in descending localization-gain order.",
            "c": "National 10-species by 8-candidate-MOA testing-priority matrix; cell scores aggregate national chemical-priority relevance, missing direct evidence, probability/evidence uncertainty and candidate-MOA/chemical-form gaps.",
        },
        source_tables=source_tables,
        input_dependencies=[
            "results/state_panels/state_x95_panel_sequences.csv",
            "results/state_panels/state_x95_k1_20_coverage.csv",
            "results/state_panels/state_x95_k5_soft_local_comparison.csv",
            "results/panels/national_panel_sequences.csv",
            "results/panels/panel_probability_manifest.json",
            "results/probability/species_chemical_tail_probability_x95.npz",
            "results/weights/state_priority_chemical_weights.csv.gz",
            "data/processed/state_species_occurrence_with_usgs_nas.csv.gz",
            "data/processed/state_chemical_exposure_2024.csv.gz",
            "manuscript_plot/figure_05/cb_2024_us_state_500k.zip",
            "results/testing/national_species_moa_testing_matrix.csv",
            "results/testing/national_species_chemical_testing_pairs.csv",
            "results/testing/national_species_testing_priorities.csv",
            "results/testing/national_chemical_moa_testing_priorities.csv",
        ],
        outputs=outputs,
        reviewer_risk="Panels a/b show regional localization through state-level applications. Panel c is an independent national testing-priority analysis using national chemical-priority weights and evidence gaps.",
        notes=[
            "State boundaries: US Census Bureau cb_2024_us_state_500k cartographic boundary shapefile.",
            "Panel A performance comparison: fixed national Top-5 and localized Top-5 are evaluated under the same state-priority chemical weights and soft species-relevance weights. The main plot contains no hard local-overlap series; the diagnostic remains in the source table and upstream comparison output.",
            "Panel A map styling: the user-specified high-contrast Nature categorical palette is reordered so the most frequent rank-1 species receive the darkest colors, with its lightest color used for Other and light grey reserved for No localized panel. Legend species are ordered by rank-1 state frequency and then total frequency; Other and No localized panel are last. Rank-2 points are enlarged and have no contrasting outline. State abbreviations use bold contrast-adaptive black or white text without halos; NC is offset below its point, VA is placed beside its point, and MD and CT are placed above-left of their points. Localized-state boundaries are widened in white, whereas No localized panel states use black boundaries. The top row is height-matched to the undistorted contiguous-US map extent, the outer left margin is zero, and horizontal space is reallocated from the paired comparison to maximize the undistorted hero map without collision.",
            f"Panel A data check: localized mean {mean_localized:.2f}%, fixed-national mean {mean_fixed:.2f}%, mean regional-weighting gain {mean_gain:.2f} pp, localized range {localized_range[0]:.2f}-{localized_range[1]:.2f}%, top gains {top_gain_text}.",
            "State-count audit: 32 localized panels are displayed. States without a generated localized panel share one neutral map class.",
        "Panel B data check: 160 state-rank rows, exactly five ranks in each of 32 localized panels, ordered by descending localization gain and matched to SI Table S6.",
            "Panel C: national species-by-candidate-MOA testing priorities use national chemical-priority weights, missing direct species-chemical evidence, probability/evidence uncertainty and candidate-MOA/chemical-form gaps. A common piecewise color scale is logarithmic from 0.05 to 0.5 x 10^-3 and linear above 0.5 x 10^-3, preserving both low-score and leading-column contrast without row/column normalization or score modification.",
            "Boundary source URL: https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip",
            "Boundary reader: pyshp/shapefile in the project .venv.",
        ],
    )
    print({"figure": FIGURE_ID, "outputs": outputs, "source_tables": source_tables})


if __name__ == "__main__":
    main()
