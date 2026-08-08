#!/usr/bin/env python3
"""Figure 4: warning-layer endpoint bridge and response-window potential."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
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
    save_contract_note,
    write_combined_source,
    write_csv,
    write_manifest,
)


FIGURE_ID = "figure_04"
TITLE = "Warning endpoints retain hazard rank and provide endpoint-specific timing and sensitivity signals"
MANUSCRIPT_CAPTION = (
    "Figure 4. The rapid-warning layer retains hazard rank and reveals endpoint-specific timing and sensitivity signals. "
    "a, Sensitivity-rank agreement between Top5-warning and EPA-framework full-data HC5 across 70 chemicals. "
    "b, Known-duration record distributions across warning and protective-apical endpoint "
    "families. c, Log10 concentration-ratio distributions for warning thresholds relative to matched "
    "protective-apical thresholds. d, Same-reference pairs classified as warning favored, equal, trade-off or "
    "protective-apical favored using reported-duration and concentration-threshold differences. Trade-off pairs had "
    "an earlier but higher warning response or a lower but later warning response. Same-reference pairs matched "
    "chemical, species, medium and ECOTOX reference."
)
HEARTBEAT_ACCENT = PALETTE["blue"]
VENTILATION_ACCENT = PALETTE["red"]

WARNING_FAMILIES = ["heartbeat", "behavior_feeding_avoidance", "other_physiology"]
PROTECTIVE_FAMILIES = [
    "mortality_survival",
    "immobilization_intoxication",
    "growth",
    "reproduction",
    "development_morphology",
]
APICAL_FAMILY_LABELS = {
    "mortality_survival": "Mortality/survival",
    "immobilization_intoxication": "Immobilization",
    "growth": "Growth",
    "reproduction": "Reproductive function",
    "development_morphology": "Development",
}
OTHER_PHYSIOLOGY_GROUPS = {
    "photosynthesis_psii": "Photosynthesis/PSII",
    "metabolism_oxygen": "Metabolism/oxygen",
    "pigment_luminescence": "Pigment/luminescence",
    "ventilation_circulation": "Ventilation/circulation",
    "other_physiology": "Other physiology",
}
BEHAVIOR_GROUPS = {
    "swimming_locomotion": "Swimming/locomotion",
    "equilibrium_behavior": "Equilibrium/behavior",
    "avoidance_taxis": "Avoidance/taxis",
    "feeding_filtering": "Feeding/filtering",
    "other_behavior": "Other behavior",
}
WARNING_ENDPOINT_ORDER = [
    "Heartbeat",
    BEHAVIOR_GROUPS["swimming_locomotion"],
    BEHAVIOR_GROUPS["equilibrium_behavior"],
    BEHAVIOR_GROUPS["avoidance_taxis"],
    BEHAVIOR_GROUPS["feeding_filtering"],
    BEHAVIOR_GROUPS["other_behavior"],
    "Photosynthesis/PSII",
    "Metabolism/oxygen",
    "Pigment/luminescence",
    "Ventilation/circulation",
    "Other physiology",
]
APICAL_ENDPOINT_ORDER = [
    "Mortality/survival",
    "Immobilization",
    "Growth",
    "Reproductive function",
    "Development",
]
ENDPOINT_ORDER = [*WARNING_ENDPOINT_ORDER, *APICAL_ENDPOINT_ORDER]
WARNING_ENDPOINTS = set(WARNING_ENDPOINT_ORDER)
DURATION_GROUP_ORDER = ["0-2 h", "2-12 h", "12-24 h", "24-48 h", "48-96 h", "96 h-7 d", "7-14 d", ">14 d", "Unknown"]
DURATION_COLORS = {
    "0-2 h": "#8B1E3F",
    "2-12 h": "#B64342",
    "12-24 h": "#E07A5F",
    "24-48 h": "#E5B85E",
    "48-96 h": "#88C0B8",
    "96 h-7 d": "#4F9AAF",
    "7-14 d": "#557EAA",
    ">14 d": "#6E6E8E",
    "Unknown": "#D9D9D9",
}
HEARTBEAT_DURATION_ORDER = ["0-2 h", "2-6 h", "6-24 h", "24-48 h", "48-96 h", "96 h-7 d", "7-14 d", ">14 d"]
HEARTBEAT_DURATION_MAP = {
    "h00_02_ultra_early": "0-2 h",
    "h02_06_early": "2-6 h",
    "h06_24_same_day": "6-24 h",
    "h24_48": "24-48 h",
    "h48_96": "48-96 h",
    "h96_168": "96 h-7 d",
    "d07_14": "7-14 d",
    "gt14d": ">14 d",
}
LIFE_STAGE_ORDER = ["embryo_egg", "larval_neonate", "juvenile", "adult", "mixed_multiple"]
LIFE_STAGE_LABELS = {
    "embryo_egg": "Embryo/egg",
    "larval_neonate": "Larval/neonate",
    "juvenile": "Juvenile",
    "adult": "Adult",
    "mixed_multiple": "Mixed",
}
RAW_DURATION_BINS = [-1e-9, 2, 12, 24, 48, 96, 168, 336, float("inf")]
RAW_DURATION_LABELS = DURATION_GROUP_ORDER[:-1]
MODEL_DURATION_LABELS = {
    "h00_02_ultra_early": "0-2 h",
    "h02_06_early": "2-12 h",
    "h06_24_same_day": "12-24 h",
    "h24_48": "24-48 h",
    "h48_96": "48-96 h",
    "h96_168": "96 h-7 d",
    "d07_14": "7-14 d",
    "gt14d": ">14 d",
}
MODEL_TIME_RANK = {
    "h00_02_ultra_early": 1,
    "h02_06_early": 2,
    "h06_24_same_day": 3,
    "h24_48": 4,
    "h48_96": 5,
    "h96_168": 6,
    "d07_14": 7,
    "gt14d": 8,
}
PANEL_C_QUADRANT_ORDER = [
    "warning_more_sensitive",
    "trade_off",
    "equal",
    "protective_apical_more_sensitive",
]
PANEL_C_QUADRANT_LABELS = {
    "warning_more_sensitive": "Warning favored",
    "trade_off": "Trade-off",
    "equal": "Equal",
    "protective_apical_more_sensitive": "Protective-apical favored",
}
PANEL_C_QUADRANT_COLORS = {
    "warning_more_sensitive": "#8B1E3F",
    "trade_off": "#E5B85E",
    "equal": "#C9C9C9",
    "protective_apical_more_sensitive": "#3775BA",
}
DISPLAY_DURATION_MID_H = {
    "0-2 h": 1.0,
    "2-12 h": 6.0,
    "12-24 h": 18.0,
    "24-48 h": 36.0,
    "48-96 h": 72.0,
    "96 h-7 d": 132.0,
    "7-14 d": 252.0,
    ">14 d": 504.0,
}
DURATION_EQUAL_ATOL_H = 1e-9
CONCENTRATION_EQUAL_ATOL_LOG10 = 1e-12
PHOTOSYNTHESIS_CODES = {"PSYN", "PSII", "PRSY"}
METABOLISM_OXYGEN_CODES = {"OXYG", "RESP", "SBNF", "NFIX", "NAFX", "NRXN", "FLUX", "ASML", "GPHY"}
PIGMENT_LUMINESCENCE_CODES = {"PIGM", "BLUM"}
VENTILATION_CODES = {"VENT", "TEVG"}
BEHAVIOR_SWIMMING_CODES = {"SWIM", "LOCO", "MOTL", "ACTV", "NMVM", "MMTE", "STLT", "SACT", "ACTP", "MSRB"}
BEHAVIOR_EQUILIBRIUM_CODES = {"EQUL", "GBHV", "FCNS", "STRS", "VISP", "NRES"}
BEHAVIOR_AVOIDANCE_TAXIS_CODES = {"PHTR", "CHEM", "VACL", "ESCR", "STIM", "AGGT"}
BEHAVIOR_FEEDING_CODES = {"FDNG", "FLTR", "GFDB", "FOOT"}


def aligned_panel_label(ax: plt.Axes, label: str, x_offset: float = -38, y_offset: float = 22) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12.0,
        color="black",
        clip_on=False,
    )


def measurement_code(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.upper() if ch.isalnum())


def endpoint_group_label(effect_family: object, measurement: object = "") -> str:
    family = str(effect_family)
    code = measurement_code(measurement)
    if family == "heartbeat":
        return "Heartbeat"
    if family == "behavior_feeding_avoidance":
        if code in BEHAVIOR_SWIMMING_CODES:
            return BEHAVIOR_GROUPS["swimming_locomotion"]
        if code in BEHAVIOR_EQUILIBRIUM_CODES:
            return BEHAVIOR_GROUPS["equilibrium_behavior"]
        if code in BEHAVIOR_AVOIDANCE_TAXIS_CODES:
            return BEHAVIOR_GROUPS["avoidance_taxis"]
        if code in BEHAVIOR_FEEDING_CODES:
            return BEHAVIOR_GROUPS["feeding_filtering"]
        return BEHAVIOR_GROUPS["other_behavior"]
    if family == "other_physiology":
        if code in PHOTOSYNTHESIS_CODES:
            return OTHER_PHYSIOLOGY_GROUPS["photosynthesis_psii"]
        if code in METABOLISM_OXYGEN_CODES:
            return OTHER_PHYSIOLOGY_GROUPS["metabolism_oxygen"]
        if code in PIGMENT_LUMINESCENCE_CODES:
            return OTHER_PHYSIOLOGY_GROUPS["pigment_luminescence"]
        if code in VENTILATION_CODES:
            return OTHER_PHYSIOLOGY_GROUPS["ventilation_circulation"]
        return OTHER_PHYSIOLOGY_GROUPS["other_physiology"]
    return APICAL_FAMILY_LABELS.get(family, "Other apical")


def endpoint_display_label(value: str) -> str:
    return value


def format_signed_two_decimals(value: float) -> str:
    magnitude = f"{abs(float(value)):.2f}"
    if magnitude == "0.00":
        return "0.00"
    return f"+{magnitude}" if value > 0 else f"−{magnitude}"


def duration_window_from_hours(hours: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                hours.notna() & (hours > 0) & (hours <= 2),
                hours.notna() & (hours > 2) & (hours <= 6),
                hours.notna() & (hours > 6) & (hours <= 24),
                hours.notna() & (hours > 24) & (hours <= 48),
                hours.notna() & (hours > 48) & (hours <= 96),
                hours.notna() & (hours > 96) & (hours <= 168),
                hours.notna() & (hours > 168) & (hours <= 336),
                hours.notna() & (hours > 336),
            ],
            [
                "h00_02_ultra_early",
                "h02_06_early",
                "h06_24_same_day",
                "h24_48",
                "h48_96",
                "h96_168",
                "d07_14",
                "gt14d",
            ],
            default="duration_unknown",
        ),
        index=hours.index,
    )


def add_endpoint_band_v16(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ef = out["effect_family"].fillna("").astype(str)
    lay = out["effect_evidence_layer"].fillna("").astype(str)
    fam = out["endpoint_family"].fillna("").astype(str)
    lev = out["endpoint_level_band"].fillna("").astype(str)
    out["endpoint_band_v16"] = np.select(
        [
            ef.eq("heartbeat"),
            lay.eq("mechanism_support"),
            lay.eq("early_warning"),
            lay.ne("protective_apical"),
            fam.isin({"no_observed_effect", "lowest_observed_effect", "derived_chronic_threshold"})
            | lev.isin({"no_effect", "lowest_effect", "low_effect_1_20", "derived_threshold", "zero_effect"}),
            lev.eq("median_effect_40_60") | fam.isin({"lethal_concentration", "lethal_dose"}),
        ],
        [
            "heartbeat_early_warning",
            "mechanistic_support",
            "other_early_warning",
            "other_context",
            "apical_low_effect_threshold",
            "apical_median_effect",
        ],
        default="apical_other_effect_level",
    )
    return out


def model_endpoint_subgroup_map(root: Path) -> pd.DataFrame:
    columns = [
        "result_id",
        "dtxsid",
        "latin_name",
        "effect_family",
        "effect_evidence_layer",
        "endpoint_family",
        "endpoint_level_band",
        "measurement",
        "exposure_duration_h",
        "medium_family",
        "waterborne_main",
    ]
    frames = []
    for path in sorted((root / "data/intermediate/ecotox_canonical_partitions").glob("part_*.csv.gz")):
        part = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
        part = part[
            part["waterborne_main"].fillna(False).astype(bool)
            & part["effect_family"].isin([*WARNING_FAMILIES, *PROTECTIVE_FAMILIES])
        ].copy()
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)
    raw["duration_window_v16"] = duration_window_from_hours(pd.to_numeric(raw["exposure_duration_h"], errors="coerce"))
    raw = add_endpoint_band_v16(raw)
    raw["endpoint_group_raw"] = [endpoint_group_label(row.effect_family, row.measurement) for row in raw.itertuples(index=False)]
    keys = ["dtxsid", "latin_name", "effect_family", "endpoint_band_v16", "duration_window_v16", "medium_family"]
    counts = raw.groupby([*keys, "endpoint_group_raw"], dropna=False, as_index=False).agg(n_raw_records=("result_id", "size"))
    totals = counts.groupby(keys, dropna=False)["n_raw_records"].sum().rename("n_raw_records_for_model_cell").reset_index()
    n_groups = counts.groupby(keys, dropna=False)["endpoint_group_raw"].nunique().rename("n_endpoint_subgroups").reset_index()
    dominant = counts.sort_values([*keys, "n_raw_records", "endpoint_group_raw"], ascending=[True, True, True, True, True, True, False, True])
    dominant = dominant.groupby(keys, dropna=False, as_index=False).first()
    dominant = dominant.merge(totals, on=keys, how="left").merge(n_groups, on=keys, how="left")
    dominant["dominant_subgroup_fraction"] = np.divide(
        dominant["n_raw_records"],
        dominant["n_raw_records_for_model_cell"],
        out=np.zeros(len(dominant), dtype=float),
        where=dominant["n_raw_records_for_model_cell"].to_numpy() > 0,
    )
    return dominant


def load_reference_rank(root: Path) -> pd.DataFrame:
    ref = pd.read_csv(root / "results/warning_hc5_bridge/chemical_medium_reference_hc5.csv")
    ref = ref.dropna(subset=["reference_hc5_typical_log10"]).copy()
    ref["epa_reference_rank_universe_n"] = len(ref)
    return ref


def build_panel_a(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    warning = pd.read_csv(root / "results/warning_hc5_bridge/top5_panel_warning_threshold_vs_reference_hc5.csv").dropna(
        subset=["reference_hc5_typical_log10", "panel_min_warning_log10"]
    )
    warning = warning.copy()
    warning["epa_hc5_sensitivity_rank"] = percentile_rank_high(warning["reference_hc5_typical_log10"])
    warning["top5_warning_trigger_rank"] = percentile_rank_high(warning["panel_min_warning_log10"])
    warning["log10_warning_to_epa_hc5_ratio"] = (
        warning["panel_min_warning_log10"] - warning["reference_hc5_typical_log10"]
    )
    warning["warning_to_epa_hc5_ratio"] = 10 ** warning["log10_warning_to_epa_hc5_ratio"]
    warning["rank_algorithm"] = (
        "x is rank(-panel_min_warning_log10, pct=True) * 100; y is rank(-reference_hc5_typical_log10, pct=True) * 100; "
        "both ranks are computed within chemicals with Top5-warning and EPA-framework full-data HC5 values"
    )
    ref = load_reference_rank(root)
    warning["epa_reference_rank_universe_n"] = len(ref)

    binned = warning.copy()
    binned["top5_warning_decile"] = pd.qcut(
        binned["top5_warning_trigger_rank"],
        q=10,
        labels=False,
        duplicates="drop",
    ).astype(int) + 1
    binned_summary = (
        binned.groupby("top5_warning_decile", as_index=False)
        .agg(
            n_pairs=("dtxsid", "size"),
            top5_warning_trigger_rank_median=("top5_warning_trigger_rank", "median"),
            epa_hc5_sensitivity_rank_median=("epa_hc5_sensitivity_rank", "median"),
        )
        .sort_values("top5_warning_decile")
    )
    binned_summary["summary_algorithm"] = "10 equal-count bins of Top5-warning sensitivity rank; y is median EPA-framework full-data HC5 sensitivity rank"

    metrics = pd.read_csv(root / "results/warning_hc5_bridge/top5_panel_warning_hc5_metrics.csv")
    metrics = metrics.loc[metrics["reference"].eq("reference_hc5_typical_log10")].copy()
    metrics.insert(0, "metric_source", "top5_warning_vs_epa_framework_full_data_hc5")
    metrics["epa_reference_rank_universe_n"] = len(ref)
    return warning, pd.concat([metrics, binned_summary], ignore_index=True, sort=False)


def build_panel_b(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "result_id",
        "test_id",
        "reference_number",
        "dtxsid",
        "latin_name",
        "common_name",
        "effect_family",
        "measurement",
        "endpoint",
        "exposure_duration_h",
        "broad_life_stage",
        "waterborne_main",
    ]
    frames = []
    for path in sorted((root / "data/intermediate/ecotox_canonical_partitions").glob("part_*.csv.gz")):
        part = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
        part = part[
            part["waterborne_main"].fillna(False).astype(bool)
            & part["effect_family"].isin([*WARNING_FAMILIES, *PROTECTIVE_FAMILIES])
        ].copy()
        frames.append(part)
    source = pd.concat(frames, ignore_index=True)
    source["endpoint_group"] = [endpoint_group_label(row.effect_family, row.measurement) for row in source.itertuples(index=False)]
    duration_h = pd.to_numeric(source["exposure_duration_h"], errors="coerce")
    source["duration_group"] = pd.cut(
        duration_h,
        bins=RAW_DURATION_BINS,
        labels=RAW_DURATION_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")
    source["duration_group"] = source["duration_group"].fillna("Unknown")
    reproductive_short = source[
        source["endpoint_group"].eq("Reproductive function")
        & source["duration_group"].eq("0-2 h")
    ].copy()
    reproductive_audit = (
        reproductive_short.groupby(
            ["latin_name", "common_name", "measurement", "endpoint", "broad_life_stage"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_records=("result_id", "size"),
            n_tests=("test_id", "nunique"),
            n_references=("reference_number", "nunique"),
            n_chemicals=("dtxsid", "nunique"),
            min_duration_h=("exposure_duration_h", "min"),
            max_duration_h=("exposure_duration_h", "max"),
        )
        .sort_values(["n_records", "latin_name", "measurement"], ascending=[False, True, True])
    )
    reproductive_audit["audit_note"] = (
        "0-2 h reproductive-function records are short reproductive-process assays, "
        "mainly sperm motility/velocity, fertilization or viability; they are not interpreted as standard reproduction-cycle tests."
    )
    rows = source.groupby(["endpoint_group", "duration_group"], as_index=False).agg(
        n_records=("result_id", "size"),
        n_tests=("test_id", "nunique"),
        n_references=("reference_number", "nunique"),
        n_chemicals=("dtxsid", "nunique"),
        n_species=("latin_name", "nunique"),
    )
    index = pd.MultiIndex.from_product([ENDPOINT_ORDER, DURATION_GROUP_ORDER], names=["endpoint_group", "duration_group"])
    out = rows.set_index(["endpoint_group", "duration_group"]).reindex(index).reset_index()
    for column in ["n_records", "n_tests", "n_references", "n_species", "n_chemicals"]:
        out[column] = out[column].fillna(0).astype(int)
    totals = out.groupby("endpoint_group")["n_records"].transform("sum")
    out["group_total_records"] = totals
    out["group_known_duration_records"] = out["endpoint_group"].map(
        out.loc[out["duration_group"].ne("Unknown")].groupby("endpoint_group")["n_records"].sum()
    ).fillna(0).astype(int)
    out["duration_fraction_within_group"] = np.divide(
        out["n_records"],
        totals,
        out=np.zeros(len(out), dtype=float),
        where=totals.to_numpy() > 0,
    )
    out["duration_fraction_among_known"] = np.divide(
        out["n_records"],
        out["group_known_duration_records"],
        out=np.zeros(len(out), dtype=float),
        where=out["group_known_duration_records"].to_numpy() > 0,
    )
    out.loc[out["duration_group"].eq("Unknown"), "duration_fraction_among_known"] = np.nan
    out["endpoint_group_order"] = out["endpoint_group"].map({name: index for index, name in enumerate(ENDPOINT_ORDER)})
    out["duration_group_order"] = out["duration_group"].map({name: index for index, name in enumerate(DURATION_GROUP_ORDER)})
    out["evidence_zone"] = np.where(out["endpoint_group"].isin(WARNING_ENDPOINTS), "warning", "protective_apical")
    displayed_duration_labels = [
        duration
        for duration in RAW_DURATION_LABELS
        if out.loc[out["duration_group"].eq(duration), "n_records"].sum() > 0
    ]
    endpoint_summary = pd.DataFrame({"endpoint_group": ENDPOINT_ORDER})
    zero_two = (
        out.loc[out["duration_group"].eq("0-2 h"), ["endpoint_group", "duration_fraction_among_known", "n_records"]]
        .rename(columns={"duration_fraction_among_known": "zero_to_two_h_share_among_known", "n_records": "zero_to_two_h_records"})
    )
    nonzero_bins = (
        out.loc[out["duration_group"].isin(displayed_duration_labels) & out["n_records"].gt(0)]
        .groupby("endpoint_group", as_index=False)
        .agg(nonzero_displayed_duration_bins=("duration_group", "nunique"))
    )
    endpoint_summary = endpoint_summary.merge(zero_two, on="endpoint_group", how="left").merge(nonzero_bins, on="endpoint_group", how="left")
    endpoint_summary["zero_to_two_h_share_among_known"] = endpoint_summary["zero_to_two_h_share_among_known"].fillna(0.0)
    endpoint_summary["zero_to_two_h_records"] = endpoint_summary["zero_to_two_h_records"].fillna(0).astype(int)
    endpoint_summary["nonzero_displayed_duration_bins"] = endpoint_summary["nonzero_displayed_duration_bins"].fillna(0).astype(int)
    endpoint_summary["displayed_duration_bin_count"] = len(displayed_duration_labels)
    endpoint_summary["zero_to_two_h_share_rank"] = (
        endpoint_summary["zero_to_two_h_share_among_known"].rank(method="min", ascending=False).astype(int)
    )
    endpoint_summary["duration_annotation"] = ""
    endpoint_summary.loc[
        endpoint_summary["endpoint_group"].eq("Ventilation/circulation")
        & endpoint_summary["zero_to_two_h_share_rank"].eq(1),
        "duration_annotation",
    ] = "Highest 0-2 h share"
    endpoint_summary.loc[
        endpoint_summary["endpoint_group"].eq("Heartbeat")
        & endpoint_summary["nonzero_displayed_duration_bins"].ge(
            endpoint_summary["displayed_duration_bin_count"].sub(1).clip(lower=0)
        ),
        "duration_annotation",
    ] = "Broad cross-duration coverage"
    endpoint_summary["duration_fraction_basis"] = (
        "Raw waterborne known-duration records; duration_fraction_among_known excludes Unknown records from the denominator."
    )
    out = out.merge(endpoint_summary, on="endpoint_group", how="left")
    return out.sort_values(["endpoint_group_order", "duration_group_order"]).reset_index(drop=True), reproductive_audit


def build_panel_c(root: Path) -> pd.DataFrame:
    model_columns = [
        "dtxsid",
        "latin_name",
        "medium_family",
        "effect_family",
        "endpoint_band_v16",
        "duration_window_v16",
        "mu_log10_umol_L",
        "mle_status",
        "n_records",
    ]
    model = pd.read_csv(root / "results/toxicity/censored_model_cells.csv.gz", usecols=lambda column: column in model_columns, low_memory=False)
    model = model[
        model["mle_status"].astype(str).str.startswith("identified")
        & np.isfinite(model["mu_log10_umol_L"])
        & model["duration_window_v16"].isin(MODEL_TIME_RANK)
        & model["effect_family"].isin([*WARNING_FAMILIES, *PROTECTIVE_FAMILIES])
    ].copy()
    model["cell_id"] = np.arange(len(model))
    model["time_rank"] = model["duration_window_v16"].map(MODEL_TIME_RANK).astype(int)
    model["duration_group"] = model["duration_window_v16"].map(MODEL_DURATION_LABELS)

    raw_columns = [
        "result_id",
        "test_id",
        "reference_number",
        "dtxsid",
        "latin_name",
        "effect_family",
        "effect_evidence_layer",
        "endpoint_family",
        "endpoint_level_band",
        "measurement",
        "exposure_duration_h",
        "medium_family",
        "waterborne_main",
    ]
    frames = []
    for path in sorted((root / "data/intermediate/ecotox_canonical_partitions").glob("part_*.csv.gz")):
        part = pd.read_csv(path, usecols=lambda column: column in raw_columns, low_memory=False)
        part = part[
            part["waterborne_main"].fillna(False).astype(bool)
            & part["effect_family"].isin([*WARNING_FAMILIES, *PROTECTIVE_FAMILIES])
        ].copy()
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)
    raw["reported_duration_h"] = pd.to_numeric(raw["exposure_duration_h"], errors="coerce")
    raw["duration_window_v16"] = duration_window_from_hours(raw["reported_duration_h"])
    raw = add_endpoint_band_v16(raw)
    raw["endpoint_group_raw"] = [endpoint_group_label(row.effect_family, row.measurement) for row in raw.itertuples(index=False)]

    keys = ["dtxsid", "latin_name", "medium_family", "effect_family", "endpoint_band_v16", "duration_window_v16"]
    cell_reference = (
        raw.dropna(subset=["reference_number"])
        .groupby([*keys, "reference_number"], dropna=False, as_index=False)
        .agg(
            min_reported_duration_h=("reported_duration_h", "min"),
            median_reported_duration_h=("reported_duration_h", "median"),
            max_reported_duration_h=("reported_duration_h", "max"),
            n_reference_records=("result_id", "size"),
            n_reference_duration_records=("reported_duration_h", "count"),
        )
    )
    subgroup_counts = raw.groupby([*keys, "endpoint_group_raw"], dropna=False, as_index=False).agg(n_raw_records=("result_id", "size"))
    subgroup_counts = (
        subgroup_counts.sort_values([*keys, "n_raw_records", "endpoint_group_raw"], ascending=[True, True, True, True, True, True, False, True])
        .groupby(keys, as_index=False)
        .first()
    )

    mapped = model.merge(subgroup_counts[[*keys, "endpoint_group_raw", "n_raw_records"]], on=keys, how="left")
    mapped["endpoint_group"] = mapped["effect_family"].map(
        {
            "heartbeat": "Heartbeat",
            **APICAL_FAMILY_LABELS,
        }
    )
    behavior_mask = mapped["effect_family"].eq("behavior_feeding_avoidance") & mapped["endpoint_group_raw"].notna()
    mapped.loc[behavior_mask, "endpoint_group"] = mapped.loc[behavior_mask, "endpoint_group_raw"]
    mapped.loc[mapped["effect_family"].eq("behavior_feeding_avoidance") & mapped["endpoint_group"].isna(), "endpoint_group"] = "Other behavior"
    other_mask = mapped["effect_family"].eq("other_physiology") & mapped["endpoint_group_raw"].notna()
    mapped.loc[other_mask, "endpoint_group"] = mapped.loc[other_mask, "endpoint_group_raw"]
    mapped.loc[mapped["effect_family"].eq("other_physiology") & mapped["endpoint_group"].isna(), "endpoint_group"] = "Other physiology"
    mapped = mapped.merge(cell_reference, on=keys, how="inner")

    warning = mapped[mapped["effect_family"].isin(WARNING_FAMILIES)].rename(
        columns={
            "cell_id": "warning_cell_id",
            "effect_family": "warning_effect_family",
            "endpoint_group": "warning_endpoint_group",
            "duration_window_v16": "warning_duration_window_v16",
            "duration_group": "warning_duration_group",
            "time_rank": "warning_time_rank",
            "mu_log10_umol_L": "warning_log10_threshold",
            "n_records": "n_warning_records",
            "min_reported_duration_h": "warning_min_reported_duration_h",
            "median_reported_duration_h": "warning_median_reported_duration_h",
            "max_reported_duration_h": "warning_max_reported_duration_h",
            "n_reference_records": "n_warning_reference_records",
            "n_reference_duration_records": "n_warning_reference_duration_records",
        }
    )
    apical = mapped[mapped["effect_family"].isin(PROTECTIVE_FAMILIES)].rename(
        columns={
            "cell_id": "apical_cell_id",
            "effect_family": "apical_effect_family",
            "endpoint_group": "apical_endpoint_group",
            "duration_window_v16": "apical_duration_window_v16",
            "duration_group": "apical_duration_group",
            "time_rank": "apical_time_rank",
            "mu_log10_umol_L": "apical_log10_threshold",
            "n_records": "n_apical_records",
            "min_reported_duration_h": "apical_min_reported_duration_h",
            "median_reported_duration_h": "apical_median_reported_duration_h",
            "max_reported_duration_h": "apical_max_reported_duration_h",
            "n_reference_records": "n_apical_reference_records",
            "n_reference_duration_records": "n_apical_reference_duration_records",
        }
    )
    all_pairs = warning.merge(
        apical,
        on=["dtxsid", "latin_name", "medium_family", "reference_number"],
        how="inner",
        suffixes=("", "_apical"),
    )
    all_pairs = all_pairs[all_pairs["warning_cell_id"].ne(all_pairs["apical_cell_id"])].copy()
    all_pairs = all_pairs.sort_values(
        [
            "warning_cell_id",
            "reference_number",
            "apical_log10_threshold",
            "apical_min_reported_duration_h",
            "apical_cell_id",
        ]
    )
    pairs = all_pairs.groupby(["warning_cell_id", "reference_number"], as_index=False).first()
    pairs["paired_duration_rank_delta"] = pairs["warning_time_rank"].astype(float) - pairs["apical_time_rank"].astype(float)
    pairs["paired_reported_duration_delta_h"] = (
        pairs["warning_min_reported_duration_h"].astype(float) - pairs["apical_min_reported_duration_h"].astype(float)
    )
    pairs["log10_warning_to_matched_protective_apical_ratio"] = pairs["warning_log10_threshold"] - pairs["apical_log10_threshold"]
    pairs["warning_to_matched_protective_apical_ratio"] = 10 ** pairs["log10_warning_to_matched_protective_apical_ratio"]
    duration_delta = pairs["paired_reported_duration_delta_h"].to_numpy(dtype=float)
    log_ratio = pairs["log10_warning_to_matched_protective_apical_ratio"].to_numpy(dtype=float)
    same_reported_duration = np.isclose(duration_delta, 0.0, atol=DURATION_EQUAL_ATOL_H)
    same_threshold = np.isclose(log_ratio, 0.0, atol=CONCENTRATION_EQUAL_ATOL_LOG10)
    pairs["warning_is_earlier"] = duration_delta < -DURATION_EQUAL_ATOL_H
    pairs["warning_is_later"] = duration_delta > DURATION_EQUAL_ATOL_H
    pairs["warning_has_same_reported_duration"] = same_reported_duration
    pairs["warning_is_lower_concentration"] = log_ratio < -CONCENTRATION_EQUAL_ATOL_LOG10
    pairs["warning_is_higher_concentration"] = log_ratio > CONCENTRATION_EQUAL_ATOL_LOG10
    pairs["warning_has_equal_threshold"] = same_threshold
    pairs["warning_is_at_or_below_apical"] = pairs["warning_is_lower_concentration"] | pairs["warning_has_equal_threshold"]
    pairs["warning_is_equal_time_and_threshold"] = pairs["warning_has_same_reported_duration"] & pairs["warning_has_equal_threshold"]
    same_or_earlier = pairs["warning_is_earlier"] | pairs["warning_has_same_reported_duration"]
    same_or_lower = pairs["warning_is_lower_concentration"] | pairs["warning_has_equal_threshold"]
    warning_more_sensitive = same_or_earlier & same_or_lower & ~pairs["warning_is_equal_time_and_threshold"]
    trade_off = (pairs["warning_is_earlier"] & pairs["warning_is_higher_concentration"]) | (
        pairs["warning_is_later"] & pairs["warning_is_lower_concentration"]
    )
    pairs["trade_off_type"] = np.select(
        [
            pairs["warning_is_later"] & pairs["warning_is_lower_concentration"],
            pairs["warning_is_earlier"] & pairs["warning_is_higher_concentration"],
        ],
        ["lower_concentration_but_later", "earlier_response_but_higher_concentration"],
        default="none",
    )
    pairs["lead_quadrant"] = np.select(
        [
            pairs["warning_is_equal_time_and_threshold"],
            warning_more_sensitive,
            trade_off,
        ],
        ["equal", "warning_more_sensitive", "trade_off"],
        default="protective_apical_more_sensitive",
    )
    pairs["lead_quadrant_label"] = pairs["lead_quadrant"].map(PANEL_C_QUADRANT_LABELS)
    pairs["concentration_comparison_basis"] = (
        "Same chemical-species-medium-reference pair; concentration sensitivity uses log10 warning threshold minus matched protective-apical threshold."
    )
    pairs["timing_comparison_basis"] = (
        "Earlier timing uses the shorter minimum reported exposure_duration_h among raw records linking each model cell to the same ECOTOX reference."
    )
    pairs["equal_rule"] = (
        f"Equal if reported-duration delta is within {DURATION_EQUAL_ATOL_H:g} h and log10 threshold ratio is within {CONCENTRATION_EQUAL_ATOL_LOG10:g}."
    )
    pairs["combined_category_rule"] = (
        "Warning favored means warning reported duration is shorter or equal and warning threshold is lower or equal, "
        "with at least one strict warning advantage; Equal means both differences are within the stated tolerances; "
        "Trade-off means warning is earlier but higher concentration or later but lower concentration; "
        "Protective-apical favored means warning reported duration is longer or equal and warning threshold is higher or equal, "
        "with at least one strict protective-apical advantage."
    )
    pairs["panel_use"] = "Pair-level source for Figure 4c concentration-ratio distributions and Figure 4d combined categories."
    pairs["endpoint_group_order"] = pairs["warning_endpoint_group"].map({name: index for index, name in enumerate(WARNING_ENDPOINT_ORDER)})
    pairs["pairing_scope"] = "same ECOTOX reference_number; lowest protective-apical threshold per warning cell/reference"
    keep = [
        "dtxsid",
        "panel_use",
        "latin_name",
        "medium_family",
        "reference_number",
        "warning_cell_id",
        "apical_cell_id",
        "warning_effect_family",
        "warning_endpoint_group",
        "apical_effect_family",
        "apical_endpoint_group",
        "warning_duration_window_v16",
        "warning_duration_group",
        "apical_duration_window_v16",
        "apical_duration_group",
        "warning_min_reported_duration_h",
        "warning_median_reported_duration_h",
        "warning_max_reported_duration_h",
        "apical_min_reported_duration_h",
        "apical_median_reported_duration_h",
        "apical_max_reported_duration_h",
        "warning_time_rank",
        "apical_time_rank",
        "paired_duration_rank_delta",
        "paired_reported_duration_delta_h",
        "warning_log10_threshold",
        "apical_log10_threshold",
        "log10_warning_to_matched_protective_apical_ratio",
        "warning_to_matched_protective_apical_ratio",
        "n_warning_records",
        "n_apical_records",
        "n_warning_reference_records",
        "n_warning_reference_duration_records",
        "n_apical_reference_records",
        "n_apical_reference_duration_records",
        "warning_is_earlier",
        "warning_is_later",
        "warning_has_same_reported_duration",
        "warning_is_lower_concentration",
        "warning_is_higher_concentration",
        "warning_has_equal_threshold",
        "warning_is_at_or_below_apical",
        "warning_is_equal_time_and_threshold",
        "trade_off_type",
        "lead_quadrant",
        "lead_quadrant_label",
        "concentration_comparison_basis",
        "timing_comparison_basis",
        "equal_rule",
        "combined_category_rule",
        "pairing_scope",
        "endpoint_group_order",
    ]
    rename_for_source = {
        "apical_cell_id": "protective_apical_cell_id",
        "apical_effect_family": "protective_apical_effect_family",
        "apical_endpoint_group": "protective_apical_endpoint_group",
        "apical_duration_window_v16": "protective_apical_duration_window_v16",
        "apical_duration_group": "protective_apical_duration_group",
        "apical_min_reported_duration_h": "protective_apical_min_reported_duration_h",
        "apical_median_reported_duration_h": "protective_apical_median_reported_duration_h",
        "apical_max_reported_duration_h": "protective_apical_max_reported_duration_h",
        "apical_time_rank": "protective_apical_time_rank",
        "apical_log10_threshold": "protective_apical_log10_threshold",
        "n_apical_records": "n_protective_apical_records",
        "n_apical_reference_records": "n_protective_apical_reference_records",
        "n_apical_reference_duration_records": "n_protective_apical_reference_duration_records",
        "warning_is_at_or_below_apical": "warning_is_at_or_below_protective_apical",
    }
    return (
        pairs[keep]
        .sort_values(["endpoint_group_order", "reference_number", "warning_cell_id"])
        .rename(columns=rename_for_source)
        .reset_index(drop=True)
    )


def draw_panel_a(ax: plt.Axes, panel_a: pd.DataFrame, panel_a_metrics: pd.DataFrame) -> None:
    ax.scatter(
        panel_a["top5_warning_trigger_rank"],
        panel_a["epa_hc5_sensitivity_rank"],
        s=22,
        color=PALETTE["red"],
        alpha=0.74,
        edgecolor="white",
        linewidth=0.25,
        label="Chemical pairs",
        zorder=3,
    )
    binned = panel_a_metrics.dropna(subset=["top5_warning_trigger_rank_median", "epa_hc5_sensitivity_rank_median"])
    ax.plot(
        binned["top5_warning_trigger_rank_median"],
        binned["epa_hc5_sensitivity_rank_median"],
        marker="o",
        ms=3.2,
        lw=1.3,
        color=PALETTE["neutral_dark"],
        label="Binned median",
        zorder=4,
    )
    ax.plot([0, 100], [0, 100], ls="--", lw=0.85, color=PALETTE["neutral"], label="Rank agreement", zorder=1)
    metrics = panel_a_metrics.dropna(subset=["spearman_r"]).iloc[0]
    ax.text(
        4,
        96,
        f"n = {int(metrics['n'])}; Spearman ρ = {float(metrics['spearman_r']):.3f}",
        fontsize=5.7,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.8),
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Top5-warning sensitivity rank (%)")
    ax.set_ylabel("EPA-framework full-data HC5 sensitivity rank (%)")
    ax.set_title("Top5-warning rank agreement with EPA-framework HC5")
    handles, labels = ax.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))
    legend_order = ["Chemical pairs", "Rank agreement", "Binned median"]
    ax.legend(
        handles=[handle_map[label] for label in legend_order],
        labels=legend_order,
        fontsize=5.2,
        loc="lower right",
        ncol=2,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.3,
        columnspacing=0.65,
        borderpad=0.35,
        frameon=True,
        facecolor="white",
        edgecolor=PALETTE["neutral_light"],
        linewidth=0.6,
        framealpha=1.0,
    )
    clean_axis(ax, grid=True)
    aligned_panel_label(ax, "a")


def draw_panel_b(ax: plt.Axes, panel_b: pd.DataFrame) -> None:
    endpoint_order = ["__warning_header__", *WARNING_ENDPOINT_ORDER, "__apical_header__", *APICAL_ENDPOINT_ORDER]
    shown_durations = [
        duration
        for duration in DURATION_GROUP_ORDER
        if duration != "Unknown" and panel_b.loc[panel_b["duration_group"].eq(duration), "n_records"].sum() > 0
    ]

    pivot = (
        panel_b.pivot(index="endpoint_group", columns="duration_group", values="duration_fraction_among_known")
        .reindex(index=endpoint_order, columns=shown_durations)
        .fillna(0)
    )
    totals = panel_b.drop_duplicates("endpoint_group").set_index("endpoint_group").reindex(endpoint_order)["group_known_duration_records"].fillna(0).astype(int)
    y = np.arange(len(endpoint_order))
    warning_end = len(WARNING_ENDPOINT_ORDER)
    apical_start = warning_end + 1
    ax.axhspan(-0.5, warning_end + 0.5, facecolor=PALETTE["red_light"], alpha=0.22, zorder=-2)
    ax.axhspan(apical_start - 0.5, len(endpoint_order) - 0.5, facecolor=PALETTE["blue_light"], alpha=0.36, zorder=-2)
    ax.text(
        0.012,
        0,
        "Warning endpoints",
        va="center",
        ha="left",
        fontsize=6.1,
        fontweight="bold",
        color=PALETTE["red"],
    )
    ax.text(
        0.012,
        apical_start,
        "Protective-apical endpoints",
        va="center",
        ha="left",
        fontsize=6.1,
        fontweight="bold",
        color=PALETTE["blue2"],
    )
    left = np.zeros(len(endpoint_order), dtype=float)
    for duration in shown_durations:
        values = pivot[duration].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.66,
            color=DURATION_COLORS[duration],
            edgecolor="white",
            linewidth=0.45,
            label=duration,
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(["", *[endpoint_display_label(name) for name in WARNING_ENDPOINT_ORDER], "", *APICAL_ENDPOINT_ORDER])
    for tick, label in zip(ax.get_yticklabels(), endpoint_order):
        if label == "Heartbeat":
            tick.set_fontweight("bold")
            tick.set_color(HEARTBEAT_ACCENT)
        elif label == "Ventilation/circulation":
            tick.set_fontweight("bold")
            tick.set_color(VENTILATION_ACCENT)
    meta = panel_b.drop_duplicates("endpoint_group").set_index("endpoint_group")
    ax.set_ylim(len(endpoint_order) - 0.45, -0.70)
    ax.set_xlim(0, 1.0)
    ax.axhline(apical_start - 0.5, color=PALETTE["neutral_dark"], lw=0.75)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Fraction of known-duration records")
    ax.set_title("Known-duration support by endpoint family", pad=10)
    for ypos, group in zip(y, endpoint_order):
        if group.startswith("__"):
            continue
        color = PALETTE["neutral_dark"]
        fontweight = "normal"
        statistic = f"n = {int(totals.loc[group]):,}"
        if group == "Heartbeat":
            statistic = (
                f"n = {int(totals.loc[group]):,}\n"
                f"{int(meta.loc[group, 'nonzero_displayed_duration_bins'])}/"
                f"{int(meta.loc[group, 'displayed_duration_bin_count'])} windows"
            )
            color = HEARTBEAT_ACCENT
            fontweight = "bold"
        elif group == "Ventilation/circulation":
            statistic = (
                f"n = {int(totals.loc[group]):,}\n"
                f"0–2 h: {float(meta.loc[group, 'zero_to_two_h_share_among_known']):.1%}"
            )
            color = VENTILATION_ACCENT
            fontweight = "bold"
        ax.text(
            1.018,
            ypos,
            statistic,
            va="center",
            ha="left",
            fontsize=5.1,
            fontweight=fontweight,
            color=color,
            linespacing=0.90,
            clip_on=False,
        )
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ax.legend(
        handles=[label_to_handle[duration] for duration in shown_durations],
        labels=shown_durations,
        fontsize=6.15,
        ncol=len(shown_durations),
        loc="upper center",
        bbox_to_anchor=(0.50, 1.040),
        columnspacing=0.30,
        handlelength=0.70,
        handletextpad=0.18,
        labelspacing=0.18,
        borderpad=0.0,
        frameon=False,
    )
    clean_axis(ax, grid=False)
    ax.spines["bottom"].set_visible(True)
    aligned_panel_label(ax, "b")


def paired_endpoint_summary(pairs: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    endpoint_order = [name for name in WARNING_ENDPOINT_ORDER if name in set(pairs["warning_endpoint_group"])]
    summary_rows = []
    for endpoint in endpoint_order:
        group = pairs[pairs["warning_endpoint_group"].eq(endpoint)]
        row = {
            "warning_endpoint_group": endpoint,
            "n_pairs": len(group),
            "n_chemicals": group["dtxsid"].nunique(),
            "n_references": group["reference_number"].nunique(),
            "median_log10_ratio": group["log10_warning_to_matched_protective_apical_ratio"].median(),
            "q25_log10_ratio": group["log10_warning_to_matched_protective_apical_ratio"].quantile(0.25),
            "q75_log10_ratio": group["log10_warning_to_matched_protective_apical_ratio"].quantile(0.75),
            "pct_lower": group["warning_is_lower_concentration"].mean(),
            "pct_at_or_below": group["warning_is_at_or_below_protective_apical"].mean(),
            "pct_earlier": group["warning_is_earlier"].mean(),
            "median_reported_duration_delta_h": group["paired_reported_duration_delta_h"].median(),
        }
        counts = group["lead_quadrant"].value_counts(normalize=True)
        for quadrant in PANEL_C_QUADRANT_ORDER:
            row[quadrant] = counts.get(quadrant, 0.0)
        summary_rows.append(row)
    return endpoint_order, pd.DataFrame(summary_rows)


def build_panel_c_ratio_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    _, summary = paired_endpoint_summary(pairs)
    summary = summary.copy()
    summary.insert(0, "panel_use", "figure_04c_paired_concentration_ratio_distributions")
    summary["x_axis_definition"] = "log10(warning / matched protective-apical)"
    summary["negative_values_mean"] = "Warning threshold lower in the same chemical-species-medium-reference pair"
    summary["positive_values_mean"] = "Protective-apical threshold lower in the same chemical-species-medium-reference pair"
    return summary


def build_panel_d_category_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    endpoint_order, summary = paired_endpoint_summary(pairs)
    rows = []
    for endpoint in endpoint_order:
        group = pairs[pairs["warning_endpoint_group"].eq(endpoint)]
        counts = group["lead_quadrant"].value_counts()
        trade_counts = group["trade_off_type"].value_counts()
        for category in PANEL_C_QUADRANT_ORDER:
            n = int(counts.get(category, 0))
            rows.append(
                {
                    "panel_use": "figure_04d_combined_timing_concentration_categories",
                    "warning_endpoint_group": endpoint,
                    "lead_quadrant": category,
                    "lead_quadrant_label": PANEL_C_QUADRANT_LABELS[category],
                    "n_pairs": n,
                    "fraction_of_endpoint_pairs": n / len(group) if len(group) else np.nan,
                    "endpoint_pair_total": len(group),
                    "n_chemicals": group["dtxsid"].nunique(),
                    "n_references": group["reference_number"].nunique(),
                    "trade_off_earlier_but_higher_concentration": int(
                        trade_counts.get("earlier_response_but_higher_concentration", 0)
                    )
                    if category == "trade_off"
                    else 0,
                    "trade_off_lower_concentration_but_later": int(trade_counts.get("lower_concentration_but_later", 0))
                    if category == "trade_off"
                    else 0,
                    "category_definition": (
                        "Warning favored: warning reported duration is shorter or equal and warning threshold is lower or equal, "
                        "with at least one strict warning advantage; "
                        "Equal: both reported-duration and model-threshold differences are within the stated tolerances; "
                        "Trade-off: warning is earlier but higher concentration, or warning is later but lower concentration; "
                        "Protective-apical favored: warning reported duration is longer or equal and warning threshold is higher or equal, "
                        "with at least one strict protective-apical advantage."
                    ),
                }
            )
    return pd.DataFrame(rows)


def draw_panel_d_categories(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    endpoint_order, summary = paired_endpoint_summary(pairs)
    y = np.arange(len(endpoint_order))

    ax.axvspan(0, 1, facecolor="#FFF3F1", alpha=0.70, zorder=-3)
    left = np.zeros(len(summary), dtype=float)
    for quadrant in PANEL_C_QUADRANT_ORDER:
        values = summary[quadrant].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.64,
            color=PANEL_C_QUADRANT_COLORS[quadrant],
            edgecolor="white",
            linewidth=0.45,
            label=PANEL_C_QUADRANT_LABELS[quadrant],
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([endpoint_display_label(name) for name in endpoint_order])
    for tick, endpoint in zip(ax.get_yticklabels(), endpoint_order):
        if endpoint == "Ventilation/circulation":
            tick.set_fontweight("bold")
            tick.set_color(VENTILATION_ACCENT)
    ax.invert_yaxis()
    ax.set_ylim(len(endpoint_order) - 0.5, -1.52)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Fraction of same-reference pairs")
    ax.set_title("Combined timing-concentration categories", pad=4)
    for idx, row in summary.iterrows():
        is_ventilation = row["warning_endpoint_group"] == "Ventilation/circulation"
        ax.text(
            1.018,
            idx,
            f"n = {int(row['n_pairs']):,}",
            va="center",
            ha="left",
            fontsize=5.1,
            fontweight="bold" if is_ventilation else "normal",
            color=VENTILATION_ACCENT if is_ventilation else PALETTE["neutral_dark"],
            clip_on=False,
        )
    legend_order = ["warning_more_sensitive", "trade_off", "equal", "protective_apical_more_sensitive"]
    ax.legend(
        handles=[Patch(facecolor=PANEL_C_QUADRANT_COLORS[q], label=PANEL_C_QUADRANT_LABELS[q]) for q in legend_order],
        fontsize=6.05,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.000),
        columnspacing=0.52,
        handlelength=0.74,
        handletextpad=0.22,
        labelspacing=0.18,
        borderpad=0.0,
        frameon=False,
    )
    clean_axis(ax, grid=False)
    ax.spines["bottom"].set_visible(True)
    aligned_panel_label(ax, "d")


def draw_panel_c_ratio(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    endpoint_order, summary = paired_endpoint_summary(pairs)
    y = np.arange(len(endpoint_order))

    ax.axvspan(-3, 0, facecolor=PALETTE["red_light"], alpha=0.20, zorder=-3)
    ax.axvspan(0, 3, facecolor=PALETTE["blue_light"], alpha=0.18, zorder=-3)
    ax.axvline(0, color=PALETTE["blue2"], lw=0.9, ls="--", zorder=2)
    rng = np.random.default_rng(202604)
    for idx, endpoint in enumerate(endpoint_order):
        group = pairs[pairs["warning_endpoint_group"].eq(endpoint)].copy()
        values = group["log10_warning_to_matched_protective_apical_ratio"].clip(-3, 3).to_numpy(dtype=float)
        if len(values):
            sample = values if len(values) <= 240 else rng.choice(values, size=240, replace=False)
            jitter = rng.normal(0, 0.055, size=len(sample))
            ax.scatter(sample, idx + jitter, s=5.5, color=PALETTE["neutral_dark"], alpha=0.12, linewidth=0, zorder=1)
        row = summary.loc[summary["warning_endpoint_group"].eq(endpoint)].iloc[0]
        q25 = np.clip(row["q25_log10_ratio"], -3, 3)
        q75 = np.clip(row["q75_log10_ratio"], -3, 3)
        med = np.clip(row["median_log10_ratio"], -3, 3)
        color = VENTILATION_ACCENT if endpoint == "Ventilation/circulation" and row["median_log10_ratio"] < 0 else PALETTE["neutral_dark"]
        ax.plot([q25, q75], [idx, idx], color=color, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter([med], [idx], s=20, color=color, edgecolor="white", linewidth=0.35, zorder=4)
        ax.text(
            3.08,
            idx,
            format_signed_two_decimals(row["median_log10_ratio"]),
            va="center",
            ha="left",
            fontsize=5.1,
            color=color,
            clip_on=False,
        )
    ax.set_xlim(-3, 3)
    ax.set_ylim(len(endpoint_order) - 0.5, -1.52)
    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
    ax.set_xticklabels(["<=-3", "-1.5", "0", "+1.5", ">=+3"], fontsize=5.3)
    ax.set_yticks(y)
    ax.set_yticklabels([endpoint_display_label(name) for name in endpoint_order])
    for tick, endpoint in zip(ax.get_yticklabels(), endpoint_order):
        if endpoint == "Ventilation/circulation":
            tick.set_fontweight("bold")
            tick.set_color(VENTILATION_ACCENT)
    ax.set_xlabel("log10(warning / matched protective-apical)")
    ax.set_title("Paired concentration-ratio distributions", pad=4)
    ax.text(
        0.01,
        0.992,
        "Warning threshold\nlower",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.55,
        fontweight="bold",
        color=PALETTE["red"],
    )
    ax.text(
        0.99,
        0.992,
        "Protective-apical\nthreshold lower",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.55,
        fontweight="bold",
        color=PALETTE["blue2"],
    )
    clean_axis(ax, grid=True)
    ax.spines["bottom"].set_visible(True)
    aligned_panel_label(ax, "c")


def draw(root: Path, out_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    apply_style()
    panel_a, panel_a_metrics = build_panel_a(root)
    panel_b, reproductive_short_audit = build_panel_b(root)
    panel_pairs = build_panel_c(root)
    panel_c_ratio_summary = build_panel_c_ratio_summary(panel_pairs)
    panel_d_category_summary = build_panel_d_category_summary(panel_pairs)

    fig = plt.figure(figsize=(7.65, 7.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.20], width_ratios=[1.0, 1.30], wspace=0.58, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, panel_a, panel_a_metrics)
    draw_panel_b(ax_b, panel_b)
    draw_panel_c_ratio(ax_c, panel_pairs)
    draw_panel_d_categories(ax_d, panel_pairs)

    outputs = export_figure(fig, out_dir / "figure_04_warning_endpoint_bridge")
    return outputs, {
        "figure_04a_warning_rank_percentile": panel_a,
        "figure_04a_warning_hc5_metrics": panel_a_metrics,
        "figure_04b_endpoint_duration_distribution": panel_b,
        "figure_04b_reproductive_function_0_2h_audit": reproductive_short_audit,
        "figure_04c_paired_concentration_ratio_summary": panel_c_ratio_summary,
        "figure_04d_combined_timing_concentration_categories": panel_d_category_summary,
        "figure_04c_same_reference_warning_apical_pairs": panel_pairs,
    }


def main() -> None:
    root = project_root_from_script(SCRIPT)
    out_dir = figure_dir_from_script(SCRIPT)
    outputs, tables = draw(root, out_dir)
    source_tables = []
    for name, table in tables.items():
        source_tables.append(write_csv(table, out_dir / f"{name}.csv"))
    source_tables.append(write_combined_source(tables, out_dir / "source_data.csv"))
    rank_metrics = tables["figure_04a_warning_hc5_metrics"].dropna(subset=["spearman_r"]).iloc[0]
    duration_meta = tables["figure_04b_endpoint_duration_distribution"].drop_duplicates("endpoint_group").set_index("endpoint_group")
    vent_meta = duration_meta.loc["Ventilation/circulation"]
    heartbeat_meta = duration_meta.loc["Heartbeat"]
    pairs = tables["figure_04c_same_reference_warning_apical_pairs"]
    trade_type_counts = pairs["trade_off_type"].value_counts()
    save_contract_note(
        out_dir,
        [
            f"# {FIGURE_ID} contract",
            "Core conclusion: Top5-warning preserves EPA-framework full-data HC5 hazard rank, while known-duration records and same-reference warning-protective-apical pairs separate endpoint-specific timing and concentration signals.",
            "Archetype: quantitative grid with endpoint-family duration comparison.",
            "Panel map: a Top5-warning/EPA-framework full-data HC5 sensitivity-rank agreement; b known-duration raw record distributions across expanded warning and protective-apical endpoint families; c log10 warning / matched protective-apical threshold-ratio distributions; d combined timing-concentration categories for the same pairs.",
            (
                f"Audit: panel a uses n = {int(rank_metrics['n'])} rows and Spearman rho = {float(rank_metrics['spearman_r']):.6f}; "
                f"panel b uses raw known-duration records; ventilation/circulation 0-2 h share = {float(vent_meta['zero_to_two_h_share_among_known']):.6f} "
                f"(rank {int(vent_meta['zero_to_two_h_share_rank'])}); heartbeat covers {int(heartbeat_meta['nonzero_displayed_duration_bins'])} "
                f"of {int(heartbeat_meta['displayed_duration_bin_count'])} displayed duration bins."
            ),
            (
                f"Panel d trade-off audit: earlier-but-higher concentration = {int(trade_type_counts.get('earlier_response_but_higher_concentration', 0))} pairs; "
                f"lower-concentration-but-later = {int(trade_type_counts.get('lower_concentration_but_later', 0))} pairs."
            ),
            f"Manuscript caption: {MANUSCRIPT_CAPTION}",
        ],
    )
    write_manifest(
        out_dir,
        figure_id=FIGURE_ID,
        title=TITLE,
        manuscript_caption=MANUSCRIPT_CAPTION,
        core_conclusion="Top5-warning preserves EPA-framework full-data HC5 hazard rank, while endpoint families show distinct known-duration and matched-pair concentration signals.",
        archetype="quantitative grid with endpoint-family duration comparison",
        panel_map={
            "a": "Clean diagonal rank distribution between Top5-warning sensitivity rank and EPA-framework full-data HC5 sensitivity rank, without endpoint-family coloring.",
            "b": "Known-duration raw record distribution for expanded warning endpoint families and protective-apical endpoint families, using raw exposure_duration_h bins.",
            "c": "Distribution of log10 warning threshold divided by the lowest matched protective-apical threshold for the same chemical, species, medium and ECOTOX reference.",
            "d": "Same-reference warning-protective-apical pair fractions classified by combined reported-duration timing and concentration ordering.",
        },
        source_tables=source_tables,
        input_dependencies=[
            "results/warning_hc5_bridge/top5_panel_warning_threshold_vs_reference_hc5.csv",
            "results/warning_hc5_bridge/top5_panel_warning_hc5_metrics.csv",
            "results/warning_hc5_bridge/chemical_medium_reference_hc5.csv",
            "results/toxicity/censored_model_cells.csv.gz",
            "data/intermediate/ecotox_canonical_partitions/part_*.csv.gz",
        ],
        outputs=outputs,
        reviewer_risk="The figure supports endpoint-specific warning-layer screening and study-design signals; protective-apical evidence remains the basis for formal benchmark follow-up.",
        notes=[
            "Panel A uses current Top5-warning rows paired to EPA-framework full-data HC5 rows from stage 16.",
        f"Panel A correlation audit: Spearman rho on plotted rank variables = {float(rank_metrics['spearman_r']):.6f}. Pearson r and concentration-ratio diagnostics remain in the source metrics and SI Table S5.",
            "Panel B uses all waterborne raw records with known exposure_duration_h for displayed fractions; unknown-duration records remain in source data but are not plotted.",
            "Panels B-D split the broad behavior_feeding_avoidance family by dominant raw measurement code: swimming/locomotion, equilibrium/behavior, avoidance/taxis, feeding/filtering and other behavior.",
            "Panel B expands other physiology using raw measurement-code families: photosynthesis/PSII, metabolism/oxygen, pigment/luminescence, ventilation/circulation and other physiology.",
            "Panel B bins raw exposure_duration_h as right-closed intervals: 0-2 h, 2-12 h, 12-24 h, 24-48 h, 48-96 h, 96 h-7 d, 7-14 d and >14 d.",
            f"Panel B short-window annotation is data-derived: ventilation/circulation 0-2 h share = {float(vent_meta['zero_to_two_h_share_among_known']):.6f}, rank {int(vent_meta['zero_to_two_h_share_rank'])} among displayed endpoint families.",
            f"Panel B heartbeat annotation is data-derived: heartbeat has nonzero known-duration records in {int(heartbeat_meta['nonzero_displayed_duration_bins'])} of {int(heartbeat_meta['displayed_duration_bin_count'])} displayed duration bins.",
            "Panel B 0-2 h reproductive-function records are short reproductive-process assays, mainly sperm motility/velocity, fertilization or viability; they are audited in a separate source table and are not interpreted as standard reproduction-cycle tests.",
            "Panels C-D use same-reference warning-protective-apical pairs only: chemical, species, medium and ECOTOX reference_number must match; for each warning cell/reference, the comparator is the lowest matched protective-apical threshold, with the shorter minimum reported protective-apical duration used as a tie-breaker.",
            "Panel C uses log10(warning / matched protective-apical) from censored model-cell thresholds; raw ECOTOX reference_number constrains the pairing and contributes reported-duration timing metadata.",
            "Panel D combines two explicit dimensions: concentration ordering is the same-pair lower model threshold, and timing ordering is the shorter minimum reported exposure_duration_h among raw records that link each model cell to the same ECOTOX reference.",
            "Panel D favored-category definitions are exact: Warning favored requires warning duration to be shorter or equal and warning threshold to be lower or equal, with at least one strict warning advantage; Protective-apical favored requires the converse ordering, with at least one strict protective-apical advantage.",
            f"Panel D equal uses tolerances of {DURATION_EQUAL_ATOL_H:g} h for reported-duration delta and {CONCENTRATION_EQUAL_ATOL_LOG10:g} log10 units for threshold ratio.",
            f"Panel D trade-off subtypes are recorded separately in source data: earlier-but-higher concentration = {int(trade_type_counts.get('earlier_response_but_higher_concentration', 0))} pairs and lower-concentration-but-later = {int(trade_type_counts.get('lower_concentration_but_later', 0))} pairs.",
            "The external Daphnia magna 1 h heartbeat dataset is not included.",
        ],
    )
    print({"figure": FIGURE_ID, "outputs": outputs, "source_tables": source_tables})


if __name__ == "__main__":
    main()
