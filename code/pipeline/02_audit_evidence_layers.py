#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WARNING_BANDS = {"heartbeat_early_warning", "other_early_warning"}
WARNING_DISPLAY_ORDER = [
    "h00_02_ultra_early",
    "h02_06_early",
    "h06_24_same_day",
    "h24_48",
    "h48_168",
    "gt7d",
    "duration_unknown",
]


def derive_acute_standard(d: pd.DataFrame) -> pd.Series:
    cl = d["class"].fillna("").astype(str)
    sp = d.latin_name.fillna("").str.lower()
    kg = d.kingdom.fillna("").astype(str)
    ef = d.effect_family.fillna("").astype(str)
    dw = d.duration_window_v16.fillna("").astype(str)
    branch = (
        ef.isin({"mortality_survival", "immobilization_intoxication"})
        & (cl.eq("Branchiopoda") | sp.str.contains("daphnia|ceriodaphnia", regex=True, na=False))
        & dw.eq("h24_48")
    )
    fish = ef.eq("mortality_survival") & cl.eq("Actinopterygii") & dw.eq("h48_96")
    algae = (
        ef.eq("growth")
        & cl.isin({"Chlorophyceae", "Trebouxiophyceae", "Cyanophyceae", "Cyanobacteriia"})
        & dw.eq("h48_96")
    )
    plant = ef.eq("growth") & kg.eq("Plantae") & dw.isin({"h96_168", "d07_14"})
    return pd.Series(
        np.select(
            [branch, fish, algae, plant],
            [
                "standard_48h_branchiopod",
                "standard_96h_fish",
                "standard_72_96h_algae",
                "standard_4_14d_plant",
            ],
            default="not_in_recognized_standard_window",
        ),
        index=d.index,
    )


def warning_display_window(s: pd.Series) -> pd.Series:
    """Map granular model bins into the six reporting windows requested for warning-response analysis."""
    return s.map(
        {
            "h00_02_ultra_early": "h00_02_ultra_early",
            "h02_06_early": "h02_06_early",
            "h06_24_same_day": "h06_24_same_day",
            "h24_48": "h24_48",
            "h48_96": "h48_168",
            "h96_168": "h48_168",
            "d07_14": "gt7d",
            "gt14d": "gt7d",
            "duration_unknown": "duration_unknown",
        }
    ).fillna("duration_unknown")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.model, low_memory=False)
    valid = d.mle_status.astype(str).str.startswith("identified")
    d = d[valid].copy()
    d["acute_standard_group"] = derive_acute_standard(d)

    # Warning/response evidence is retained across the full time axis. It remains separate from
    # acute-standard apical toxicity and does not include the external 1 h Daphnia experiment.
    warning = d[d.endpoint_band_v16.isin(WARNING_BANDS)].copy()
    warning["warning_response_window"] = warning_display_window(warning.duration_window_v16)
    warning_support = (
        warning.groupby(
            ["endpoint_band_v16", "effect_family", "warning_response_window"], dropna=False
        )
        .agg(
            n_model_cells=("dtxsid", "size"),
            n_chemicals=("dtxsid", "nunique"),
            n_species=("latin_name", "nunique"),
        )
        .reset_index()
    )
    warning_support["warning_response_window"] = pd.Categorical(
        warning_support.warning_response_window,
        categories=WARNING_DISPLAY_ORDER,
        ordered=True,
    )
    warning_support = warning_support.sort_values(
        ["warning_response_window", "endpoint_band_v16", "effect_family"]
    )
    warning_support.to_csv(out / "rapid_warning_full_duration_support.csv", index=False)
    # Backward-compatible explicit hourly subset.
    warning_support[
        warning_support.warning_response_window.isin(WARNING_DISPLAY_ORDER[:3])
    ].to_csv(out / "rapid_warning_hourly_support_only.csv", index=False)

    acute = d[
        d.acute_standard_group.str.startswith("standard_")
        & d.endpoint_band_v16.astype(str).str.startswith("apical_")
    ].copy()
    acute.groupby(
        ["acute_standard_group", "effect_family", "endpoint_band_v16"], dropna=False
    ).agg(
        n_model_cells=("dtxsid", "size"),
        n_chemicals=("dtxsid", "nunique"),
        n_species=("latin_name", "nunique"),
    ).reset_index().to_csv(out / "acute_standard_toxicity_support_only.csv", index=False)

    rows = []
    fields = [
        "effect_family",
        "endpoint_band_v16",
        "duration_window_v16",
        "medium_family",
        "mle_status",
        "kingdom",
        "phylum_division",
        "class",
        "tax_order",
        "family",
        "genus",
    ]
    for field in fields:
        series = d[field].astype("string")
        missing = series.isna() | series.str.strip().eq("")
        unknown = series.fillna("").str.lower().str.contains(
            "unknown|unmapped|not_assessable", regex=True
        )
        rows.append(
            {
                "field": field,
                "unit": "identified_model_cell",
                "n_units": len(d),
                "n_missing_literal": int(missing.sum()),
                "n_explicit_unknown_or_unmapped": int(unknown.sum()),
                "n_unique_nonmissing": int(series[~missing].nunique()),
                "mapping_policy": "missing, explicitly unspecified, and unmapped are never silently merged",
            }
        )
    pd.DataFrame(rows).to_csv(out / "categorical_mapping_completeness_audit.csv", index=False)

    by_window = warning.groupby("warning_response_window").agg(
        n_model_cells=("dtxsid", "size"),
        n_chemicals=("dtxsid", "nunique"),
        n_species=("latin_name", "nunique"),
    )
    manifest = {
        "rapid_warning_model_cells_all_durations": int(len(warning)),
        "rapid_warning_model_cells_le_24h": int(
            warning.warning_response_window.isin(WARNING_DISPLAY_ORDER[:3]).sum()
        ),
        "rapid_warning_window_counts": {
            k: int(by_window.loc[k, "n_model_cells"]) if k in by_window.index else 0
            for k in WARNING_DISPLAY_ORDER
        },
        "acute_standard_apical_model_cells": int(len(acute)),
        "separation_rule": (
            "warning/response tables contain heartbeat, behavior, and other early-warning endpoints "
            "across 0-2 h, 2-6 h, 6-24 h, 24-48 h, 48 h-7 d, >7 d, and unknown duration; "
            "acute-standard tables contain only apical toxicity cells matching recognized taxon/endpoint windows"
        ),
        "external_daphnia_1h_data_included": False,
        "audit_unit": "identified model cells; record-level mapping remains in upstream ECOTOX-derived tables",
    }
    (out / "evidence_layer_separation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
