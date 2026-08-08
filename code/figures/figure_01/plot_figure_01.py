#!/usr/bin/env python3
"""Figure 1: evidence imbalance and empirical SSD sentinel contrasts."""

from __future__ import annotations

import sys
import math
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgba
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parents[1]))

from plot_common import (  # noqa: E402
    PALETTE,
    apply_style,
    clean_axis,
    compact_moa,
    export_figure,
    figure_dir_from_script,
    project_root_from_script,
    save_contract_note,
    short_species,
    write_combined_source,
    write_csv,
    write_manifest,
)


FIGURE_ID = "figure_01"
TITLE = "ECOTOX evidence imbalance and empirical SSD sentinel contrasts"
MANUSCRIPT_CAPTION = (
    "Figure 1. Imbalanced coverage of toxicity evidence and sensitive SSD positions of selected sentinel candidates. "
    "a, Shares of all toxicity records represented by the five most frequent species, effect/endpoint families and "
    "chemicals. The contrasting record shares show strong evidence imbalance across data fields and why raw record "
    "abundance alone cannot provide a fair basis for cross-species sensitivity ranking. b, Taxon–mechanism record "
    "matrix for the full ECOTOX-derived corpus. Cells show record counts by finer taxonomic group and resolved "
    "mechanism group, with unresolved MOA retained as an explicit category. c, Seven illustrative measured protective "
    "SSDs spanning distinct pollutant classes. Open circles show all measured protective species used to fit each SSD. "
    "Filled triangles mark observed members of the national Top-10 sentinel sequence, and open diamonds mark observed "
    "routine toxicity-test species identified from EPA WET methods, OECD guidelines and commonly applied laboratory "
    "assays; the species represented by diamonds vary among SSD contexts according to data availability. In every "
    "example, at least one observed national Top-10 sentinel lies in the lower 5% tail, and all observed Top-10 "
    "sentinels remain within the lower 40% of the fitted distribution. In six of the seven examples, at least one "
    "routine toxicity-test species occurs above the median sensitivity percentile, with a maximum percentile of 80.8%."
)
PRIMARY_TARGET_X = 0.95
TOP_N_SENTINELS = 10
SENTINEL_COLORS = {
    1: "#4E79A7",
    2: "#F28E2B",
    3: "#59A14F",
    4: "#E15759",
    5: "#B07AA1",
    6: "#76B7B2",
    7: "#EDC948",
    8: "#9C755F",
    9: "#BAB0AC",
    10: "#2F6F8F",
}
PANEL_A_TARGET_CLASSES = [
    "cyanotoxin",
    "insecticide",
    "disinfection oxidant",
    "heavy metal",
    "herbicide",
    "emerging PFAS",
    "industrial organic",
]
PANEL_A_PREFERRED_CONTEXTS = {
    "cyanotoxin": {"chemical_name": "Microcystin LR", "effect_family": "mortality_survival"},
    "insecticide": {"chemical_name": "Phosphamidon", "effect_family": "mortality_survival"},
    "disinfection oxidant": {"chemical_name": "Chloramine", "effect_family": "mortality_survival"},
    "heavy metal": {"chemical_name": "Cadmium", "effect_family": "mortality_survival"},
    "industrial organic": {"chemical_name": "Pentachlorophenol", "effect_family": "growth"},
}
PANEL_B_TOP_TAXON_GROUPS = 12
OTHER_TAXA_LABEL = "Other Taxa"
OTHER_MOA_LABEL = "Other Resolved MOA"
UNRESOLVED_MOA_LABEL = "MOA_UNRESOLVED"
TYPOGRAPHY_NOTE = (
    "Typography: base matplotlib font size 8.0 pt; aligned panel labels 12 pt; "
    "axis titles 6.8-7.4 pt; dense tick labels 5.8-6.3 pt; "
    "Panel A labels all Top-5 bars with enlarged two-line item names and percentages; Panel B resolved-MOA cells omit internal numeric annotations; "
    "full values are retained in source data."
)


def short_text(value: object, max_len: int = 22) -> str:
    text = "" if pd.isna(value) else str(value).replace("_", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def normal_cdf_percent(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (x - float(mu)) / (sigma * math.sqrt(2.0))
    return 50.0 * (1.0 + np.vectorize(math.erf)(z))


def panel_a_curve_label(meta: pd.Series) -> str:
    chemical = str(meta["chemical_name"])
    pollutant_class = str(meta["pollutant_class"])
    if chemical == "Perfluorooctanesulfonic acid":
        chemical_display = chemical
        class_display = "PFAS"
    elif chemical.startswith("Uranium nitrate oxide"):
        chemical_display = "Uranium nitrate oxide"
        class_display = pollutant_class
    else:
        chemical_display = short_text(chemical, 18)
        class_display = pollutant_class
    return (
        f"{chemical_display} ({class_display}; "
        f"n={int(meta['n_species_in_ssd'])}, Top-10 obs. {int(meta['n_national_top10_observed'])}/10)"
    )


def pollutant_class_from_row(row: pd.Series) -> str:
    name = str(row.get("PREFERRED_NAME", "")).lower()
    form = str(row.get("chemical_form_class", "")).lower()
    primary = str(row.get("primary_moa", "")).lower()
    functional = str(row.get("moa_level_functional", "")).lower()
    metal_form = ("metal" in form and "nonmetal" not in form) or "organometallic" in form
    if any(token in name for token in ["perfluoro", "fluorooctane", "pfos", "pfoa"]):
        return "emerging PFAS"
    if "microcystin" in name:
        return "cyanotoxin"
    if any(token in name for token in ["chloramine", "chlorite", "chlorate"]):
        return "disinfection oxidant"
    if any(
        token in name
        for token in [
            "cadmium",
            "copper",
            "zinc",
            "nickel",
            "lead",
            "chromium",
            "chromate",
            "dichromate",
            "mercury",
            "mercuric",
            "silver",
            "uranium",
        ]
    ) or metal_form:
        return "heavy metal"
    if any(
        token in name
        for token in [
            "azinphos",
            "chlorpyrifos",
            "carbaryl",
            "dichlorvos",
            "fenitrothion",
            "malathion",
            "parathion",
            "phosphonothioate",
            "propoxur",
            "imidacloprid",
            "permethrin",
            "cypermethrin",
            "cyfluthrin",
            "dieldrin",
            "naled",
            "diazinon",
            "methomyl",
            "coumaphos",
            "trichlorfon",
            "phorate",
            "profenofos",
            "disulfoton",
            "phosphamidon",
            "aminocarb",
            "diflubenzuron",
        ]
    ):
        return "insecticide"
    if any(
        token in name
        for token in [
            "glyphosate",
            "atrazine",
            "diuron",
            "monolinuron",
            "bromacil",
            "simazine",
            "metolachlor",
        ]
    ) or "photosynthesis" in functional:
        return "herbicide"
    if any(token in name for token in ["phenol", "aniline", "nitrophenol"]):
        return "industrial organic"
    if "reactive" in functional:
        return "reactive organic"
    if "xenobiotic" in functional or "receptor" in functional:
        return "mechanism-active organic"
    return "other organic"


def title_case_label(value: str) -> str:
    keep_upper = {"MOA", "PFAS"}
    words = str(value).replace("_", " ").split()
    titled = []
    for word in words:
        stripped = word.strip()
        if stripped.upper() in keep_upper:
            titled.append(stripped.upper())
        elif stripped.lower() in {"and", "or", "of", "to"}:
            titled.append(stripped.lower())
        else:
            titled.append(stripped[:1].upper() + stripped[1:])
    return " ".join(titled)


def fine_taxonomic_group(row: pd.Series) -> str:
    klass = str(row.get("class", "") or "").strip()
    order = str(row.get("tax_order", "") or "").strip()
    family = str(row.get("family", "") or "").strip()
    phylum = str(row.get("phylum_division", "") or "").strip()

    if family == "Daphniidae" or (klass == "Branchiopoda" and order == "Diplostraca"):
        return "Cladocerans"
    if klass == "Branchiopoda":
        return "Other Branchiopods"
    if klass == "Actinopterygii":
        if order in {"Cypriniformes", "Salmoniformes", "Perciformes", "Cyprinodontiformes", "Siluriformes"}:
            return f"{order} Fishes"
        return "Other Ray-Finned Fishes"
    if klass == "Insecta":
        if order == "Diptera":
            return "Dipteran Insects"
        if order in {"Ephemeroptera", "Trichoptera", "Odonata"}:
            return "Aquatic Insects"
        return "Other Insects"
    if klass == "Chlorophyceae":
        return "Green Algae"
    if klass == "Cyanophyceae":
        return "Cyanobacteria"
    if klass == "Amphibia":
        return "Anurans"
    if klass == "Malacostraca":
        if order == "Amphipoda":
            return "Amphipods"
        if order == "Decapoda":
            return "Decapods"
        return "Other Malacostracans"
    if klass in {"Gastropoda", "Bivalvia"}:
        return f"{klass} Molluscs"
    if klass in {"Liliopsida", "Magnoliopsida", "Filicopsida"}:
        return "Aquatic Plants"
    if klass in {"Ciliatea", "Monogononta", "Bacillariophyceae", "Euglenophyceae"}:
        return title_case_label(klass)
    if klass and klass != "nan":
        return title_case_label(klass)
    if phylum and phylum != "nan":
        return title_case_label(phylum)
    return "Unassigned"


def spread_label_positions(values: np.ndarray, *, low: float = 3.2, high: float = 46.0, min_gap: float = 3.8) -> np.ndarray:
    if len(values) == 0:
        return values
    order = np.argsort(values)
    placed = np.clip(values[order].astype(float), low, high)
    for i in range(1, len(placed)):
        if placed[i] - placed[i - 1] < min_gap:
            placed[i] = placed[i - 1] + min_gap
    overflow = placed[-1] - high
    if overflow > 0:
        placed -= overflow
    for i in range(len(placed) - 2, -1, -1):
        if placed[i + 1] - placed[i] < min_gap:
            placed[i] = placed[i + 1] - min_gap
    underflow = low - placed[0]
    if underflow > 0:
        placed += underflow
    result = np.empty_like(placed)
    result[order] = placed
    return result


def load_cells(root: Path) -> pd.DataFrame:
    cells = pd.read_csv(root / "results/toxicity/censored_model_cells.csv.gz")
    cells = cells.rename(columns={"dtxsid": "DTXSID"})
    hierarchy = pd.read_csv(
        root / "results/chemistry/chemical_moa_hierarchy.csv.gz",
        usecols=[
            "DTXSID",
            "PREFERRED_NAME",
            "chemical_form_class",
            "primary_moa",
            "moa_level_functional",
            "moa_level_mie",
        ],
    ).drop_duplicates("DTXSID")
    cells = cells.merge(hierarchy, on="DTXSID", how="left")
    candidate_path = root / "results/probability/candidate_universe_locked.csv"
    if candidate_path.exists():
        method_flags = pd.read_csv(
            candidate_path,
            usecols=["latin_name", "official_wet_method_species", "official_or_common_method_species"],
        ).drop_duplicates("latin_name")
        cells = cells.merge(method_flags, on="latin_name", how="left")
    else:
        cells["official_wet_method_species"] = False
        cells["official_or_common_method_species"] = False
    for column in ["official_wet_method_species", "official_or_common_method_species"]:
        cells[column] = cells[column].where(cells[column].notna(), False).astype(bool)
    cells["PREFERRED_NAME"] = cells["PREFERRED_NAME"].fillna(cells["DTXSID"])
    cells["taxonomic_class"] = cells["class"].fillna("Unassigned").replace("", "Unassigned")
    return cells


def load_national_top10(root: Path) -> pd.DataFrame:
    seq = pd.read_csv(root / "results/panels/national_panel_sequences.csv")
    seq["protection_target_x"] = pd.to_numeric(seq["protection_target_x"])
    top10 = seq[
        seq["universe"].eq("priority")
        & seq["method"].eq("data_driven")
        & np.isclose(seq["protection_target_x"], PRIMARY_TARGET_X)
        & seq["rank"].le(TOP_N_SENTINELS)
    ].copy()
    if len(top10) < TOP_N_SENTINELS:
        raise ValueError("Could not locate the national x=0.95 data-driven Top-10 sentinel sequence.")
    top10["rank"] = top10["rank"].astype(int)
    return top10.sort_values("rank")


def prepare_protective_species_contexts(cells: pd.DataFrame) -> pd.DataFrame:
    protective = cells[
        cells["endpoint_band_v16"].astype(str).str.contains("apical", case=False, na=False)
        & cells["mu_log10_umol_L"].notna()
    ].copy()
    group_cols = ["DTXSID", "PREFERRED_NAME", "effect_family", "medium_family", "latin_name"]
    # One empirical SSD point per species in a chemical/effect/medium context.  If a
    # species has multiple apical model cells, use the most sensitive measured cell
    # already estimated by the censored-likelihood pipeline.
    species_context = (
        protective.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            mu_log10_umol_L=("mu_log10_umol_L", "min"),
            n_model_cells=("mu_log10_umol_L", "size"),
            n_records=("n_records", "sum"),
            n_studies=("n_studies", "sum"),
            taxonomic_class=("taxonomic_class", "first"),
            primary_moa=("primary_moa", "first"),
            chemical_form_class=("chemical_form_class", "first"),
            moa_level_functional=("moa_level_functional", "first"),
            official_wet_method_species=("official_wet_method_species", "max"),
            official_or_common_method_species=("official_or_common_method_species", "max"),
        )
    )
    return species_context


def choose_diverse_contexts(candidates: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    candidates = candidates[~candidates["chemical_name"].astype(str).str.contains("sodium chloride", case=False, na=False)].copy()
    class_pools: list[pd.DataFrame] = []
    for target_class in PANEL_A_TARGET_CLASSES:
        sub = candidates[candidates["pollutant_class"].eq(target_class)].copy()
        if sub.empty:
            continue
        preferred = PANEL_A_PREFERRED_CONTEXTS.get(target_class)
        if preferred is not None:
            preferred_sub = sub.copy()
            for column, value in preferred.items():
                preferred_sub = preferred_sub[preferred_sub[column].eq(value)]
            if not preferred_sub.empty:
                sub = preferred_sub
        sub = sub.sort_values(
            [
                "n_standard_test_species_above_50pct",
                "n_national_top10_observed",
                "top10_observed_min_percentile",
                "top10_observed_max_percentile",
                "n_species_in_ssd",
            ],
            ascending=[False, False, True, True, False],
        ).head(12)
        class_pools.append(sub)

    if len(class_pools) == len(PANEL_A_TARGET_CLASSES):
        best_score = -np.inf
        best_indices: list[int] | None = None
        for rows in itertools.product(*[list(pool.index) for pool in class_pools]):
            combo = candidates.loc[list(rows)].copy()
            mu = np.sort(combo["ssd_fit_mu_log10"].astype(float).to_numpy())
            if len(mu) < 2:
                continue
            min_gap = float(np.min(np.diff(mu)))
            span = float(mu[-1] - mu[0])
            std_gt50 = float(combo["n_standard_test_species_above_50pct"].sum())
            top_obs = float(combo["n_national_top10_observed"].sum())
            score = min_gap * 8.0 + span * 1.25 + std_gt50 * 0.35 + top_obs * 0.08
            if score > best_score:
                best_score = score
                best_indices = list(rows)
        if best_indices is not None:
            selected = candidates.loc[best_indices].copy()
            selected["selection_score_spread"] = float(best_score)
            selected["selection_min_fit_mu_gap"] = float(np.min(np.diff(np.sort(selected["ssd_fit_mu_log10"].astype(float).to_numpy()))))
            selected["selection_fit_mu_span"] = float(selected["ssd_fit_mu_log10"].max() - selected["ssd_fit_mu_log10"].min())
            return selected

    # Fallback if a target class is absent in an older dataset.
    ordered = candidates.sort_values(
        [
            "n_standard_test_species_above_50pct",
            "n_national_top10_observed",
            "top10_observed_min_percentile",
            "top10_observed_max_percentile",
            "n_species_in_ssd",
        ],
        ascending=[False, False, True, True, False],
    )
    selected_indices: list[int] = []
    used_classes: set[str] = set()
    for idx, row in ordered.iterrows():
        pollutant_class = str(row["pollutant_class"])
        if pollutant_class not in used_classes:
            selected_indices.append(idx)
            used_classes.add(pollutant_class)
        if len(selected_indices) >= n_examples:
            break
    return candidates.loc[selected_indices].copy()


def select_empirical_ssd_examples(species_context: pd.DataFrame, top10: pd.DataFrame, n_examples: int = len(PANEL_A_TARGET_CLASSES)) -> tuple[pd.DataFrame, pd.DataFrame]:
    top10_ranks = dict(zip(top10["latin_name"], top10["rank"]))
    top10_species = set(top10["latin_name"])
    context_rows: list[dict[str, object]] = []
    ssd_rows: list[pd.DataFrame] = []
    strict_contexts_found = 0

    for context_key, group in species_context.groupby(["DTXSID", "PREFERRED_NAME", "effect_family", "medium_family"], dropna=False):
        g = group.sort_values("mu_log10_umol_L").reset_index(drop=True).copy()
        if len(g) < 15:
            continue
        g["ssd_rank"] = np.arange(1, len(g) + 1)
        g["ssd_percentile"] = g["ssd_rank"] / len(g) * 100.0
        g["national_top10_rank"] = g["latin_name"].map(top10_ranks)
        g["is_national_top10"] = g["national_top10_rank"].notna()
        top_observed = g[g["is_national_top10"]].copy()
        n_observed = len(top_observed)
        if n_observed == 0:
            continue
        all_observed_within_40 = bool(top_observed["ssd_percentile"].max() <= 40.0)
        any_observed_within_5 = bool(top_observed["ssd_percentile"].min() <= 5.0)
        strict_pass = bool(n_observed == TOP_N_SENTINELS and all_observed_within_40 and any_observed_within_5)
        if strict_pass:
            strict_contexts_found += 1
        if not (all_observed_within_40 and any_observed_within_5):
            continue
        standard_species = g[g["official_or_common_method_species"].fillna(False).astype(bool)].copy()
        wet_method_species = g[g["official_wet_method_species"].fillna(False).astype(bool)].copy()
        standard_species_sorted = standard_species.sort_values(["ssd_rank", "latin_name"]).copy()
        if len(standard_species_sorted):
            most_sensitive_standard = standard_species_sorted.iloc[0]
            standard_above_median = standard_species_sorted[standard_species_sorted["ssd_percentile"].gt(50.0)].copy()
            standard_above_median_names = "; ".join(
                f"{row.latin_name} ({float(row.ssd_percentile):.1f}%)"
                for row in standard_above_median.sort_values(["ssd_percentile", "latin_name"]).itertuples(index=False)
            )
        else:
            most_sensitive_standard = None
            standard_above_median = pd.DataFrame()
            standard_above_median_names = ""

        row = {
            "context_id": "|".join(str(x) for x in context_key),
            "DTXSID": context_key[0],
            "chemical_name": context_key[1],
            "effect_family": context_key[2],
            "medium_family": context_key[3],
            "n_species_in_ssd": len(g),
            "n_national_top10_observed": n_observed,
            "top10_observed_min_percentile": float(top_observed["ssd_percentile"].min()),
            "top10_observed_max_percentile": float(top_observed["ssd_percentile"].max()),
            "n_standard_test_species_observed": int(len(standard_species)),
            "n_standard_test_species_above_50pct": int((standard_species["ssd_percentile"] > 50.0).sum()),
            "standard_test_species_max_percentile": float(standard_species["ssd_percentile"].max()) if len(standard_species) else np.nan,
            "most_sensitive_standard_test_species": str(most_sensitive_standard["latin_name"]) if most_sensitive_standard is not None else "",
            "most_sensitive_standard_test_species_rank": int(most_sensitive_standard["ssd_rank"]) if most_sensitive_standard is not None else np.nan,
            "most_sensitive_standard_test_species_percentile": float(most_sensitive_standard["ssd_percentile"]) if most_sensitive_standard is not None else np.nan,
            "most_sensitive_standard_test_species_is_top10": bool(most_sensitive_standard["latin_name"] in top10_species) if most_sensitive_standard is not None else False,
            "standard_test_species_above_50pct_names": standard_above_median_names,
            "n_official_wet_species_observed": int(len(wet_method_species)),
            "n_official_wet_species_above_50pct": int((wet_method_species["ssd_percentile"] > 50.0).sum()),
            "strict_all_top10_present_pass": strict_pass,
            "strict_contexts_found_in_dataset": strict_contexts_found,
            "x_range_log10": float(g["mu_log10_umol_L"].max() - g["mu_log10_umol_L"].min()),
            "pollutant_class": pollutant_class_from_row(g.iloc[0]),
            "ssd_fit_mu_log10": float(g["mu_log10_umol_L"].mean()),
            "ssd_fit_sigma_log10": float(g["mu_log10_umol_L"].std(ddof=1)),
        }
        context_rows.append(row)
        g = g.assign(**row)
        ssd_rows.append(g)

    if not context_rows:
        raise ValueError("No empirical SSD context had observed national Top-10 members within the requested sensitivity window.")

    context_summary = pd.DataFrame(context_rows)
    strict_total = int(context_summary["strict_all_top10_present_pass"].sum())
    context_summary["strict_contexts_found_in_dataset"] = strict_total
    if strict_total > 0:
        selected = choose_diverse_contexts(context_summary[context_summary["strict_all_top10_present_pass"]].copy(), n_examples)
        selection_rule = "strict: all 10 national Top-10 observed, all within first 40%, at least one within first 5%"
    else:
        selected = choose_diverse_contexts(context_summary.copy(), n_examples)
        selection_rule = (
            "fallback: no real-data context contains all 10 national Top-10 sentinels; "
            "selected contexts maximize observed Top-10 support while all observed Top-10 members are within first 40% and at least one is within first 5%"
        )
    selected = selected.sort_values("ssd_fit_mu_log10", ascending=True).reset_index(drop=True)
    selected = selected.assign(context_order=np.arange(1, len(selected) + 1), selection_rule=selection_rule)

    all_ssd = pd.concat(ssd_rows, ignore_index=True)
    selected_points = all_ssd.merge(
        selected[["context_id", "context_order", "selection_rule", "strict_contexts_found_in_dataset"]],
        on="context_id",
        how="inner",
        suffixes=("", "_selected"),
    )
    selected_points["national_top10_rank"] = selected_points["national_top10_rank"].astype("Int64")
    selected_points["top10_marker"] = selected_points["national_top10_rank"].map(lambda x: "" if pd.isna(x) else f"T{int(x)}")
    selected_points["top10_color"] = selected_points["national_top10_rank"].map(
        lambda x: "" if pd.isna(x) else SENTINEL_COLORS.get(int(x), PALETTE["red"])
    )
    selected_points["is_official_method_species"] = selected_points["official_wet_method_species"].fillna(False).astype(bool)
    selected_points["is_standard_test_species"] = selected_points["official_or_common_method_species"].fillna(False).astype(bool)
    return selected_points, selected


def resolved_mechanism_label(row: pd.Series) -> str | None:
    primary = row.get("primary_moa")
    if not isinstance(primary, str) or primary.strip() == "" or primary.strip().upper() == "MOA_UNRESOLVED":
        return None
    functional = row.get("moa_level_functional")
    if isinstance(functional, str) and functional.startswith("MOAFAM::"):
        return compact_moa(functional)
    if isinstance(primary, str) and primary.startswith("MIE::"):
        return compact_moa(primary)
    mie = row.get("moa_level_mie")
    if isinstance(mie, str) and mie.startswith("MIE::"):
        return compact_moa(mie)
    return compact_moa(primary)


def build_panel_b(cells: pd.DataFrame) -> pd.DataFrame:
    d = cells.copy()
    d["mechanism_group"] = d.apply(resolved_mechanism_label, axis=1)
    d["mechanism_group"] = d["mechanism_group"].fillna(UNRESOLVED_MOA_LABEL)
    d["fine_taxon_group"] = d.apply(fine_taxonomic_group, axis=1)
    total_model_cells = len(d)
    unresolved_cells = int(d["mechanism_group"].eq(UNRESOLVED_MOA_LABEL).sum())

    top_taxa = d.groupby("fine_taxon_group").size().sort_values(ascending=False).head(PANEL_B_TOP_TAXON_GROUPS).index.tolist()
    top_mechanisms = (
        d.loc[~d["mechanism_group"].eq(UNRESOLVED_MOA_LABEL)]
        .groupby("mechanism_group")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    d["taxon_group"] = np.where(d["fine_taxon_group"].isin(top_taxa), d["fine_taxon_group"], OTHER_TAXA_LABEL)
    d["mechanism_display"] = np.where(
        d["mechanism_group"].eq(UNRESOLVED_MOA_LABEL),
        UNRESOLVED_MOA_LABEL,
        np.where(d["mechanism_group"].isin(top_mechanisms), d["mechanism_group"], OTHER_MOA_LABEL),
    )

    summary = (
        d.groupby(["taxon_group", "mechanism_display"], as_index=False)
        .agg(
            n_model_cells=("DTXSID", "size"),
            n_species=("latin_name", "nunique"),
            n_chemicals=("DTXSID", "nunique"),
            n_effect_families=("effect_family", "nunique"),
        )
    )
    class_order = d.groupby("taxon_group").size().sort_values(ascending=False).index.tolist()
    mech_order = d.groupby("mechanism_display").size().sort_values(ascending=False).index.tolist()
    full_index = pd.MultiIndex.from_product([class_order, mech_order], names=["taxon_group", "mechanism_display"])
    summary = summary.set_index(["taxon_group", "mechanism_display"]).reindex(full_index).reset_index()
    for column in ["n_model_cells", "n_species", "n_chemicals", "n_effect_families"]:
        summary[column] = summary[column].fillna(0).astype(int)
    summary["log10_1p_model_cells"] = np.log10(summary["n_model_cells"] + 1.0)
    summary["total_model_cells"] = total_model_cells
    summary["moa_unresolved_cells"] = unresolved_cells
    summary["data_scope"] = "all censored_model_cells; fine taxon groups derive from ECOTOX class/order/family labels; MOA_UNRESOLVED displayed as the rightmost mechanism column"
    return summary


def build_panel_c(cells: pd.DataFrame) -> pd.DataFrame:
    total = len(cells)
    d = cells.copy()
    d["chemical_label"] = d["PREFERRED_NAME"].fillna(d["DTXSID"])
    specs = [
        ("Species", "latin_name", "Species"),
        ("Effect/endpoint", "effect_family", "Effect family"),
        ("Chemical", "chemical_label", "Chemical"),
    ]
    rows: list[dict[str, object]] = []
    for dimension, column, display in specs:
        counts = d.groupby(column, dropna=False).size().sort_values(ascending=False).head(5)
        for rank, (label, count) in enumerate(counts.items(), start=1):
            if dimension == "Species":
                short = short_species(str(label), max_len=22)
            elif dimension == "Effect/endpoint":
                short = short_text(title_case_label(str(label)), max_len=24)
            else:
                short = short_text(label, max_len=22)
            rows.append(
                {
                    "dimension": dimension,
                    "dimension_label": display,
                    "rank": rank,
                    "item": str(label),
                    "item_short": short,
                    "n_model_cells": int(count),
                    "total_model_cells": int(total),
                    "percent_total_model_cells": float(count / total * 100.0),
                }
            )
    return pd.DataFrame(rows)


def axis_mechanism_label(value: str) -> str:
    if value == UNRESOLVED_MOA_LABEL:
        return "MOA\nUnresolved"
    label = value.replace("MOA: ", "").replace("MIE: ", "").replace("FORM: ", "")
    label = label.replace("nuclear receptor xenobiotic", "receptor/\nxenobiotic")
    label = label.replace("neurotransmission ion channel", "neurotransmission")
    label = label.replace("xenobiotic metabolism", "xenobiotic\nmetabolism")
    label = label.replace("reactive chemistry", "reactive\nchemistry")
    label = label.replace(OTHER_MOA_LABEL, "Other\nResolved MOA")
    if "\n" in label:
        return "\n".join(title_case_label(part) for part in label.split("\n"))
    return title_case_label(label)


def axis_taxon_label(value: str) -> str:
    if value == "Cladocerans":
        return "Cladocerans"
    if value == "Other Ray-Finned Fishes":
        return "Other Ray-Finned\nFishes"
    return short_text(value, 24)


def short_sci_tick(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    if abs(mantissa - round(mantissa)) < 0.05:
        mantissa_text = f"{mantissa:.0f}"
    else:
        mantissa_text = f"{mantissa:.1f}"
    return f"{mantissa_text}e{exponent}"


def short_count_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def top5_bar_label(item: object, item_short: object, dimension: str) -> str:
    """Compact but readable labels for the Top-5 record-share bars."""
    raw = "" if pd.isna(item) else str(item)
    short = "" if pd.isna(item_short) else str(item_short)
    if dimension == "Species":
        return short_text(short, max_len=18)
    if dimension == "Effect/endpoint":
        effect_labels = {
            "mortality_survival": "Mortality/surv.",
            "population_ecosystem": "Population",
            "growth": "Growth",
            "biochemical_enzyme_endocrine": "Biochem./enzyme",
            "development_morphology": "Development",
        }
        return effect_labels.get(raw, short_text(title_case_label(raw), max_len=15))
    chemical_labels = {
        "Copper sulfate": "Cu sulfate",
        "Cadmium chloride": "Cd chloride",
        "Zinc sulfate": "Zn sulfate",
    }
    return chemical_labels.get(raw, short_text(short or raw, max_len=13))


def aligned_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-28, 18),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12.0,
        color="black",
        clip_on=False,
    )


def annotate_top10_callouts(ax: plt.Axes, top: pd.DataFrame, context_index: int) -> None:
    if top.empty:
        return
    d = top.sort_values(["ssd_percentile", "mu_log10_umol_L"]).copy()
    d["label_y"] = spread_label_positions(d["ssd_percentile"].to_numpy(), low=3.2, high=46.0, min_gap=3.9)
    x0, x1 = ax.get_xlim()
    x_span = max(x1 - x0, 1e-6)
    preferred_side = 1 if context_index % 2 == 0 else -1
    base_dx = (0.070 + 0.014 * (context_index % 3)) * x_span
    for row in d.itertuples(index=False):
        point_x = float(row.mu_log10_umol_L)
        point_y = float(row.ssd_percentile)
        side = preferred_side
        label_x = point_x + side * base_dx
        if label_x > x1 - 0.025 * x_span:
            side = -1
            label_x = point_x - base_dx
        if label_x < x0 + 0.025 * x_span:
            side = 1
            label_x = point_x + base_dx
        label_x = min(max(label_x, x0 + 0.018 * x_span), x1 - 0.018 * x_span)
        ax.annotate(
            short_species(row.latin_name, max_len=15),
            xy=(point_x, point_y),
            xytext=(label_x, float(row.label_y)),
            textcoords="data",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=5.0,
            color=PALETTE["red"],
            arrowprops={
                "arrowstyle": "-",
                "color": PALETTE["red"],
                "lw": 0.45,
                "alpha": 0.72,
                "shrinkA": 1.5,
                "shrinkB": 2.6,
                "connectionstyle": "angle3,angleA=0,angleB=90",
            },
            path_effects=[pe.withStroke(linewidth=1.9, foreground="white")],
            zorder=7,
        )


def annotate_top10_callouts_global(ax: plt.Axes, label_frames: list[pd.DataFrame]) -> None:
    if not label_frames:
        return
    d = pd.concat(label_frames, ignore_index=True)
    if d.empty:
        return
    d = d.sort_values(["latin_name", "ssd_percentile", "n_species_in_ssd"], ascending=[True, True, False]).drop_duplicates(
        "latin_name",
        keep="first",
    )
    x0, x1 = ax.get_xlim()
    x_span = max(x1 - x0, 1e-6)
    d["label_side"] = np.where(d["label_context_index"].astype(int) % 2 == 0, 1, -1)
    d.loc[d["mu_log10_umol_L"] < x0 + 0.20 * x_span, "label_side"] = 1
    d.loc[d["mu_log10_umol_L"] > x0 + 0.80 * x_span, "label_side"] = -1
    d["label_column"] = np.floor((d["mu_log10_umol_L"] - x0) / x_span * 4).clip(0, 3).astype(int)
    d["desired_label_y"] = d["ssd_percentile"].astype(float) + (d["label_context_index"].astype(float) - d["label_context_index"].astype(float).mean()) * 0.75
    d["label_y"] = np.nan
    for (_, _), sub in d.groupby(["label_side", "label_column"], sort=False):
        d.loc[sub.index, "label_y"] = spread_label_positions(
            sub["desired_label_y"].to_numpy(),
            low=7.4,
            high=48.0,
            min_gap=3.8,
        )
    for row in d.sort_values(["label_column", "label_side", "label_y"]).itertuples(index=False):
        point_x = float(row.mu_log10_umol_L)
        point_y = float(row.ssd_percentile)
        side = int(row.label_side)
        offset = (0.055 + 0.012 * (int(row.label_context_index) % 3)) * x_span
        label_x = point_x + side * offset
        if label_x > x1 - 0.018 * x_span:
            label_x = x1 - 0.018 * x_span
        if label_x < x0 + 0.018 * x_span:
            label_x = x0 + 0.018 * x_span
        ha = "left" if label_x >= point_x else "right"
        ax.annotate(
            short_species(row.latin_name, max_len=15),
            xy=(point_x, point_y),
            xytext=(label_x, float(row.label_y)),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=5.0,
            color=PALETTE["red"],
            arrowprops={
                "arrowstyle": "-",
                "color": PALETTE["red"],
                "lw": 0.45,
                "alpha": 0.72,
                "shrinkA": 1.5,
                "shrinkB": 2.6,
                "connectionstyle": "angle3,angleA=0,angleB=90",
            },
            path_effects=[pe.withStroke(linewidth=1.9, foreground="white")],
            zorder=7,
        )


def draw_panel_a(
    ax: plt.Axes,
    ssd_points: pd.DataFrame,
    selected_contexts: pd.DataFrame,
    top10: pd.DataFrame,
    *,
    panel_label: str = "c",
) -> None:
    colors = ["#7A5195", PALETTE["blue2"], "#EF5675", PALETTE["teal"], "#A55194", PALETTE["gold"], PALETTE["green"]]
    x_min = float(ssd_points["mu_log10_umol_L"].min())
    x_max = float(ssd_points["mu_log10_umol_L"].max())
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.20 * x_range, x_max - 0.08 * x_range)
    ax.set_ylim(0, 100)
    ax.axhspan(0, 5, color=PALETTE["red_light"], alpha=0.34, lw=0, zorder=0)
    ax.axhline(5, color=PALETTE["red"], lw=0.9, ls=(0, (3, 2)), alpha=0.90, zorder=1)
    ax.axhline(40, color=PALETTE["neutral"], lw=0.85, ls="--", alpha=0.90, zorder=1)
    ax.text(
        0.985,
        0.072,
        "HC5-sensitive tail",
        transform=ax.transAxes,
        fontsize=6.4,
        color=PALETTE["red"],
        ha="right",
        va="center",
        zorder=8,
    )
    ax.text(
        0.985,
        0.415,
        "40% sensitivity",
        transform=ax.transAxes,
        fontsize=6.3,
        color=PALETTE["neutral_dark"],
        ha="right",
        va="bottom",
        zorder=8,
    )

    curve_handles: list[Line2D] = []
    ordered_contexts = selected_contexts.sort_values("context_order").reset_index(drop=True)
    for idx, meta in ordered_contexts.iterrows():
        context_id = meta["context_id"]
        group = ssd_points[ssd_points["context_id"].eq(context_id)].copy()
        group = group.sort_values("mu_log10_umol_L")
        color = colors[idx % len(colors)]
        label = panel_a_curve_label(meta)
        x_min = float(group["mu_log10_umol_L"].min())
        x_max = float(group["mu_log10_umol_L"].max())
        fit_x = np.linspace(x_min, x_max, 220)
        fit_y = normal_cdf_percent(fit_x, float(meta["ssd_fit_mu_log10"]), float(meta["ssd_fit_sigma_log10"]))
        ax.plot(fit_x, fit_y, color=color, lw=1.7, alpha=0.98)
        ax.scatter(
            group["mu_log10_umol_L"],
            group["ssd_percentile"],
            s=10,
            facecolor="white",
            edgecolor=color,
            alpha=0.62,
            lw=0.80,
            zorder=3,
        )
        routine = group[group["is_standard_test_species"]].copy()
        if not routine.empty:
            ax.scatter(
                routine["mu_log10_umol_L"],
                routine["ssd_percentile"],
                s=22,
                marker="D",
                facecolor="none",
                edgecolor=PALETTE["neutral_dark"],
                alpha=0.95,
                lw=0.95,
                zorder=4,
            )
        top = group[group["is_national_top10"]].copy()
        if not top.empty:
            marker_colors = [SENTINEL_COLORS.get(int(rank), PALETTE["red"]) for rank in top["national_top10_rank"]]
            ax.scatter(
                top["mu_log10_umol_L"],
                top["ssd_percentile"],
                s=22,
                marker="^",
                color=marker_colors,
                edgecolor="black",
                lw=0.75,
                zorder=5,
            )
        curve_handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))

    measured_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor=PALETTE["neutral_dark"],
        markeredgewidth=0.80,
        markersize=3.9,
        label="All measured SSD species",
    )
    routine_handle = Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="none",
        markeredgecolor=PALETTE["neutral_dark"],
        markeredgewidth=0.95,
        markersize=4.0,
        label="Routine toxicity-test species",
    )
    sentinel_shape_handle = Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor=PALETTE["neutral_dark"],
        markeredgecolor="black",
        markeredgewidth=0.75,
        markersize=4.8,
        label="National Top-10 sentinel",
    )
    top10_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=SENTINEL_COLORS[int(row.rank)],
            markeredgecolor="black",
            markeredgewidth=0.75,
            markersize=4.8,
            label=f"  T{int(row.rank)} {short_species(row.latin_name, max_len=16)}",
        )
        for row in top10.sort_values("rank").itertuples(index=False)
    ]

    ax.set_ylabel("Empirical SSD percentile (% species)", fontsize=7.4)
    ax.set_xlabel("Measured toxicity threshold (log10 µmol/L; lower = more sensitive)")
    ax.set_ylim(0, 100)
    x_min = float(ssd_points["mu_log10_umol_L"].min())
    x_max = float(ssd_points["mu_log10_umol_L"].max())
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.20 * x_range, x_max - 0.08 * x_range)
    ax.set_xlabel("Measured toxicity threshold (log10 umol/L; lower = more sensitive)", fontsize=7.4)
    curve_legend = ax.legend(
        handles=curve_handles + [measured_handle, routine_handle, sentinel_shape_handle],
        loc="upper left",
        ncol=1,
        fontsize=5.9,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    ax.add_artist(curve_legend)
    ax.legend(
        handles=top10_handles,
        loc="upper left",
        bbox_to_anchor=(0.018, 0.507),
        ncol=1,
        fontsize=5.7,
        handlelength=1.0,
        handletextpad=0.30,
        labelspacing=0.30,
        borderaxespad=0.2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        fancybox=False,
    )
    clean_axis(ax, grid=True)
    aligned_panel_label(ax, panel_label)


def draw_panel_b(ax: plt.Axes, panel_b: pd.DataFrame, fig: plt.Figure, *, panel_label: str = "b") -> None:
    row_order = panel_b.groupby("taxon_group")["n_model_cells"].sum().sort_values(ascending=False).index.tolist()
    if OTHER_TAXA_LABEL in row_order:
        row_order = [row for row in row_order if row != OTHER_TAXA_LABEL] + [OTHER_TAXA_LABEL]
    col_order = panel_b.groupby("mechanism_display")["n_model_cells"].sum().sort_values(ascending=False).index.tolist()
    if UNRESOLVED_MOA_LABEL in col_order:
        col_order = [col for col in col_order if col != UNRESOLVED_MOA_LABEL] + [UNRESOLVED_MOA_LABEL]
    pivot_counts = (
        panel_b.pivot(index="taxon_group", columns="mechanism_display", values="n_model_cells")
        .reindex(index=row_order, columns=col_order)
        .fillna(0)
    )
    matrix_raw = pivot_counts.to_numpy(dtype=float)
    unresolved_j = col_order.index(UNRESOLVED_MOA_LABEL) if UNRESOLVED_MOA_LABEL in col_order else None
    resolved_columns = [col for col in col_order if col != UNRESOLVED_MOA_LABEL]
    if resolved_columns:
        resolved_matrix = pivot_counts[resolved_columns].to_numpy(dtype=float)
        positive_resolved = resolved_matrix[resolved_matrix > 0]
        color_vmax = float(np.percentile(positive_resolved, 95)) if positive_resolved.size else float(matrix_raw.max())
    else:
        color_vmax = float(matrix_raw.max()) if matrix_raw.size else 1.0
    color_vmax = max(color_vmax, 1.0)
    matrix_display = np.minimum(matrix_raw, color_vmax)
    if unresolved_j is not None:
        matrix_display[:, unresolved_j] = np.nan
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "bhbt_evidence_blue",
        ["#FFFFFF", "#E5F0FA", "#9ECAE1", "#4292C6", "#08519C", "#08306B"],
    )
    heatmap_cmap.set_bad("#F2F2F2")
    heatmap_norm = PowerNorm(gamma=0.72, vmin=0.0, vmax=max(color_vmax, 1e-6))
    im = ax.imshow(matrix_display, aspect="auto", cmap=heatmap_cmap, norm=heatmap_norm)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels([axis_mechanism_label(c) for c in col_order], rotation=38, ha="right", va="top", rotation_mode="anchor", fontsize=5.8)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([axis_taxon_label(r) for r in row_order], fontsize=5.8)
    if unresolved_j is not None:
        unresolved_values = pivot_counts.iloc[:, unresolved_j].to_numpy(dtype=float)
        unresolved_vmax = max(float(np.nanmax(unresolved_values)) if unresolved_values.size else 1.0, 1.0)
        unresolved_norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=unresolved_vmax)
        unresolved_cmap = LinearSegmentedColormap.from_list(
            "bhbt_unresolved_grey",
            ["#F7F7F7", "#CFCFCF", "#5F5F5F"],
        )
        for i, value in enumerate(unresolved_values):
            fill_color = unresolved_cmap(unresolved_norm(float(value)))
            ax.add_patch(
                plt.Rectangle(
                    (unresolved_j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=fill_color,
                    hatch="///",
                    edgecolor=(0.25, 0.25, 0.25, 0.40),
                    linewidth=0.20,
                    zorder=3,
                )
            )
            luminance = 0.2126 * fill_color[0] + 0.7152 * fill_color[1] + 0.0722 * fill_color[2]
            text_color = "white" if luminance < 0.54 else "#222222"
            ax.text(
                unresolved_j,
                i,
                short_count_label(float(value)),
                ha="center",
                va="center",
                fontsize=4.9,
                color=text_color,
                fontweight="bold" if value >= np.percentile(unresolved_values, 75) else "normal",
                zorder=5,
            )
        ax.axvline(unresolved_j - 0.5, color=PALETTE["neutral_dark"], lw=0.75, alpha=0.90, zorder=4)
    ax.set_title("Taxon–mechanism record matrix", fontsize=7.4)
    ax.set_xlabel("Mechanism group", fontsize=6.9)
    ax.set_ylabel("Finer taxonomic group", fontsize=6.9, labelpad=0.3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.012)
    cbar.set_label("")
    cbar.ax.set_title("Resolved\nentries (n)", fontsize=5.4, pad=4)
    cbar.set_ticks([0.0, color_vmax / 2.0, color_vmax])
    cbar.formatter = FuncFormatter(lambda x, _pos: short_sci_tick(float(x)))
    cbar.update_ticks()
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=5.2, labelleft=False, labelright=True, pad=0.8)
    aligned_panel_label(ax, panel_label)


def draw_panel_c(ax: plt.Axes, panel_c: pd.DataFrame, *, panel_label: str = "a") -> None:
    dims = ["Species", "Effect/endpoint", "Chemical"]
    colors = {"Species": PALETTE["blue2"], "Effect/endpoint": PALETTE["teal"], "Chemical": PALETTE["violet"]}
    group_gap = 1.15
    bar_width = 0.72
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_centers: list[float] = []
    for dim in dims:
        d = panel_c[panel_c["dimension"].eq(dim)].sort_values("rank")
        base = len(x_positions) + (group_gap if x_positions else 0)
        xs = base + np.arange(len(d))
        alphas = np.linspace(0.95, 0.55, len(d))
        ax.bar(
            xs,
            d["percent_total_model_cells"],
            width=bar_width,
            color=[to_rgba(colors[dim], float(alpha)) for alpha in alphas],
            edgecolor="white",
            linewidth=0.5,
            label=dim,
        )
        label_offset = max(1.00, float(panel_c["percent_total_model_cells"].max()) * 0.022)
        for row, x in zip(d.itertuples(index=False), xs):
            value = float(row.percent_total_model_cells)
            item_label = top5_bar_label(row.item, row.item_short, dim)
            ax.text(
                float(x) - 0.10,
                value + label_offset,
                f"{item_label}\n{value:.1f}%",
                ha="left",
                va="bottom",
                fontsize=5.40,
                color=PALETTE["neutral_dark"],
                rotation=66,
                rotation_mode="anchor",
                clip_on=False,
                linespacing=0.90,
                path_effects=[pe.withStroke(linewidth=1.25, foreground="white", alpha=0.86)],
            )
        x_positions.extend(xs.tolist())
        x_labels.extend([f"Top {int(r)}" for r in d["rank"]])
        group_centers.append(float(np.mean(xs)))
    max_pct = float(panel_c["percent_total_model_cells"].max())
    ax.set_ylim(0, max_pct * 1.44)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(((i % 5) + 1)) for i, _ in enumerate(x_positions)], rotation=0, fontsize=5.9)
    for center, dim in zip(group_centers, dims):
        ax.text(center, -0.16, dim, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.6, color=colors[dim], fontweight="bold")
    for boundary in [(group_centers[0] + group_centers[1]) / 2, (group_centers[1] + group_centers[2]) / 2]:
        ax.axvline(boundary, color=PALETTE["neutral_light"], lw=0.7)
    ax.set_ylabel("Share of all records (%)", fontsize=6.9)
    ax.set_xlabel("Rank by record share", fontsize=6.9, labelpad=18)
    ax.set_title("Top-5 record shares by field", fontsize=7.4)
    clean_axis(ax, grid=False)
    aligned_panel_label(ax, panel_label)


def draw(root: Path, out_dir: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    apply_style(font_size=8.0)
    cells = load_cells(root)
    top10 = load_national_top10(root)
    species_context = prepare_protective_species_contexts(cells)
    panel_a, selected_contexts = select_empirical_ssd_examples(species_context, top10)
    panel_b = build_panel_b(cells)
    panel_c = build_panel_c(cells)
    routine_summary_columns = [
        "context_order",
        "chemical_name",
        "pollutant_class",
        "effect_family",
        "medium_family",
        "n_species_in_ssd",
        "n_national_top10_observed",
        "top10_observed_min_percentile",
        "top10_observed_max_percentile",
        "n_standard_test_species_observed",
        "most_sensitive_standard_test_species",
        "most_sensitive_standard_test_species_rank",
        "most_sensitive_standard_test_species_percentile",
        "most_sensitive_standard_test_species_is_top10",
        "n_standard_test_species_above_50pct",
        "standard_test_species_above_50pct_names",
        "standard_test_species_max_percentile",
        "n_official_wet_species_observed",
    ]
    routine_position_summary = selected_contexts[
        [column for column in routine_summary_columns if column in selected_contexts.columns]
    ].copy()

    fig = plt.figure(figsize=(7.45, 6.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.52], width_ratios=[1.18, 0.82], hspace=0.54, wspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    draw_panel_c(ax_a, panel_c, panel_label="a")
    draw_panel_b(ax_b, panel_b, fig, panel_label="b")
    draw_panel_a(ax_c, panel_a, selected_contexts, top10, panel_label="c")

    outputs = export_figure(fig, out_dir / "figure_01_ssd_sentinel_concept")
    tables = {
        "figure_01a_top5_record_share": panel_c,
        "figure_01b_taxon_mechanism_matrix": panel_b,
        "figure_01c_empirical_ssd_points": panel_a,
        "figure_01c_selected_ssd_contexts": selected_contexts,
        "figure_01c_routine_species_position_summary": routine_position_summary,
    }
    return outputs, tables


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
            "Core conclusion: The complete ECOTOX-derived record universe is unevenly distributed across taxa, mechanisms, endpoints and chemicals, and illustrative SSD examples show that observed national Top-10 sentinels can occupy sensitive measured positions while routine toxicity-test species are often less sensitive.",
            "Archetype: asymmetric mixed quantitative figure.",
            "Panel map: a grouped Top-5 record-share bar chart for species, effect/endpoint and chemical, with compact two-line item-name and percentage labels above all bars and the x-axis showing rank by record share; b complete fine-taxon-by-mechanism evidence-entry matrix with resolved MOA entries shown on the blue scale and MOA_UNRESOLVED retained as a separate grey hatched rightmost column with direct k-count labels; c full-width measured species points with fitted SSD curves across seven pollutant classes selected for visibly separated fitted SSD positions where possible, national Top-10 sentinels as enlarged color-coded triangles, all measured species as open circles, routine toxicity-test species as open diamonds, and a compact Top-10 sentinel colour sublegend.",
            "Reviewer risk: no context in the current measured dataset contains all 10 national x=0.95 Top-10 sentinel species, so panel c colors observed Top-10 members only and records strict_n=0 in source data.",
            TYPOGRAPHY_NOTE,
            f"Manuscript caption: {MANUSCRIPT_CAPTION}",
        ],
    )
    write_manifest(
        out_dir,
        figure_id=FIGURE_ID,
        title=TITLE,
        manuscript_caption=MANUSCRIPT_CAPTION,
        core_conclusion=(
            "The complete ECOTOX-derived record universe is unevenly distributed across taxa, mechanisms, endpoints and chemicals, "
            "and illustrative SSD examples show that observed national Top-10 sentinels can occupy sensitive measured positions while routine toxicity-test species are often less sensitive."
        ),
        archetype="asymmetric mixed quantitative figure",
        panel_map={
            "a": "Grouped Top-5 bar chart showing the share of all records contributed by the five most frequent species, effect/endpoint families and chemicals; all bars report compact item names and percentages as two-line labels above the bars, and the x-axis reports rank by record share.",
            "b": "Complete censored_model_cells evidence-entry matrix by finer ECOTOX-derived class/order/family taxon group and mechanism group, with Cladocerans shown as a concise row label, resolved MOA entries scaled by the blue colorbar, MOA_UNRESOLVED retained as a separate grey hatched rightmost column with direct k-count labels, and Other Taxa placed last.",
            "c": "Full-width empirical SSD examples: all measured species points are shown as open circles, normal SSD curves are fitted on log10 thresholds, examples are selected to increase fitted-SSD separation across pollutant classes where possible, observed national Top-10 sentinels are plotted as enlarged color-coded triangles with a compact species-colour sublegend, and routine toxicity-test species are overlaid as open diamonds. The source tables report the most sensitive routine toxicity-test species and any routine species above the median SSD percentile in each displayed example.",
        },
        source_tables=source_tables,
        input_dependencies=[
            "results/toxicity/censored_model_cells.csv.gz",
            "results/chemistry/chemical_moa_hierarchy.csv.gz",
            "results/panels/national_panel_sequences.csv",
            "results/probability/candidate_universe_locked.csv",
        ],
        outputs=outputs,
        reviewer_risk=(
            "Panel C is empirical but not a universal claim for all chemicals: strict all-10 Top-10 co-occurrence is absent in the measured data, "
            "so the figure transparently shows observed Top-10 members only and does not impute missing absolute toxicity values."
        ),
        notes=[
            "Panel C uses measured protective apical model cells estimated by the censored-likelihood pipeline; missing Top-10 species are not imputed.",
            "Panel C uses official_or_common_method_species from candidate_universe_locked.csv to draw routine toxicity-test species diamonds; these species-level metadata include official-method and additional common-test organisms and are used as one comparator group within each chemical SSD context.",
            "Panel C source data include a routine-species position summary for the illustrative SSD examples; formal model-estimated cover-probability comparisons among panels are reported in Figure 2b rather than repeated in Figure 1.",
            "Panel C uses user-requested preferred real-data contexts where eligible: Phosphamidon mortality/survival for insecticide and Cadmium mortality/survival for heavy metal; Microcystin LR mortality/survival is included as a second emerging-concern example beyond PFAS.",
            "Within each target pollutant class, Panel C candidate ranking prioritizes contexts where observed routine toxicity-test species include more species above the 50th SSD percentile; the final seven-context combination then prioritizes wider fitted-SSD separation where possible while all observed national Top-10 sentinels remain within the first 40% and at least one is within the first 5%.",
            "Panel C target classes are cyanotoxin, insecticide, disinfection oxidant, heavy metal, herbicide, emerging PFAS and industrial organic; Sodium chloride is excluded because it is not treated as a pollutant example.",
            "Panel A/B use the complete censored_model_cells table; Panel A ranks the five most frequent records within species, effect/endpoint and chemical fields and labels all bars with compact item names plus percentages. Panel B retains MOA_UNRESOLVED as its own rightmost mechanism column but excludes it from the resolved-MOA blue colorbar so the resolved mechanism structure remains visible; the unresolved column uses a separate grey hatched encoding with in-cell k-count labels, and the Daphniidae/Diplostraca branchiopod row is simplified to Cladocerans for readability.",
            TYPOGRAPHY_NOTE,
            "No raw inputs were overwritten; derived tables are stored in this figure folder.",
        ],
    )
    print({"figure": FIGURE_ID, "outputs": outputs, "source_tables": source_tables})


if __name__ == "__main__":
    main()
