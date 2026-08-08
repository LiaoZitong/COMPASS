#!/usr/bin/env python3
"""Shared helpers for the revised BHBT manuscript main figures.

The figure-specific scripts live in manuscript_plot/figure_XX and read only
project data/results. Generated source-data tables stay beside the figure that
uses them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "blue_light": "#B4C0E4",
    "teal": "#42949E",
    "green": "#2E9E44",
    "green_light": "#AADCA9",
    "red": "#B64342",
    "red_light": "#E9A6A1",
    "gold": "#C99000",
    "violet": "#9A4D8E",
    "neutral_dark": "#4D4D4D",
    "neutral": "#767676",
    "neutral_light": "#CFCECE",
    "neutral_ultra": "#F2F2F2",
}

TARGET_COLORS = {
    0.80: "#42949E",
    0.90: "#3775BA",
    0.95: "#B64342",
}


def project_root_from_script(script_file: str | Path) -> Path:
    """Return the public package root identified by its marker file."""

    script = Path(script_file).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "COMPASS_PACKAGE.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate COMPASS_PACKAGE.toml above figure script")


def figure_dir_from_script(script_file: str | Path) -> Path:
    root = project_root_from_script(script_file)
    output = root / "outputs" / "figures" / Path(script_file).resolve().parent.name
    output.mkdir(parents=True, exist_ok=True)
    return output


def apply_style(font_size: float = 7.0) -> None:
    """Publication-oriented matplotlib defaults with editable vector text."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": font_size,
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


def clean_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.tick_params(width=0.8, length=3)
    if grid:
        ax.grid(True, color="#E6E6E6", lw=0.5, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=8)


def short_species(name: str, max_len: int = 24) -> str:
    if not isinstance(name, str):
        return str(name)
    parts = name.split()
    if len(parts) >= 2:
        text = f"{parts[0][0]}. {' '.join(parts[1:])}"
    else:
        text = name
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def short_method(method: str) -> str:
    mapping = {
        "data_driven": "Optimized Top-5",
        "EPA_WQC_taxonomic_requirements": "Op. EPA WQC proxy",
        "Canada_Type_A_composition": "Op. CCME Type A proxy",
        "EPA_WET_fixed_method_species": "US EPA WET species",
        "US_EPA_WET_species": "US EPA WET species",
        "taxonomy_diversity_baseline": "Taxonomic diversity",
        "random_all_species_probability": "Random 5-species",
    }
    return mapping.get(method, str(method).replace("_", " "))


def compact_moa(value: str) -> str:
    """Collapse long mechanism labels to compact manuscript-figure labels."""

    if not isinstance(value, str) or value.strip() == "":
        return "Unassigned"
    v = value.strip()
    if v.startswith("FORM::"):
        return "FORM: " + v.split("::", 1)[1].replace("_", " ")
    if v.startswith("MOAFAM::"):
        return "MOA: " + v.split("::", 1)[1].replace("_", " ")
    low = v.lower()
    if "acetylcholinesterase" in low or "sodium channel" in low or "ionotrop" in low or "neuro" in low:
        return "MOA: neurotransmission"
    if "thyro" in low:
        return "MOA: thyroid axis"
    if "deposition of energy" in low or "mitochond" in low:
        return "MOA: energy"
    if "photosystem" in low or "photosynthesis" in low:
        return "MOA: photosynthesis"
    if "ahr" in low or "nuclear" in low or "estrogen" in low or "androgen" in low:
        return "MOA: receptor/xenobiotic"
    if "alkylation" in low or "reactive" in low or "oxid" in low:
        return "MOA: reactive chemistry"
    if v.startswith("MIE::"):
        return "MIE: " + v.split("::", 1)[1].split(",")[0][:24]
    return v[:30]


def percentile_rank_high(values: pd.Series) -> pd.Series:
    """Percentile rank where smaller concentrations become higher concern."""

    return (-values.astype(float)).rank(method="average", pct=True) * 100.0


def q_panel(p: np.ndarray, species_indices: Sequence[int]) -> np.ndarray:
    """Joint panel score q=1-prod_s(1-p_s)."""

    if len(species_indices) == 0:
        return np.zeros(p.shape[1], dtype=float)
    sub = p[np.asarray(species_indices, dtype=int), :].astype(float)
    return 1.0 - np.prod(1.0 - sub, axis=0)


def write_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path.name


def write_combined_source(tables: Mapping[str, pd.DataFrame], out_path: Path) -> str:
    frames: list[pd.DataFrame] = []
    for panel, frame in tables.items():
        f = frame.copy()
        f.insert(0, "panel", panel)
        frames.append(f)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return write_csv(combined, out_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_figure(fig: plt.Figure, out_base: Path, dpi_png: int = 300, dpi_tiff: int = 600) -> list[str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi_png, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=dpi_tiff, bbox_inches="tight")
    plt.close(fig)
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        saved.append(out_base.with_suffix(suffix).name)
    return saved


def write_manifest(
    out_dir: Path,
    *,
    figure_id: str,
    title: str,
    manuscript_caption: str | None = None,
    core_conclusion: str,
    archetype: str,
    panel_map: Mapping[str, str],
    source_tables: Sequence[str],
    input_dependencies: Sequence[str],
    outputs: Sequence[str],
    reviewer_risk: str,
    notes: Sequence[str] | None = None,
) -> None:
    payload = {
        "figure_id": figure_id,
        "title": title,
        "manuscript_caption": manuscript_caption,
        "core_conclusion": core_conclusion,
        "archetype": archetype,
        "backend": "Python/Matplotlib",
        "final_size_inches": "double-column manuscript figure; see script figsize",
        "panel_map": dict(panel_map),
        "source_tables": list(source_tables),
        "input_dependencies": list(input_dependencies),
        "outputs": list(outputs),
        "reviewer_risk": reviewer_risk,
        "qa_checks": {
            "editable_svg_text": True,
            "pdf_truetype_text": True,
            "png_preview_dpi": 300,
            "tiff_dpi": 600,
            "source_data_in_figure_folder": True,
            "red_green_not_sole_encoding": True,
        },
        "notes": list(notes or []),
    }
    checksums = {}
    for name in [*source_tables, *outputs]:
        p = out_dir / name
        if p.exists() and p.is_file():
            checksums[name] = sha256_file(p)
    payload["sha256"] = checksums
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(root: Path, relative: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(root / relative, **kwargs)


def save_contract_note(out_dir: Path, lines: Iterable[str]) -> None:
    (out_dir / "figure_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
