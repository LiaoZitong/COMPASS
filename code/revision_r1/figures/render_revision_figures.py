#!/usr/bin/env python3
"""Render the R1-updated main/SI figures and AP04 SI table in release order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    package_root = Path(__file__).resolve().parents[3]
    revision_root = package_root / "results/revision_r1"
    scripts = Path(__file__).resolve().parent
    revision_scripts = scripts.parent
    if not (revision_root / "07_g3_cross_package_integration/gate_g3.json").is_file():
        raise FileNotFoundError("Run the complete R1 analysis and G3 integration gate before figures")
    robustness = revision_root / "08_strict_loo_robustness"
    if not (robustness / "gate_strict_loo_robustness.json").is_file():
        raise FileNotFoundError("Run the strict-LOO robustness stage before figures")
    base_samples = package_root / "outputs/figures/figure_02/figure_02b_feasible_combination_samples.csv"
    if not base_samples.is_file():
        raise FileNotFoundError(
            "Run code/figures/figure_02/plot_figure_02.py first; its preserved seeded "
            "feasible-panel samples are an input to the R1 Figure 2 rescoring."
        )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "render_r1_si_figure_s6.py"),
            "--analysis-dir",
            str(robustness),
            "--output-dir",
            str(revision_root / "08_r1_exhibits" / "si_figure_s6"),
        ],
        cwd=package_root,
        check=True,
    )
    for name in (
        "render_r1_main_figure_02.py",
        "render_r1_main_figure_03.py",
        "render_r1_main_figure_05.py",
        "render_r1_si_figure_s3.py",
        "render_r1_si_figure_s4.py",
        "render_r1_si_figure_s7.py",
    ):
        subprocess.run([sys.executable, str(scripts / name)], cwd=package_root, check=True)
    ap04 = revision_root / "06_ap04_localization_sensitivity"
    s8 = ap04 / "si_figure_s8"
    table_s7 = ap04 / "si_table_s7"
    subprocess.run(
        [sys.executable, str(revision_scripts / "render_ap04_si_figure.py"), "--input-dir", str(ap04), "--output-dir", str(s8)],
        cwd=package_root,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(revision_scripts / "build_ap04_si_table.py"), "--input-dir", str(ap04), "--output-dir", str(table_s7)],
        cwd=package_root,
        check=True,
    )
    subprocess.run([sys.executable, str(scripts / "validate_r1_exhibits.py")], cwd=package_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
