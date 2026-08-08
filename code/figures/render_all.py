#!/usr/bin/env python3
"""Render the five frozen COMPASS main figures from supplied analysis outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = [
        here / "figure_01" / "plot_figure_01.py",
        here / "figure_02" / "plot_figure_02.py",
        here / "figure_03" / "plot_figure_03.py",
        here / "figure_04" / "plot_figure_04.py",
        here / "figure_05" / "plot_figure_05.py",
    ]
    for script in scripts:
        print(f"Rendering {script.parent.name} from {script.name}")
        subprocess.run(
            [sys.executable, str(script)],
            cwd=here.parents[1],
            check=True,
        )


if __name__ == "__main__":
    main()
