#!/usr/bin/env python3
"""Merge calibrated x80/x90/x95 probability targets.

This wrapper gives the second merge pass its own sequential, semantic entry.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("07_merge_uncalibrated_probability_targets.py")), run_name="__main__")
