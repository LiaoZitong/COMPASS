#!/usr/bin/env python3
"""Run the calibrated pass of the measured-tail probability estimator.

This wrapper preserves a sequential, content-specific pipeline entry while
delegating the shared estimator implementation to step 06.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("06_estimate_uncalibrated_tail_probabilities.py")), run_name="__main__")
