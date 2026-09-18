#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "code/revision_r1/run_ap03_module_a.py"
SPEC = importlib.util.spec_from_file_location("ap03a", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
AP03A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AP03A)


def test_correlation_metrics_preserve_rank_direction() -> None:
    data = pd.DataFrame({"x": [4.0, 1.0, 3.0, 2.0], "y": [40.0, 10.0, 30.0, 20.0]})
    observed = AP03A.correlation_metrics(data, "x", "y")
    assert observed["n"] == 4
    assert np.isclose(observed["spearman_rho"], 1.0)
    assert np.isclose(observed["kendall_tau"], 1.0)


def test_correlation_metrics_do_not_impute_missing_values() -> None:
    data = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    observed = AP03A.correlation_metrics(data, "x", "y")
    assert observed["n"] == 3
    assert np.isclose(observed["spearman_rho"], 1.0)


if __name__ == "__main__":
    tests = [
        test_correlation_metrics_preserve_rank_direction,
        test_correlation_metrics_do_not_impute_missing_values,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
