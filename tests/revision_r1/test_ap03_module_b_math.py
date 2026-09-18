#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "code/revision_r1/run_ap03_module_b.py"
SPEC = importlib.util.spec_from_file_location("ap03b", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
AP03B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AP03B)


def test_panel_score_matches_union_formula() -> None:
    probabilities = np.array([[0.2, 0.1], [0.3, 0.4]])
    observed = AP03B.panel_score(probabilities)
    expected = np.array([1 - 0.8 * 0.7, 1 - 0.9 * 0.6])
    assert np.allclose(observed, expected)


def test_balanced_uses_supported_families_only_and_missing_nonmortality_is_not_zero() -> None:
    targets = pd.DataFrame(
        {
            "dtxsid": ["A", "A", "B"],
            "target_id": ["A|mortality_survival", "A|growth", "B|mortality_survival"],
            "effect_family": ["mortality_survival", "growth", "mortality_survival"],
            "panel_target_score": [0.9, 0.3, 0.8],
        }
    )
    observed = AP03B.aggregate_rankings(targets, "test").set_index("dtxsid")
    assert np.isclose(observed.loc["A", "test_balanced_score"], 0.6)
    assert np.isclose(observed.loc["A", "test_mortality_excluded_score"], 0.3)
    assert np.isnan(observed.loc["B", "test_mortality_excluded_score"])
    assert observed.loc["B", "test_mortality_excluded_status"] == "NOT SCOREABLE"


if __name__ == "__main__":
    tests = [
        test_panel_score_matches_union_formula,
        test_balanced_uses_supported_families_only_and_missing_nonmortality_is_not_zero,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
