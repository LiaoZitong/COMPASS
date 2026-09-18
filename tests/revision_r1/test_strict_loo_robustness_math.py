from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "code/revision_r1/run_strict_loo_robustness.py"
SPEC = importlib.util.spec_from_file_location("run_strict_loo_robustness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_dependence_matches_independent_union() -> None:
    probability = np.array(
        [
            [0.10, 0.30, 0.80],
            [0.25, 0.20, 0.15],
            [0.40, 0.05, 0.10],
        ],
        dtype=float,
    )
    expected = 1.0 - np.prod(1.0 - probability, axis=0)
    observed = MODULE.copula_union(probability, rho=0.0)
    assert np.allclose(observed, expected, rtol=0, atol=1e-10)


def test_selection_shift_has_mar_anchor_and_expected_direction() -> None:
    probability = np.array([0.05, 0.25, 0.80], dtype=float)
    reverse = MODULE.apply_selection_odds_ratio(probability, 0.25)
    anchor = MODULE.apply_selection_odds_ratio(probability, 1.0)
    preferential = MODULE.apply_selection_odds_ratio(probability, 4.0)
    assert np.allclose(anchor, probability, rtol=0, atol=1e-12)
    assert np.all(reverse > anchor)
    assert np.all(preferential < anchor)


def test_greedy_tie_breaking_is_deterministic() -> None:
    probability = np.full((3, 4), 0.2, dtype=float)
    weights = np.repeat(0.25, 4)
    species = ["Species c", "Species a", "Species b"]
    selected = MODULE.deterministic_greedy(probability, weights, species, k=2)
    assert [species[index] for index in selected] == ["Species a", "Species b"]


if __name__ == "__main__":
    tests = [
        test_zero_dependence_matches_independent_union,
        test_selection_shift_has_mar_anchor_and_expected_direction,
        test_greedy_tie_breaking_is_deterministic,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
