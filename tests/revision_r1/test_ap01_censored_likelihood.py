from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PACKAGE_ROOT / "code/pipeline/01_prepare_censored_toxicity.py"
SPEC = importlib.util.spec_from_file_location("prepare_censored_toxicity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixed_sigma_loglik(mu: np.ndarray, kinds, points, lowers, uppers, sigma: float) -> np.ndarray:
    mu = np.asarray(mu, float)
    out = np.zeros_like(mu)
    for kind, point, lower, upper in zip(kinds, points, lowers, uppers):
        if kind in {"exact", "approximate_exact"}:
            out += -0.5 * ((point - mu) / sigma) ** 2
        elif kind == "left_censored":
            out += log_ndtr((upper - mu) / sigma)
        elif kind == "right_censored":
            out += log_ndtr((mu - lower) / sigma)
        elif kind == "interval_censored":
            upper_cdf = np.exp(log_ndtr((upper - mu) / sigma))
            lower_cdf = np.exp(log_ndtr((lower - mu) / sigma))
            out += np.log(np.maximum(upper_cdf - lower_cdf, 1e-300))
        else:
            raise AssertionError(kind)
    return out


def test_left_censored_observation_is_not_half_limit_substitution() -> None:
    kinds = np.array(["exact", "left_censored", "exact"], object)
    points = np.array([0.2, np.nan, 1.4])
    lowers = np.array([0.2, np.nan, 1.4])
    uppers = np.array([0.2, -0.3, 1.4])
    sigma = 0.65
    estimate, status, *_ = MODULE.censored_normal_em(kinds, points, lowers, uppers, sigma)
    grid = np.linspace(-2.5, 2.5, 20001)
    grid_mle = float(grid[np.argmax(fixed_sigma_loglik(grid, kinds, points, lowers, uppers, sigma))])
    naive_half_limit = float(np.mean([0.2, -0.3 - np.log10(2), 1.4]))
    assert status == "identified_censored_mle"
    assert abs(estimate - grid_mle) < 5e-4
    assert abs(estimate - naive_half_limit) > 1e-2


def test_interval_observation_is_not_midpoint_substitution() -> None:
    kinds = np.array(["exact", "interval_censored", "exact"], object)
    points = np.array([-0.8, np.nan, 1.1])
    lowers = np.array([-0.8, -0.2, 1.1])
    uppers = np.array([-0.8, 0.9, 1.1])
    sigma = 0.55
    estimate, status, *_ = MODULE.censored_normal_em(kinds, points, lowers, uppers, sigma)
    grid = np.linspace(-2.5, 2.5, 20001)
    grid_mle = float(grid[np.argmax(fixed_sigma_loglik(grid, kinds, points, lowers, uppers, sigma))])
    midpoint_mean = float(np.mean([-0.8, (-0.2 + 0.9) / 2, 1.1]))
    assert status == "identified_censored_mle"
    assert abs(estimate - grid_mle) < 5e-4
    assert abs(estimate - midpoint_mean) > 1e-3


def test_one_sided_only_cell_remains_a_bound() -> None:
    kinds = np.array(["left_censored", "left_censored"], object)
    points = np.array([np.nan, np.nan])
    lowers = np.array([np.nan, np.nan])
    uppers = np.array([-0.7, -0.2])
    estimate, status, _se, bound, _iterations = MODULE.censored_normal_em(
        kinds, points, lowers, uppers, sigma=0.65
    )
    assert np.isnan(estimate)
    assert status == "upper_bound_only"
    assert bound == -0.7


if __name__ == "__main__":
    tests = [
        test_left_censored_observation_is_not_half_limit_substitution,
        test_interval_observation_is_not_midpoint_substitution,
        test_one_sided_only_cell_remains_a_bound,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
