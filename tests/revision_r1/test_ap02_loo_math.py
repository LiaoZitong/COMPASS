from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "code/revision_r1/run_ap02_strict_loo.py"
SPEC = importlib.util.spec_from_file_location("run_ap02_strict_loo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_leave_one_out_mean_and_sample_sd_match_brute_force() -> None:
    values = np.array([-1.2, -0.4, 0.1, 0.9, 1.7, 2.2], dtype=float)
    means, sds = MODULE.leave_one_out_stats(values)
    for index in range(len(values)):
        retained = np.delete(values, index)
        assert np.isclose(means[index], retained.mean(), rtol=0, atol=1e-12)
        assert np.isclose(sds[index], retained.std(ddof=1), rtol=0, atol=1e-12)


def test_leave_one_out_handles_constant_remaining_values() -> None:
    values = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    means, sds = MODULE.leave_one_out_stats(values)
    assert means[0] == 1.0
    assert sds[0] == 0.0
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(sds))


if __name__ == "__main__":
    tests = [
        test_leave_one_out_mean_and_sample_sd_match_brute_force,
        test_leave_one_out_handles_constant_remaining_values,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
