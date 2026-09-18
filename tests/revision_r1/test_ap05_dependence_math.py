#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "code/revision_r1/run_ap05_dependence_audit.py"
SPEC = importlib.util.spec_from_file_location("ap05", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
AP05 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AP05)


def test_weighted_median_uses_shared_context_counts() -> None:
    observed = AP05.weighted_median(
        np.array([-0.2, 0.1, 0.3]), np.array([1.0, 10.0, 1.0])
    )
    assert observed == 0.1


def test_pair_filter_and_fallback() -> None:
    candidates = pd.DataFrame(
        {
            "latin_name": ["Species a", "Species b"],
            "n_contexts": [25, 25],
        }
    )
    rows = []
    for index in range(19):
        rows.extend(
            [
                {"context_id": f"c{index}", "latin_name": "Species a", "tail": index % 2},
                {"context_id": f"c{index}", "latin_name": "Species b", "tail": index % 2},
            ]
        )
    summary, pairs, _top = AP05.estimate_working_rho_with_pairs(
        pd.DataFrame(rows),
        candidates,
        tail_column="tail",
        variant="test",
        target=0.95,
    )
    assert pairs.empty
    assert summary["fallback_used"] is True
    assert summary["rho_working"] == 0.15


if __name__ == "__main__":
    tests = [
        test_weighted_median_uses_shared_context_counts,
        test_pair_filter_and_fallback,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
