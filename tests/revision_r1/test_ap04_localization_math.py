#!/usr/bin/env python3
"""Direct assertion tests for AP04 helper mathematics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "code/revision_r1/run_ap04_localization_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("ap04", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AP04 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AP04)


def run() -> None:
    assert AP04.jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert AP04.jaccard(["a"], ["b"]) == 0.0
    assert AP04.normalized_top5_footrule(list("abcde"), list("abcde")) == 1.0
    assert AP04.normalized_top5_footrule(list("abcde"), list("fghij")) == 0.0
    reversed_score = AP04.normalized_top5_footrule(list("abcde"), list("edcba"))
    assert 0.0 < reversed_score < 1.0

    frame = pd.DataFrame(
        {
            "detection_rank_state": [1.0, 0.0],
            "concentration_rank_state": [0.0, 1.0],
            "site_rank_state": [0.5, 0.5],
            "analytical_invisibility_proxy": [0.0, 0.0],
        }
    )
    weights = AP04.component_weights(frame, [0.45, 0.35, 0.10, 0.10])
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]

    support_frame = pd.DataFrame(
        {
            "n_result_records": [1, 100, 10, 5],
            "n_sites": [1, 20, 2, 3],
            "DTXSID": ["D", "A", "C", "B"],
        }
    )
    retained = AP04.retained_group(support_frame, 0.5)
    assert len(retained) == 2
    assert "A" in set(retained["DTXSID"])

    species_frame = pd.DataFrame(
        {
            "soft_local_species_weight": [0.6, 0.6, 0.04],
            "soft_local_evidence_class": ["state_occurrence", "model_transfer_low_weight", "neighbor_occurrence"],
            "eligible_state_candidate": [True, False, True],
            "exact_state_occurrence": [True, False, True],
            "current_or_unresolved_official_nas": [False, True, False],
        }
    )
    assert AP04.candidate_mask(species_frame, "soft_baseline").tolist() == [True, True, False]
    assert AP04.candidate_mask(
        species_frame, "hard_exclude_current_or_unresolved_official_nas"
    ).tolist() == [True, False, False]
    assert AP04.candidate_mask(species_frame, "state_confirmed_only").tolist() == [True, False, True]
    assert AP04.candidate_mask(species_frame, "regional_breadth_or_better").tolist() == [True, False, False]

    print("AP04 helper tests: PASS")


if __name__ == "__main__":
    run()
