#!/usr/bin/env python3
"""Direct assertion checks for the completed G3 integration package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g3-dir", required=True)
    args = parser.parse_args()
    root = Path(args.g3_dir).resolve()
    gate = json.loads((root / "gate_g3.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "g3_authoritative_numeric_freeze.json").read_text(encoding="utf-8"))
    decisions = pd.read_csv(root / "g3_analysis_decision_register.csv")
    claims = pd.read_csv(root / "g3_claim_boundary_matrix.csv")

    assert gate["status"] == "PASS"
    assert gate["primary_analysis_update_required"] is True
    assert gate["manuscript_or_si_edits_performed"] is False
    assert freeze["strict_loo_primary"]["top5_membership_changed"] is False
    assert freeze["strict_loo_primary"]["top5_order_changed"] is True
    assert freeze["available_member_validation"]["fully_observed_top5_rows"] == 0
    assert freeze["followup_ranking"]["mortality_excluded_not_scoreable"] == 494
    assert freeze["localization"]["fallback_states"] == ["PA", "TN", "WV"]
    assert set(decisions["package"]) == {"AP01", "AP02", "AP05", "AP03A", "AP03B", "AP04"}
    assert len(claims) == 4
    print("G3 integration output tests: PASS")


if __name__ == "__main__":
    main()
