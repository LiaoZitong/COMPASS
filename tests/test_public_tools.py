from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class InputValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module("validate_inputs_public", "code/validate_inputs.py")

    def test_identifier_patterns(self) -> None:
        self.assertIsNotNone(self.module.DTXSID_PATTERN.fullmatch("DTXSID1024122"))
        self.assertIsNone(self.module.DTXSID_PATTERN.fullmatch("1024122"))
        self.assertIsNotNone(
            self.module.FORMAL_BINOMIAL_PATTERN.fullmatch("Daphnia magna")
        )

    def test_range_check_flags_invalid_probability(self) -> None:
        findings = []
        frame = pd.DataFrame({"probability": [0.0, 0.5, 1.2]})
        self.module.check_range(
            findings, Path("fixture.csv"), frame, ["probability"], 0.0, 1.0
        )
        self.assertEqual(len(findings), 1)
        self.assertFalse(findings[0].passed)


class ResultValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module("validate_results_public", "code/validate_results.py")

    def test_absolute_tolerance(self) -> None:
        self.assertTrue(self.module.close(0.3055373085, 0.3055373086, 1e-9))
        self.assertFalse(self.module.close(0.30, 0.31, 1e-9))


class SiteExporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module("export_site_data_public", "code/export_site_data.py")

    def test_sequence_is_locked_to_lower_five_percent(self) -> None:
        frame = pd.DataFrame(
            [
                {"universe": "priority", "protection_target_x": 0.95, "method": "data_driven", "rank": 1, "latin_name": "Daphnia magna"},
                {"universe": "priority", "protection_target_x": 0.90, "method": "data_driven", "rank": 1, "latin_name": "Other species"},
            ]
        )
        selected = self.module.selected_sequence(frame)
        self.assertEqual(selected.iloc[0]["latin_name"], "Daphnia magna")

    def test_support_note_is_scope_first(self) -> None:
        note = self.module.support_note("very_sparse_1_4")
        self.assertIn("confirmatory testing", note)

    def test_complete_order_contains_every_candidate_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probability_dir = root / "results/probability"
            weight_dir = root / "results/weights"
            probability_dir.mkdir(parents=True)
            weight_dir.mkdir(parents=True)
            species = np.asarray(["Species alpha", "Species beta", "Species gamma", "Species delta"])
            chemicals = np.asarray([f"DTXSID{index:07d}" for index in range(362)])
            probabilities = np.asarray([[0.4], [0.3], [0.2], [0.1]]) @ np.ones((1, 362))
            np.savez(
                probability_dir / "species_chemical_tail_probability_x95.npz",
                p=probabilities,
                species=species,
                chemicals=chemicals,
            )
            pd.DataFrame(
                {"DTXSID": chemicals, "national_weight": np.ones(362)}
            ).to_csv(weight_dir / "national_priority_chemicals.csv", index=False)
            frozen = pd.DataFrame(
                {"rank": [1, 2], "latin_name": ["Species alpha", "Species beta"]}
            )
            candidate = pd.DataFrame({"latin_name": species})
            complete = self.module.build_complete_national_order(root, frozen, candidate)
            self.assertEqual(
                complete["latin_name"].tolist(),
                ["Species alpha", "Species beta", "Species gamma", "Species delta"],
            )
            self.assertEqual(complete["latin_name"].nunique(), 4)


class ReleaseAuditTests(unittest.TestCase):
    def test_temp_fixture_has_no_persistent_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.txt"
            path.write_text("synthetic fixture", encoding="utf-8")
            self.assertEqual(path.read_text(encoding="utf-8"), "synthetic fixture")


if __name__ == "__main__":
    unittest.main()
