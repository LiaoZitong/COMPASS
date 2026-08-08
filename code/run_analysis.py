#!/usr/bin/env python3
"""Run the public COMPASS analysis stages in their frozen semantic order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "code" / "pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["full", "upstream", "core", "robustness", "smoke"],
        default="full",
    )
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--profile-draws", type=int, default=500)
    parser.add_argument("--mnar-or", default="0.25,0.5,1,2,4")
    parser.add_argument("--traditional-random-replicates", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def result(path: str) -> Path:
    return PROJECT_ROOT / "results" / path


def upstream_inputs() -> list[Path]:
    return [
        PROJECT_ROOT / "data/intermediate/ecotox_canonical_partitions",
        PROJECT_ROOT / "data/processed/chemical_master.csv.gz",
        PROJECT_ROOT / "data/processed/state_chemical_exposure_2024.csv.gz",
        PROJECT_ROOT / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz",
        PROJECT_ROOT / "data/reference/AOP_MOA_data.zip",
        PROJECT_ROOT / "data/reference/species_traits_curated.csv",
        PROJECT_ROOT / "data/reference/lifestage_codes.txt",
        PROJECT_ROOT / "data/reference/regulatory_baseline_catalog.csv",
    ]


def core_inputs() -> list[Path]:
    return [
        PROJECT_ROOT / "results/toxicity/censored_model_cells.csv.gz",
        PROJECT_ROOT / "results/chemistry/chemical_moa_form_master.csv.gz",
        PROJECT_ROOT / "results/chemistry/species_traits_taxonomy.csv.gz",
        PROJECT_ROOT / "results/weights/national_priority_chemicals.csv",
        PROJECT_ROOT / "results/weights/state_priority_chemical_weights.csv.gz",
        PROJECT_ROOT / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz",
        PROJECT_ROOT / "data/reference/regulatory_baseline_catalog.csv",
    ]


def robustness_inputs() -> list[Path]:
    return [
        PROJECT_ROOT / "results/toxicity/censored_record_level_compact.csv.gz",
        PROJECT_ROOT / "results/probability/species_chemical_tail_probability_x95.npz",
        PROJECT_ROOT / "results/panels/national_panel_sequences.csv",
        PROJECT_ROOT / "results/hc5_framework_comparison/hc5_framework_context_estimates.csv.gz",
        PROJECT_ROOT / "results/validation/moa_simplified_oof_predictions.csv.gz",
    ]


def required_inputs(args: argparse.Namespace, mode: str) -> list[Path]:
    required: list[Path] = []
    if mode in {"full", "upstream"} and not args.skip_upstream:
        required.extend(upstream_inputs())
    if mode in {"full", "core"}:
        if args.skip_upstream or mode == "core":
            required.extend(core_inputs())
        else:
            required.append(PROJECT_ROOT / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz")
            required.append(PROJECT_ROOT / "data/reference/regulatory_baseline_catalog.csv")
    if mode in {"full", "robustness"} and (args.skip_upstream or mode == "robustness"):
        required.extend(robustness_inputs())
    return list(dict.fromkeys(required))


def ensure_layout(
    args: argparse.Namespace,
    mode: str,
    *,
    strict_inputs: bool,
    create_outputs: bool,
) -> list[str]:
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_inputs(args, mode)
        if not path.exists()
    ]
    if strict_inputs and missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    if not create_outputs:
        return missing
    for name in ("toxicity", "chemistry", "weights", "probability", "panels", "state_panels", "testing", "figures", "validation", "complementarity", "warning_hc5_bridge", "warning_lead_potential", "mechanism_evidence", "hc5_framework_comparison"):
        result(name).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "manifests").mkdir(exist_ok=True)
    return missing


def py(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(PIPELINE / script), *arguments]


def run(label: str, command: list[str], args: argparse.Namespace) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"[{label}] {printable}", flush=True)
    if args.dry_run:
        return
    log_path = PROJECT_ROOT / "logs" / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=runtime_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            # Windows PowerShell can expose a GBK console stream.  Preserve the
            # UTF-8 log verbatim and make terminal forwarding loss-tolerant so
            # an informational character can never terminate a scientific run.
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                safe = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace")
                print(safe, end="", flush=True)
            log.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def run_probability_pass(prefix: str, estimator: str, merger: str, calibration: Path | None, args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix=f"bhbt_{prefix}_") as temporary:
        work = Path(temporary)
        for label, target in (("80", "0.80"), ("90", "0.90"), ("95", "0.95")):
            command = py(estimator, "--model", str(result("toxicity/censored_model_cells.csv.gz")), "--chem", str(result("chemistry/chemical_moa_form_master.csv.gz")), "--traits", str(result("chemistry/species_traits_taxonomy.csv.gz")), "--occurrence", str(PROJECT_ROOT / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz"), "--weights-dir", str(result("weights")), "--out-prob", str(work / f"x{label}/prob"), "--out-panels", str(work / f"x{label}/panels"), "--target", target)
            if calibration is not None:
                command.extend(["--calibration-json", str(calibration)])
            run(f"{prefix}_{label}_{estimator.removesuffix('.py')}", command, args)
        run(f"{prefix}_{merger.removesuffix('.py')}", py(merger, "--x80-dir", str(work / "x80"), "--x90-dir", str(work / "x90"), "--x95-dir", str(work / "x95"), "--out-prob", str(result("probability")), "--out-panels", str(result("panels"))), args)


def upstream(args: argparse.Namespace) -> None:
    parts = str(PROJECT_ROOT / "data/intermediate/ecotox_canonical_partitions/part_*.csv.gz")
    run("01_prepare_censored_toxicity", py("01_prepare_censored_toxicity.py", "--parts", parts, "--life-codes", str(PROJECT_ROOT / "data/reference/lifestage_codes.txt"), "--out", str(result("toxicity"))), args)
    run("02_audit_evidence_layers", py("02_audit_evidence_layers.py", "--model", str(result("toxicity/censored_model_cells.csv.gz")), "--out", str(result("toxicity"))), args)
    run("03_build_chemical_annotations", py("03_build_chemical_annotations.py", "--chemical-master", str(PROJECT_ROOT / "data/processed/chemical_master.csv.gz"), "--aop-data", str(PROJECT_ROOT / "data/reference/AOP_MOA_data.zip"), "--traits", str(PROJECT_ROOT / "data/reference/species_traits_curated.csv"), "--out", str(result("chemistry"))), args)
    run(
        "04_refine_moa_annotations",
        py(
            "04_refine_moa_annotations.py",
            "--aop-zip", str(PROJECT_ROOT / "data/reference/AOP_MOA_data.zip"),
            "--chemical-master", str(PROJECT_ROOT / "data/processed/chemical_master.csv.gz"),
            "--chemical-annotations", str(result("chemistry/chemical_moa_form_master.csv.gz")),
            "--out", str(result("chemistry")),
        ),
        args,
    )
    run("05_build_national_weights", py("05_build_national_weights.py", "--exposure", str(PROJECT_ROOT / "data/processed/state_chemical_exposure_2024.csv.gz"), "--model-cells", str(result("toxicity/censored_model_cells.csv.gz")), "--chem", str(result("chemistry/chemical_moa_form_master.csv.gz")), "--out", str(result("weights"))), args)


def core(args: argparse.Namespace) -> None:
    run_probability_pass("06", "06_estimate_uncalibrated_tail_probabilities.py", "07_merge_uncalibrated_probability_targets.py", None, args)
    run("08_validate_chemical_holdout", py("08_validate_chemical_holdout.py", "--root", str(PROJECT_ROOT), "--model", str(result("toxicity/censored_model_cells.csv.gz")), "--chem", str(result("chemistry/chemical_moa_form_master.csv.gz")), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)), args)
    run("09_validate_moa_core_entry", py("09_validate_moa_core_entry.py", "--root", str(PROJECT_ROOT)), args)
    calibration = result("validation/hc5_core_calibration_parameters.json")
    run("10_prepare_hc5_core_calibration", py("10_prepare_hc5_core_calibration.py", "--input", str(result("validation/probability_calibration_parameters.json")), "--output", str(calibration)), args)
    run_probability_pass("11", "11_estimate_calibrated_tail_probabilities.py", "12_merge_calibrated_probability_targets.py", calibration, args)
    run("13_build_hc5_panels", py("13_build_hc5_panels.py", "--root", str(PROJECT_ROOT), "--prob-npz", str(result("probability/species_chemical_tail_probability_x95.npz"))), args)
    run("14_build_random_baselines", py("14_build_random_baselines.py", "--root", str(PROJECT_ROOT), "--replicates", str(args.traditional_random_replicates), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed), "--targets", "0.95", "--universes", "priority,all_chemicals"), args)
    run("15_quantify_taxonomic_complementarity", py("15_quantify_taxonomic_complementarity.py", "--root", str(PROJECT_ROOT), "--target", "0.95"), args)
    run("16_evaluate_warning_hc5_bridge", py("16_evaluate_warning_hc5_bridge.py", "--root", str(PROJECT_ROOT), "--bootstrap", str(args.bootstrap)), args)
    run("17_assess_warning_apical_lead_potential", py("17_assess_warning_apical_lead_potential.py", "--root", str(PROJECT_ROOT), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)), args)
    run("18_evaluate_mechanism_evidence_and_testing", py("18_evaluate_mechanism_evidence_and_testing.py", "--root", str(PROJECT_ROOT), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)), args)
    run("19_compare_hc5_frameworks", py("19_compare_hc5_frameworks.py", "--root", str(PROJECT_ROOT), "--random-replicates", str(args.traditional_random_replicates), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)), args)
    run("20_build_state_panels_and_testing", py("20_build_state_panels_and_testing.py", "--prob", str(result("probability/species_chemical_tail_probability_x95.npz")), "--occurrence", str(PROJECT_ROOT / "data/processed/state_species_occurrence_with_usgs_nas.csv.gz"), "--state-weights", str(result("weights/state_priority_chemical_weights.csv.gz")), "--candidate-summary", str(result("probability/candidate_universe_locked.csv")), "--priority", str(result("weights/national_priority_chemicals.csv")), "--chem", str(result("chemistry/chemical_moa_form_master.csv.gz")), "--under-tested-registry", str(result("probability/under_tested_species_registry.csv")), "--out-state", str(result("state_panels")), "--out-testing", str(result("testing"))), args)


def robustness(args: argparse.Namespace) -> None:
    run("21_propagate_hc5_profile_uncertainty", py("21_propagate_hc5_profile_uncertainty.py", "--root", str(PROJECT_ROOT), "--draws", str(args.profile_draws), "--seed", str(args.seed)), args)
    run("22_assess_mnar_sensitivity", py("22_assess_mnar_sensitivity.py", "--root", str(PROJECT_ROOT), "--odds-ratios", args.mnar_or, "--seed", str(args.seed)), args)


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        args.bootstrap = min(args.bootstrap, 5)
        args.profile_draws = min(args.profile_draws, 25)
        args.traditional_random_replicates = min(args.traditional_random_replicates, 3)
        mode = "full"
    else:
        mode = args.mode
    missing = ensure_layout(
        args,
        mode,
        strict_inputs=not args.dry_run,
        create_outputs=not args.dry_run,
    )
    if args.dry_run and missing:
        print("[dry-run] Missing inputs are expected in a code-only checkout:")
        for path in missing:
            print(f"  - {path}")
    if mode in {"full", "upstream"} and not args.skip_upstream:
        upstream(args)
    if mode in {"full", "core"}:
        core(args)
    if mode in {"full", "robustness"}:
        robustness(args)


if __name__ == "__main__":
    main()
