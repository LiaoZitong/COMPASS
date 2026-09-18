#!/usr/bin/env python3
"""Run the reviewer-requested R1 analyses in their frozen dependency order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STEPS = (
    ("01_ap01_eligibility", "build_ap01_eligibility_ledger.py"),
    ("02_ap02_strict_loo", "run_ap02_strict_loo.py"),
    ("03_ap05_dependence", "run_ap05_dependence_audit.py"),
    ("04_ap03_module_a", "run_ap03_module_a.py"),
    ("05_ap03_module_b", "run_ap03_module_b.py"),
    ("06_ap04_localization_sensitivity", "run_ap04_localization_sensitivity.py"),
    ("07_g3_cross_package_integration", "run_g3_cross_package_integration.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="COMPASS repository containing the completed base-stage results/ tree.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/revision_r1"),
        help="R1 output directory; relative paths are resolved below --analysis-root.",
    )
    parser.add_argument(
        "--localization-config",
        type=Path,
        default=Path("config/r1_localization_sensitivity.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--from-step",
        choices=[name for name, _script in STEPS],
        help="Resume at this step after verifying all earlier gate files.",
    )
    return parser.parse_args()


def resolved_below(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path must remain below analysis root: {resolved}")
    return resolved


def command_for(
    scripts: Path,
    name: str,
    root: Path,
    out: Path,
    config: Path,
) -> list[str]:
    directories = {step: out / step for step, _script in STEPS}
    script = scripts / dict(STEPS)[name]
    if name == "01_ap01_eligibility":
        return [sys.executable, str(script), "--bhbt-root", str(root), "--out", str(directories[name])]
    if name == "02_ap02_strict_loo":
        return [sys.executable, str(script), "--bhbt-root", str(root), "--out", str(directories[name])]
    if name == "03_ap05_dependence":
        return [
            sys.executable, str(script), "--bhbt-root", str(root),
            "--ap02-dir", str(directories["02_ap02_strict_loo"]),
            "--out", str(directories[name]),
        ]
    if name == "04_ap03_module_a":
        return [
            sys.executable, str(script), "--bhbt-root", str(root),
            "--ap02-dir", str(directories["02_ap02_strict_loo"]),
            "--ap05-dir", str(directories["03_ap05_dependence"]),
            "--out", str(directories[name]),
        ]
    if name == "05_ap03_module_b":
        return [
            sys.executable, str(script), "--bhbt-root", str(root),
            "--ap02-dir", str(directories["02_ap02_strict_loo"]),
            "--ap03a-dir", str(directories["04_ap03_module_a"]),
            "--out", str(directories[name]),
        ]
    if name == "06_ap04_localization_sensitivity":
        return [
            sys.executable, str(script), "--project-root", str(root),
            "--analysis-root", str(out), "--out", str(directories[name]),
            "--config", str(config),
        ]
    if name == "07_g3_cross_package_integration":
        return [
            sys.executable, str(script), "--analysis-root", str(out),
            "--out", str(directories[name]),
        ]
    raise KeyError(name)


def require_resume_gates(out: Path, start_index: int) -> None:
    gates = {
        "01_ap01_eligibility": "gate_ap01.json",
        "02_ap02_strict_loo": "gate_ap02.json",
        "03_ap05_dependence": "gate_ap05.json",
        "04_ap03_module_a": "gate_ap03a.json",
        "05_ap03_module_b": "gate_ap03b.json",
        "06_ap04_localization_sensitivity": "gate_ap04.json",
    }
    for name, _script in STEPS[:start_index]:
        gate_name = gates.get(name)
        if gate_name is None:
            continue
        path = out / name / gate_name
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume: missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"Cannot resume: gate is not PASS: {path}")


def main() -> int:
    args = parse_args()
    root = args.analysis_root.resolve()
    if not (root / "results").is_dir() or not (root / "code/pipeline").is_dir():
        raise FileNotFoundError("--analysis-root must contain results/ and code/pipeline/")
    out = resolved_below(root, args.out)
    config = resolved_below(root, args.localization_config)
    if not config.is_file():
        raise FileNotFoundError(config)
    scripts = Path(__file__).resolve().parent
    start_index = 0 if args.from_step is None else [name for name, _ in STEPS].index(args.from_step)
    require_resume_gates(out, start_index)
    out.mkdir(parents=True, exist_ok=True)

    plan = []
    for name, _script in STEPS[start_index:]:
        command = command_for(scripts, name, root, out, config)
        plan.append({"step": name, "command": command})
        print(f"[{name}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)

    manifest = {
        "schema_version": 1,
        "analysis": "COMPASS R1 strict focal-species leave-one-out revision",
        "ordered_steps": [name for name, _ in STEPS],
        "executed_from_step": STEPS[start_index][0],
        "dry_run": bool(args.dry_run),
        "commands": plan,
    }
    if not args.dry_run:
        (out / "r1_run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
