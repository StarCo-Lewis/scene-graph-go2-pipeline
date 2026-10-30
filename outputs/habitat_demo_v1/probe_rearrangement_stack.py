#!/usr/bin/env python3
"""Inspect whether the active Python can run Habitat-Lab Rearrangement."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def package_version(*names: str):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    errors = []

    try:
        import habitat_sim
        checks["habitat_sim_import"] = True
        checks["habitat_sim_version"] = package_version("habitat-sim", "habitat_sim")
        checks["bullet_build_flag"] = getattr(
            getattr(habitat_sim, "bindings", None), "built_with_bullet", None)
    except Exception as exc:
        checks["habitat_sim_import"] = False
        errors.append(f"habitat_sim: {type(exc).__name__}: {exc}")

    checks["habitat_lab_module_found"] = importlib.util.find_spec("habitat") is not None
    try:
        from habitat.tasks.rearrange.rearrange_grasp_manager import RearrangeGraspManager
        from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
        checks["rearrange_import"] = True
        checks["grasp_manager"] = RearrangeGraspManager.__name__
        checks["rearrange_sim"] = RearrangeSim.__name__
    except Exception as exc:
        checks["rearrange_import"] = False
        errors.append(f"habitat_lab_rearrange: {type(exc).__name__}: {exc}")
    checks["habitat_lab_version"] = package_version("habitat-lab", "habitat_lab")

    ready = bool(checks.get("habitat_sim_import") and checks.get("rearrange_import"))
    result = {
        "schema_version": "1.0",
        "status": "software_ready_assets_unverified" if ready else "missing_rearrangement_components",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "checks": checks,
        "errors": errors,
        "next_requirement": (
            "Select and validate an articulated mobile-manipulator asset and task config."
            if ready else
            "Install a Habitat-Lab version compatible with the existing Habitat-Sim build."
        ),
        "claim": "This import probe does not demonstrate IK, arm motion, grasp, or task execution.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
