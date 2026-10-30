#!/usr/bin/env python3
"""Check Spot-with-arm software, configuration, URDF and IK readiness."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path


REL_URDF = Path("data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
REL_AGENT_CONFIG = Path("habitat/config/habitat/simulator/agents/spot.yaml")
SPOT_IMPORTS = (
    ("habitat.articulated_agents.robots", "SpotRobot"),
    ("habitat.robots.spot_robot", "SpotRobot"),
)


def unique_existing(paths):
    seen = set()
    output = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen and resolved.is_file():
            seen.add(key)
            output.append(key)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = []
    checks = {}
    habitat_file = None
    try:
        import habitat
        habitat_file = Path(habitat.__file__).resolve()
        checks["habitat_package_file"] = str(habitat_file)
    except Exception as exc:
        errors.append(f"habitat: {type(exc).__name__}: {exc}")
    import_errors = []
    spot_class = None
    for module_name, class_name in SPOT_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            spot_class = getattr(module, class_name)
            checks["spot_robot_import_path"] = f"{module_name}.{class_name}"
            break
        except Exception as exc:
            import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    checks["spot_robot_import"] = spot_class is not None
    checks["spot_import_attempts"] = import_errors
    if spot_class is not None:
        checks["spot_robot_class_file"] = inspect.getfile(spot_class)
    else:
        errors.append("SpotRobot was unavailable from both current and Habitat 0.2.3 namespaces")

    checks["pybullet_available_for_ik"] = importlib.util.find_spec("pybullet") is not None
    roots = [Path.cwd(), Path.home() / "habitat-lab", Path.home() / "Workspace" / "habitat-lab",
             Path(sys.prefix), Path("/opt/habitat-lab")]
    if habitat_file is not None:
        roots.extend(habitat_file.parents[:4])
    urdfs = unique_existing(root / REL_URDF for root in roots)
    configs = unique_existing(root / REL_AGENT_CONFIG for root in roots)
    # Installed package layouts normally omit the leading habitat/ directory.
    if habitat_file is not None:
        configs.extend(unique_existing([
            habitat_file.parent / "config/habitat/simulator/agents/spot.yaml",
            habitat_file.parent / "config/default_structured_configs.py",
        ]))
        configs = list(dict.fromkeys(configs))
    checks["spot_urdf_candidates"] = urdfs
    checks["spot_config_candidates"] = configs
    checks["spot_urdf_found"] = bool(urdfs)
    checks["spot_config_found"] = bool(configs)

    ready = bool(checks.get("spot_robot_import") and checks["pybullet_available_for_ik"]
                 and checks["spot_urdf_found"])
    missing = [name for name in ("spot_robot_import", "pybullet_available_for_ik",
                                  "spot_urdf_found")
               if not checks.get(name)]
    if ready:
        status = "spot_assets_and_ik_ready"
    elif checks.get("spot_robot_import") and checks.get("spot_urdf_found"):
        status = "spot_code_and_urdf_ready_ik_missing"
    elif checks.get("spot_robot_import"):
        status = "spot_code_ready_assets_or_ik_missing"
    else:
        status = "spot_setup_incomplete"
    result = {
        "schema_version": "1.0",
        "status": status,
        "python": sys.executable,
        "checks": checks,
        "missing": missing,
        "errors": errors,
        "next_step": (
            "Instantiate SpotRobot in a minimal headless Scene201 arm-control smoke test."
            if ready else
            "Install or locate only the listed required Spot components before scene loading."
        ),
        "note": "A standalone Spot YAML is version-dependent and is not required when the legacy class provides parameters in code.",
        "claim": "This is an asset/import check; it does not prove collision-free reach or grasp success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
