"""Explicit collision inflation for this meter-scale Scene201 proxy stage.

Original mesh/config files are never changed. The new run records its changed
physics parameter; this is NOT ignoring stage collisions or forgiving contacts.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from run_navigation_demo import read_json, write_json

STAGE_MARGIN_M = .001


def prepare(source, output_dir):
    source, root = Path(source).resolve(), Path(output_dir).resolve()
    target = root / "runtime_scene201.stage_config.json"
    if source == target or target.exists():
        raise ValueError("Refusing to overwrite a source/existing runtime stage config")
    data = read_json(source)
    if not isinstance(data, dict):
        raise ValueError("Stage config must be a JSON object")
    if data.get('units_to_meters', 1.) != 1. or data.get('scale', [1.,1.,1.]) != [1.,1.,1.]:
        raise ValueError("Stage margin policy only supports the meter-scale Scene201 asset")
    if data.get('is_collidable', True) is not True:
        raise ValueError("Stage collisions must remain enabled")
    original_margin = data.get('margin')
    assets = {}
    for field in ('render_asset', 'collision_asset'):
        value = data.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing stage asset: {field}")
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        if not path.is_file():
            raise ValueError(f"Missing stage asset: {path}")
        data[field] = str(path)
        assets[field] = str(path)
    # Optional file references must keep their original base directory too.
    for field in ('semantic_asset', 'nav_asset', 'semantic_descriptor_filename'):
        value = data.get(field)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            data[field] = str((source.parent / value).resolve())
    data.update(margin=STAGE_MARGIN_M, is_collidable=True)
    write_json(target, data)
    report = {"schema_version":"1.0", "policy":"scene201_explicit_static_mesh_margin_v1",
              "source_config":str(source), "source_config_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "runtime_config":str(target), "source_config_margin":original_margin,
              "habitat_0_2_3_default_when_unspecified_m":.04,
              "requested_stage_margin_m":STAGE_MARGIN_M, "loaded_stage_margin_m":None,
              "status":"configuration_written_not_loaded", "geometry_assets":assets,
              "geometry_assets_modified":False, "contact_rejection_tolerance_modified":False,
              "original_config_modified":False, "training_eligible":False}
    write_json(root / "stage_collision_configuration.json", report)
    return target, report


def verify_loaded(sim, report, output_dir):
    """Fail closed if the instantiated stage did not consume the explicit margin."""
    try:
        template = sim.get_stage_initialization_template()
        if template is None:
            raise ValueError("STAGE_COLLISION_CONFIG_UNVERIFIED: no initialization template")
        margin = float(template.margin)
        collidable = bool(sim.get_stage_is_collidable())
        report.update(loaded_stage_margin_m=margin if math.isfinite(margin) else None,
                      loaded_stage_collidable=collidable)
        if not math.isfinite(margin) or abs(margin-STAGE_MARGIN_M) > 1e-7 or not collidable:
            raise ValueError("STAGE_COLLISION_CONFIG_MISMATCH")
        report['status'] = 'loaded_parameters_verified'
        return report
    except Exception as exc:
        report.update(status='loaded_parameters_rejected', error=str(exc))
        raise ValueError(f"Cannot verify stage collision configuration: {exc}") from exc
    finally:
        write_json(Path(output_dir) / "stage_collision_configuration.json", report)
