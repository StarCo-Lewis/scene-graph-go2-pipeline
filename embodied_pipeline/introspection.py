"""Read-only architecture and legacy integrity inspection; standard library only."""

import hashlib
import json
from pathlib import Path

from ._legacy import ROOT
from .contracts import MODULES

MANIFEST = Path(__file__).with_name("legacy_manifest.json")


def verify_legacy():
    baseline = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed, missing = [], []
    for entry in baseline["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            changed.append(entry["path"])
    return dict(status="unchanged" if not changed and not missing else "changed",
        checked_file_count=len(baseline["files"]), changed=changed, missing=missing,
        scope=baseline["scope"], engine_snapshot_copied=False)


def describe():
    return dict(schema_version="modular_compatibility_layout_v1",
        modules=[vars(spec) for spec in MODULES],
        orchestration="Existing feedback flow and batch/suite dispatch",
        contracts="Interface protocols and module registry; unchanged legacy wire formats",
        defaults="Inherited from the selected legacy runner; typed5 is explicitly selected",
        new_llm_stages=0, new_recovery_policy=False, room_topology_inference=False,
        robot_backend_connected=False, legacy_files_moved=False,
        note="Compatibility adapters reuse engines; proposal/commit/flow are structurally extracted.")


def write_run_manifest(directory):
    """Separate provenance sidecar; never add architecture metadata to LLM input."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    source_files = {str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in Path(__file__).parent.rglob("*.py")}
    value = dict(**describe(), module_sources_sha256=source_files, legacy_integrity=verify_legacy())
    (directory / "architecture_manifest.json").write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="ascii")

