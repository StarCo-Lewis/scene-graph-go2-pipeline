"""Load existing engines without copying, moving or changing their files."""

from importlib import import_module
from pathlib import Path
import sys
from threading import RLock

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIRS = (
    "vlm_full_scene_v1", "habitat_demo_v1", "object_alignment_v1",
    "skill_matching_v1", "llm_audit_v1", "scene_memory_v1",
)
_LOCK = RLock()


def bootstrap():
    """Make the legacy flat imports available; do not import engines eagerly."""
    with _LOCK:
        for directory in reversed(ENGINE_DIRS):
            path = str(ROOT / "outputs" / directory)
            if path not in sys.path:
                sys.path.insert(0, path)


def load(name):
    """Reject module-name collisions instead of silently using unrelated code."""
    with _LOCK:
        bootstrap()
        expected = [ROOT / "outputs" / directory / (name + ".py")
                    for directory in ENGINE_DIRS]
        expected = [path.resolve() for path in expected if path.is_file()]
        if len(expected) != 1:
            raise ImportError(f"Expected one legacy engine module for {name!r}")
        cached = sys.modules.get(name)
        if cached is not None and Path(getattr(cached, "__file__", "")).resolve() != expected[0]:
            raise ImportError(f"Legacy module name collision: {name}")
        module = import_module(name)
        if Path(module.__file__).resolve() != expected[0]:
            raise ImportError(f"Legacy module loaded from an unexpected path: {name}")
        return module


def resolve(exports, name):
    if name not in exports:
        raise AttributeError(name)
    module, attribute = exports[name]
    return getattr(load(module), attribute)

