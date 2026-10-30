import copy
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_audit as va


class Pixels:
    shape = (2, 2, 3)
    dtype = "uint8"
    def tobytes(self, order="C"):
        return bytes(range(12))


class Executor:
    def __init__(self):
        self.state = {"robot_position": [1., 0., 2.], "robot_rotation_wxyz": [1., 0., 0., 0.],
                      "objects": {"67": {"center": [1., 1., 2.], "com": [1., .9, 2.], "support_id": None}},
                      "held_object_id": 67, "memory_version": 1, "lifted": True}
    def snapshot(self):
        return copy.deepcopy(self.state)


class AuditTests(unittest.TestCase):
    def make(self, root, enabled=True, **kwargs):
        args = SimpleNamespace(output_dir=Path(root), visual_audit=enabled,
                               audit_stride=kwargs.get("stride", 2), audit_max_frames=kwargs.get("limit", 30))
        summary = {"status": "passed"}
        audit = va.VisualAudit(args, summary)
        executor = Executor()
        audit.bind(SimpleNamespace(sim=object()), executor)
        return audit, summary, executor

    def test_png_roundtrip_chunks(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "frame.png"
            va.save_rgb(path, Pixels())
            data = path.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            offset, chunks = 8, {}
            while offset < len(data):
                n = struct.unpack(">I", data[offset:offset+4])[0]
                kind, value = data[offset+4:offset+8], data[offset+8:offset+8+n]
                self.assertEqual(struct.unpack(">I", data[offset+8+n:offset+12+n])[0], zlib.crc32(kind+value))
                chunks[kind] = value
                offset += n + 12
            self.assertEqual(zlib.decompress(chunks[b"IDAT"]), b"\0" + bytes(range(6)) + b"\0" + bytes(range(6,12)))

    def test_disabled_no_render_no_directory(self):
        with tempfile.TemporaryDirectory() as root, patch.object(va, "render_rgb") as render:
            audit, summary, _ = self.make(root, False)
            with audit:
                audit.capture("before_action")
            render.assert_not_called()
            self.assertNotIn("visual_audit", summary)
            self.assertFalse(audit.root.exists())

    def test_stride_boundaries_and_frame_budget(self):
        with tempfile.TemporaryDirectory() as root, patch.object(va, "render_rgb", return_value=Pixels()):
            audit, summary, executor = self.make(root, limit=3)
            before = executor.snapshot()
            with audit:
                audit.capture("before_action")
                audit.capture("navigation_post_step", force=False)
                audit.capture("navigation_post_step", force=False)
                audit.capture("after_action")
                audit.capture("round_begin")
            self.assertEqual(executor.snapshot(), before)
            self.assertEqual(len(audit.frames), 3)
            self.assertTrue(summary["visual_audit"]["frame_limit_reached"])
            self.assertEqual(summary["visual_audit"]["status"], "partial")

    def test_execution_failure_preserved(self):
        with tempfile.TemporaryDirectory() as root, patch.object(va, "render_rgb", return_value=Pixels()):
            audit, summary, _ = self.make(root)
            with self.assertRaisesRegex(ValueError, "placement failed"):
                with audit:
                    audit.capture("before_action")
                    raise ValueError("placement failed")
            data = json.loads((audit.root / "audit.json").read_text())
            self.assertEqual(data["execution_status"], "failed")
            self.assertEqual(data["execution_error"], "placement failed")
            self.assertTrue((audit.root / "index.html").is_file())

    def test_render_failure_state_only_no_success_claim(self):
        with tempfile.TemporaryDirectory() as root, patch.object(va, "render_rgb", side_effect=ValueError("GPU unavailable")) as render:
            audit, summary, _ = self.make(root)
            with audit:
                audit.capture("before_action")
                audit.capture("after_action")
            self.assertEqual(render.call_count, 1)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["visual_audit"]["status"], "partial")
            self.assertIsNone(audit.frames[0]["image"])

    def test_state_mutation_stops_execution(self):
        with tempfile.TemporaryDirectory() as root:
            audit, _, executor = self.make(root)
            def mutate(sim, **kwargs):
                executor.state["robot_position"][0] += 1
                return Pixels()
            with patch.object(va, "render_rgb", side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, "changed runtime state"):
                    with audit:
                        audit.capture("before_action")

    def test_historic_boundary_only_and_script_injection_escaped(self):
        state = Executor().snapshot()
        summary = {"status": "failed", "error": "timeout", "rounds": [{"round": 1,
            "instruction": "</script><script>alert(1)</script>", "actions": [{"step_id": "a1", "skill_id": "navigate_to_object",
            "status": "arrived", "before": state, "after": state}]}]}
        report = va.historic_report(summary)
        self.assertEqual(report["evidence_source"], "historic_action_boundary_readback")
        self.assertEqual(len(report["frames"]), 2)
        self.assertIsNone(report["frames"][0]["image"])
        with tempfile.TemporaryDirectory() as root:
            va.write_viewer(Path(root), report)
            html = (Path(root) / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("</script><script>alert", html)
            self.assertIn("\\u003c/script", html)

    def test_config_only_adds_camera(self):
        hs = SimpleNamespace(CameraSensorSpec=lambda: SimpleNamespace(), SensorType=SimpleNamespace(COLOR="rgb"))
        cfg = SimpleNamespace(action_space={"forward": 1})
        va.configure_camera(hs, cfg)
        self.assertEqual(cfg.action_space, {"forward": 1})
        self.assertEqual(cfg.sensor_specifications[0].resolution, [360, 640])

    def test_invalid_budget_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                self.make(root, limit=0)

    def test_nested_serial_executor_state(self):
        state = Executor().snapshot()
        nested = {"held_object_id": 67, "lifted": True,
                  "backend": {k: v for k, v in state.items() if k not in ("held_object_id", "lifted")}}
        self.assertEqual(va.compact_state(nested), va.compact_state(state))

    def test_camera_blocked_frame_can_recover_without_fake_rgb(self):
        with tempfile.TemporaryDirectory() as root, patch.object(va, 'render_rgb',
                side_effect=[va.AuditCameraRejected('AUDIT_CAMERA_MAST_BLOCKED'),Pixels()]):
            audit,summary,_=self.make(root)
            with audit:
                audit.capture('navigation_post_step')
                audit.capture('navigation_post_step')
            self.assertIsNone(audit.frames[0]['image'])
            self.assertTrue(audit.frames[1]['image'].endswith('.png'))
            self.assertEqual(summary['visual_audit']['status'],'partial')


if __name__ == "__main__":
    unittest.main()
