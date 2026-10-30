"""Continuation integrity and stage selection, without network or WSL access."""

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline.orchestration import real_scene_resume as resume
from embodied_pipeline.orchestration import real_scene
from embodied_pipeline.perception.real_rgbd import write, sha


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        prepared = root / "prepared"
        map_file = prepared / "dataset/pcd_saves/full_pcd_test_overlap_post.pkl.gz"
        map_file.parent.mkdir(parents=True)
        with gzip.open(map_file, "wb") as stream:
            stream.write(b"offline checkpoint fixture")
        audit = root / "input_audit.json"
        write(audit, {"fixture": True})
        adapter = root / "adapters/run_qwen_conceptgraph_captions.py"
        adapter.parent.mkdir()
        adapter.write_text("# offline fixture\n")
        old_root = root / "old"
        stages = []
        for name in ("detections", "mapping"):
            record = dict(stage=name, exit_code=0)
            write(old_root / "logs" / (name + ".json"), record)
            stages.append(record)
        report = dict(schema_version="recorded_real_scene_run_v1", status="failed_or_interrupted",
            current_stage="captions", conceptgraphs_mapping_completed=True, stages=stages,
            prepared_input=str(prepared), prepared_input_audit=dict(path=str(audit), sha256=sha(audit)),
            map_file=str(map_file), detection_variant="test", source_code_hashes={str(adapter): sha(adapter)},
            source=str(root / "source"), semantic_model="offline-fixture")
        summary = old_root / "run_summary.json"
        write(summary, report)
        return summary, report, adapter

    def test_checkpoint_rejects_failed_mapping_changed_code_and_corrupt_map(self):
        for mutation in ("none", "mapping", "code", "map"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                summary, report, adapter = self.fixture(Path(tmp))
                if mutation == "mapping":
                    report["stages"][1]["exit_code"] = 1
                    write(summary, report)
                elif mutation == "code":
                    adapter.write_text("changed")
                elif mutation == "map":
                    Path(report["map_file"]).write_bytes(b"broken gzip")
                with patch.object(resume, "validate_prepared") as validate:
                    if mutation == "none":
                        old, evidence = resume.inspect_checkpoint(summary)
                        self.assertEqual(old["semantic_model"], "offline-fixture")
                        self.assertTrue(evidence["gzip_integrity_verified"])
                        validate.assert_called_once()
                    else:
                        with self.assertRaises((ValueError, OSError)):
                            resume.inspect_checkpoint(summary)

    def test_only_semantic_stages_run_old_files_preserved_and_output_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, old, adapter = self.fixture(root)
            original = summary.read_bytes()
            args = SimpleNamespace(from_summary=summary, output_dir=root / "new",
                                   base_url="https://invalid.test")
            with patch.object(resume.sys, "platform", "linux"), \
                    patch.dict(resume.os.environ, {"DASHSCOPE_API_KEY": "offline-fixture"}), \
                    patch.object(resume, "validate_prepared"), \
                    patch.object(real_scene, "run_stage", return_value={"stage": "captions", "exit_code": 0}) as stage, \
                    patch.object(resume, "export_scene", return_value={"fixture": True}) as export, \
                    patch.object(resume, "import_scene", return_value={"execution_ready": False}), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Live API forbidden")):
                self.assertEqual(resume.resume(args), 0)
                self.assertEqual(stage.call_count, 1)
                self.assertEqual(stage.call_args.args[0], "captions")
                self.assertIn("offline-fixture", stage.call_args.args[1])
                export.assert_called_once()
                new = json.loads((args.output_dir / "run_summary.json").read_text())
                self.assertEqual(new["status"], "graph_generated_and_imported_not_executed")
                self.assertTrue(new["map_unchanged"])
                self.assertEqual(summary.read_bytes(), original)
                with self.assertRaises(FileExistsError):
                    resume.resume(args)
                self.assertEqual(stage.call_count, 1)

    def test_caption_failure_recorded_without_export_or_old_run_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, old, adapter = self.fixture(root)
            original = summary.read_bytes()
            args = SimpleNamespace(from_summary=summary, output_dir=root / "failed",
                                   base_url="https://invalid.test")
            with patch.object(resume.sys, "platform", "linux"), \
                    patch.dict(resume.os.environ, {"DASHSCOPE_API_KEY": "offline-fixture"}), \
                    patch.object(resume, "validate_prepared"), \
                    patch.object(real_scene, "run_stage", side_effect=RuntimeError("offline failure")), \
                    patch.object(resume, "export_scene") as export:
                with self.assertRaisesRegex(RuntimeError, "offline failure"):
                    resume.resume(args)
                export.assert_not_called()
                new = json.loads((args.output_dir / "run_summary.json").read_text())
                self.assertEqual(new["status"], "failed_or_interrupted")
                self.assertEqual(new["current_stage"], "captions")
                self.assertEqual(summary.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
