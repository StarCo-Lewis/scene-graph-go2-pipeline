"""No real API, WSL or robot calls: all process launches below are injected."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from embodied_pipeline.ui import api_planner as a, field_backend as b, field_worker as w


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "outputs/real_scene_runs/field_ui_offline"
        self.folder.mkdir(parents=True)
        b.write(self.folder / "task.json", {"instruction": "Navigate to the table."})
        self.encoder = self.root / "fake_encoder"
        self.encoder.write_text("not executed")
        self.model_dir = self.root / "fake_model"
        self.model_dir.mkdir()
        self.config = {"schema_version": "field_api_planning_v1", "api_key": "offline-test-secret",
                       "model": "offline-model", "base_url": "https://example.invalid/v1",
                       "embedding_python": str(self.encoder), "embedding_model": str(self.model_dir),
                       "job": {"schema_version": "field_planning_job_v1", "folder": self.folder.relative_to(self.root).as_posix(),
                               "task_sha256": b.sha(self.folder / "task.json")}}

    def test_api_validation(self):
        a.validate_api("model", "https://example.invalid/v1", "secret")
        for url in ["http://example.invalid", "https://secret@example.invalid", "https://example.invalid?key=secret", "https://example.invalid/#key"]:
            with self.assertRaises(ValueError):
                a.validate_api("model", url, "secret")
        for model, key in [("", "secret"), ("model\n--execute", "secret"), ("model", "")]:
            with self.assertRaises(ValueError):
                a.validate_api(model, "https://example.invalid", key)

    def test_argv_no_shell_secret_or_robot(self):
        with patch.object(a, "ROOT", Path("C:/My Project")):
            cmd = a.launch_command("/home/lewis/a b/python", "Ubuntu-24.04")
        self.assertEqual(cmd[:3], ["wsl.exe", "--distribution", "Ubuntu-24.04"])
        self.assertIn("/home/lewis/a b/python", cmd)
        self.assertNotIn(self.config["api_key"], str(cmd))
        self.assertNotIn("--execute", cmd)
        self.assertEqual(cmd[-1], "/mnt/c/My Project")

    def test_public_config_omits_secret(self):
        data = a.public_config(self.config)
        self.assertNotIn("api_key", data)
        self.assertNotIn(self.config["api_key"], json.dumps(data))

    def test_desktop_launch_sends_key_only_via_stdin(self):
        child = Mock(returncode=2)
        child.communicate.return_value = (b"error offline-test-secret", None)
        with patch.object(a.subprocess, "Popen", return_value=child) as spawn:
            result = a.launch_wsl(["wsl.exe", "fake-python"], self.config)
        self.assertNotIn(self.config["api_key"], str(spawn.call_args))
        self.assertEqual(json.loads(child.communicate.call_args.args[0]), self.config)
        self.assertNotIn(self.config["api_key"], result["message"])
        self.assertNotIn("shell", spawn.call_args.kwargs)
        spawn.assert_called_once()

    def test_existing_pipeline_invoked_and_logs_redacted(self):
        child = Mock()
        child.stdout = io.StringIO("START decomposition\nsecret offline-test-secret\nEND\n")
        child.wait.return_value = 0
        with patch.object(a, "ROOT", self.root), patch.object(w, "ROOT", self.root), patch.object(a.subprocess, "Popen", return_value=child) as spawn:
            self.assertEqual(a.run_config(self.config), 0)
        args, kw = spawn.call_args
        self.assertIn("real-scene-plan", args[0])
        self.assertIn("--model", args[0])
        self.assertNotIn(self.config["api_key"], str(args))
        self.assertEqual(kw["env"]["DASHSCOPE_API_KEY"], self.config["api_key"])
        self.assertNotIn("--execute", args[0])
        for name in ["worker.console.log", "api_result.json", "planning_owner.lock"]:
            self.assertNotIn(self.config["api_key"], (self.folder/name).read_text())
        self.assertFalse(b.read(self.folder / "api_result.json")["robot_executed"])

    def test_duplicate_reserved_run_rejected(self):
        a.claim_folder(self.folder)
        with patch.object(a, "ROOT", self.root), patch.object(w, "ROOT", self.root), patch.object(a.subprocess, "Popen") as spawn:
            with self.assertRaises(FileExistsError):
                a.run_config(self.config)
            spawn.assert_not_called()

    def test_invalid_hash_never_spawn(self):
        for mode in ["hash"]:
            cfg = json.loads(json.dumps(self.config))
            if mode == "hash":
                cfg["job"]["task_sha256"] = "wrong"
            else:
                cfg["embedding_python"] = str(self.root / "missing")
            with patch.object(w, "ROOT", self.root), patch.object(a.subprocess, "Popen") as spawn:
                with self.assertRaises(ValueError):
                    a.run_config(cfg)
                spawn.assert_not_called()

    def test_spawn_failure_report_no_retry_or_secret(self):
        self.config["embedding_python"] = str(self.root / "missing")
        self.config["embedding_model"] = str(self.root / "missing-model")
        with patch.object(a, "ROOT", self.root), patch.object(w, "ROOT", self.root), patch.object(a.subprocess, "Popen", side_effect=OSError("offline-test-secret")) as spawn:
            self.assertEqual(a.run_config(self.config), 2)
            spawn.assert_called_once()
        value = b.read(self.folder / "api_result.json")
        self.assertEqual(value["status"], "planning_process_failed")
        self.assertNotIn(self.config["api_key"], json.dumps(value))
        self.assertTrue((self.folder / "planning_owner.lock").exists())

    def test_api_run_cannot_be_queued_again(self):
        b.write(self.folder / "api_launch.json", a.public_config(self.config))
        with patch.object(b, "worker_status", return_value={"status": "ready"}):
            with self.assertRaises(ValueError):
                b.submit(self.folder)


if __name__ == "__main__":
    unittest.main()
