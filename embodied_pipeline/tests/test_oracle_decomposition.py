"""Offline Oracle-runner checks; fixture replay is never live model evidence."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT
from embodied_pipeline import memory, planning
from embodied_pipeline.orchestration import oracle_decomposition as oracle
from embodied_pipeline.orchestration.proposal import propose_once


def options(**extra):
    return SimpleNamespace(**dict(
        bundle=oracle.DEFAULT_BUNDLE, approval=oracle.DEFAULT_APPROVAL,
        case_ids=None, model="offline-fixture", base_url="https://invalid.test",
        matrix_contract="typed5_schema", alignment_wire_format="matrix",
        top_k=2, cosine_threshold=.35, cosine_margin=.05, **extra))


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c["case_id"]: c for c in oracle.selected_cases(options())}

    def setUp(self):
        self.http = patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden"))
        self.process = patch("subprocess.Popen", side_effect=AssertionError("Live subprocess forbidden"))
        self.http.start()
        self.process.start()
        self.addCleanup(self.http.stop)
        self.addCleanup(self.process.stop)

    def test_approval_scope_and_hash(self):
        self.assertEqual(len(self.cases), 11)
        args = options()
        args.case_ids = ["A03"]
        with self.assertRaisesRegex(ValueError, "injectable"):
            oracle.selected_cases(args)
        with tempfile.TemporaryDirectory() as tmp:
            approval = oracle.read(args.approval)
            approval["payload_sha256_by_case"]["C01"] = "changed"
            args.approval = Path(tmp) / "approval.json"
            args.case_ids = ["C01"]
            memory.write_json(args.approval, approval)
            with self.assertRaisesRegex(ValueError, "Payload changed"):
                oracle.selected_cases(args)

    def test_english_decoded_values_not_just_ascii_serialization(self):
        oracle.check_english_input({"instruction": "Bring the book here."})
        with self.assertRaisesRegex(ValueError, "Chinese text"):
            oracle.check_english_input(json.loads('{"instruction":"\\u4e66"}'))

    def test_all_inputs_preserved_and_only_oracle_prechecks_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in self.cases.values():
                for arm in ("original_replay", "oracle_decomposition"):
                    folder, record = oracle.prepare_branch(case, arm, Path(tmp) / case["case_id"] / arm)
                    for name in oracle.INPUT_NAMES:
                        self.assertEqual(oracle.sha(folder / name), case["source_files"][name]["sha256"])
                    self.assertEqual(record["english_input_check"], "passed_no_chinese_text")
                    self.assertFalse(record["private_evaluation_forwarded"])
                    if arm == "oracle_decomposition":
                        self.assertEqual(record["decomposition_precheck"], "accepted")
                    if case["case_id"] == "C01" and arm == "original_replay":
                        self.assertEqual(record["decomposition_precheck"], "rejected")
                    self.assertFalse((folder / "llm_binding_proposal.json").exists())

    def test_saved_decomposition_cli_never_injects_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = oracle.prepare_branch(self.cases["C01"], "original_replay", Path(tmp))
            def fake_process(command, **kwargs):
                self.assertIn("--decomposition-json", command)
                self.assertIn("--identifier-allocation-json", command)
                self.assertNotIn("--instruction", command)
                self.assertNotIn("--binding-proposal-json", command)
                self.assertEqual(command[command.index("--matrix-contract")+1], "typed5_schema")
                memory.write_json(folder / "object_alignment.json",
                    {"status": "error", "error": {"message": "offline invalid original",
                     "category": "model_or_input_validation_error"}})
                return SimpleNamespace(returncode=1)
            summary = {"stages": []}
            with patch.object(oracle.subprocess, "run", side_effect=fake_process):
                with self.assertRaisesRegex(ValueError, "offline invalid original"):
                    oracle.alignment_from_saved(options(output_dir=folder),
                        oracle.read(folder / "round_input.json"), summary)
            self.assertEqual(summary["upstream_error"]["category"], "model_or_input_validation_error")

    def test_custom_provider_does_not_call_decomposition_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = oracle.prepare_branch(self.cases["C01"], "original_replay", Path(tmp))
            def rejected(*args):
                raise ValueError("offline saved decomposition rejected")
            with patch.object(planning, "decompose_and_bind") as live:
                result, plan, alignment = propose_once(options(), folder, None, alignment_provider=rejected)
            live.assert_not_called()
            self.assertIsNone(plan)
            self.assertEqual(result["status"], "proposal_unavailable")

    def test_no_plan_commits_failed_turn_without_changing_physical_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = self.cases["C01"]
            folder, _ = oracle.prepare_branch(case, "original_replay", root)
            initial = oracle.read(folder / "runtime_state.json")
            proposal = {"status": "proposal_unavailable", "error": "offline rejection", "token_usage": {}}
            with patch.object(oracle, "propose_once", return_value=(proposal, None, None)):
                result = oracle.execute_branch(options(), case, "original_replay", root, folder, None)
            self.assertFalse(result["private_reference_passed"])
            self.assertEqual(result["status"], "proposal_unavailable")
            candidate = oracle.read(root / "memory_after_feedback.json")
            expected = copy.deepcopy(initial)
            expected["backend"]["memory_version"] = case["round"]
            self.assertEqual(candidate, expected)
            checkpoint = oracle.read(root / "memory_checkpoint.json")
            self.assertEqual(len(checkpoint["history"]), case["round"])
            self.assertEqual(checkpoint["history"][-1]["actions"], [])
            self.assertEqual(oracle.read(folder / "runtime_state.json"), initial)

    def test_successful_archived_control_fixture_and_private_negative_checks(self):
        case = self.cases["C00"]
        source = oracle.source_path(case["source_files"]["runtime_state.json"]).parent
        plan = oracle.read(source / "skill_plan_proposed.json")
        alignment = oracle.read(source / "object_alignment.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, _ = oracle.prepare_branch(case, "oracle_decomposition", root)
            initial = oracle.read(folder / "runtime_state.json")
            proposal = {"status": "proposal_ready_not_semantically_validated", "token_usage": {}}
            with patch.object(oracle, "propose_once", return_value=(proposal, plan, alignment)):
                result = oracle.execute_branch(options(), case, "oracle_decomposition", root, folder, None)
            self.assertTrue(result["private_reference_passed"])
            feedback = oracle.read(root / "execution_feedback.json")
            altered = copy.deepcopy(case)
            altered["private_evaluation"]["placement_events_in_order"].reverse()
            self.assertFalse(oracle.evaluate_private(altered, initial, feedback)["private_reference_passed"])
            altered = copy.deepcopy(case)
            altered["private_evaluation"]["protected_object_ids"] = [86]
            self.assertFalse(oracle.evaluate_private(altered, initial, feedback)["checks"]["protected_objects_untouched"])


if __name__ == "__main__":
    unittest.main()
