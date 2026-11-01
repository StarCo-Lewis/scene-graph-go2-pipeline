"""Every claimed collection task delivers engineering_plan.json; offline mocks only.

No model/provider/ROS/WSL calls: network is blocked, the alignment subprocess and the
skill request are replaced by fixtures.
"""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline.data_collection import plan_artifact
from embodied_pipeline.orchestration import task_collection_worker as worker
# Module import keeps unittest discovery from collecting CollectionWorkerTests twice.
from embodied_pipeline.tests import test_task_collection_worker as base


def no_subprocess(*args, **kwargs):
    raise AssertionError("No alignment subprocess may start in this fixture")


class DeliveryTests(unittest.TestCase):
    # Reuse the existing offline scene, alignment and matrix fixtures without rerunning their tests.
    source = base.CollectionWorkerTests.source
    alignment = base.CollectionWorkerTests.alignment
    matrix = base.CollectionWorkerTests.matrix

    def setUp(self):
        base.CollectionWorkerTests.setUp(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.scene = self.source(self.temp)
        self.root = self.temp / "task"

    def run_task(self, args=None):
        return worker.run_task(args or self.args, self.task, self.scene, self.root, None)

    def envelope(self):
        return worker.read(self.root / "engineering_plan.json")

    def fake_alignment_process(self, text, decomposition=None, returncode=1):
        def run(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text(text, encoding="utf-8")
            if decomposition is not None:
                worker.write(Path(command[command.index("--decomposition-output") + 1]), decomposition)
            return subprocess.CompletedProcess(command, returncode)
        return run

    def assert_incomplete(self, result, stage):
        self.assertFalse(result["planning_completed"])
        self.assertFalse(result["engineering_json_generated"])
        self.assertTrue(result["engineering_plan_file_written"])
        self.assertEqual(result["artifact_completeness"], "incomplete")
        envelope = self.envelope()
        self.assertEqual(envelope["artifact_completeness"], "incomplete")
        self.assertFalse(envelope["planning_completed"])
        self.assertFalse(envelope["usable_as_plan"])
        self.assertFalse(envelope["execution_ready"])
        self.assertEqual(envelope["status"], "incomplete_planning_artifact_not_executable")
        self.assertEqual(envelope["failed_stage"], stage)
        self.assertIn("planning_incomplete", envelope["quality_flags"])
        self.assertFalse(plan_artifact.has_complete(self.root / "engineering_plan.json"))
        return envelope

    def test_malformed_alignment_json_delivers_incomplete_envelope(self):
        with patch.object(worker.subprocess, "run", side_effect=self.fake_alignment_process("{not json")), \
                patch.object(worker.skills, "request_matrix") as request:
            result = self.run_task()
        request.assert_not_called()
        envelope = self.assert_incomplete(result, "decomposition_and_object_binding")
        self.assertEqual(envelope["error_type"], "JSONDecodeError")
        self.assertIsNone(envelope["plan"])
        self.assertEqual(envelope["evidence_read_errors"], {"object_alignment.json": "JSONDecodeError"})
        self.assertIn("object_alignment.json", [row["path"] for row in envelope["evidence_files"]])

    def test_binding_clarification_keeps_decomposition_evidence(self):
        decomposition = self.alignment()["task_decomposition"]
        stopped = json.dumps({"status": "clarification_required", "task_decomposition": decomposition,
                              "error": {"code": "ambiguous_target"}})
        with patch.object(worker.subprocess, "run",
                          side_effect=self.fake_alignment_process(stopped, decomposition, returncode=0)), \
                patch.object(worker.skills, "request_matrix") as request:
            result = self.run_task()
        request.assert_not_called()
        envelope = self.assert_incomplete(result, "decomposition_and_object_binding")
        evidence = envelope["partial_evidence"]
        self.assertEqual(evidence["object_alignment_status"], "clarification_required")
        self.assertEqual(evidence["object_alignment_error"], {"code": "ambiguous_target"})
        self.assertEqual(evidence["task_decomposition"]["instruction"], self.task["instruction"])
        self.assertIsNone(evidence["object_bindings"])
        self.assertIn("object_binding_incomplete", envelope["quality_flags"])
        self.assertIn("decomposition_audit", result)
        # A reasonable clarification is kept, but never usable as a plan.
        self.assertFalse(result["usable_as_plan"])

    def test_missing_key_delivers_without_any_request(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}), \
                patch.object(worker.subprocess, "run", side_effect=no_subprocess), \
                patch.object(worker.skills, "request_matrix") as request:
            result = self.run_task()
        request.assert_not_called()
        envelope = self.assert_incomplete(result, "credential_check")
        self.assertEqual(envelope["error_type"], "RuntimeError")
        self.assertIn("decomposition_missing", envelope["quality_flags"])
        self.assertEqual(envelope["response_evidence"]["local_request_records"], 0)
        self.assertIsNotNone(envelope["source_graph_sha256"])

    def test_skill_request_exception_keeps_decomposition_and_bindings(self):
        alignment = self.alignment()
        with patch.object(worker, "collect_alignment", return_value=alignment), \
                patch.object(worker.skills, "request_matrix",
                             side_effect=ConnectionError("offline reset")) as request:
            result = self.run_task()
        self.assertEqual(request.call_count, 1)  # no retry
        envelope = self.assert_incomplete(result, "skill_binding")
        evidence = envelope["partial_evidence"]
        self.assertEqual(evidence["object_bindings"], alignment["context_bindings"])
        self.assertEqual(evidence["task_decomposition"], alignment["task_decomposition"])
        self.assertIsNone(evidence["raw_skill_matrix_proposal"])
        self.assertIn("skill_plan_missing", envelope["quality_flags"])
        self.assertNotIn("object_binding_incomplete", envelope["quality_flags"])

    def test_audit_exception_never_blocks_delivery(self):
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix", side_effect=self.matrix) as request, \
                patch.object(worker.plan_semantics, "audit", side_effect=RuntimeError("audit bug")):
            result = self.run_task()
        self.assertEqual(request.call_count, 1)
        self.assertTrue(result["planning_completed"])
        self.assertTrue(result["engineering_json_generated"])
        self.assertEqual(result["status"], "engineering_json_generated_requires_semantic_review_ungraded")
        self.assertEqual(result["decomposition_audit"]["audit_error_type"], "RuntimeError")
        envelope = self.envelope()
        self.assertEqual(envelope["artifact_completeness"], "complete")
        self.assertTrue(envelope["semantic_review_required"])
        self.assertIn("semantic_audit_error", envelope["quality_flags"])
        self.assertIn("plan_semantics_audit_error", envelope["quality_flags"])
        self.assertFalse(envelope["usable_as_plan"])
        self.assertFalse(result["usable_as_plan"])
        self.assertFalse(envelope["execution_ready"])
        self.assertTrue(plan_artifact.has_complete(self.root / "engineering_plan.json"))

    def test_audit_exception_in_block_mode_fails_closed_but_delivers(self):
        args = SimpleNamespace(**dict(vars(self.args), semantic_audit_mode="block"))
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix") as request, \
                patch.object(worker.plan_semantics, "audit", side_effect=RuntimeError("audit bug")):
            result = self.run_task(args)
        request.assert_not_called()
        self.assert_incomplete(result, "decomposition_semantics_audit")

    def test_late_audit_exception_after_stopped_binding_still_delivers(self):
        def stopped(args, task, destination):
            worker.write(Path(destination) / "task_decomposition.json", self.alignment()["task_decomposition"])
            raise ValueError("Existing alignment contract blocked continuation")
        with patch.object(worker, "collect_alignment", side_effect=stopped), \
                patch.object(worker.plan_semantics, "audit", side_effect=RuntimeError("audit bug")):
            result = self.run_task()
        envelope = self.assert_incomplete(result, "decomposition_and_object_binding")
        self.assertEqual(result["decomposition_audit"]["audit_error_type"], "RuntimeError")
        self.assertIn("semantic_audit_error", envelope["quality_flags"])

    def test_unplanned_subtasks_in_built_plan_are_flagged_in_final_envelope(self):
        plan = {"subtask_plans": [{"subtask_id": "s1", "steps": [
                    {"step_id": "s1_01", "skill_id": "grasp_object",
                     "arguments": {"object": {"role_id": "table", "element_id": 179}}}]}],
                "unplanned_subtasks": ["s2"], "action_count": 1}
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix", side_effect=self.matrix), \
                patch.object(worker.skills, "build_result", return_value=plan):
            result = self.run_task()
        self.assertTrue(result["planning_completed"])
        self.assertEqual(result["status"], "engineering_json_generated_requires_semantic_review_ungraded")
        self.assertIn("unplanned_subtasks_remain", result["quality_flags"])
        envelope = self.envelope()
        self.assertEqual(envelope["artifact_completeness"], "complete")
        self.assertEqual(envelope["status"], "ungraded_proposal_requires_semantic_review")
        self.assertTrue(envelope["semantic_review_required"])
        self.assertIn("unplanned_subtasks_remain", envelope["quality_flags"])
        self.assertTrue(envelope["decomposition_audit"]["passed"])
        self.assertEqual(envelope["plan_semantics_audit"]["defect_codes"], ["unplanned_subtasks_remain"])
        self.assertFalse(envelope["plan_semantics_audit"]["blocks_delivery"])
        saved = worker.read(self.root / "plan_semantics_audit.json")
        self.assertEqual(saved["audit_stage"], "skill_plan")
        self.assertEqual(saved["defect_codes"], ["unplanned_subtasks_remain"])
        # Defect found only on the final plan: still delivered and counted complete, but not usable.
        self.assertFalse(envelope["usable_as_plan"])
        self.assertFalse(result["usable_as_plan"])
        self.assertTrue(plan_artifact.has_complete(self.root / "engineering_plan.json"))

    def test_success_is_complete_and_audited_on_the_plan(self):
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix", side_effect=self.matrix):
            result = self.run_task()
        self.assertTrue(result["planning_completed"])
        self.assertEqual(result["artifact_completeness"], "complete")
        self.assertTrue(plan_artifact.has_complete(self.root / "engineering_plan.json"))
        self.assertTrue(plan_artifact.summary_planning_completed(result))
        self.assertTrue((self.root / "plan_semantics_audit.json").is_file())
        self.assertIn("plan_semantics_audit", self.envelope())
        envelope = self.envelope()
        self.assertTrue(envelope["usable_as_plan"])
        self.assertTrue(result["usable_as_plan"])
        self.assertFalse(envelope["human_semantic_review_performed"])
        self.assertFalse(envelope["execution_ready"])
        self.assertEqual(envelope["status"], "ungraded_proposal_not_executable")

    def test_existing_directory_is_rejected_and_left_untouched(self):
        self.root.mkdir()
        old = self.root / "engineering_plan.json"
        old.write_text('{"old": true}\n', encoding="utf-8")
        before = sorted(p.name for p in self.root.iterdir())
        with patch.object(worker, "collect_alignment") as alignment, \
                patch.object(worker.skills, "request_matrix") as request:
            with self.assertRaises(FileExistsError):
                self.run_task()
        alignment.assert_not_called()
        request.assert_not_called()
        self.assertEqual(old.read_text(encoding="utf-8"), '{"old": true}\n')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), before)

    def test_interrupt_is_recorded_delivered_and_reraised(self):
        with patch.object(worker, "collect_alignment", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_task()
        summary = worker.read(self.root / "planning_summary.json")
        self.assertEqual(summary["status"], "collection_interrupted_ungraded")
        self.assertTrue(summary["engineering_plan_file_written"])
        envelope = self.envelope()
        self.assertTrue(envelope["interrupted"])
        self.assertIn("collection_interrupted", envelope["quality_flags"])

    def test_envelope_write_failure_is_reported_not_hidden(self):
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix", side_effect=self.matrix), \
                patch.object(worker, "write_new", side_effect=OSError("disk full")):
            result = self.run_task()
        self.assertTrue(result["planning_completed"])
        self.assertFalse(result["engineering_plan_file_written"])
        self.assertFalse(result["engineering_json_generated"])
        self.assertIsNone(result["artifact_completeness"])
        self.assertEqual(result["status"], "engineering_plan_delivery_failed_ungraded")
        self.assertEqual(result["engineering_plan_delivery_error"]["error_type"], "OSError")
        self.assertFalse((self.root / "engineering_plan.json").exists())
        self.assertTrue((self.root / "skill_plan.json").is_file())

    def test_unserialisable_evidence_falls_back_to_minimal_incomplete_envelope(self):
        with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                patch.object(worker.skills, "request_matrix", side_effect=self.matrix), \
                patch.object(worker, "build_envelope", side_effect=TypeError("not serialisable")):
            result = self.run_task()
        self.assertTrue(result["engineering_plan_minimal_fallback"])
        self.assertEqual(result["artifact_completeness"], "incomplete")
        self.assertFalse(result["engineering_json_generated"])
        self.assertEqual(result["status"], "engineering_plan_delivery_failed_ungraded")
        envelope = self.envelope()
        self.assertIsNone(envelope["plan"])
        self.assertEqual(envelope["evidence_serialization_error_type"], "TypeError")

    def test_decomposition_contract_carries_ordered_visit_terminal_goal_policy(self):
        worker.claim(self.root, self.task)
        library, _, _ = worker.prepare_inputs(self.task, self.scene, self.root)
        contract = worker.read(self.root / "execution_contract.json")
        self.assertIn(worker.ORDERED_VISIT_POLICY, contract["factual_boundary"])
        for phrase in ("only robot near B in goal_state", "A -> B -> A",
                       "require only robot near A at the end", "explicit dependency"):
            self.assertIn(phrase, contract["factual_boundary"])
        self.assertEqual(contract["ordered_navigation_goal_policy_version"], "ordered_visit_terminal_goal_v1")
        self.assertEqual(library["execution_contract"]["ordered_navigation_goal_policy_version"],
                         "ordered_visit_terminal_goal_v1")
        config = worker.read(self.root / "generation_config.json")
        self.assertEqual(config["decomposition_goal_policy"], "ordered_visit_terminal_goal_v1")
        self.assertEqual(config["prompt_version"], "complete_intent_relevant_constraints_v2")
        self.assertEqual(config["semantic_retries"], 0)
        self.assertEqual(config["max_http_attempts_per_stage"], 1)
        # The user instruction itself is never rewritten.
        self.assertEqual(worker.read(self.root / "round_input.json")["instruction"], self.task["instruction"])
        command = worker.alignment_command(self.args, self.task, self.root)
        self.assertEqual(command[command.index("--execution-contract") + 1],
                         str(self.root / "execution_contract.json"))

    def test_decomposition_prompt_carries_ordered_visit_rule(self):
        from embodied_pipeline.planning import intent_prompt
        self.assertEqual(intent_prompt.VERSION, "complete_intent_relevant_constraints_v2")
        composed = intent_prompt.compose("BASE", "intent_constraints_prompt")
        for phrase in ("put only the robot\nnear the last destination",
                       "do not\nalso keep them as final robot-near conditions",
                       "require only robot near A at the\nend",
                       "dependency on the preceding visit",
                       "ask for\nclarification"):
            self.assertIn(phrase, composed)
        # The baseline arm stays unchanged for comparisons; the worker uses the intent arm.
        self.assertEqual(intent_prompt.compose("BASE", "baseline_prompt"), "BASE")
        command = worker.alignment_command(self.args, self.task, self.root)
        self.assertEqual(command[command.index("--prompt-arm") + 1], "intent_constraints_prompt")

    def test_unknown_audit_mode_rejected_before_claim(self):
        args = SimpleNamespace(**dict(vars(self.args), semantic_audit_mode="strict"))
        with self.assertRaisesRegex(ValueError, "Unknown semantic audit mode"):
            self.run_task(args)
        self.assertFalse(self.root.exists())


CLEAN = {"passed": True, "has_defect": False, "defect_codes": [], "issue_codes": [], "mode": "annotate"}


class EnvelopeUsabilityTests(unittest.TestCase):
    """Direct build_envelope boundaries; pure data, no worker run and no files read."""
    task = {"record_id": "fixture_usable", "scene_id": "fixture_scene", "instruction": "Go to table ID 1."}

    def build(self, audit=CLEAN, plan_audit=CLEAN, unchanged=True):
        state = {"plan": {"subtask_plans": [], "action_count": 0}, "metadata": {}, "hashes": {},
                 "plan_audit": plan_audit}
        with tempfile.TemporaryDirectory() as temp:
            return worker.build_envelope(self.task, temp, state, audit, None, unchanged, None)

    def assert_kept_but_not_usable(self, envelope, review=True):
        self.assertEqual(envelope["artifact_completeness"], "complete")  # completeness counting unchanged
        self.assertTrue(plan_artifact.is_complete(envelope))
        self.assertIsNotNone(envelope["plan"])
        self.assertFalse(envelope["usable_as_plan"])
        self.assertFalse(envelope["execution_ready"])
        self.assertEqual(envelope["semantic_review_required"], review)
        if review:
            self.assertEqual(envelope["status"], "ungraded_proposal_requires_semantic_review")

    def test_both_clean_audits_and_confirmed_inputs_are_usable(self):
        envelope = self.build()
        self.assertTrue(envelope["usable_as_plan"])
        self.assertFalse(envelope["semantic_review_required"])
        self.assertEqual(envelope["usability_blockers"], [])
        self.assertFalse(envelope["human_semantic_review_performed"])
        self.assertFalse(envelope["execution_ready"])
        self.assertEqual(envelope["status"], "ungraded_proposal_not_executable")

    def test_missing_unknown_or_unclean_audits_are_not_usable(self):
        cases = {
            "none": (None, CLEAN), "empty": ({}, CLEAN), "plan_audit_none": (CLEAN, None),
            "plan_audit_empty": (CLEAN, {}),
            "passed_false_without_defects": (dict(CLEAN, passed=False), CLEAN),
            "passed_null": (dict(CLEAN, passed=None), CLEAN),
            "passed_missing": ({k: v for k, v in CLEAN.items() if k != "passed"}, CLEAN),
            "has_defect_null": (dict(CLEAN, has_defect=None), CLEAN),
            "mode_off": (dict(CLEAN, mode="off"), CLEAN),
            "mode_skipped": (CLEAN, dict(CLEAN, mode="skipped")),
            "defect_codes_present": (CLEAN, dict(CLEAN, defect_codes=["unplanned_subtasks_remain"])),
            "audit_error": (worker.audit_failure(RuntimeError("x")), CLEAN),
        }
        for name, (audit, plan_audit) in cases.items():
            with self.subTest(name):
                envelope = self.build(audit, plan_audit)
                self.assert_kept_but_not_usable(envelope)
                self.assertTrue(envelope["usability_blockers"])

    def test_unconfirmed_inputs_block_usability_but_keep_the_artifact(self):
        for unchanged in (False, None):
            with self.subTest(unchanged=unchanged):
                envelope = self.build(unchanged=unchanged)
                self.assert_kept_but_not_usable(envelope, review=False)
                self.assertEqual(envelope["usability_blockers"], ["input_integrity_not_confirmed"])
        self.assertIn("input_integrity_error", self.build(unchanged=False)["quality_flags"])

    def test_shared_rule_matches_summary_and_envelope(self):
        self.assertEqual(worker.plan_usability(True, CLEAN, CLEAN, True), (True, False, []))
        self.assertEqual(worker.plan_usability(False, CLEAN, CLEAN, True),
                         (False, False, ["planning_incomplete"]))
        usable, review, blockers = worker.plan_usability(True, {}, None, True)
        self.assertFalse(usable)
        self.assertTrue(review)
        self.assertEqual(blockers, ["decomposition_audit_missing", "plan_semantics_audit_missing"])
        self.assertFalse(worker.audit_clean({"passed": True, "has_defect": False, "mode": "off"}))


class PlanArtifactTests(unittest.TestCase):
    def test_completeness_rules(self):
        self.assertTrue(plan_artifact.is_complete({"plan": {"subtask_plans": []}}))  # legacy success
        self.assertFalse(plan_artifact.is_complete({"plan": None, "artifact_completeness": "incomplete"}))
        self.assertFalse(plan_artifact.is_complete({"plan": {}, "artifact_completeness": "incomplete",
                                                    "planning_completed": False}))
        self.assertTrue(plan_artifact.is_complete({"plan": {}, "artifact_completeness": "complete",
                                                   "planning_completed": True}))
        self.assertFalse(plan_artifact.is_complete([]))

    def test_files_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "engineering_plan.json"
            self.assertFalse(plan_artifact.has_complete(path))
            path.write_text("{broken", encoding="utf-8")
            self.assertFalse(plan_artifact.has_complete(path))
            path.write_text(json.dumps({"plan": None, "artifact_completeness": "incomplete"}), encoding="utf-8")
            self.assertFalse(plan_artifact.has_complete(path))
        self.assertTrue(plan_artifact.summary_planning_completed({"engineering_json_generated": True}))
        self.assertFalse(plan_artifact.summary_planning_completed(
            {"engineering_json_generated": True, "planning_completed": False}))
        self.assertFalse(plan_artifact.summary_planning_completed(None))


if __name__ == "__main__":
    unittest.main()
