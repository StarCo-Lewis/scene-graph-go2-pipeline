"""Offline contract regressions; authored migrations are not new LLM results."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT, load
from embodied_pipeline import execution, instruction_skills as skills, memory
from embodied_pipeline.planning import process_contract as contract
from embodied_pipeline.validation.process_constraints import audit
from embodied_pipeline.orchestration import intent_comparison as experiment
from embodied_pipeline.orchestration.proposal import propose_once
from embodied_pipeline.tests.test_oracle_decomposition import options

RUN = ROOT / "outputs/vlm_full_scene_v1/runs/intent_constraints_20261020_194829_455377/cases"


def read(path):
    path = str(path.resolve())
    if os.name == "nt":
        path = "\\\\?\\" + path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixture(case):
    folder = RUN / case / "repeat_01/intent_constraints_prompt/planning"
    response = read(folder / "llm_decomposition_attempt_1.json")
    # This historical artifact is a raw response envelope, not decoded JSON.
    raw = json.loads(response["response_content"])
    return folder, raw


def migrated(case):
    folder, raw = fixture(case)
    allocation = read(folder / "identifier_allocation.json")
    edited = copy.deepcopy(raw)
    # Explicit test fixture authored from the observed protection task. Never
    # run this transformation on live LLM text; it would infer task semantics.
    edited["tasks"] = edited["tasks"][:1]
    edited["tasks"][0][1] += " Keep the protected object untouched throughout."
    edited["task_roles"] = [r for r in edited["task_roles"] if r[0] == 50001]
    edited["dependencies"] = []
    edited["process_constraint_contract"] = contract.VERSION
    edited["process_constraints"] = [[20003, 1]]
    with contract.installed():
        decomposition = load("typed_matrix_ids").decode_decomposition(edited, allocation)
        decomposition = load("align_objects").validate_decomposition(decomposition)
    alignment = read(folder / "object_alignment.json")
    alignment["task_decomposition"] = decomposition
    library = read(folder / "execution_skill_library.json")
    library["execution_contract"]["process_constraints"] = contract.grounded_restrictions(alignment)
    book = skills.make_codebook(alignment, library)
    matrix = read(folder / "llm_matrix_proposal_semantic_1_attempt_1.json")
    matrix["codebook_sha256"] = book["sha256"]  # New test plan version, same seven action rows.
    decoded = load("action_matrix").decode_matrix(matrix, book, alignment, library)
    plan = skills.build_result(alignment, library, decoded)
    return folder, edited, allocation, alignment, library, book, matrix, plan


class ProcessContractTests(unittest.TestCase):
    def setUp(self):
        self.http = patch("urllib.request.urlopen", side_effect=AssertionError("No live API calls"))
        self.http.start()
        self.addCleanup(self.http.stop)

    def test_c01_a01_keep_seven_actions_without_inventing_protection_actions(self):
        for case in ("C01", "A01"):
            with self.subTest(case=case):
                folder, _, _, alignment, library, _, matrix, plan = migrated(case)
                self.assertEqual(len(matrix["matrix"]), 7)
                self.assertEqual(len(alignment["task_decomposition"]["subtasks"]), 1)
                initial = read(folder / "runtime_state.json")
                before = copy.deepcopy(initial)
                nodes = read(folder / "input_memory/scene/scene_graph.json")
                compact = lambda state: load("dialogue_chain_cases").compact_state(state, (86, 121))
                feedback = execution.execute_proxy(plan, alignment, library, nodes, initial, compact)
                self.assertEqual(feedback["status"], "proxy_actions_completed")
                self.assertTrue(feedback["process_constraint_evaluation"]["passed"],
                                feedback["process_constraint_evaluation"])
                self.assertEqual(initial, before)
                expected = (86, 2) if case == "C01" else (121, 69)
                self.assertEqual(feedback["resulting_state"]["backend"]["objects"][str(expected[0])]["support_id"],
                                 expected[1])
                self.assertIsNone(feedback["resulting_state"]["held_object_id"])

    def test_historical_failures_still_reject_and_action_coverage_is_not_relaxed(self):
        for case in ("C01", "A01"):
            folder, _, _, alignment, library, book, matrix, _ = migrated(case)
            historical = read(folder / "object_alignment.json")
            with self.assertRaisesRegex(ValueError, "cover every subtask"):
                load("action_matrix").decode_matrix(
                    read(folder / "llm_matrix_proposal_semantic_1_attempt_1.json"),
                    read(folder / "action_codebook_semantic_1.json"), historical, library)
            # A genuine second action task must still get its own action rows.
            second = copy.deepcopy(alignment["task_decomposition"]["subtasks"][0])
            second["subtask_id"] = "st_002"
            second["description"] = "Return to the source table after placement."
            alignment["task_decomposition"]["subtasks"].append(second)
            alignment["task_decomposition"]["identifier_context"]["task_codes"]["st_002"] = 50002
            book = skills.make_codebook(alignment, library)
            matrix["codebook_sha256"] = book["sha256"]
            with self.assertRaisesRegex(ValueError, "cover every subtask"):
                load("action_matrix").decode_matrix(matrix, book, alignment, library)

    def test_strict_extension_rejects_wrong_types_undefined_roles_and_rules(self):
        _, raw, allocation, *_ = migrated("C01")
        for rows in ([[20099, 1]], [["20003", 1]], [[20003, 2]],
                     [[20003, 1], [20003, 1]], [[10001, 1]], [[20003, True]]):
            with self.subTest(rows=rows), contract.installed():
                invalid = dict(raw, process_constraints=rows)
                with self.assertRaises(ValueError):
                    load("typed_matrix_ids").decode_decomposition(invalid, allocation)

    def test_process_only_role_is_grounded_without_becoming_an_action_participant(self):
        _, raw, allocation, *_ = migrated("C01")
        raw["task_roles"] = [r for r in raw["task_roles"] if r[1] != 20003]
        raw["constraints"] = [r for r in raw["constraints"] if 20003 not in (r[2], r[4])]
        for i, row in enumerate(raw["constraints"], 60001):
            row[0] = i
        with contract.installed():
            result = load("align_objects").validate_decomposition(
                load("typed_matrix_ids").decode_decomposition(raw, allocation))
        self.assertIn("role_003", {r["role_id"] for r in result["task_roles"]})
        self.assertIn("role_003", {r["role_id"] for r in result["constraint_roles"]})
        self.assertNotIn("role_003", {r["role_id"] for r in result["subtasks"][0]["required_roles"]})

    def test_actual_generation_schema_and_cleanup(self):
        folder, raw, *_ = migrated("C01")
        engine, typed = load("align_objects"), load("typed_matrix_ids")
        original_schema, original_decode, original_validate = typed.schema, typed.decode_decomposition, engine.validate_decomposition
        captured = []
        def respond(request, **kwargs):
            payload = json.loads(request.data)
            captured.append(payload)
            content = copy.deepcopy(raw)
            content["codebook_sha256"] = json.loads(payload["messages"][1]["content"])["identifier_allocation"]["sha256"]
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}]}
        with patch.dict(os.environ, {"PROCESS_TEST_KEY": "offline-placeholder"}), \
             patch.object(engine, "request_json", side_effect=respond), \
             patch.object(engine, "DECOMPOSITION_PROMPT", contract.compose_process_prompt(engine.DECOMPOSITION_PROMPT)), \
             contract.installed():
            result = engine.call_llm(raw["instruction"], "offline", "https://invalid.test", "PROCESS_TEST_KEY",
                max_attempts=1, execution_contract=read(folder / "execution_contract.json"),
                decomposition_format="matrix", matrix_contract="typed5_schema")
        self.assertEqual(len(result["process_constraints"]), 1)
        schema = captured[0]["response_format"]["json_schema"]["schema"]
        self.assertIn("process_constraints", schema["required"])
        self.assertIs(typed.schema, original_schema)
        self.assertIs(typed.decode_decomposition, original_decode)
        self.assertIs(engine.validate_decomposition, original_validate)
        with self.assertRaisesRegex(RuntimeError, "fixture"), contract.installed():
            raise RuntimeError("fixture")
        self.assertIs(typed.schema, original_schema)

    def test_move_then_restore_and_failed_manipulation_are_not_clean_passes(self):
        folder, _, _, alignment, library, _, _, plan = migrated("C01")
        initial = read(folder / "runtime_state.json")
        compact = lambda state: load("dialogue_chain_cases").compact_state(state, (86, 121))
        feedback = execution.execute_proxy(plan, alignment, library,
            read(folder / "input_memory/scene/scene_graph.json"), initial, compact)
        altered = copy.deepcopy(feedback)
        altered["trace"][0]["end_state"]["objects"]["121"]["support_id"] = 2
        self.assertEqual(altered["resulting_state"], feedback["resulting_state"])
        self.assertFalse(audit(alignment, initial, altered)["passed"])
        altered = copy.deepcopy(feedback)
        altered["trace"][0].update(skill_id="grasp_object",
            arguments={"target": {"role_id": "role_003", "element_id": 121}},
            status="proxy_action_failed")
        self.assertFalse(audit(alignment, initial, altered)["passed"])
        # Using an unchanged protected support as a destination does not mean
        # manipulating that support. Payload and surface arguments differ.
        surface_alignment = copy.deepcopy(alignment)
        surface_alignment["task_decomposition"]["process_constraints"] = [
            dict(role_id="role_002", restriction="do_not_manipulate")]
        tracked = tuple(int(k) for k in initial["backend"]["objects"])
        surface_feedback = execution.execute_proxy(plan, surface_alignment, library,
            read(folder / "input_memory/scene/scene_graph.json"), initial,
            lambda state: load("dialogue_chain_cases").compact_state(state, tracked))
        self.assertTrue(surface_feedback["process_constraint_evaluation"]["passed"])
        altered = copy.deepcopy(feedback)
        altered["trace"][0]["skill_id"] = "future_unknown_skill"
        self.assertFalse(audit(alignment, initial, altered)["passed"])
        altered = copy.deepcopy(feedback)
        altered["trace"].pop()
        self.assertFalse(audit(alignment, initial, altered)["passed"])

    def test_proposal_supplies_grounded_restrictions_to_action_request(self):
        folder, _, _, alignment, _, _, _, _ = migrated("C01")
        case = next(c for c in experiment.oracle.selected_cases(options()) if c["case_id"] == "C01")
        with tempfile.TemporaryDirectory() as tmp:
            planning, _ = experiment.prepare_branch(case, "intent_constraints_prompt", Path(tmp),
                                                    process_constraints=True)
            library_before = read(planning / "execution_skill_library.json")
            def candidates(aligned, library, encoder, **kwargs):
                return {"subtasks": [dict(subtask_id=t["subtask_id"],
                    candidate_skill_ids=[s["skill_id"] for s in library["skills"]])
                    for t in aligned["task_decomposition"]["subtasks"]]}
            def request(aligned, library, book, *args, **kwargs):
                payload = skills.matrix_payload(aligned, library, book, "offline",
                    kwargs["retrieval"], False, matrix_contract="typed5_schema")
                value = json.loads(payload["messages"][1]["content"])
                self.assertEqual(value["execution_contract"]["process_constraints"]["restrictions"][0]["element_ids"], [121])
                raise ValueError("offline stop after request inspection")
            with patch.object(skills, "candidates", side_effect=candidates), \
                 patch.object(skills, "request_matrix", side_effect=request):
                summary, plan, _ = propose_once(options(candidate_policy="legacy"), planning, None,
                    alignment_provider=lambda *a: copy.deepcopy(alignment))
            self.assertEqual(summary["error"], "offline stop after request inspection")
            self.assertIsNone(plan)
            self.assertTrue((planning / "process_constraints_grounded.json").is_file())
            self.assertEqual(read(planning / "execution_skill_library.json"), library_before)

    def test_new_cli_defaults_and_legacy_resume_version(self):
        seen = []
        def run(args, **kwargs):
            seen.append((args.process_constraints, kwargs["experiment"]["profile"]))
            return 0
        with patch.object(experiment.oracle, "run", side_effect=run):
            experiment.main(["--preflight-only", "--case-ids", "C01"])
            experiment.main(["--legacy-process-contract", "--preflight-only", "--case-ids", "C01"])
        self.assertEqual(seen, [(True, experiment.PROCESS_PROFILE), (False, experiment.VERSION)])


if __name__ == "__main__":
    unittest.main()
