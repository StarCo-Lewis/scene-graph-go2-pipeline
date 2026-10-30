"""Offline parity with unchanged engines. Fixture actions are NOT model results."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT, bootstrap, load
from embodied_pipeline import perception, planning, instruction_skills as skills
from embodied_pipeline import validation, execution, errors, memory
from embodied_pipeline.contracts import MODULES, EpisodeServices
from embodied_pipeline.introspection import verify_legacy
from embodied_pipeline.orchestration import feedback as modular_feedback
from embodied_pipeline.orchestration import proposal as modular_proposal
from embodied_pipeline.orchestration import batch as modular_batch
from embodied_pipeline.orchestration import matrix_followup as modular_followup

bootstrap()
legacy = load("dialogue_execution_feedback")
pilot = load("run_crossroom_task")
fixtures = load("test_dialogue_state_recovery")
typed_fixtures = load("test_typed9_integration")
wire = load("typed_matrix_ids")


def cleaned(value, root):
    if isinstance(value, dict):
        return {k: cleaned(v, root) for k, v in value.items()
                if k not in ("duration_seconds", "recorded_at")}
    if isinstance(value, (list, tuple)):
        return type(value)(cleaned(v, root) for v in value)
    if isinstance(value, str):
        return value.replace(str(root), "<RUN>").replace(root.as_posix(), "<RUN>")
    return value


def artifacts(root):
    """Compare all JSON content, ignoring only time, run root and audit filename UUIDs."""
    result = {}
    for path in root.rglob("*.json"):
        key = path.relative_to(root).as_posix()
        if key.startswith("llm_response_audit/validation_"):
            key = key.rsplit("_", 1)[0] + "_<UUID>.json"
        value = cleaned(json.loads(path.read_text(encoding="utf-8")), root)
        if key in result:
            raise AssertionError("Unexpected audit-name collision in parity fixture")
        result[key] = value
    return result


class StructuralTests(unittest.TestCase):
    def test_seven_modules_and_no_new_llm_stages(self):
        self.assertEqual(len(MODULES), 7)
        self.assertEqual({s.name for s in MODULES}, {
            "perception", "planning", "instruction_skills", "validation", "execution", "errors", "memory"})

    def test_legacy_sources_unchanged(self):
        result = verify_legacy()
        self.assertEqual(result["status"], "unchanged", result)
        self.assertEqual(result["checked_file_count"], 309)

    def test_adapters_are_original_functions_not_reimplementations(self):
        for module in (perception, planning, skills, validation, execution, errors, memory):
            for name, (source, attr) in module._EXPORTS.items():
                with self.subTest(module=module.__name__, attribute=name):
                    self.assertIs(getattr(module, name), getattr(load(source), attr))

    def test_robot_placeholder_never_claims_execution(self):
        with self.assertRaises(NotImplementedError):
            execution.RobotBackend().execute(None, None, None, None, None, None)

    def test_import_does_not_load_engines_or_start_calls(self):
        code = ("import sys; import embodied_pipeline; import embodied_pipeline.orchestration; "
                "assert 'dialogue_execution_feedback' not in sys.modules; "
                "assert 'align_objects' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_name_collision_is_rejected(self):
        with patch.dict(sys.modules, {"matrix_type_contract": SimpleNamespace(__file__=str(ROOT / "wrong.py"))}):
            with self.assertRaisesRegex(ImportError, "collision"):
                load("matrix_type_contract")


class OfflineParityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RecoveryTests()
        self.cfg, self.nodes, self.edges, self.held, self.instruction, self.args = self.fixture.inputs()
        self.args.preflight_only = False
        self.args.inference_mode = "bounded_inference"
        self.args.memory_mode = "execution_feedback"
        self.args.alignment_wire_format = "matrix"
        self.args.matrix_contract = "typed5_schema"
        # Unexpected transport/subprocess use must fail locally, never reach a provider.
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden in parity tests"))
        self.process = patch("subprocess.Popen", side_effect=AssertionError("Live subprocess forbidden in parity tests"))
        self.network.start()
        self.process.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.process.stop)

    def typed_request(self, recovered, bad_hash, capture):
        def request(alignment, library, book, *args, **kwargs):
            old_book = skills.make_codebook(wire.without_context(alignment), library)
            raw = self.fixture.matrix(old_book, recovered)
            maps = {key: {a["code"]: b["code"] for a, b in zip(old_book[key], book[key])}
                    for key in ("subtasks", "skills", "objects", "values")}
            slots = dict(maps["objects"])
            slots.update(maps["values"])
            slots[0] = 0
            output = dict(schema_version="action_matrix_typed5_v1",
                codebook_sha256="stale-plan" if bad_hash else book["sha256"],
                matrix=[[maps["subtasks"][r[0]], maps["skills"][r[1]], slots[r[2]], slots[r[3]]]
                        for r in raw["matrix"]])
            capture.append(skills.matrix_payload(alignment, library, book, args[0],
                                                kwargs["retrieval"], False, matrix_contract="typed5_schema"))
            return output
        return request

    def proposal_run(self, folder, modular, recovered=False, bad_hash=False, infrastructure=False):
        perception.prepare_feedback(folder, dict(round=1, instruction=self.instruction), self.held,
            [], self.cfg, self.nodes, self.edges, self.args)
        capture = []
        def decompose(args, descriptor, summary):
            if infrastructure:
                summary["upstream_error"] = {"category": "infrastructure_error", "message": "offline timeout"}
                raise ValueError("LLM transport failed: offline timeout")
            return typed_fixtures.alignment(self.instruction, self.nodes, recovered, digits=5)
        target_planning = planning if modular else legacy
        target_skills = skills if modular else legacy
        with patch.object(target_planning, "decompose_and_bind", side_effect=decompose) as d, \
                patch.object(target_skills, "request_matrix",
                    side_effect=self.typed_request(recovered, bad_hash, capture)) as m:
            fn = modular_proposal.propose_once if modular else legacy.propose_once
            result = fn(self.args, folder, fixtures.Encoder())
        return result, capture, (d.call_count, m.call_count)

    def test_proposal_payload_and_all_artifacts_match(self):
        for recovered, bad_hash, infrastructure in ((False, False, False), (True, False, False),
                                                    (True, True, False), (False, False, True)):
            with self.subTest(recovered=recovered, bad_hash=bad_hash, infrastructure=infrastructure), \
                    tempfile.TemporaryDirectory() as tmp, \
                    patch.object(pilot, "sha_identity_root", return_value="offline-equivalence-scope"):
                a, b = Path(tmp) / "old", Path(tmp) / "new"
                old, old_payload, old_calls = self.proposal_run(a, False, recovered, bad_hash, infrastructure)
                new, new_payload, new_calls = self.proposal_run(b, True, recovered, bad_hash, infrastructure)
                self.assertEqual(cleaned(old, a), cleaned(new, b))
                self.assertEqual(old_payload, new_payload)
                self.assertEqual(old_calls, new_calls)
                self.assertEqual(old_calls, (1, 0 if infrastructure else 1))
                self.assertEqual(artifacts(a), artifacts(b))

    def test_preflight_matches_both_scenes_and_makes_no_proposal(self):
        self.args.preflight_only = True
        for scene in ("scene200", "scene202"):
            episode = dict(episode_id="natural_start", scene=scene,
                           rounds=[dict(round=1, instruction="Bring the object here.")])
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(pilot, "sha_identity_root", return_value="offline-equivalence-scope"), \
                    patch.object(legacy, "propose_once") as old_propose, \
                    patch.object(modular_feedback, "propose_once") as new_propose:
                a, b = Path(tmp) / "old", Path(tmp) / "new"
                old = legacy.run_episode_feedback(self.args, episode, a, None)
                new = modular_feedback.run_episode_feedback(self.args, episode, b, None)
                old_propose.assert_not_called()
                new_propose.assert_not_called()
                self.assertEqual(cleaned(old, a), cleaned(new, b))
                self.assertEqual(artifacts(a), artifacts(b))

    def episode_run(self, root, modular, first="partial", audit_error=False, mutation=False):
        episode = dict(episode_id="natural_start", scene="scene202", rounds=[
            dict(round=1, instruction=self.instruction), dict(round=2, instruction="Continue the next request.")])
        seen = []
        def proposer(args, folder, encoder):
            state = memory.read_json(folder / "runtime_state.json")
            seen.append(copy.deepcopy(state))
            if first == "infrastructure":
                return {"status": "infrastructure_error", "error": "offline timeout"}, None, None
            if first == "interrupt":
                raise KeyboardInterrupt
            if first == "unavailable" or len(seen) == 2:
                return {"status": "proposal_unavailable", "error": "offline rejection"}, None, None
            library = memory.read_json(folder / "execution_skill_library.json")
            alignment = self.fixture.alignment(self.instruction, recovered=first == "success")
            alignment, _ = planning.close_runtime_objects(alignment, library, state, self.nodes)
            book = skills.make_codebook(alignment, library)
            decoded = validation.decode_matrix(self.fixture.matrix(book, recovered=first == "success"),
                                               book, alignment, library)
            plan = skills.build_result(alignment, library, decoded)
            memory.write_json(folder / "skill_plan_proposed.json", plan)
            if mutation:
                state["held_object_id"] = 86
                memory.write_json(folder / "runtime_state.json", state)
            return {"status": "proposal_ready_not_semantically_validated"}, plan, alignment
        args = copy.copy(self.args)
        if audit_error:
            args.feedback_evaluator = lambda *a: (_ for _ in ()).throw(ValueError("offline audit error"))
        with patch.object(pilot, "initial_state", return_value=copy.deepcopy(self.held)):
            if modular:
                result = modular_feedback.run_episode_feedback(args, episode, root, None,
                    services=EpisodeServices(proposer=proposer))
            else:
                with patch.object(legacy, "propose_once", side_effect=proposer):
                    result = legacy.run_episode_feedback(args, episode, root, None)
        return result, seen

    def test_rollout_memory_failure_prefix_and_audit_match(self):
        cases = [("partial", False, False), ("success", False, False), ("unavailable", False, False),
                 ("unavailable", True, False), ("infrastructure", False, False),
                 ("interrupt", False, False), ("partial", False, True)]
        for first, audit_error, mutation in cases:
            with self.subTest(first=first, audit_error=audit_error, mutation=mutation), \
                    tempfile.TemporaryDirectory() as tmp, \
                    patch.object(pilot, "sha_identity_root", return_value="offline-equivalence-scope"):
                a, b = Path(tmp) / "old", Path(tmp) / "new"
                old, old_seen = self.episode_run(a, False, first, audit_error, mutation)
                new, new_seen = self.episode_run(b, True, first, audit_error, mutation)
                self.assertEqual(cleaned(old, a), cleaned(new, b))
                self.assertEqual(old_seen, new_seen)
                self.assertEqual(artifacts(a), artifacts(b))
                if first == "partial" and not mutation:
                    checkpoint = memory.read_json(b / "memory_checkpoint.json")
                    self.assertEqual(checkpoint["state"]["held_object_id"], 121)
                    self.assertEqual(checkpoint["state"]["backend"]["memory_version"], 2)
                    self.assertEqual(len(checkpoint["history"][0]["actions"]), 2)
                    self.assertEqual(checkpoint["history"][1]["actions"], [])

    def test_memory_does_not_commit_when_history_validation_fails(self):
        for modular in (False, True):
            with self.subTest(modular=modular), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                episode = dict(episode_id="natural_start", scene="scene202",
                               rounds=[dict(round=1, instruction="No plan available")])
                proposer = lambda *a: ({"status": "proposal_unavailable", "error": "offline"}, None, None)
                # The commit-time context is the only call here with a nonempty history.
                original = load("planning_start_contract").conversation_context
                def check(cfg, nodes):
                    if cfg["conversation_context"].get("proxy_execution_history"):
                        raise ValueError("offline invalid history")
                    return original(cfg, nodes)
                target = validation if modular else legacy
                with patch.object(target, "conversation_context", side_effect=check), \
                        patch.object(load("planning_start_contract"), "conversation_context", side_effect=check):
                    if modular:
                        result = modular_feedback.run_episode_feedback(self.args, episode, root, None,
                            services=EpisodeServices(proposer=proposer))
                    else:
                        with patch.object(legacy, "propose_once", side_effect=proposer):
                            result = legacy.run_episode_feedback(self.args, episode, root, None)
                self.assertEqual(result["status"], "harness_error")
                self.assertEqual(memory.read_json(root / "memory_checkpoint.json")["history"], [])

    def test_batch_preflight_remains_first_turn_only(self):
        for runner in (load("run_dialogue_chain"), modular_batch):
            with tempfile.TemporaryDirectory() as tmp:
                args = copy.copy(self.args)
                args.preflight_only, args.episode_ids, args.repeat = True, None, 1
                args.output_dir = Path(tmp) / "batch"
                self.assertEqual(runner.run(args), 0)
                result = memory.read_json(args.output_dir / "dialogue_summary.json")
                self.assertEqual(result["planned_round_count"], 16)
                self.assertEqual(result["processed_round_count"], 0)
                self.assertFalse(result["fresh_llm_calls"])
                self.assertFalse(result["planning_semantic_gate"])

    def test_full_batch_offline_fixtures_and_infrastructure_stop_match(self):
        reference = load("test_dialogue_chain").reference_replay
        for infrastructure in (False, True):
            with self.subTest(infrastructure=infrastructure), tempfile.TemporaryDirectory() as tmp:
                saved = []
                for modular in (False, True):
                    root = Path(tmp) / ("new" if modular else "old")
                    args = copy.copy(self.args)
                    args.output_dir, args.episode_ids, args.repeat = root, None, 1
                    args.embedding_python, args.embedding_model_dir = Path(__file__), Path(tmp)
                    def proposer(args, folder, encoder):
                        if infrastructure:
                            return {"status": "infrastructure_error", "error": "offline timeout"}, None, None
                        cfg, nodes, _, _ = perception.scene_fixture("scene202")
                        descriptor = memory.read_json(folder / "round_input.json")
                        state = memory.read_json(folder / "runtime_state.json")
                        _, plan = reference(descriptor["round"], state, nodes, cfg)
                        alignment = {"context_bindings": {f"id_{i}": {"element_ids": [i]}
                                     for i in (86, 121, 101, 69, 2)}}
                        memory.write_json(folder / "skill_plan_proposed.json", plan)
                        return {"status": "proposal_ready_not_semantically_validated"}, plan, alignment
                    target = modular_feedback if modular else legacy
                    runner = modular_batch if modular else load("run_dialogue_chain")
                    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-not-a-key"}), \
                            patch.object(pilot, "LocalEncoder") as encoder, \
                            patch.object(pilot, "model_fingerprint", return_value="offline"), \
                            patch.object(pilot, "sha_identity_root", return_value="offline-equivalence-scope"), \
                            patch.object(target, "propose_once", side_effect=proposer):
                        encoder.return_value.stats.return_value = {}
                        code = runner.run(args)
                    summary = memory.read_json(root / "dialogue_summary.json")
                    self.assertEqual(code, 2 if infrastructure else 0)
                    self.assertEqual(summary["processed_round_count"], 0 if infrastructure else 16)
                    self.assertEqual(summary["successful_execution_round_count"], 0 if infrastructure else 16)
                    saved.append(artifacts(root))
                self.assertEqual(saved[0], saved[1])

    def test_followup_suite_preflight_retains_all_74_planned_turns(self):
        for runner in (load("run_matrix_followup"), modular_followup):
            with tempfile.TemporaryDirectory() as tmp:
                args = copy.copy(self.args)
                args.preflight_only, args.episode_ids, args.repeat, args.group = True, None, 1, "all"
                args.output_dir = Path(tmp) / "batch"
                self.assertEqual(runner.run(args), 0)
                result = memory.read_json(args.output_dir / "dialogue_summary.json")
                self.assertEqual(result["planned_episode_count"], 11)
                self.assertEqual(result["planned_round_count"], 74)
                self.assertEqual(result["processed_round_count"], 0)
                self.assertTrue(result["historical_source_summary_unchanged"])


class SavedEvidenceTests(unittest.TestCase):
    """Read archived outputs without changing, repairing or re-requesting them."""

    BATCH = ROOT / "outputs/vlm_full_scene_v1/runs/matrix_followup_v7_typed5_20261019_183854_885579"

    def read(self, relative):
        path = self.BATCH / relative
        # Windows archived audit paths can exceed the classic MAX_PATH limit.
        name = str(path.resolve())
        if os.name == "nt":
            name = "\\\\?\\" + name
        with open(name, encoding="utf-8") as stream:
            return json.load(stream)

    def test_saved_invalid_states_and_conflicting_goals_still_rejected(self):
        for episode, number, expected in (("t02", 1, "expected 0/1"),
                                          ("t05", 1, "expected 0/1"),
                                          ("t02", 2, "Conflicting required state goals"),
                                          ("t06", 3, "Conflicting required state goals")):
            with self.subTest(episode=episode, round=number):
                base = f"episodes/{episode}_r01/round_{number}/planning/"
                raw = json.loads(self.read(base + "llm_decomposition_attempt_1.json")["response_content"])
                allocation = self.read(base + "identifier_allocation.json")
                before = copy.deepcopy(raw)
                exception = errors.AlignmentError if "Conflicting" in expected else ValueError
                with self.assertRaisesRegex(exception, expected):
                    validation.validate_decomposition(validation.decode_decomposition(raw, allocation))
                self.assertEqual(raw, before)

    def test_saved_duplicate_reasons_are_not_silently_repaired(self):
        base = "episodes/t06_r01/round_5/planning/"
        raw = self.read(base + "llm_binding_matrix_proposal_attempt_1.json")
        roles = self.read(base + "task_decomposition.json")["task_roles"]
        nodes = self.read(base + "input_memory/scene/scene_graph.json")
        book = self.read(base + "object_binding_codebook.json")
        self.assertEqual(len(raw["reasons"]), 6)
        self.assertEqual(len({tuple(row) for row in raw["reasons"]}), 3)
        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            validation.from_binding_matrix(raw, roles, nodes, book)
        self.assertEqual(len(raw["reasons"]), 6)

    def test_saved_a12_execution_exactly_replays_failure_prefix(self):
        base = "episodes/t06_r01/round_6/"
        plan = self.read(base + "planning/skill_plan_proposed.json")
        alignment = self.read(base + "planning/object_alignment.json")
        library = self.read(base + "planning/execution_skill_library.json")
        nodes = self.read(base + "planning/input_memory/scene/scene_graph.json")
        state = self.read(base + "planning/runtime_state.json")
        expected = self.read(base + "execution_feedback.json")
        before = copy.deepcopy(state)
        result = execution.execute_proxy(plan, alignment, library, nodes, state,
            compact=lambda value: memory.compact_state(value, (124, 152, 230)))
        self.assertEqual(result, expected)
        self.assertEqual(state, before)
        self.assertEqual(result["resulting_state"]["held_object_id"], 152)
        self.assertEqual(result["attempted_action_count"], 3)
        self.assertEqual(result["error"], "LIFTED_HELD_OBJECT_REQUIRED")


if __name__ == "__main__":
    unittest.main()
