"""Offline evidence isolation and prompt-routing checks; no live API calls."""

import copy
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import load
from embodied_pipeline import introspection, memory
from embodied_pipeline.planning import intent_alignment_worker as worker
from embodied_pipeline.planning.intent_prompt import ARMS, ADDENDUM, compose
from embodied_pipeline.orchestration import intent_comparison as experiment
from embodied_pipeline.orchestration import oracle_decomposition as oracle
from embodied_pipeline.tests.test_oracle_decomposition import options


FIXTURE_BASELINE = {"outputs/fixture_a.json": '{"fixture": "a"}\n',
                    "outputs/fixture_b.json": '{"fixture": "b"}\n'}


@contextlib.contextmanager
def controlled_legacy_baseline(root, *, break_fixture=None, remove_fixture=None):
    """Point verify_legacy at a temporary baseline instead of the workspace manifest.

    The real verify_legacy still hashes every listed file; only the baseline moves, so
    this offline test no longer fails when a production artifact is updated under a
    separate review. Manifest paths are absolute because ROOT / absolute keeps the
    absolute path, which leaves introspection.ROOT and write_run_manifest untouched.
    """
    base = Path(root) / "legacy_fixture"
    entries = []
    for relative, text in FIXTURE_BASELINE.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        entries.append(dict(path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if break_fixture is not None:
        (base / break_fixture).write_text('{"fixture": "edited offline"}\n', encoding="utf-8")
    if remove_fixture is not None:
        (base / remove_fixture).unlink()
    manifest = Path(root) / "legacy_manifest_fixture.json"
    manifest.write_text(json.dumps(dict(
        scope="Offline test baseline over temporary fixtures; not the production manifest.",
        files=entries), indent=2) + "\n", encoding="utf-8")
    with patch.object(introspection, "MANIFEST", manifest):
        yield base


class IntentComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = oracle.selected_cases(options())

    def test_all_22_prepared_branches_preserve_facts_without_injected_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in self.cases:
                for arm in ARMS:
                    folder, record = experiment.prepare_branch(case, arm, Path(tmp) / case["case_id"] / arm)
                    for name in experiment.INPUT_NAMES:
                        self.assertEqual(oracle.sha(folder / name), case["source_files"][name]["sha256"])
                    self.assertFalse((folder / "injected_decomposition.json").exists())
                    self.assertFalse((folder / "identifier_allocation.json").exists())
                    self.assertFalse(record["oracle_payload_injected"])
                    self.assertEqual(record["input_precheck"], "accepted")
                    prompt = oracle.read(folder.parent / "decomposition_prompt.json")["system_prompt"]
                    self.assertEqual("COMPLETE INTENTS AND RELEVANT CONSTRAINTS" in prompt, arm == ARMS[1])

    def test_real_request_construction_uses_treatment_only_in_system_prompt(self):
        case = next(c for c in self.cases if c["case_id"] == "A06")
        engine = load("align_objects")
        contract = oracle.read(oracle.source_path(case["source_files"]["execution_contract.json"]))
        captured = []
        def fake_request(request, **kwargs):
            payload = json.loads(request.data)
            captured.append(payload)
            user = json.loads(payload["messages"][1]["content"])
            fixture = copy.deepcopy(case["oracle_payload"])
            fixture["codebook_sha256"] = user["identifier_allocation"]["sha256"]
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(fixture)}}]}
        with patch.dict(os.environ, {"INTENT_TEST_KEY": "offline-placeholder"}), \
             patch.object(engine, "request_json", side_effect=fake_request), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden")):
            for arm in ARMS:
                with patch.object(engine, "DECOMPOSITION_PROMPT", compose(engine.DECOMPOSITION_PROMPT, arm)):
                    engine.call_llm(case["instruction_en"], "offline", "https://invalid.test", "INTENT_TEST_KEY",
                        max_attempts=1, execution_contract=contract, decomposition_format="matrix",
                        matrix_contract="typed5_schema")
        baseline, treatment = captured
        self.assertNotIn(ADDENDUM.strip(), baseline["messages"][0]["content"])
        self.assertIn(ADDENDUM.strip(), treatment["messages"][0]["content"])
        left, right = [json.loads(p["messages"][1]["content"]) for p in captured]
        self.assertEqual(left["instruction"], case["instruction_en"])
        self.assertEqual(left["execution_contract"], right["execution_contract"])
        self.assertNotIn("oracle_payload", left)
        self.assertNotIn("private_evaluation", left)
        self.assertNotEqual(left["identifier_allocation"], right["identifier_allocation"])

    def test_worker_restores_engine_prompt_even_on_failure(self):
        engine = load("align_objects")
        before = engine.DECOMPOSITION_PROMPT
        def fail():
            self.assertEqual(engine.DECOMPOSITION_PROMPT, compose(before, ARMS[1]))
            raise ValueError("offline fixture")
        with patch.object(engine, "main", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "offline fixture"):
                worker.main(["--prompt-arm", ARMS[1], "--instruction", "English instruction"])
        self.assertEqual(engine.DECOMPOSITION_PROMPT, before)

    def test_live_provider_has_no_saved_decomposition_or_allocation_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = options(output_dir=folder, prompt_arm=ARMS[1])
            def fake_run(command, **kwargs):
                self.assertIn("--instruction", command)
                self.assertIn("--prompt-arm", command)
                self.assertNotIn("--decomposition-json", command)
                self.assertNotIn("--identifier-allocation-json", command)
                self.assertNotIn("--binding-proposal-json", command)
                memory.write_json(folder / "object_alignment.json", dict(status="error",
                    error=dict(message="offline rejection", category="model_or_input_validation_error")))
                return type("Result", (), {"returncode": 1})()
            with patch.object(experiment.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "offline rejection"):
                    experiment.alignment_from_live(args, {"instruction": "Bring the bottle."}, {"stages": []})

    def resume_fixture(self, root):
        base = load("align_objects").DECOMPOSITION_PROMPT
        rows = [dict(case_id=c, repeat=1, arm=a,
                     prompt_sha256=oracle.content_hash(load("typed_matrix_ids").decomp_prompt(compose(base, a), 5)))
                for c in ("A10", "A11") for a in ARMS]
        source = root / "old_summary.json"
        memory.write_json(source, dict(profile=experiment.VERSION, arms=list(ARMS),
            status="harness_error", engine_unchanged=True, archived_sources_unchanged=True,
            copied_inputs_unchanged=True, candidate_policy="full_inventory_no_cosine_v1",
            model="offline-fixture", repeats=1, case_ids=["A10", "A11"],
            planned_branch_count=4, processed_branch_count=1, preflight=rows,
            branches=[dict(rows[0], status="proposal_unavailable", private_reference_passed=False)]))
        return options(resume_from=source, repeat=None, arms=list(ARMS),
                       candidate_policy="full_inventory_no_cosine_v1", preflight_only=True,
                       output_dir=root / "new")

    def test_resume_skips_completed_failures_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden")), \
             controlled_legacy_baseline(Path(tmp)):
            args = self.resume_fixture(Path(tmp))
            before = args.resume_from.read_bytes()
            keys, info = experiment.resume_selection(args)
            self.assertEqual(keys, {("A10", 1, ARMS[1]), ("A11", 1, ARMS[0]), ("A11", 1, ARMS[1])})
            self.assertEqual(oracle.run(args, prepare=experiment.prepare_branch,
                experiment=dict(profile=experiment.VERSION, arms=list(ARMS), **info),
                input_names=experiment.INPUT_NAMES, branch_keys=keys), 0)
            summary = oracle.read(args.output_dir / "comparison_summary.json")
            self.assertEqual(summary["legacy_integrity"]["status"], "unchanged")
            self.assertEqual(summary["legacy_integrity"]["checked_file_count"], len(FIXTURE_BASELINE))
            self.assertEqual(summary["planned_branch_count"], 3)
            self.assertEqual({(r["case_id"], r["repeat"], r["arm"]) for r in summary["preflight"]}, keys)
            self.assertFalse(summary["fresh_llm_calls"])
            self.assertEqual(args.resume_from.read_bytes(), before)

    def test_resume_rejects_prompt_drift_and_duplicate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.resume_fixture(Path(tmp))
            saved = oracle.read(args.resume_from)
            saved["preflight"][1]["prompt_sha256"] = "changed"
            memory.write_json(args.resume_from, saved)
            with self.assertRaisesRegex(ValueError, "prompt changed"):
                experiment.resume_selection(args)
            args = self.resume_fixture(Path(tmp))
            saved = oracle.read(args.resume_from)
            saved["branches"].append(saved["branches"][0])
            saved["processed_branch_count"] = 2
            memory.write_json(args.resume_from, saved)
            with self.assertRaisesRegex(ValueError, "accounting"):
                experiment.resume_selection(args)

    def test_resume_subset_defers_timeout_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.resume_fixture(Path(tmp))
            args.case_ids = ["A11"]
            keys, info = experiment.resume_selection(args)
            self.assertEqual(keys, {("A11", 1, a) for a in ARMS})
            self.assertEqual(info["deferred_branch_keys"],
                             [dict(case_id="A10", repeat=1, arm=ARMS[1])])

    def test_transport_continues_but_unknown_runtime_and_mutation_stop(self):
        for failure in ("transport", "unknown", "mutation"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = options(repeat=1, arms=list(ARMS), candidate_policy="full_inventory_no_cosine_v1",
                    preflight_only=False, output_dir=root / "run", continue_on_transport_error=True,
                    embedding_python=root / "python", embedding_model_dir=root / "model")
                args.case_ids = ["A11"]
                memory.write_json(args.embedding_python, {})
                args.embedding_model_dir.mkdir()
                calls = []
                proposal = dict(status="infrastructure_error", failed_stage="decomposition_and_binding",
                    error="LLM transport failed after 2 attempts: TimeoutError",
                    token_usage=dict(transport_failures_unknown_usage=2))
                def execute(opts, case, arm, folder, planning, encoder):
                    calls.append(arm)
                    if len(calls) == 1:
                        if failure == "unknown":
                            raise RuntimeError("Unexpected engine failure")
                        if failure == "mutation":
                            memory.write_json(planning / "runtime_state.json", {})
                            raise oracle.BranchTransportError(proposal)
                        with patch.object(oracle, "propose_once", return_value=(proposal, None, None)):
                            return oracle.execute_branch(opts, case, arm, folder, planning, encoder)
                    return dict(case_id=case["case_id"], arm=arm, status="proxy_actions_completed",
                                private_reference_passed=True)
                retrieval = load("skill_retrieval")
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-placeholder"}), \
                     patch.object(retrieval, "model_fingerprint", return_value="offline"), \
                     patch.object(retrieval, "LocalEncoder") as encoder, \
                     patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden")), \
                     controlled_legacy_baseline(root):
                    encoder.return_value.stats.return_value = {}
                    result = oracle.run(args, prepare=experiment.prepare_branch, execute=execute,
                        experiment=dict(profile=experiment.VERSION, arms=list(ARMS)),
                        input_names=experiment.INPUT_NAMES)
                saved = oracle.read(args.output_dir / "comparison_summary.json")
                if failure == "transport":
                    self.assertEqual(result, 0)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(saved["status"], "completed_with_transport_errors")
                    self.assertEqual(saved["processed_branch_count"], 2)
                    row = saved["branches"][0]
                    self.assertIsNone(row["private_reference_passed"])
                    self.assertFalse(row["memory_committed"])
                    self.assertEqual(row["token_usage"]["transport_failures_unknown_usage"], 2)
                    self.assertFalse((args.output_dir / row["path"] / "memory_after_feedback.json").exists())
                else:
                    self.assertEqual(result, 2)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(saved["status"], "invalid_integrity" if failure == "mutation" else "harness_error")

    def guard_args(self, root):
        args = options(repeat=1, arms=list(ARMS), candidate_policy="full_inventory_no_cosine_v1",
                       preflight_only=True, output_dir=root / "run")
        args.case_ids = ["A11"]
        return args

    def test_unmodified_fixture_baseline_is_really_hashed_and_unchanged(self):
        original_manifest = introspection.MANIFEST
        original_root = introspection.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            with controlled_legacy_baseline(Path(tmp)):
                integrity = introspection.verify_legacy()
        self.assertEqual(introspection.MANIFEST, original_manifest)
        self.assertEqual(introspection.ROOT, original_root)
        self.assertEqual(integrity["status"], "unchanged")
        self.assertEqual(integrity["checked_file_count"], len(FIXTURE_BASELINE))
        self.assertEqual((integrity["changed"], integrity["missing"]), ([], []))

    def test_changed_or_missing_baseline_file_rejects_run_before_any_output(self):
        """The integrity guard is never skipped; only its baseline is controlled here."""
        for kind, kwargs in (("changed", {"break_fixture": "outputs/fixture_a.json"}),
                             ("missing", {"remove_fixture": "outputs/fixture_b.json"})):
            with self.subTest(kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.guard_args(root)
                original_manifest = introspection.MANIFEST
                with controlled_legacy_baseline(root, **kwargs) as fixture_root, \
                     patch("urllib.request.urlopen", side_effect=AssertionError("Live HTTP forbidden")):
                    integrity = introspection.verify_legacy()
                    self.assertEqual(integrity["status"], "changed")
                    self.assertEqual(integrity["checked_file_count"], len(FIXTURE_BASELINE))
                    self.assertEqual(integrity[kind], [str(fixture_root / next(iter(kwargs.values())))])
                    self.assertEqual(integrity["missing" if kind == "changed" else "changed"], [])
                    with self.assertRaisesRegex(ValueError, "Legacy source hashes differ"):
                        oracle.run(args, prepare=experiment.prepare_branch,
                                   experiment=dict(profile=experiment.VERSION, arms=list(ARMS)),
                                   input_names=experiment.INPUT_NAMES)
                self.assertEqual(introspection.MANIFEST, original_manifest)
                self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
