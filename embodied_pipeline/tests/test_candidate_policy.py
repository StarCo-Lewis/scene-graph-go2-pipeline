"""Offline candidate repair tests using historical failures, never live calls."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT
from embodied_pipeline import instruction_skills as skills, memory
from embodied_pipeline.instruction_skills.candidate_policy import complete_candidates
from embodied_pipeline.orchestration.proposal import propose_once
from embodied_pipeline.orchestration import oracle_decomposition as oracle

ARCHIVE = ROOT / "outputs/vlm_full_scene_v1/runs/oracle_d_20261020_151108_506717/cases"


class CandidatePolicyTests(unittest.TestCase):
    def folder(self, case):
        return ARCHIVE / case / "repeat_01/oracle_decomposition/planning"

    def read(self, case, name):
        return oracle.read(self.folder(case) / name)

    def test_historical_candidates_and_actual_typed5_prompt(self):
        for case in ("C01", "A06"):
            with self.subTest(case=case):
                library = self.read(case, "execution_skill_library.json")
                retrieval = self.read(case, "skill_retrieval_semantic_1.json")
                originals = copy.deepcopy((library, retrieval))
                repaired, audit = complete_candidates(retrieval, library, "prerequisite_complete")
                all_ids = {s["skill_id"] for s in library["skills"]}
                for before, after, note in zip(retrieval["subtasks"], repaired["subtasks"], audit["subtasks"]):
                    self.assertEqual(set(after["candidate_skill_ids"]), all_ids)
                    self.assertEqual(set(note["added_skill_ids"]), {"grasp_object", "lift_held_object"})
                    self.assertEqual(before["ranking"], after["ranking"])
                alignment = self.read(case, "object_alignment.json")
                book = self.read(case, "action_codebook_semantic_1.json")
                payload = skills.matrix_payload(alignment, library, book, "offline", repaired, False,
                                               matrix_contract="typed5_schema")
                prompt = json.loads(payload["messages"][-1]["content"])
                self.assertEqual({s["skill_id"] for s in prompt["skills"]}, all_ids)
                for row in prompt["subtasks"]:
                    self.assertEqual(set(row["candidate_skill_codes"]), {40001, 40002, 40003, 40004, 40005})
                self.assertEqual((library, retrieval), originals)

    def test_default_and_missing_dependency(self):
        library = self.read("C01", "execution_skill_library.json")
        retrieval = self.read("C01", "skill_retrieval_semantic_1.json")
        result, audit = complete_candidates(retrieval, library)
        self.assertIs(result, retrieval)
        self.assertIsNone(audit)
        library["skills"] = [s for s in library["skills"] if s["skill_id"] != "grasp_object"]
        with self.assertRaisesRegex(ValueError, "Missing prerequisite skill"):
            complete_candidates(retrieval, library, "prerequisite_complete")

    def test_proposal_routes_repaired_candidates_without_inserting_actions(self):
        args = SimpleNamespace(model="offline", base_url="https://invalid.test",
            matrix_contract="typed5_schema", alignment_wire_format="matrix",
            top_k=2, cosine_threshold=.35, cosine_margin=.05,
            candidate_policy="prerequisite_complete")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in oracle.INPUT_NAMES:
                memory.write_json(folder / name, self.read("C01", name))
            alignment = self.read("C01", "object_alignment.json")
            def capture(*positional, **kw):
                self.assertFalse(kw["widen_inventory"])
                for row in kw["retrieval"]["subtasks"]:
                    self.assertIn("grasp_object", row["candidate_skill_ids"])
                    self.assertIn("lift_held_object", row["candidate_skill_ids"])
                raise ValueError("Offline stop before model output")
            with patch.object(skills, "request_matrix", side_effect=capture) as request:
                summary, plan, _ = propose_once(args, folder, None,
                    alignment_provider=lambda *unused: copy.deepcopy(alignment))
            self.assertEqual(request.call_count, 1)
            self.assertIsNone(plan)
            self.assertEqual(summary["status"], "proposal_unavailable")
            self.assertEqual(summary["candidate_policy"], "full_inventory_no_cosine_v1")
            self.assertFalse((folder / "skill_plan_proposed.json").exists())

    def test_cli_selects_only_requested_oracle_arms(self):
        with patch.object(oracle, "run", return_value=0) as runner:
            oracle.main(["--case-ids", "C01", "A06", "--arms", "oracle_decomposition",
                         "--candidate-policy", "prerequisite_complete", "--preflight-only"])
        args = runner.call_args.args[0]
        self.assertEqual(args.arms, ["oracle_decomposition"])
        self.assertEqual(args.case_ids, ["C01", "A06"])
        self.assertEqual(args.candidate_policy, "prerequisite_complete")


if __name__ == "__main__":
    unittest.main()
