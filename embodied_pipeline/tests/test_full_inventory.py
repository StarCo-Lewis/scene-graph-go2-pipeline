"""Full inventory coverage and prompt integration, offline only."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
from embodied_pipeline import instruction_skills as skills
from embodied_pipeline.tests import test_real_scene_plan as fixtures


class FullInventoryTests(unittest.TestCase):
    def test_no_encoder_no_cutoff_and_no_mutation(self):
        library = {"skills": [{"skill_id": s} for s in ["navigate", "grasp", "lift", "place", "future_skill"]]}
        alignment = {"task_decomposition": {"subtasks": [
            {"subtask_id": "s1", "description": "Move the box"},
            {"subtask_id": "s2", "description": "Put it down"}]}}
        before = copy.deepcopy((library, alignment))
        encoder = Mock()
        result = skills.candidates(alignment, library, encoder, top_k=1, threshold=1, margin=1)
        encoder.encode.assert_not_called()
        for row in result["subtasks"]:
            self.assertEqual(row["candidate_skill_ids"], [s["skill_id"] for s in library["skills"]])
            self.assertNotIn("ranking", row)
        self.assertFalse(result["embedding_used"])
        self.assertEqual(before, (library, alignment))

    def test_full_catalogue_reaches_typed_matrix_prompt(self):
        fixture = fixtures.PlanningOnlyTests()
        with tempfile.TemporaryDirectory() as temp:
            root, (task, bundle, library, source, hashes) = fixture.prepare(temp)
            alignment = fixture.alignment(task)
            retrieval = skills.candidates(alignment, library)
            book = skills.make_codebook(alignment, library)
            payload = skills.matrix_payload(alignment, library, book, "offline", retrieval, False,
                                           matrix_contract="typed5_schema")
            import json
            prompt = json.loads(payload["messages"][-1]["content"])
            self.assertEqual(len(prompt["skills"]), len(library["skills"]))
            for row in prompt["subtasks"]:
                self.assertEqual(set(row["candidate_skill_codes"]), {s["code"] for s in prompt["skills"]})

    def test_duplicate_catalogue_rejected(self):
        with self.assertRaises(ValueError):
            skills.candidates({}, {"skills": [{"skill_id": "x"}, {"skill_id": "x"}]})

    def test_live_entry_does_not_start_embedding(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from embodied_pipeline._legacy import load
        smoke = fixtures.smoke
        with tempfile.TemporaryDirectory() as temp:
            fixture = fixtures.PlanningOnlyTests()
            root, unused = fixture.prepare(temp)
            args = SimpleNamespace(task_json=Path(temp) / "task.json", output_dir=Path(temp) / "new",
                                   preflight_only=False, model="offline", base_url="https://invalid.test")
            def generated(args, root, task, bundle, library, summary, encoder):
                self.assertIsNone(encoder)
                summary["status"] = "planning_json_generated_not_executed"
            with patch.dict(os.environ, DASHSCOPE_API_KEY="offline"), \
                 patch.object(load("skill_retrieval"), "LocalEncoder", side_effect=AssertionError("Embedding forbidden")), \
                 patch.object(smoke, "generate", side_effect=generated):
                self.assertEqual(smoke.run(args), 0)
