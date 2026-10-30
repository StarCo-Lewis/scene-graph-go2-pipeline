import unittest
from run_sideboard_task import fixture


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.a = {"status": "success", "context_bindings": {"sideboard": {"element_ids": [13]}}}
        self.p = {"subtask_plans": [{"steps": [{"skill_id": "navigate_to_object",
            "arguments": {"target": {"role_id": "sideboard", "element_id": 13}}}]}]}
        self.n = [{"id": 13, "bbox_center": [0, 0, 1]}]

    def test_fixture_is_explicit_and_does_not_mutate_source(self):
        a, p, node = fixture(self.a, self.p, self.n)
        self.assertEqual(node["id"], 13)
        self.assertIn("not_llm_generated", p["plan_origin"])
        self.assertNotIn("instruction", self.p)

    def test_missing_node_rejected(self):
        with self.assertRaisesRegex(ValueError, "NO object"):
            fixture(self.a, self.p, [])

    def test_inconsistent_source_target_rejected(self):
        self.p["subtask_plans"][0]["steps"][0]["arguments"]["target"]["element_id"] = 67
        with self.assertRaises(ValueError):
            fixture(self.a, self.p, self.n)
