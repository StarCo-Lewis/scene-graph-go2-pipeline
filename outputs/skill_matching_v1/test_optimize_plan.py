import unittest

from optimize_plan import candidate_plan, decide


def original():
    return {"subtask_plans": [{"subtask_id": "a", "steps": [
        {"step_id": "n1", "skill_id": "navigate_to_object", "arguments": {"target": {"element_id": 1}}},
        {"step_id": "i1", "skill_id": "inspect_object", "arguments": {"target": {"element_id": 1}}},
        {"step_id": "n2", "skill_id": "navigate_to_object", "arguments": {"target": {"element_id": 2}}},
    ]}, {"subtask_id": "b", "steps": [
        {"step_id": "n3", "skill_id": "navigate_to_object", "arguments": {"target": {"element_id": 1}}},
        {"step_id": "g1", "skill_id": "grasp_object", "arguments": {"target": {"element_id": 1}}},
    ]}]}


class OptimizeTests(unittest.TestCase):
    def test_nonobject_critique_rejected_with_explicit_reason(self):
        for proposal in ([], None, 'bad review'):
            with self.subTest(proposal=proposal), self.assertRaisesRegex(ValueError, 'JSON object'):
                candidate_plan(original(), proposal)

    def test_may_reorder_and_remove_navigation(self):
        candidate = candidate_plan(original(), {"ordered_step_ids": ["n2", "n1", "i1", "g1"]})
        self.assertEqual([s["step_id"] for p in candidate["subtask_plans"] for s in p["steps"]],
                         ["n2", "n1", "i1", "g1"])

    def test_cannot_remove_inspection_or_manipulation(self):
        with self.assertRaisesRegex(ValueError, "only navigation"):
            candidate_plan(original(), {"ordered_step_ids": ["n1", "n2", "n3", "g1"]})

    def test_cannot_move_step_across_subtask_order(self):
        with self.assertRaisesRegex(ValueError, "subtask order"):
            candidate_plan(original(), {"ordered_step_ids": ["n3", "n1", "i1", "n2", "g1"]})

    def test_unknown_or_duplicate_steps_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate step"):
            candidate_plan(original(), {"ordered_step_ids": ["n1", "n1"]})
        with self.assertRaisesRegex(ValueError, "unknown"):
            candidate_plan(original(), {"ordered_step_ids": ["n1", "made_up"]})


if __name__ == "__main__":
    unittest.main()
