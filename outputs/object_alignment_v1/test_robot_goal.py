import unittest
from align_objects import validate_constraints, AlignmentError


class RobotGoalTests(unittest.TestCase):
    def goal(self, subject="robot", relation="near", target="sideboard"):
        return [{"constraint_id": "g1", "subject_role": subject,
                 "relation": relation, "object_role": target, "required": True}]

    def test_reserved_executor_does_not_need_scene_node(self):
        result = validate_constraints(self.goal(), "goal_state", {"sideboard"})
        self.assertEqual(result[0]["subject_role"], "robot")

    def test_unknown_destination_still_rejected(self):
        with self.assertRaises(AlignmentError):
            validate_constraints(self.goal(target="apple"), "goal_state", {"sideboard"})

    def test_arbitrary_actor_and_unsupported_robot_relation_rejected(self):
        for subject, relation in (("person", "near"), ("robot", "on")):
            with self.subTest(subject=subject, relation=relation), self.assertRaises(AlignmentError):
                validate_constraints(self.goal(subject, relation), "goal_state", {"sideboard"})
