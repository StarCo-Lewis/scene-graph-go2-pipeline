import unittest
from align_objects import AlignmentError, validate_planning_capability_claims


class CapabilityClaimTests(unittest.TestCase):
    def setUp(self):
        self.contract = {'planning_capabilities': {'validate_supplied_on_in_edges': True}}
        self.data = {
            'subtasks': [{'subtask_id': 'st_01', 'description': 'Navigate to the red can on the main table.'}],
            'initial_scene_constraints': [{'relation': 'on', 'required': True}],
        }

    def test_rejects_observed_bug_without_deleting_relation(self):
        self.data['subtasks'][0]['description'] = (
            'Navigate to the can. Unresolved requirement: using only observed edges '
            'to determine which can is not supported by navigation-only execution.')
        with self.assertRaisesRegex(AlignmentError, 'planning capability'):
            validate_planning_capability_claims(self.data, self.contract)
        self.assertEqual(self.data['initial_scene_constraints'][0]['relation'], 'on')

    def test_valid_relation_description(self):
        self.assertIs(validate_planning_capability_claims(self.data, self.contract), self.data)

    def test_real_inspection_limitation_remains(self):
        self.data['subtasks'][0]['description'] += ' Physical safety inspection is unsupported by robot execution.'
        self.assertIs(validate_planning_capability_claims(self.data, self.contract), self.data)

    def test_no_false_claim_without_capability(self):
        self.data['subtasks'][0]['description'] = 'Edges are not supported by navigation execution.'
        self.assertIs(validate_planning_capability_claims(self.data, {}), self.data)


if __name__ == '__main__':
    unittest.main()
