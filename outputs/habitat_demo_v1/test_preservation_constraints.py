"""Offline protocol tests; no API calls, robot execution or scene truth claims."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_json_long_horizon import nodes, initial_state
from preservation_constraints import (extract, constraint_only, check_state, check_action,
                                      no_action_leaf, for_review)
from hierarchical_json_planner import combine_leaves
from test_json_long_horizon import row
from json_symbolic_review import review_symbolic

TEXT = 'Leave the kettle (ID 67) on the sideboard (ID 13).'


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = nodes()
        self.state = initial_state(self.nodes)
        self.state['backend']['objects']['67']['support_id'] = 13
        self.constraints = extract(TEXT)

    def test_exact_ids_and_source_span(self):
        text = 'Visit cabinet ID 62. ' + TEXT
        items = extract(text)
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0]['object_id'], items[0]['surface_id']), (67, 13))
        a, b = items[0]['source_span']
        self.assertEqual(text[a:b], items[0]['source_text'])
        self.assertFalse(constraint_only(text))
        self.assertTrue(constraint_only(TEXT))

    def test_unsupported_or_ambiguous_text_never_gets_no_action_exemption(self):
        for text in ('Place the kettle (ID 67) on the sideboard (ID 13).',
                     'Leave it there.', 'Leave the kettle (ID 67) on a sideboard.',
                     'Do not leave the kettle (ID 67) on the sideboard (ID 13).',
                     'Leave the kettle (ID 67) and cup (ID 68) on the sideboard (ID 13).',
                     TEXT + ' Pick up the kettle (ID 67).'):
            with self.subTest(text=text):
                self.assertFalse(constraint_only(text))
                with self.assertRaises(ValueError):
                    no_action_leaf(text, self.state, self.nodes)

    def test_satisfied_constraint_keeps_full_state_and_has_no_atomic_actions(self):
        before = copy.deepcopy(self.state)
        leaf = no_action_leaf(TEXT, self.state, self.nodes)
        self.assertEqual(leaf['begin_state'], leaf['end_state'])
        self.assertEqual(leaf['plan']['subtask_plans'], [])
        self.assertFalse(leaf['constraint_report']['fresh_llm_calls'])
        self.assertFalse(leaf['constraint_report']['simulation_executed'])
        self.assertEqual(self.state, before)

    def test_unsatisfied_or_missing_support_fails_not_repaired(self):
        for support in (None, 62):
            self.state['backend']['objects']['67']['support_id'] = support
            with self.assertRaisesRegex(ValueError, 'NOT_SATISFIED'):
                no_action_leaf(TEXT, self.state, self.nodes)

    def test_missing_inactive_and_self_referencing_bindings_fail(self):
        with self.assertRaisesRegex(ValueError, 'SELF_SUPPORT'):
            extract('Keep the kettle (ID 67) on the kettle (ID 67).')
        with self.assertRaisesRegex(ValueError, 'BINDING_UNAVAILABLE'):
            check_state(extract('Keep the kettle (ID 67) on the desk (ID 999).'), self.state, self.nodes)
        self.nodes[0]['active'] = False
        with self.assertRaisesRegex(ValueError, 'BINDING_UNAVAILABLE'):
            check_state(self.constraints, self.state, self.nodes)

    def test_unknown_pose_and_held_protected_object_fail(self):
        self.state['held_object_id'] = 67
        with self.assertRaisesRegex(ValueError, 'IS_HELD'):
            check_state(self.constraints, self.state, self.nodes)
        self.state['held_object_id'] = None
        self.state['backend']['objects']['67']['center'] = None
        with self.assertRaisesRegex(ValueError, 'POSE_UNKNOWN'):
            check_state(self.constraints, self.state, self.nodes)

    def test_manipulation_is_rejected_even_if_final_support_would_be_restored(self):
        for skill, key in (('grasp_object', 'target'), ('lift_held_object', 'object'),
                           ('place_object_on_surface', 'object')):
            step = {'skill_id': skill, 'arguments': {key: {'element_id': 67}}}
            with self.subTest(skill=skill), self.assertRaisesRegex(ValueError, 'MANIPULATION_FORBIDDEN'):
                check_action(self.constraints, step, self.state)

    def test_protected_support_and_carry_are_also_guarded(self):
        with self.assertRaisesRegex(ValueError, 'MANIPULATION_FORBIDDEN'):
            check_action(self.constraints, {'skill_id':'grasp_object',
                'arguments':{'target':{'element_id':13}}}, self.state)
        self.state['held_object_id'] = 67
        with self.assertRaisesRegex(ValueError, 'MANIPULATION_FORBIDDEN'):
            check_action(self.constraints, {'skill_id':'navigate_to_object',
                'arguments':{'target':{'element_id':62}}}, self.state)

    def test_unrelated_actions_allowed_but_each_boundary_checks_pose(self):
        check_action(self.constraints, {'skill_id':'navigate_to_object',
            'arguments':{'target':{'element_id':62}}}, self.state)
        changed = copy.deepcopy(self.state)
        changed['backend']['objects']['67']['center'][0] += .01
        with self.assertRaisesRegex(ValueError, 'STATE_CHANGED'):
            check_state(self.constraints, changed, self.nodes, reference=self.state)

    def test_original_constraint_applies_even_in_an_earlier_action_leaf(self):
        policy = {'hierarchy_context':{'original_instruction':'Visit the desk. ' + TEXT}}
        inherited = for_review('Visit the desk.', policy)
        self.assertEqual(len(inherited), 1)
        self.assertEqual(inherited[0]['object_id'], 67)
        self.assertEqual(len(for_review(TEXT, policy)), 1)

    def test_empty_llm_plan_cannot_masquerade_as_a_validated_constraint(self):
        leaf = no_action_leaf(TEXT, self.state, self.nodes)
        _, library = row([('navigate_to_object', 13)])
        alignment, plan = combine_leaves(TEXT, [leaf], library)
        plan['proposal_origin'] = 'fresh_llm_matrix_decoded'
        with self.assertRaisesRegex(ValueError, 'EMPTY_PLAN_WITHOUT_VALIDATED'):
            review_symbolic({'instruction':TEXT,'alignment':alignment,'plan':plan},
                            library,self.nodes,self.state,1.5)

    def test_action_text_cannot_be_erased_by_combining_only_constraint_leaves(self):
        _, library = row([('navigate_to_object', 13)])
        leaf = no_action_leaf(TEXT, self.state, self.nodes)
        with self.assertRaisesRegex(ValueError, 'No executable leaves'):
            combine_leaves('Visit cabinet ID 62. ' + TEXT, [leaf], library)


if __name__ == '__main__':
    unittest.main()
