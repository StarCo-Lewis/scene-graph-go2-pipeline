"""Offline tests: proximity is a symbolic hypothesis, never physical evidence."""
import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_json_long_horizon import nodes, row
from run_json_long_horizon import initial_state
from json_symbolic_review import (SymbolicBackend, SymbolicGoalError,
    SYMBOLIC_GOAL_POLICY, check_symbolic_near, review_symbolic)
from action_matrix import make_codebook, decode_matrix
from match_skills import build_result
from sequential_proxy_core import PARAMETERS


def goal(subject='kettle', target='office', required=True):
    return {'constraint_id': 'goal_object_near', 'subject_role': subject,
        'object_role': target, 'relation': 'near', 'required': required}


class SymbolicNearGoalTests(unittest.TestCase):
    def setup_world(self):
        descriptor, library = row([('navigate_to_object', 13)], held=True, visits=[13])
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True, prepared_target_id=67)
        return descriptor, library, state

    def check(self, state=None, bindings=None, predicate=None, reach=1.5):
        descriptor, _, original = self.setup_world()
        backend = SymbolicBackend(state or original, nodes(), reach)
        return check_symbolic_near(predicate or goal(),
            bindings if bindings is not None else descriptor['alignment']['context_bindings'], backend)

    def test_held_object_carry_satisfies_object_and_robot_goals(self):
        descriptor, library, state = self.setup_world()
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal()]
        saved = copy.deepcopy((descriptor, library, state))
        report = review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual(report['symbolic_goal_policy'], SYMBOLIC_GOAL_POLICY)
        self.assertEqual(report['goal_checks'][-1]['distances_m'], [0.])
        self.assertEqual(report['goal_checks'][-1]['status'], 'satisfied_under_symbolic_hypothesis')
        self.assertEqual(report['predicted_end_state']['held_object_id'], 67)
        self.assertTrue(report['predicted_end_state']['lifted'])
        self.assertFalse(report['simulation_executed'])
        self.assertFalse(report['geometry_verified'])
        self.assertFalse(report['training_eligible'])
        self.assertFalse(report['goal_checks'][-1]['geometry_or_navigation_verified'])
        self.assertEqual(descriptor, saved[0])
        self.assertEqual(library, saved[1])
        self.assertEqual(state, saved[2])

    def test_decoded_matrix_and_nested_json_use_same_goal_checker(self):
        descriptor, library, state = self.setup_world()
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal()]
        library = copy.deepcopy(library)
        for skill in library['skills']:
            skill['parameters'] = [p for p in skill['parameters'] if p['name'] in PARAMETERS[skill['skill_id']]]
        alignment = descriptor['alignment']
        book = make_codebook(alignment, library)
        skill_code = next(s['code'] for s in book['skills'] if s['skill_id'] == 'navigate_to_object')
        target_code = next(o['code'] for o in book['objects'] if o['role_id'] == 'office' and o['element_id'] == 13)
        proposal = decode_matrix({'matrix': [[1, skill_code, target_code, 0]]}, book, alignment, library)
        matrix_row = dict(descriptor, plan=build_result(alignment, library, proposal))
        nested = review_symbolic(descriptor, library, nodes(), state, 1.5)
        matrix = review_symbolic(matrix_row, library, nodes(), state, 1.5)
        self.assertEqual(nested['goal_checks'], matrix['goal_checks'])
        self.assertEqual(nested['predicted_end_state'], matrix['predicted_end_state'])

    def test_navigation_without_holding_does_not_move_object_to_goal(self):
        descriptor, library = row([('navigate_to_object', 13)], held=False)
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal()]
        with self.assertRaises(SymbolicGoalError) as caught:
            review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        self.assertEqual(caught.exception.details['reason'], 'outside_horizontal_reach')
        self.assertIn('goal_object_near', str(caught.exception))

    def test_horizontal_threshold_boundary_is_inclusive(self):
        _, _, state = self.setup_world()
        state['backend']['objects']['13']['center'] = [0., 100., 0.]
        for distance, expected in ((1.5, 'satisfied_under_symbolic_hypothesis'),
                                   (1.500001, 'not_satisfied')):
            with self.subTest(distance=distance):
                state['backend']['objects']['67']['center'] = [distance, -100., 0.]
                result = self.check(state)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['distances_m'], [distance])
                self.assertEqual(result['distance_metric'], 'horizontal_xz')

    def test_far_object_in_z_direction_is_rejected(self):
        _, _, state = self.setup_world()
        state['backend']['objects']['13']['center'] = [0., 0., 0.]
        state['backend']['objects']['67']['center'] = [0., 1., 2.]
        self.assertEqual(self.check(state)['status'], 'not_satisfied')

    def test_missing_and_invalid_coordinates_never_pass(self):
        _, _, state = self.setup_world()
        for invalid in (None, [], [0., 0.], [0., 0., 0., 0.], ['0', 0., 0.],
                        [True, 0., 0.], [math.nan, 0., 0.], [0., math.inf, 0.], [0., 0., -math.inf]):
            for identity in ('67', '13'):
                with self.subTest(identity=identity, invalid=invalid):
                    bad = copy.deepcopy(state)
                    bad['backend']['objects'][identity]['center'] = invalid
                    self.assertEqual(self.check(bad)['reason'], 'missing_or_invalid_position')
        bad = copy.deepcopy(state)
        del bad['backend']['objects']['67']
        self.assertEqual(self.check(bad)['reason'], 'missing_or_invalid_position')

    def test_unresolved_empty_duplicate_boolean_and_unmapped_ids_reject(self):
        descriptor, _, _ = self.setup_world()
        for invalid in ({}, {'element_ids': []}, {'element_ids': [67, 67]},
                        {'element_ids': [True]}, {'element_ids': [999]},
                        {'element_ids': ['67']}, {'element_ids': (67,)}, None):
            with self.subTest(invalid=invalid):
                bindings = copy.deepcopy(descriptor['alignment']['context_bindings'])
                bindings['kettle'] = invalid
                self.assertEqual(self.check(bindings=bindings)['reason'], 'unresolved_or_invalid_binding')
        self.assertEqual(self.check(bindings={})['status'], 'unknown')

    def test_invalid_threshold_never_passes(self):
        for invalid in (0, -1, math.nan, math.inf, True, '1.5', None):
            with self.subTest(reach=invalid):
                self.assertEqual(self.check(reach=invalid)['reason'], 'invalid_positive_finite_reach')

    def test_multiple_objects_require_all_pairs_not_one_lucky_match(self):
        descriptor, _, state = self.setup_world()
        bindings = copy.deepcopy(descriptor['alignment']['context_bindings'])
        bindings['kettle']['element_ids'] = [67, 62]
        state['backend']['objects']['13']['center'] = [0., 0., 0.]
        state['backend']['objects']['67']['center'] = [0., 1., 0.]
        state['backend']['objects']['62']['center'] = [4., 1., 0.]
        self.assertEqual(self.check(state, bindings)['status'], 'not_satisfied')
        state['backend']['objects']['62']['center'] = [1., 1., 0.]
        result = self.check(state, bindings)
        self.assertEqual(result['status'], 'satisfied_under_symbolic_hypothesis')
        self.assertEqual(result['distances_m'], [0., 1.])

    def test_all_target_pairs_are_checked(self):
        descriptor, _, state = self.setup_world()
        bindings = copy.deepcopy(descriptor['alignment']['context_bindings'])
        bindings['office']['element_ids'] = [13, 62]
        state['backend']['objects']['13']['center'] = [0., 0., 0.]
        state['backend']['objects']['67']['center'] = [0., 1., 0.]
        state['backend']['objects']['62']['center'] = [4., 1., 0.]
        result = self.check(state, bindings)
        self.assertEqual(result['status'], 'not_satisfied')
        self.assertEqual(len(result['pair_checks']), 2)

    def test_object_cannot_satisfy_near_itself(self):
        self.assertEqual(self.check(predicate=goal('kettle', 'kettle'))['reason'], 'object_near_self_reference')

    def test_robot_goal_retains_single_target_and_checks_position(self):
        descriptor, _, state = self.setup_world()
        state['backend']['robot_position'] = list(state['backend']['objects']['13']['center'])
        self.assertEqual(self.check(state, predicate=goal('robot'))['status'], 'satisfied_under_symbolic_hypothesis')
        bad = copy.deepcopy(state)
        bad['backend']['robot_position'] = [math.nan, 0., 0.]
        self.assertEqual(self.check(bad, predicate=goal('robot'))['status'], 'unknown')
        bindings = copy.deepcopy(descriptor['alignment']['context_bindings'])
        bindings['office']['element_ids'] = [13, 62]
        self.assertEqual(self.check(state, bindings, goal('robot'))['reason'], 'robot_near_requires_single_destination')

    def test_optional_unknown_goal_is_reported_not_marked_satisfied(self):
        descriptor, library = row([('navigate_to_object', 13)], held=False)
        state = initial_state(nodes())
        state['backend']['objects']['67']['center'] = None
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal(required=False)]
        report = review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual(report['goal_checks'][-1]['status'], 'unknown')
        self.assertFalse(report['goal_checks'][-1]['required'])

    def test_required_unknown_position_fails_closed(self):
        descriptor, library = row([('navigate_to_object', 13)], held=False)
        state = initial_state(nodes())
        state['backend']['objects']['67']['center'] = None
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal()]
        with self.assertRaises(SymbolicGoalError) as caught:
            review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual(caught.exception.details['status'], 'unknown')
        self.assertEqual(caught.exception.details['reason'], 'missing_or_invalid_position')

    def test_other_unsupported_required_relation_still_rejects(self):
        descriptor, library, state = self.setup_world()
        invalid = dict(goal(), relation='under')
        descriptor['alignment']['task_decomposition']['goal_state'] += [invalid]
        with self.assertRaises(SymbolicGoalError) as caught:
            review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual(caught.exception.details['reason'], 'unsupported_goal_relation')


if __name__ == '__main__':
    unittest.main()
