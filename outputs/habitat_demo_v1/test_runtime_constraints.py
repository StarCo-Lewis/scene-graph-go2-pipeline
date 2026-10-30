import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_constraints import check_runtime_initial_constraints, RuntimeConstraintError
import test_live_llm_sequence as live_tests
inputs = live_tests.inputs
from run_continuous_sequence import build_fixture_episode
from live_plan_review import review_live_plan
import run_continuous_sequence as continuous


def initial_near(required=True):
    return {'constraint_id': 'robot_near_kettle', 'subject_role': 'robot',
        'relation': 'near', 'object_role': 'kettle', 'required': required}


class RuntimeConstraintTests(unittest.TestCase):
    def test_near_uses_habitat_horizontal_axes_not_graph_edges_or_vertical_height(self):
        library, snapshot, nodes = inputs()
        row = build_fixture_episode(library)['rounds'][1]
        row['alignment']['task_decomposition']['initial_scene_constraints'] = [initial_near()]
        original = copy.deepcopy(snapshot)
        report = review_live_plan(row, library, nodes, snapshot, 1.5)
        self.assertEqual(report['initial_runtime_constraint_checks'][0]['status'], 'confirmed_in_runtime_snapshot')
        self.assertEqual(report['initial_runtime_constraint_checks'][0]['distances_m'], [0.])
        self.assertFalse(report['initial_runtime_constraint_checks'][0]['geometry_or_navigation_verified'])
        self.assertEqual(snapshot, original)

    def test_false_initial_near_rejected_before_future_navigation_could_establish_it(self):
        library, snapshot, _ = inputs()
        row = build_fixture_episode(library)['rounds'][3]
        row['alignment']['task_decomposition']['initial_scene_constraints'] = [initial_near()]
        snapshot['backend']['robot_position'][0] += 10
        with self.assertRaises(RuntimeConstraintError) as error:
            check_runtime_initial_constraints(row['alignment'], snapshot, 1.5)
        self.assertEqual(error.exception.details['status'], 'not_satisfied')

    def test_missing_and_nonfinite_positions_unknown_not_guessed(self):
        library, snapshot, _ = inputs()
        alignment = build_fixture_episode(library)['rounds'][1]['alignment']
        alignment['task_decomposition']['initial_scene_constraints'] = [initial_near()]
        for position in (None, [0., float('nan'), 0.], [0., 0., float('inf')], [True, 0., 0.]):
            saved = copy.deepcopy(snapshot)
            saved['backend']['robot_position'] = position
            with self.subTest(position=position), self.assertRaises(RuntimeConstraintError) as error:
                check_runtime_initial_constraints(alignment, saved, 1.5)
            self.assertEqual(error.exception.details['status'], 'unknown')
        saved = copy.deepcopy(snapshot)
        saved['backend']['objects'].pop('67')
        with self.assertRaises(RuntimeConstraintError):
            check_runtime_initial_constraints(alignment, saved, 1.5)

    def test_unbound_role_not_inferred_from_held_pointer(self):
        library, snapshot, _ = inputs()
        alignment = build_fixture_episode(library)['rounds'][1]['alignment']
        alignment['task_decomposition']['initial_scene_constraints'] = [initial_near()]
        alignment['context_bindings'].pop('kettle')
        snapshot.update(held_object_id=67, lifted=True)
        with self.assertRaises(RuntimeConstraintError):
            check_runtime_initial_constraints(alignment, snapshot, 1.5)

    def test_optional_initial_near_recorded_but_not_enforced_and_no_robot_means_no_new_requirement(self):
        library, snapshot, _ = inputs()
        alignment = build_fixture_episode(library)['rounds'][1]['alignment']
        self.assertEqual(check_runtime_initial_constraints(alignment, snapshot, 1.5), [])
        alignment['task_decomposition']['initial_scene_constraints'] = [initial_near(False)]
        snapshot['backend']['robot_position'][0] += 10
        self.assertEqual(check_runtime_initial_constraints(alignment, snapshot, 1.5)[0]['status'], 'not_satisfied')

    def test_object_on_relation_not_replaced_by_runtime_proximity(self):
        library, snapshot, _ = inputs()
        alignment = build_fixture_episode(library)['rounds'][1]['alignment']
        alignment['task_decomposition']['initial_scene_constraints'] = [
            dict(constraint_id='on', subject_role='kettle', relation='on', object_role='sideboard', required=True)]
        self.assertEqual(check_runtime_initial_constraints(alignment, snapshot, 1.5), [])

    def test_live_dispatch_initial_failure_runs_no_actions(self):
        # Reuse the fake continuous harness; no API or actual Habitat physics.
        details = {'constraint_id': 'fixture', 'status': 'not_satisfied'}
        calls = []
        def reject(alignment, snapshot, reach):
            calls.append(snapshot)
            raise RuntimeConstraintError(details)
        with patch.object(continuous, 'check_runtime_initial_constraints', side_effect=reject), \
                patch.object(continuous.SerialExecutor, 'execute') as dispatch, \
                self.assertRaises(RuntimeConstraintError):
            live_tests.LiveSequenceTests().exercise_live_loop()
        self.assertEqual(len(calls), 1)
        dispatch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
