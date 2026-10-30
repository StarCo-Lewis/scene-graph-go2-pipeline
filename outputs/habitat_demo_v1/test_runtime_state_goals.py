"""Offline state-goal protocol tests; no model, network or Habitat required."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_json_long_horizon import nodes, row
from test_live_llm_sequence import inputs
from test_continuous_sequence import FakeBackend
from run_json_long_horizon import initial_state
from json_symbolic_review import review_symbolic, SymbolicGoalError
from live_plan_review import review_live_plan
from run_continuous_sequence import evaluate_goals, build_fixture_episode, validate_episode, contract
from sequential_proxy_core import SerialExecutor
from runtime_goal_protocol import (check_state_goal, validate_state_goal_set, normalize_state_goal,
                                   PROMPT, CONTRACT)
from align_objects import validate_constraints, validate_decomposition, AlignmentError
from decomposition_wire import from_flat_roles, FLAT_PROMPT
from action_matrix import make_codebook, decode_matrix
from match_skills import build_result


def goal(relation='held', expected=True, role='kettle', required=True):
    return {'constraint_id': relation+'_'+role, 'subject_role': 'robot', 'relation': relation,
            'object_role': role, 'expected_value': expected, 'required': required}


class RuntimeStateGoalsTests(unittest.TestCase):
    def check(self, predicate=None, state=None, bindings=None, source='symbolic_hypothesis_not_observation'):
        return check_state_goal(predicate or goal(), bindings if bindings is not None else
            {'kettle': {'element_ids': [67]}}, state if state is not None else
            {'held_object_id': 67, 'lifted': True}, {67, 13, 62}, source)

    def test_legacy_positive_state_goals_normalized_without_inference(self):
        predicate = goal()
        predicate.pop('expected_value')
        normalized = validate_constraints([predicate], 'goal_state', {'kettle'})
        self.assertIs(normalized[0]['expected_value'], True)
        self.assertEqual(predicate.get('expected_value'), None)

    def test_state_initial_assertions_accepted_but_are_not_graph_facts(self):
        from llm_graph_binding import partition_initial_constraints
        for predicate in (goal(), goal('lifted')):
            with self.subTest(predicate=predicate):
                value=validate_constraints([predicate], 'initial_scene_constraints', {'kettle'})
                graph,runtime=partition_initial_constraints(value)
                self.assertEqual(graph,[])
                self.assertEqual(runtime,value)

    def test_unknown_state_words_actors_roles_and_evidence_rejected(self):
        for changes in ({'relation': 'holding'}, {'relation': 'held_by'}, {'subject_role': 'kettle'},
                        {'object_role': 'unknown'}, {'verified': True}, {'required': 'false'},
                        {'expected_value': 'true'}, {'expected_value': 1}):
            with self.subTest(changes=changes), self.assertRaises(AlignmentError):
                validate_constraints([dict(goal(), **changes)], 'goal_state', {'kettle'})

    def test_spatial_predicates_cannot_silently_drop_negation(self):
        with self.assertRaises(AlignmentError):
            validate_constraints([goal('near', False)], 'goal_state', {'kettle'})

    def test_binding_requires_unique_active_integer(self):
        for ids in ([], [67, 13], [67, 67], [True], ['67'], [999], (67,)):
            with self.subTest(ids=ids):
                self.assertEqual(self.check(bindings={'kettle': {'element_ids': ids}})['status'], 'unknown')

    def test_missing_or_invalid_state_never_satisfies_negative_goal(self):
        for state in ({}, {'held_object_id': True}, {'held_object_id': '67'}, {'held_object_id': 999},
                      {'held_object_id': None, 'lifted': True}, {'held_object_id': 67, 'lifted': 'true'}):
            with self.subTest(state=state):
                self.assertEqual(self.check(goal(expected=False), state)['status'], 'unknown')
        self.assertEqual(self.check(goal('lifted', False), {'held_object_id': 67})['status'], 'unknown')

    def test_hold_identity_and_lift_flag_are_distinct(self):
        self.assertEqual(self.check(state={'held_object_id': 67, 'lifted': False})['actual_value'], True)
        self.assertEqual(self.check(goal('lifted'), {'held_object_id': 67, 'lifted': False})['status'], 'not_satisfied')
        self.assertEqual(self.check(goal('lifted'), {'held_object_id': 13, 'lifted': True})['status'], 'not_satisfied')

    def test_negative_goal_is_object_specific_not_gripper_empty(self):
        check = self.check(goal(expected=False), {'held_object_id': 13, 'lifted': True})
        self.assertEqual(check['status'], 'satisfied_under_symbolic_hypothesis')
        self.assertEqual(check['executor_state']['held_object_id'], 13)

    def test_sources_are_explicit_and_never_physical_grasp_evidence(self):
        for source, status in (('symbolic_hypothesis_not_observation', 'satisfied_under_symbolic_hypothesis'),
                               ('read_only_runtime_prediction', 'satisfied_under_runtime_prediction'),
                               ('proxy_executor_readback', 'observed_in_proxy')):
            check = self.check(source=source)
            self.assertEqual(check['status'], status)
            self.assertEqual(check['source'], source)
            self.assertFalse(check['physical_grasp_verified'])
        with self.assertRaises(ValueError):
            self.check(source='model_claim')

    def test_conflicting_required_goals_rejected_including_aliases(self):
        for predicates in ([goal(), goal(expected=False)], [goal('lifted'), goal(expected=False)],
                           [goal(), goal('lifted', role='office')]):
            with self.subTest(predicates=predicates), self.assertRaises(ValueError):
                validate_state_goal_set(predicates)
        with self.assertRaises(ValueError):
            validate_state_goal_set([goal(), goal(expected=False, role='alias')],
                                    {'kettle': {'element_ids': [67]}, 'alias': {'element_ids': [67]}})
        validate_state_goal_set([goal(), goal('lifted', False)])  # held but not lifted is valid.

    def pickup(self, lifted=True):
        steps = [('navigate_to_object', 67), ('position_arm_for_object', 67), ('grasp_object', 67)]
        if lifted:
            steps.append(('lift_held_object', 67))
        descriptor, library = row(steps, held=True)
        descriptor.pop('expected_end')  # Completion must work WITHOUT hand-authored test assertions.
        descriptor['alignment']['task_decomposition']['goal_state'] = [goal(), goal('lifted')] if lifted else [goal()]
        return descriptor, library

    def test_symbolic_hold_lift_without_expected_end(self):
        descriptor, library = self.pickup()
        state = initial_state(nodes())
        saved = copy.deepcopy((descriptor, state))
        report = review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual([c['actual_value'] for c in report['goal_checks']], [True, True])
        self.assertFalse(report['simulation_executed'])
        self.assertFalse(report['training_eligible'])
        self.assertEqual((descriptor, state), saved)

    def test_matrix_decoding_and_nested_json_check_same_state_goals(self):
        descriptor, library = self.pickup()
        baseline = review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        codebook = make_codebook(descriptor['alignment'], library)
        object_code = next(r['code'] for r in codebook['objects'] if r['element_id'] == 67 and r['role_id'] == 'kettle')
        skills = {r['skill_id']: r['code'] for r in codebook['skills']}
        matrix = {'matrix': [[1, skills[s['skill_id']], object_code, 0]
                            for s in descriptor['plan']['subtask_plans'][0]['steps']]}
        descriptor['plan'] = build_result(descriptor['alignment'], library,
            decode_matrix(matrix, codebook, descriptor['alignment'], library))
        revised = review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        self.assertEqual(baseline['goal_checks'], revised['goal_checks'])

    def test_near_without_grasp_fails_explicit_held_goal(self):
        descriptor, library = row([('navigate_to_object', 67)])
        descriptor.pop('expected_end')
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal()]
        with self.assertRaises(SymbolicGoalError) as error:
            review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        self.assertEqual(error.exception.details['relation'], 'held')

    def test_grasp_without_lift_fails_explicit_lift_goal(self):
        descriptor, library = self.pickup(lifted=False)
        descriptor['alignment']['task_decomposition']['goal_state'].append(goal('lifted'))
        with self.assertRaises(SymbolicGoalError) as error:
            review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        self.assertEqual(error.exception.details['relation'], 'lifted')

    def test_placement_release_checked_without_expected_end(self):
        descriptor, library = row([('navigate_to_object', 13), ('position_arm_for_object', 13),
                                   ('place_object_on_surface', 67, 13)], support=13)
        descriptor.pop('expected_end')
        descriptor['alignment']['task_decomposition']['goal_state'] += [goal(expected=False), goal('lifted', False)]
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True)
        report = review_symbolic(descriptor, library, nodes(), state, 1.5)
        self.assertEqual([g['actual_value'] for g in report['goal_checks'][1:]], [False, False])
        self.assertIsNone(report['predicted_end_state']['held_object_id'])

    def test_optional_unsatisfied_goal_reported_not_fabricated(self):
        descriptor, library = row([('navigate_to_object', 67)])
        descriptor['alignment']['task_decomposition']['goal_state'].append(goal(required=False))
        report = review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)
        self.assertEqual(report['goal_checks'][-1]['status'], 'not_satisfied')

    def test_live_rehearsal_checks_state_without_expected_end(self):
        library, state, graph = inputs()
        descriptor = build_fixture_episode(library)['rounds'][1]
        descriptor.pop('expected_end')
        descriptor['alignment']['task_decomposition']['goal_state'] = [goal(), goal('lifted')]
        report = review_live_plan(descriptor, library, graph, state, 1.5)
        self.assertTrue(all(g['status'] == 'satisfied_under_runtime_prediction' for g in report['goal_checks']))

    def test_runtime_readback_reads_executor_not_geometry_or_model(self):
        backend = FakeBackend()
        executor = SerialExecutor(backend)
        alignment = {'context_bindings': {'kettle': {'element_ids': [67]}},
                     'task_decomposition': {'goal_state': [goal(), goal('lifted')]}}
        checks = evaluate_goals(alignment, backend, executor)
        self.assertEqual([c['status'] for c in checks], ['not_satisfied', 'not_satisfied'])
        executor.held_object_id, executor.lifted = 67, True
        checks = evaluate_goals(alignment, backend, executor)
        self.assertEqual([c['status'] for c in checks], ['observed_in_proxy', 'observed_in_proxy'])
        self.assertTrue(all(not c['physical_grasp_verified'] for c in checks))
        self.assertEqual(evaluate_goals(alignment, backend)[0]['status'], 'unknown')

    def test_runtime_episode_rejects_inactive_and_multi_object_state_goals(self):
        library, _, graph = inputs()
        for ids in ([999], [67, 13], [True]):
            descriptor = build_fixture_episode(library)['rounds'][1]
            descriptor['alignment']['task_decomposition']['goal_state'] = [goal(role='bad')]
            descriptor['alignment']['context_bindings']['bad'] = {'element_ids': ids}
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_episode({'rounds': [descriptor]}, library, graph)

    def test_prompts_and_contract_advertise_same_protocol(self):
        from strict_prompt_v2 import CONTRACT as strict
        self.assertIn('proxy_state_goals_v1', FLAT_PROMPT)
        self.assertIn('robot held OBJECT_ROLE', strict)
        self.assertEqual(contract()['goal_protocol'], CONTRACT)
        self.assertNotIn('near only', FLAT_PROMPT)

    def flat(self):
        from test_decomposition_wire import sample
        value = sample()
        value['goal_state'] = [goal(), goal('lifted')]
        return value

    def test_flat_role_state_goals_expand_without_task_or_action_changes(self):
        original = self.flat()
        saved = copy.deepcopy(original)
        decomposition = validate_decomposition(from_flat_roles(original))
        self.assertEqual(decomposition['goal_state'], original['goal_state'])
        self.assertEqual(len(decomposition['subtasks']), len(original['subtasks']))
        self.assertEqual(original, saved)

    def test_state_role_quantity_more_than_one_rejected_before_binding(self):
        value = self.flat()
        value['object_roles'][0]['quantity'] = 2
        with self.assertRaisesRegex(AlignmentError, 'quantity=1'):
            validate_decomposition(from_flat_roles(value))


if __name__ == '__main__':
    unittest.main(verbosity=2)
