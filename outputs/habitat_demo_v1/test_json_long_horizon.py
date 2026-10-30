import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_json_long_horizon import instructions, initial_state, project_scene, STATE_SOURCE
from json_symbolic_review import review_symbolic, SymbolicBackend
from run_continuous_sequence import OUT
from run_navigation_demo import read_json
from match_skills import build_result
from action_sequence import refresh_sequence
from sequential_proxy_core import PARAMETERS
import run_json_long_horizon as workflow
import replay_json_round_planning as replay_module
from run_navigation_demo import write_json


def nodes():
    # Synthetic unit fixture only; not a Scene201 geometry or perception result.
    return [{'id': i, 'object_tag': name, 'bbox_center': pos, 'bbox_extent': [.5, 1., .5],
             'json_test_context': {'room_hypothesis': room}} for i, name, pos, room in (
        (67, 'kettle', [0., 8., 1.], 'kitchen'),
        (13, 'sideboard', [0., 0., .5], 'open_office'),
        (62, 'cabinet', [0., 7., .5], 'kitchen'))]


def row(steps, support=None, held=False, visits=None):
    refs = {67: 'kettle', 13: 'office', 62: 'kitchen'}
    instruction = 'Carry the held kettle and place it.' if support is not None else 'Visit destinations.'
    if any(s[0] == 'grasp_object' for s in steps):
        instruction = 'Pick up the kettle, lift it, and carry it. Place it if requested.'
    goal = {'constraint_id': 'goal_01', 'subject_role': 'kettle' if support else 'robot',
            'relation': 'on' if support else 'near', 'object_role': refs[support or steps[-1][1]],
            'required': True}
    alignment = {'status': 'success', 'context_bindings': {role: {'element_ids': [i]} for i, role in refs.items()},
        'task_decomposition': {'instruction': instruction, 'subtasks': [{'subtask_id': 'st_01',
            'description': instruction, 'dependencies': []}], 'initial_scene_constraints': [], 'goal_state': [goal]}}
    actions = []
    for item in steps:
        skill, i = item[:2]
        key = 'object' if skill in ('lift_held_object', 'place_object_on_surface') else 'target'
        args = {key: {'role_id': refs[i], 'element_id': i}}
        if len(item) == 3:
            args['surface'] = {'role_id': refs[item[2]], 'element_id': item[2]}
        actions.append({'skill_id': skill, 'arguments': args})
    library = read_json(OUT / 'quadruped_atomic_skill_library.v0.1.json')
    library['skills'] = [s for s in library['skills'] if s['skill_id'] in PARAMETERS]
    library['execution_contract'] = {'supported_skill_ids': list(PARAMETERS), 'fixed_order_required': False}
    plan = build_result(alignment, library, {'subtask_plans': [{'subtask_id': 'st_01',
        'intent': instruction, 'steps': actions}]})
    descriptor = {'instruction': instruction, 'alignment': alignment, 'plan': plan,
                  'expected_end': {'held_object_id': 67 if held else None, 'lifted': held}}
    if support is not None:
        descriptor['expected_support_id'] = support
    if visits is not None:
        descriptor['expected_navigation_targets'] = visits
    return descriptor, library


class JSONLongTests(unittest.TestCase):
    def chain_fixture(self, command, log):
        # Hand-authored unit-test responses only; no actual LLM calls or observations.
        root = Path(command[command.index('--output-dir')+1])
        descriptor = read_json(root / 'round_input.json')
        routes = {
            1: [('navigate_to_object', i) for i in (67,13,67)],
            2: [('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67)],
            3: [('navigate_to_object',13)],
            4: [('position_arm_for_object',13),('place_object_on_surface',67,13)],
            5: [('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67),
                ('navigate_to_object',62),('position_arm_for_object',62),('place_object_on_surface',67,62)],
            6: [('navigate_to_object', i) for i in (13,67)],
            7: [('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67)],
            8: [('navigate_to_object',13),('position_arm_for_object',13),('place_object_on_surface',67,13)],
            9: [('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67),('navigate_to_object',62)],
            10: [('position_arm_for_object',62),('place_object_on_surface',67,62)],
            11: [('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67),
                 ('navigate_to_object',13),('position_arm_for_object',13),('place_object_on_surface',67,13)],
            12: [('navigate_to_object', i) for i in (67,13,67)],
        }
        fixture, library = row(routes[descriptor['round']])
        fixture['alignment']['task_decomposition']['instruction'] = descriptor['instruction']
        fixture['plan']['instruction'] = descriptor['instruction']
        # Fixture changed task identity: rebuild program-owned derived metadata.
        fixture['plan'] = refresh_sequence(fixture['plan'])
        state = read_json(root / 'runtime_state.json')
        write_json(root / 'object_alignment.json', fixture['alignment'])
        write_json(root / 'skill_plan_final.json', fixture['plan'])
        write_json(root / 'planning_summary.json', {'status': 'planning_passed',
            'validation_mode': 'json_symbolic_test',
            'input_memory_version': state['backend']['memory_version'], 'synthetic_test': True})

    def replay_fixture(self, directory):
        source = directory / 'old/round_8/planning'
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True)
        state['backend']['memory_version'] = 7
        workflow.prepare_round(source, instructions()[7], state, nodes(), 'test', 'https://invalid.test')
        write_json(source / 'original_failure.json', {'steps': [{}]})
        args = SimpleNamespace(replay_planning_dir=source, output_dir=directory/'new',
                               model='test', base_url='https://invalid.test')
        return source, args

    def test_single_round_replay_copies_input_and_keeps_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source, args = self.replay_fixture(Path(directory))
            before = replay_module.checkpoint_hashes(source)
            with patch.object(replay_module, 'execute_worker', side_effect=self.chain_fixture) as child:
                self.assertEqual(replay_module.replay(args), 0)
                self.assertEqual(child.call_count, 1)
            summary = read_json(args.output_dir / 'replay_summary.json')
            self.assertEqual(summary['status'], 'json_planning_replay_passed')
            self.assertEqual(summary['source_round'], 8)
            self.assertEqual(summary['input_projection_revision'], 7)
            self.assertFalse(summary['completed_chain_claim'])
            self.assertFalse(summary['simulation_executed'])
            self.assertTrue(summary['source_checkpoint_unchanged'])
            self.assertEqual(replay_module.checkpoint_hashes(source), before)
            self.assertEqual(read_json(source/'original_failure.json'), {'steps': [{}]})
            self.assertFalse((args.output_dir / 'round_9').exists())
            with self.assertRaises(FileExistsError):
                replay_module.replay(args)

    def test_round_replay_failing_worker_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args = self.replay_fixture(Path(directory))
            with patch.object(replay_module, 'execute_worker', side_effect=ValueError('fixture API error')):
                self.assertEqual(replay_module.replay(args), 2)
            summary = read_json(args.output_dir/'replay_summary.json')
            self.assertEqual(summary['status'], 'failed')
            self.assertNotIn('action_count', summary)
            self.assertFalse((args.output_dir/'planning/skill_plan_final.json').exists())

    def test_round_replay_rejects_changed_graph_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source, args = self.replay_fixture(Path(directory))
            path = source/'input_memory/scene/scene_graph.json'
            changed = read_json(path)
            changed[0]['bbox_center'][0] += 10
            write_json(path, changed)
            with patch.object(replay_module, 'execute_worker') as child:
                with self.assertRaisesRegex(ValueError, 'graph differs'):
                    replay_module.replay(args)
                child.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_round_replay_rejects_observed_checkpoint_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source, args = self.replay_fixture(Path(directory))
            state = read_json(source/'runtime_state.json')
            state['state_source'] = 'simulator_observation'
            write_json(source/'runtime_state.json', state)
            with patch.object(replay_module, 'execute_worker') as child:
                with self.assertRaisesRegex(ValueError, 'mode/state/contract mismatch'):
                    replay_module.replay(args)
                child.assert_not_called()

    def args(self, directory):
        base = directory / 'base.json'
        verified = directory / 'verified.json'
        write_json(base, nodes())
        write_json(verified, [n for n in nodes() if n['id'] != 62])
        return SimpleNamespace(base_graph=base, verified_graph=verified, output_dir=directory/'run',
            preflight_only=False, model='test', base_url='https://invalid.test')

    def test_mocked_full_chain_keeps_projection_revisions_and_no_simulation_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            with patch.object(workflow, 'execute_worker', side_effect=self.chain_fixture) as child:
                self.assertEqual(workflow.run(args), 0)
                self.assertEqual(child.call_count, 12)
            result = read_json(args.output_dir / 'regression_summary.json')
            self.assertFalse(result['simulation_executed'])
            self.assertTrue(result['source_files_unchanged'])
            self.assertEqual(result['completed_round_count'], 12)
            for previous, current in zip(result['rounds'], result['rounds'][1:]):
                self.assertEqual(previous['end_state'], current['begin_state'])
            self.assertEqual(result['rounds'][-1]['end_state']['backend']['memory_version'], 12)

    def test_failed_round_stops_later_json_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            def child(command, log):
                root = Path(command[command.index('--output-dir')+1])
                if read_json(root / 'round_input.json')['round'] == 3:
                    raise ValueError('synthetic test failure')
                self.chain_fixture(command, log)
            with patch.object(workflow, 'execute_worker', side_effect=child) as mocked:
                self.assertEqual(workflow.run(args), 2)
                self.assertEqual(mocked.call_count, 3)
            result = read_json(args.output_dir / 'regression_summary.json')
            self.assertEqual(result['completed_round_count'], 2)
            self.assertEqual(result['status'], 'failed')
            self.assertFalse((args.output_dir / 'round_4').exists())

    def test_twelve_rounds_have_no_prewritten_skills_and_five_placement_assertions(self):
        rows = instructions()
        self.assertEqual([r['round'] for r in rows], list(range(1, 13)))
        self.assertTrue(all('plan' not in r and 'steps' not in r for r in rows))
        self.assertEqual([rows[i-1]['expected_support_id'] for i in (4, 5, 8, 10, 11)],
                         [13, 62, 13, 62, 13])

    def test_cross_room_pickup_carry_release_and_return_are_symbolic_only(self):
        state = initial_state(nodes())
        saved = copy.deepcopy(state)
        sequences = [
            ([("navigate_to_object",67),("position_arm_for_object",67),("grasp_object",67),("lift_held_object",67)], None, True),
            ([("navigate_to_object",13)], None, True),
            ([("position_arm_for_object",13),("place_object_on_surface",67,13)], 13, False),
            ([("position_arm_for_object",67),("grasp_object",67),("lift_held_object",67),
              ("navigate_to_object",62),("position_arm_for_object",62),("place_object_on_surface",67,62)], 62, False),
        ]
        for steps, support, held in sequences:
            descriptor, library = row(steps, support, held)
            report = review_symbolic(descriptor, library, nodes(), state, 1.5)
            self.assertFalse(report['simulation_executed'])
            self.assertFalse(report['geometry_verified'])
            self.assertFalse(report['training_eligible'])
            self.assertTrue(all(t['execution_status'] == 'not_executed' for t in report['symbolic_trace']))
            state = report['predicted_end_state']
            self.assertEqual(state['state_source'], STATE_SOURCE)
        self.assertEqual(state['backend']['objects']['67']['support_id'], 62)
        self.assertEqual(state['backend']['objects']['67']['room_hypothesis'], 'kitchen')
        self.assertIsNone(saved['held_object_id'])
        self.assertEqual(saved, initial_state(nodes()))

    def test_no_grasp_cannot_satisfy_holding(self):
        descriptor, library = row([('navigate_to_object',13)], held=True)
        with self.assertRaisesRegex(ValueError, 'outcome mismatch'):
            review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)

    def test_wrong_surface_fails_outcome_even_if_valid_skill(self):
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True)
        descriptor, library = row([('navigate_to_object',62), ('position_arm_for_object',62),
            ('place_object_on_surface',67,62)], support=62)
        descriptor['expected_support_id'] = 13
        with self.assertRaisesRegex(ValueError, 'Wrong requested destination'):
            review_symbolic(descriptor, library, nodes(), state, 1.5)

    def test_held_object_cannot_be_navigation_target(self):
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True)
        descriptor, library = row([('navigate_to_object',67)], held=True)
        with self.assertRaisesRegex(ValueError, 'HELD_OBJECT_UNDEFINED'):
            review_symbolic(descriptor, library, nodes(), state, 1.5)

    def test_visit_order_cannot_be_deleted(self):
        descriptor, library = row([('navigate_to_object',13)], visits=[13,62,13])
        with self.assertRaisesRegex(ValueError, 'visit order'):
            review_symbolic(descriptor, library, nodes(), initial_state(nodes()), 1.5)

    def test_projection_is_separate_unverified_and_does_not_touch_original_nodes(self):
        original = nodes()
        saved = copy.deepcopy(original)
        state = initial_state(original)
        state['backend']['objects']['67'].update(center=[0.,1.,0.], support_id=13)
        projected, edges = project_scene(original, state)
        self.assertEqual(original, saved)
        kettle = next(n for n in projected if n['id']==67)
        self.assertFalse(kettle['position_evidence']['verified'])
        self.assertEqual(edges[0]['evidence_source'], STATE_SOURCE)
        self.assertFalse(edges[0]['verified'])

    def test_observed_state_cannot_be_accepted_as_test_hypothesis(self):
        state = initial_state(nodes())
        state['state_source'] = 'simulator_observation'
        with self.assertRaisesRegex(ValueError, 'explicitly marked hypothetical'):
            SymbolicBackend(state, nodes(), 1.5)


if __name__ == '__main__':
    unittest.main()
