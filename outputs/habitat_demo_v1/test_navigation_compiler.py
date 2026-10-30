import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_json_long_horizon import row, nodes
from run_json_long_horizon import initial_state
from run_navigation_demo import read_json, write_json
from navigation_compiler import compile_navigation, explicit_visits
from match_skills import build_result
from json_symbolic_review import review_symbolic
import plan_live_round as worker


def fixture(targets=(62,), instruction=None):
    descriptor, library = row([('navigate_to_object', i) for i in targets])
    alignment = descriptor['alignment']
    refs = {62: 'kitchen', 67: 'kettle', 13: 'office'}
    instruction = instruction or ', then '.join(f'visit ID {i}' for i in targets) + '.'
    roles = {refs[i]: {'role_id': refs[i], 'quantity': 1, 'required': True} for i in targets}
    alignment['context_bindings'] = {refs[i]: {'element_ids': [i], 'alternatives': []} for i in targets}
    alignment['task_decomposition'].update(instruction=instruction, task_roles=list(roles.values()),
        subtasks=[{'subtask_id': f'st_{j:02}', 'description': f'Visit ID {i}.',
                   'dependencies': [] if j == 1 else [f'st_{j-1:02}'],
                   'required_roles': [roles[refs[i]]]} for j, i in enumerate(targets, 1)])
    descriptor.update(instruction=instruction, expected_navigation_targets=list(targets))
    return descriptor, library, initial_state(nodes())


class NavigationCompilerTests(unittest.TestCase):
    def compile(self, descriptor, library, snapshot, scene=None):
        return compile_navigation(descriptor['alignment'], library, descriptor['instruction'],
                                  nodes() if scene is None else scene, snapshot)

    def test_saved_failure_instruction_shapes(self):
        for instruction, target, description in (
            ('Visit kitchen cabinet ID 62, ', 62, 'Visit the kitchen cabinet by moving near it.'),
            ('then visit kettle ID 67.', 67, 'Go near kettle ID 67.'),
            ('Visit open-office sideboard ID 13.', 13, 'Visit the open-office sideboard.')):
            with self.subTest(instruction=instruction):
                d, lib, state = fixture((target,), instruction)
                d['alignment']['task_decomposition']['subtasks'][0]['description'] = description
                proposal, audit = self.compile(d, lib, state)
                self.assertIsNotNone(proposal)
                self.assertEqual(audit['route'], 'compiled')
                d['plan'] = build_result(d['alignment'], lib, proposal)
                self.assertEqual(review_symbolic(d, lib, nodes(), state, 1.5)['navigation_targets'], [target])

    def test_explicit_order_and_repetition_not_deduplicated(self):
        for targets in ((62, 13, 62), (13, 13)):
            d, lib, state = fixture(targets)
            original = copy.deepcopy((d, lib, state))
            proposal, audit = self.compile(d, lib, state)
            self.assertEqual(audit['ordered_requested_ids'], list(targets))
            d['plan'] = build_result(d['alignment'], lib, proposal)
            result = review_symbolic(d, lib, nodes(), state, 1.5)
            self.assertEqual(result['navigation_targets'], list(targets))
            self.assertEqual((lib, state), original[1:])
            self.assertFalse(result['simulation_executed'])
            self.assertFalse(result['training_eligible'])

    def test_mixed_negated_conditional_and_unspecified_commands_not_compiled(self):
        for instruction in ('Visit ID 62 and grasp ID 67.', 'Do not visit ID 62.',
            'If needed, visit ID 62.', 'Visit ID 62 while holding kettle ID 67.',
            'Visit and inspect cabinet ID 62.', 'Visit the cabinet.', 'Visit ID 62 twice.',
            'Visit ID 62. Delete ID 67.', 'Visit ID 62 unless blocked.',
            'Visit ID 62 then pick up the kettle.', 'Visit ID 62 or visit ID 13.',
            'Visit (ID 62.', 'Visit ID 62).'):
            with self.subTest(instruction=instruction):
                self.assertIsNone(explicit_visits(instruction))

    def test_parenthesized_id_and_prefix_supported(self):
        self.assertEqual(explicit_visits('First, visit cabinet (ID 62), and then visit ID 13.'), [62, 13])

    def test_wrong_subtask_order_is_not_compiled(self):
        d, lib, state = fixture((62, 13))
        d['alignment']['task_decomposition']['subtasks'].reverse()
        self.assertIsNone(self.compile(d, lib, state)[0])

    def test_initial_conditions_still_checked_after_assembly(self):
        d, lib, state = fixture()
        d['alignment']['task_decomposition']['initial_scene_constraints'] = [
            {'constraint_id': 'initial_01', 'subject_role': 'robot', 'relation': 'near',
             'object_role': 'kitchen', 'required': True}]
        state['backend']['robot_position'] = [100., .1, 100.]
        proposal, audit = self.compile(d, lib, state)
        self.assertIsNotNone(proposal)
        d['plan'] = build_result(d['alignment'], lib, proposal)
        with self.assertRaises(ValueError):
            review_symbolic(d, lib, nodes(), state, 1.5)

    def test_disabled_feature_keeps_llm_proposal_and_critique(self):
        d, lib, state = fixture()
        proposal, _ = self.compile(d, lib, state)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, value in (('round_input.json', d), ('runtime_state.json', state),
                ('execution_skill_library.json', lib), ('input_memory/scene/scene_graph.json', nodes())):
                write_json(root/filename, value)
            args = SimpleNamespace(output_dir=root, round_input=root/'round_input.json', model='offline',
                base_url='https://invalid.test', reach=1.5)
            with patch.object(worker, 'decompose_and_bind', return_value=d['alignment']), \
                 patch.object(worker, 'request_plan', return_value=proposal) as propose, \
                 patch.object(worker, 'request_order', return_value={'invalid': 'keep original'}) as critique:
                self.assertEqual(worker.plan(args, reviewer=review_symbolic), 0)
            propose.assert_called_once()
            critique.assert_called_once()
            self.assertEqual(read_json(root/'skill_plan_final.json')['proposal_origin'], 'fresh_llm_skill_proposal')

    def test_unique_binding_not_guessed_from_instruction(self):
        for ids in ([67], [62, 67], [True], ['62'], []):
            d, lib, state = fixture()
            d['alignment']['context_bindings']['kitchen']['element_ids'] = ids
            self.assertIsNone(self.compile(d, lib, state)[0])
        d, lib, state = fixture()
        d['alignment']['context_bindings']['kitchen']['alternatives'] = [67]
        self.assertIsNone(self.compile(d, lib, state)[0])

    def test_deleted_held_or_unsupported_navigation_not_compiled(self):
        d, lib, state = fixture((67,))
        self.assertIsNone(self.compile(d, lib, state, [n for n in nodes() if n['id'] != 67])[0])
        state['held_object_id'] = 67
        self.assertIsNone(self.compile(d, lib, state)[0])
        state['held_object_id'] = None
        lib['execution_contract']['supported_skill_ids'].remove('navigate_to_object')
        self.assertIsNone(self.compile(d, lib, state)[0])

    def test_mismatched_decomposition_and_goal_not_replaced(self):
        for change in ('instruction', 'description', 'goals', 'unused_role', 'subtask_count'):
            d, lib, state = fixture()
            decomposition = d['alignment']['task_decomposition']
            if change == 'instruction': decomposition['instruction'] = 'Other instruction'
            if change == 'description': decomposition['subtasks'][0]['description'] = 'Visit ID 62 and inspect it.'
            if change == 'goals': decomposition['goal_state'][0]['relation'] = 'on'
            if change == 'unused_role': d['alignment']['context_bindings']['unused'] = {'element_ids': [67]}
            if change == 'subtask_count': decomposition['subtasks'].append(copy.deepcopy(decomposition['subtasks'][0]))
            self.assertIsNone(self.compile(d, lib, state)[0], change)

    def test_pipeline_compilation_preserves_snapshot_and_skips_two_api_stages(self):
        d, lib, state = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, value in (('round_input.json', d), ('runtime_state.json', state),
                ('execution_skill_library.json', lib), ('input_memory/scene/scene_graph.json', nodes())):
                write_json(root / filename, value)
            args = SimpleNamespace(output_dir=root, round_input=root/'round_input.json',
                model='offline', base_url='https://invalid.test', reach=1.5, compile_navigation=True)
            with patch.object(worker, 'decompose_and_bind', return_value=d['alignment']), \
                 patch.object(worker, 'request_plan') as propose, patch.object(worker, 'request_order') as critique:
                self.assertEqual(worker.plan(args, reviewer=review_symbolic), 0)
            propose.assert_not_called()
            critique.assert_not_called()
            self.assertEqual(read_json(root/'runtime_state.json'), state)
            final = read_json(root/'skill_plan_final.json')
            self.assertEqual(final['proposal_origin'], 'compiled_validated_navigation')
            self.assertFalse(list(root.glob('llm_skill_proposal*')))
            self.assertFalse((root/'llm_redundancy_proposal.json').exists())
            self.assertEqual(read_json(root/'planning_summary.json')['navigation_compiler']['skill_proposal_llm_requests'], 0)

    def test_compiled_plan_review_failure_stops_without_fallback(self):
        d, lib, state = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, value in (('round_input.json', d), ('runtime_state.json', state),
                ('execution_skill_library.json', lib), ('input_memory/scene/scene_graph.json', nodes())):
                write_json(root/filename, value)
            args = SimpleNamespace(output_dir=root, round_input=root/'round_input.json', model='offline',
                base_url='https://invalid.test', reach=1.5, compile_navigation=True)
            def reject(*args): raise ValueError('Independent runtime review rejected fixture')
            with patch.object(worker, 'decompose_and_bind', return_value=d['alignment']), \
                 patch.object(worker, 'request_plan') as propose, patch.object(worker, 'request_order') as critique:
                self.assertEqual(worker.plan(args, reviewer=reject), 2)
            propose.assert_not_called()
            critique.assert_not_called()
            self.assertIn('runtime review rejected', read_json(root/'planning_summary.json')['error'])


if __name__ == '__main__':
    unittest.main()
