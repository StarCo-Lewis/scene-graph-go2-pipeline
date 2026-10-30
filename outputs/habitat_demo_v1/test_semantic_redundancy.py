import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_live_llm_sequence import inputs
from run_continuous_sequence import build_fixture_episode
from run_navigation_demo import read_json, write_json
from match_skills import build_result
from live_plan_review import review_live_plan
from semantic_redundancy import SemanticRedundancyError, lift_repetition_policy
import plan_live_round as worker


def pickup_row(library, count=1, instruction=None, split=False):
    row = copy.deepcopy(build_fixture_episode(library)['rounds'][1])
    if instruction:
        row['alignment']['task_decomposition']['instruction'] = instruction
    steps = copy.deepcopy(row['plan']['subtask_plans'][0]['steps'])
    steps += [copy.deepcopy(steps[-1]) for _ in range(count - 1)]
    decomposition = row['alignment']['task_decomposition']
    if split:
        decomposition['subtasks'] = [dict(subtask_id=f'st_{i:02}', description=s['skill_id'],
            dependencies=[] if i == 1 else [f'st_{i-1:02}']) for i, s in enumerate(steps, 1)]
        proposal = {'subtask_plans': [dict(subtask_id=t['subtask_id'], intent=t['description'], steps=[s])
            for t, s in zip(decomposition['subtasks'], steps)]}
    else:
        proposal = {'subtask_plans': [dict(subtask_id='st_01', intent=decomposition['instruction'], steps=steps)]}
    row['plan'] = build_result(row['alignment'], library, proposal)
    return row


class SemanticRedundancyTests(unittest.TestCase):
    def test_double_lift_rejected_across_subtask_boundary_and_snapshot_unchanged(self):
        library, snapshot, nodes = inputs()
        original = copy.deepcopy(snapshot)
        for split in (False, True):
            row = pickup_row(library, 2, 'Pick up and lift the kettle again; keep holding it.', split)
            with self.subTest(split=split), self.assertRaises(SemanticRedundancyError) as caught:
                review_live_plan(row, library, nodes, snapshot, 1.5)
            self.assertEqual(caught.exception.details['code'], 'REDUNDANT_LIFT_ALREADY_LIFTED')
        self.assertEqual(snapshot, original)

    def test_navigation_does_not_reset_lift_cycle(self):
        library, snapshot, nodes = inputs()
        row = pickup_row(library, 2)
        steps = row['plan']['subtask_plans'][0]['steps']
        nav = copy.deepcopy(build_fixture_episode(library)['rounds'][2]['plan']['subtask_plans'][0]['steps'][0])
        steps.insert(-1, nav)
        row['plan'] = build_result(row['alignment'], library, {'subtask_plans': row['plan']['subtask_plans']})
        with self.assertRaises(SemanticRedundancyError):
            review_live_plan(row, library, nodes, snapshot, 1.5)

    def test_explicit_twice_preserved_but_no_physical_height_claim(self):
        library, snapshot, nodes = inputs()
        for instruction in ('Pick up the kettle and lift it twice.',
                            'Pick up the kettle and lift it two times.',
                            'Pick up the kettle, lift it and then lift it again.'):
            with self.subTest(instruction=instruction):
                report = review_live_plan(pickup_row(library, 2, instruction), library, nodes, snapshot, 1.5)
                repeats = report['semantic_redundancy']['explicit_proxy_repeat_calls']
                self.assertEqual(len(repeats), 1)
                self.assertFalse(repeats[0]['physical_incremental_lift_verified'])
                with self.assertRaises(SemanticRedundancyError):
                    review_live_plan(pickup_row(library, 3, instruction), library, nodes, snapshot, 1.5)
                with self.assertRaisesRegex(ValueError, 'not covered'):
                    review_live_plan(pickup_row(library, 1, instruction), library, nodes, snapshot, 1.5)

    def test_negation_does_not_authorize_repeat(self):
        for instruction in ('Lift it, not twice.', 'Do not lift it twice.', 'Avoid lifting it twice.'):
            self.assertIsNone(lift_repetition_policy(instruction)['explicit_repeat_quote'])

    def test_initial_lifted_state_needs_explicit_repeat(self):
        library, snapshot, nodes = inputs()
        snapshot.update(held_object_id=67, lifted=True, prepared_target_id=67)
        for instruction, accepted in [('Lift the held kettle.', False), ('Lift the held kettle again.', True)]:
            row = pickup_row(library, instruction=instruction)
            row['plan']['subtask_plans'][0]['steps'] = row['plan']['subtask_plans'][0]['steps'][-1:]
            row['plan'] = build_result(row['alignment'], library, {'subtask_plans': row['plan']['subtask_plans']})
            if accepted:
                report = review_live_plan(row, library, nodes, snapshot, 1.5)
                self.assertEqual(len(report['semantic_redundancy']['explicit_proxy_repeat_calls']), 1)
            else:
                with self.assertRaises(SemanticRedundancyError):
                    review_live_plan(row, library, nodes, snapshot, 1.5)

    def exercise_worker(self, corrected):
        library, snapshot, nodes = inputs()
        instruction = 'Pick up and lift the kettle again; keep holding it.'
        bad = pickup_row(library, 2, instruction, split=True)
        good = pickup_row(library, 1, instruction, split=True)
        descriptor = {'round': 4, 'instruction': instruction, 'expected_end': {'held_object_id': 67, 'lifted': True}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for file, value in [('round_input.json', descriptor), ('runtime_state.json', snapshot),
                                ('execution_skill_library.json', library), ('input_memory/scene/scene_graph.json', nodes)]:
                write_json(root / file, value)
            args = SimpleNamespace(output_dir=root, round_input=root / 'round_input.json', model='test', base_url='https://invalid.test', reach=1.5)
            second = good if corrected else bad
            with patch.object(worker, 'decompose_and_bind', side_effect=[bad['alignment'], second['alignment']]) as decompose, \
                 patch.object(worker, 'request_plan', side_effect=[{'subtask_plans': bad['plan']['subtask_plans']},
                    {'subtask_plans': second['plan']['subtask_plans']}]) as proposal, \
                 patch.object(worker, 'request_order', return_value={'ordered_step_ids': [s['step_id']
                    for t in good['plan']['subtask_plans'] for s in t['steps']], 'reason': 'Preserve all'}) as critique:
                self.assertEqual(worker.plan(args), 0 if corrected else 2)
            self.assertEqual(decompose.call_count, 2)
            self.assertEqual(decompose.call_args.args[3], 2)
            self.assertEqual(decompose.call_args.args[4]['code'], 'REDUNDANT_LIFT_ALREADY_LIFTED')
            self.assertEqual(proposal.call_count, 2)
            self.assertEqual(critique.call_count, 1 if corrected else 0)
            self.assertEqual(read_json(root / 'runtime_state.json'), snapshot)
            self.assertEqual(read_json(root / 'planning_summary.json')['semantic_redecomposition_count'], 1)
            self.assertTrue((root / 'semantic_rejection_1.json').exists())
            if corrected:
                final = read_json(root / 'skill_plan_final.json')
                self.assertEqual(len(final['subtask_plans']), 3)
                self.assertEqual(final['proposal_origin'], 'fresh_llm_skill_proposal')
            else:
                self.assertFalse((root / 'skill_plan_final.json').exists())

    def test_worker_redecomposes_instead_of_dropping_required_subtask(self):
        self.exercise_worker(corrected=True)

    def test_worker_stops_after_one_unsuccessful_semantic_redecomposition(self):
        self.exercise_worker(corrected=False)

    def test_second_decomposition_uses_original_instruction_snapshot_and_archives(self):
        library, snapshot, _ = inputs()
        row = pickup_row(library)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = {'current_runtime_state': snapshot}
            write_json(root / 'execution_contract.json', policy)
            def launch(command, **kwargs):
                out = Path(command[command.index('--output') + 1])
                write_json(out, row['alignment'])
                self.assertEqual(command[command.index('--instruction') + 1], row['plan']['instruction'])
                contract_path = Path(command[command.index('--execution-contract') + 1])
                self.assertEqual(read_json(contract_path)['current_runtime_state'], snapshot)
                self.assertEqual(read_json(contract_path)['previous_plan_semantic_error'], {'code': 'test'})
                self.assertEqual(command[command.index('--scene-dir') + 1], str(root / 'input_memory/scene'))
                return SimpleNamespace(stdout=[], wait=lambda: 0)
            args = SimpleNamespace(output_dir=root, model='test', base_url='https://invalid.test')
            with patch.object(worker.subprocess, 'Popen', side_effect=launch):
                worker.decompose_and_bind(args, {'instruction': row['plan']['instruction']}, {'stages': []}, 2, {'code': 'test'})
            self.assertEqual(read_json(root / 'execution_contract.json'), policy)
            self.assertTrue((root / 'semantic_replan_2/object_alignment.json').exists())
            self.assertTrue((root / 'object_alignment_semantic_attempt_2.json').exists())


if __name__ == '__main__':
    unittest.main()
