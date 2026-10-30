import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_action_matrix_ab import inputs
from test_json_long_horizon import nodes
from run_navigation_demo import read_json, write_json
from adapter_parameter_contract import project_library
from run_live_llm_sequence import planning_command, LivePlanner, default_instructions
from json_symbolic_review import review_symbolic
from action_matrix import make_codebook
import plan_live_round as worker


class FixtureEncoder:
    # Equal finite vectors deliberately retain the full ambiguous candidate set.
    def encode(self, texts):
        return [[1., 0.] for _ in texts]


class LiveMatrixTests(unittest.TestCase):
    def exercise(self, make_response, encoder=True):
        descriptor, library, state, _, _, _ = inputs()
        library['execution_contract']['adapter_mode'] = 'json_symbolic_test_only'
        library = project_library(library)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [('round_input.json', descriptor), ('runtime_state.json', state),
                ('execution_skill_library.json', library), ('input_memory/scene/scene_graph.json', nodes())]:
                write_json(root/name, value)
            args = SimpleNamespace(output_dir=root, round_input=root/'round_input.json', model='fixture',
                base_url='https://invalid.test', reach=1.5, skill_output_format='matrix')
            if encoder:
                args.skill_retriever = FixtureEncoder()
            observed = []
            def request(alignment, library, book, *a, **kw):
                observed.append((copy.deepcopy(book), copy.deepcopy(kw)))
                return make_response(book, len(observed))
            with patch.object(worker, 'decompose_and_bind', return_value=descriptor['alignment']), \
                 patch.object(worker, 'request_matrix', side_effect=request) as matrix_request, \
                 patch.object(worker, 'request_plan', side_effect=AssertionError('No JSON fallback')) as json_request, \
                 patch.object(worker, 'request_order', return_value={'reason': 'Keep'}), \
                 patch.object(worker, 'candidate_plan', side_effect=lambda plan, critique: copy.deepcopy(plan)):
                code = worker.plan(args, reviewer=review_symbolic)
            json_request.assert_not_called()
            self.assertEqual(read_json(root/'runtime_state.json'), state)
            artifacts = {p.name: read_json(p) for p in root.glob('*.json')}
            return code, matrix_request.call_count, artifacts, observed

    @staticmethod
    def valid(book, attempt):
        skills = {s['skill_id']: s['code'] for s in book['skills']}
        objects = {o['role_id']: o['code'] for o in book['objects']}
        return {'matrix': [[1, skills['navigate_to_object'], objects['office'], 0],
            [2, skills['position_arm_for_object'], objects['office'], 0],
            [2, skills['place_object_on_surface'], objects['kettle'], objects['office']]]}

    def test_fresh_matrix_decoded_then_existing_state_and_goal_review(self):
        code, count, files, observed = self.exercise(self.valid)
        self.assertEqual((code, count), (0, 1))
        self.assertEqual(files['skill_plan_final.json']['proposal_origin'], 'fresh_llm_matrix_decoded')
        self.assertEqual(files['planning_summary.json']['skill_output_format'], 'matrix')
        self.assertEqual(files['planning_summary.json']['skill_retrieval_policy']['top_k'], 2)
        self.assertEqual(files['symbolic_review_final.json']['predicted_end_state']['backend']['objects']['67']['support_id'], 13)
        self.assertFalse(files['planning_summary.json']['simulation_executed'])
        self.assertEqual(files['action_codebook_semantic_1.json'], observed[0][0])
        self.assertEqual(files['action_codebook_semantic_1.json']['values'], [])
        self.assertFalse(observed[0][1]['widen_inventory'])

    def test_bare_matrix_normalized_without_second_request(self):
        code, count, files, _ = self.exercise(lambda b, a: self.valid(b,a)['matrix'])
        self.assertEqual((code, count), (0, 1))
        self.assertIsInstance(files['llm_matrix_proposal_semantic_1_attempt_1.json'], list)
        self.assertTrue(files['skill_plan_final.json']['format_normalization']['changed'])

    def test_invalid_code_corrected_once_and_same_codebook(self):
        code, count, files, observed = self.exercise(lambda b,a: {'matrix': [[1,999,1,0]]} if a==1 else self.valid(b,a))
        self.assertEqual((code, count), (0, 2))
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertTrue(observed[1][1]['widen_inventory'])
        self.assertEqual(files['llm_matrix_proposal_semantic_1_attempt_1.json'], {'matrix': [[1,999,1,0]]})

    def test_two_invalid_matrices_stop_without_json_fallback(self):
        code, count, files, _ = self.exercise(lambda b,a: {'matrix': []})
        self.assertEqual((code, count), (2, 2))
        self.assertEqual(files['planning_summary.json']['status'], 'failed')
        self.assertNotIn('skill_plan_final.json', files)

    def test_row_counter_feedback_reaches_retry_without_action_deletion(self):
        def respond(book, attempt):
            matrix = self.valid(book, attempt)
            if attempt == 1:
                matrix['matrix'][-1][0] = 3
            return matrix
        code, count, files, observed = self.exercise(respond)
        self.assertEqual((code, count), (0, 2))
        self.assertIn('NOT the action row number', observed[1][1]['validation_error'])
        self.assertEqual(len(files['llm_matrix_proposal_semantic_1_attempt_2.json']['matrix']), 3)
        self.assertEqual(files['llm_matrix_proposal_semantic_1_attempt_2.json']['matrix'][-1][0], 2)
        self.assertEqual(files['planning_summary.json']['status'], 'planning_passed')

    def test_missing_surface_preparation_reaches_retry_and_is_not_silently_filled(self):
        def respond(book, attempt):
            matrix = self.valid(book, attempt)
            if attempt == 1:
                del matrix['matrix'][1]
            return matrix
        code, count, files, observed = self.exercise(respond)
        self.assertEqual((code, count), (0, 2))
        self.assertEqual(observed[1][1]['validation_error'], 'SURFACE_PROXY_PREPARATION_REQUIRED')
        self.assertEqual(len(files['llm_matrix_proposal_semantic_1_attempt_1.json']['matrix']), 2)
        self.assertEqual(len(files['llm_matrix_proposal_semantic_1_attempt_2.json']['matrix']), 3)
        self.assertEqual(observed[0][0], observed[1][0])

    def test_missing_embedding_configuration_fails_before_api(self):
        code, count, files, _ = self.exercise(self.valid, encoder=False)
        self.assertEqual((code, count), (2, 0))
        self.assertEqual(files['planning_summary.json']['failed_stage'], 'local_skill_embedding_initialization')

    def test_live_command_forwards_embedding_and_matrix_settings(self):
        args = SimpleNamespace(planning_python='planner', model='fixture', base_url='https://invalid.test',
            proxy_reach_m=1.5, skill_output_format='matrix', embedding_python='embed',
            embedding_model_dir='local_model', retrieval_top_k=2, retrieval_threshold=.35, retrieval_margin=.05)
        command = planning_command(args, Path('round_3/planning'))
        self.assertEqual(command[command.index('--skill-output-format')+1], 'matrix')
        self.assertEqual(command[command.index('--embedding-python')+1], 'embed')
        self.assertEqual(command[command.index('--round-input')+1], str(Path('round_3/planning/round_input.json')))

    def test_provider_rejects_missing_matrix_evidence(self):
        descriptor, library, state, _, _, _ = inputs()
        descriptor['round'] = 3
        state['backend']['memory_version'] = 7
        args = SimpleNamespace(planning_python='planner', model='fixture', base_url='https://invalid.test',
            proxy_reach_m=1.5, skill_output_format='matrix', embedding_python='embed',
            embedding_model_dir='local_model', retrieval_top_k=2, retrieval_threshold=.35, retrieval_margin=.05)
        backend = SimpleNamespace(memory={'version':7, 'edges':[], 'entities': {
            str(n['id']): {'node': n, 'active': True} for n in nodes()}})
        executor = SimpleNamespace(snapshot=lambda: copy.deepcopy(state), held_object_id=67)
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)/'round_3'
            def fake_worker(command, log):
                write_json(round_dir/'planning/planning_summary.json', {'status':'planning_passed', 'input_memory_version':7})
            with patch('run_live_llm_sequence.execute_worker', side_effect=fake_worker), \
                 self.assertRaisesRegex(ValueError, 'Matrix/cosine planning evidence missing'):
                LivePlanner(args, default_instructions()).plan(descriptor, executor, backend, library, round_dir)


if __name__ == '__main__':
    unittest.main()
