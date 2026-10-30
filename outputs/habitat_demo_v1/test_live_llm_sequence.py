import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_live_llm_sequence import default_instructions, validate_instructions, LivePlanner, execute_worker
from run_continuous_sequence import OUT, build_fixture_episode, contract, PARAMETERS
from run_navigation_demo import read_json, write_json
from live_plan_review import review_live_plan
import plan_live_round as worker
import run_continuous_sequence as continuous
from test_continuous_sequence import FakeBackend


def inputs():
    library = read_json(OUT / 'quadruped_atomic_skill_library.v0.1.json')
    library['skills'] = [s for s in library['skills'] if s['skill_id'] in PARAMETERS]
    library['execution_contract'] = contract()
    snapshot = {'held_object_id': None, 'lifted': False, 'prepared_target_id': None,
        'backend': {'robot_position': [-2.7, .1, -8.], 'memory_version': 0,
            'objects': {'67': {'center': [-2.7, 1.4, -8.], 'support_id': None},
                        '13': {'center': [-5.8, .5, -1.5], 'support_id': None}}}}
    return library, snapshot, [{'id': 67}, {'id': 13}]


class LiveSequenceTests(unittest.TestCase):
    def test_instructions_have_no_plan_and_explicit_outcome_checks(self):
        rows = validate_instructions(default_instructions())['rounds']
        self.assertEqual(len(rows), 5)
        self.assertFalse(any('plan' in r or 'alignment' in r for r in rows))
        self.assertEqual(rows[1]['expected_end'], {'held_object_id': 67, 'lifted': True})

    def test_precomputed_plan_and_bad_instruction_assertions_rejected(self):
        for key, value in [('plan', {}), ('expected_end', {'held_object_id': True}),
                           ('expected_navigation_targets', [999]), ('instruction', ''), ('round', 2)]:
            episode = default_instructions()
            episode['rounds'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_instructions(episode)

    def test_readonly_five_round_rehearsal_persists_held_state(self):
        library, snapshot, nodes = inputs()
        original = copy.deepcopy(snapshot)
        fixture = build_fixture_episode(library)
        for descriptor, row in zip(default_instructions()['rounds'], fixture['rounds']):
            row.update(expected_end=descriptor['expected_end'])
            if 'expected_navigation_targets' in descriptor:
                row['expected_navigation_targets'] = descriptor['expected_navigation_targets']
            report = review_live_plan(row, library, nodes, snapshot, 1.5)
            self.assertEqual(report['status'], 'conditionally_valid_proxy_sequence')
            self.assertFalse(report['geometry_verified'])
            if row['round'] in (2, 4):
                self.assertEqual(report['predicted_end_state']['held_object_id'], 67)
            snapshot = report['predicted_end_state']
        self.assertEqual(original['backend']['objects']['67']['center'], [-2.7, 1.4, -8.])
        self.assertIsNone(original['held_object_id'])

    def test_regrasp_already_held_rejected_using_actual_runtime_state(self):
        library, snapshot, nodes = inputs()
        snapshot.update(held_object_id=67, lifted=True, prepared_target_id=67)
        row = build_fixture_episode(library)['rounds'][1]
        with self.assertRaisesRegex(ValueError, 'GRIPPER_NOT_EMPTY'):
            review_live_plan(row, library, nodes, snapshot, 1.5)

    def test_no_grasp_does_not_satisfy_hold_instruction(self):
        library, snapshot, nodes = inputs()
        row = build_fixture_episode(library)['rounds'][1]
        row['plan']['subtask_plans'][0]['steps'] = row['plan']['subtask_plans'][0]['steps'][:1]
        from action_sequence import refresh_sequence
        row['plan'] = refresh_sequence(row['plan'])
        with self.assertRaisesRegex(ValueError, 'held/lifted'):
            review_live_plan(row, library, nodes, snapshot, 1.5)

    def test_required_revisit_cannot_be_deleted(self):
        library, snapshot, nodes = inputs()
        row = build_fixture_episode(library)['rounds'][0]
        row['expected_navigation_targets'] = [67, 13, 67]
        row['plan']['subtask_plans'][0]['steps'] = [row['plan']['subtask_plans'][0]['steps'][-1]]
        from action_sequence import refresh_sequence
        row['plan'] = refresh_sequence(row['plan'])
        with self.assertRaisesRegex(ValueError, 'visit order'):
            review_live_plan(row, library, nodes, snapshot, 1.5)

    def test_place_without_holding_rejected_before_geometry_prediction(self):
        library, snapshot, nodes = inputs()
        row = build_fixture_episode(library)['rounds'][2]
        with self.assertRaisesRegex(ValueError, 'LIFTED_HELD_OBJECT_REQUIRED'):
            review_live_plan(row, library, nodes, snapshot, 1.5)

    def test_worker_uses_fresh_proposal_and_never_masks_api_failure(self):
        library, snapshot, nodes = inputs()
        row = build_fixture_episode(library)['rounds'][0]
        descriptor = default_instructions()['rounds'][0]
        descriptor['instruction'] = row['plan']['instruction']
        library['execution_contract']['current_runtime_state'] = snapshot
        proposal = {'subtask_plans': row['plan']['subtask_plans']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for file, value in [('round_input.json', descriptor), ('runtime_state.json', snapshot),
                                ('execution_skill_library.json', library), ('input_memory/scene/scene_graph.json', nodes),
                                ('object_alignment.json', row['alignment'])]:
                write_json(root / file, value)
            args = SimpleNamespace(output_dir=root, round_input=root / 'round_input.json', model='test-model', base_url='https://invalid.test', reach=1.5)
            process = SimpleNamespace(stdout=[], wait=lambda: 0)
            order = [s['step_id'] for s in row['plan']['subtask_plans'][0]['steps']]
            with patch.object(worker.subprocess, 'Popen', return_value=process), \
                 patch.object(worker, 'request_plan', return_value=proposal) as request, \
                 patch.object(worker, 'request_order', return_value={'ordered_step_ids': order, 'reason': 'Keep required visits'}):
                self.assertEqual(worker.plan(args), 0)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(request.call_args.args[1]['execution_contract']['current_runtime_state'], snapshot)
                self.assertEqual(read_json(root / 'skill_plan_final.json')['proposal_origin'], 'fresh_llm_skill_proposal')
            # An old final file deliberately remains: failed summary must prevent its consumption.
            with patch.object(worker.subprocess, 'Popen', return_value=process), \
                 patch.object(worker, 'request_plan', side_effect=TimeoutError('injected timeout')):
                self.assertEqual(worker.plan(args), 2)
            self.assertEqual(read_json(root / 'planning_summary.json')['status'], 'failed')

    def test_provider_saves_current_memory_and_rejects_stale_success(self):
        library, snapshot, nodes = inputs()
        snapshot['backend']['memory_version'] = 7
        backend = SimpleNamespace(memory={'version': 7, 'entities': {str(n['id']): {'node': n, 'active': True} for n in nodes}, 'edges': []})
        executor = SimpleNamespace(snapshot=lambda: copy.deepcopy(snapshot), held_object_id=None)
        args = SimpleNamespace(proxy_reach_m=1.5, planning_python='python', model='test-model', base_url='https://invalid.test')
        planner = LivePlanner(args, default_instructions())
        with tempfile.TemporaryDirectory() as directory:
            round_dir = Path(directory) / 'round_1'
            def fake_worker(command, log):
                root = round_dir / 'planning'
                self.assertEqual(read_json(root / 'input_memory/memory_state.json')['version'], 7)
                write_json(root / 'planning_summary.json', {'status': 'planning_passed', 'input_memory_version': 6})
            with patch('run_live_llm_sequence.execute_worker', side_effect=fake_worker), \
                 self.assertRaisesRegex(ValueError, 'memory version'):
                planner.plan(default_instructions()['rounds'][0], executor, backend, library, round_dir)

    def test_failed_worker_propagates_underlying_stage_and_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'planning_summary.json', {'status': 'failed',
                'failed_stage': 'llm_decomposition_and_object_alignment', 'error': 'TimeoutError after 2 attempts'})
            with self.assertRaisesRegex(ValueError, 'llm_decomposition_and_object_alignment.*TimeoutError'):
                execute_worker([sys.executable, '-c', 'raise SystemExit(2)'], root / 'planning.console.log')

    def exercise_live_loop(self, fail_round=None):
        library, _, nodes = inputs()
        for node in nodes:
            node['bbox_center'] = [0., 0., 1.]
        fixture = build_fixture_episode(library)
        observations, lifecycle = [], {'opened': 0, 'closed': 0}
        state = SimpleNamespace(position=[0., 0., 0.])
        agent = SimpleNamespace(get_state=lambda: state, set_state=lambda s: None)
        class FakeSim:
            pathfinder = SimpleNamespace(load_nav_mesh=lambda p: True)
            def __enter__(self):
                lifecycle['opened'] += 1
                return self
            def __exit__(self, *a):
                lifecycle['closed'] += 1
            def get_physics_simulation_library(self):
                return 'bullet'
            def get_stage_initialization_template(self):
                return SimpleNamespace(margin=.001)
            def get_stage_is_collidable(self):
                return True
            def get_agent(self, i):
                return agent
        class Backend(FakeBackend):
            def __init__(self, sim, nodes, memory, *args):
                super().__init__()
                self.memory = memory
                self.sim = sim
                self.motion_evidence = []
            def sync_memory(self, *args):
                self.memory['version'] += 1
        class Provider:
            episode = default_instructions()
            def plan(self, descriptor, executor, backend, library, directory):
                observations.append((descriptor['round'], executor.held_object_id, backend.memory['version']))
                if descriptor['round'] == fail_round:
                    raise TimeoutError('injected live planning timeout')
                row = copy.deepcopy(fixture['rounds'][descriptor['round'] - 1])
                row['planning_evidence'] = {'fresh_llm': True, 'input_memory_version': backend.memory['version']}
                return row
        hs = SimpleNamespace(SimulatorConfiguration=lambda: SimpleNamespace(),
            Configuration=lambda *a: None, Simulator=lambda cfg: FakeSim())
        hs_agent = SimpleNamespace(AgentConfiguration=lambda: SimpleNamespace(),
            ActionSpec=lambda *a: None, ActuationSpec=lambda **a: None)
        hs_physics = SimpleNamespace(PhysicsSimulationLibrary=SimpleNamespace(Bullet='bullet'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'input'
            write_json(scene / 'scene_graph.json', nodes)
            write_json(scene / 'scene_graph_edges.json', [])
            for path in ['habitat_split_assets_v2/scene201_without_kettle.stage_config.json',
                         'habitat_split_assets_v2/kettle_dynamic.object_config.json',
                         'habitat_split_assets_v2/sideboard_static.object_config.json',
                         'scene_package/hf_house_201/hf_house_201.navmesh']:
                write_json(root / path, {})
            for name in ('stage_render.glb', 'stage_collision.glb'):
                (root / 'habitat_split_assets_v2' / name).write_bytes(b'offline fake asset')
            write_json(root / 'habitat_split_assets_v2/scene201_without_kettle.stage_config.json',
                       {'render_asset':'stage_render.glb', 'collision_asset':'stage_collision.glb'})
            args = SimpleNamespace(scene_dir=scene, output_dir=root / 'output', episode_json=None,
                test_fault='none', preflight_only=False, seed=203, proxy_reach_m=1.5, max_actions=500, settle_seconds=10.)
            summary = {'rounds': [], 'simulator_creation_count': 0}
            memory = {'version': 0, 'entities': {str(n['id']): {'node': n, 'active': True} for n in nodes}, 'edges': []}
            with patch.dict(sys.modules, {'habitat_sim': hs, 'habitat_sim.agent': hs_agent, 'habitat_sim.physics': hs_physics}), \
                 patch.object(continuous, 'SCENE', root), patch.object(continuous, 'HabitatBackend', Backend), \
                 patch.object(continuous, 'initialize', return_value=memory), \
                 patch.object(continuous, 'select_coordinates', return_value=(None, [{'nav_goal': [0., 0., 0.]}])), \
                 patch.object(continuous, 'choose_start', return_value=[0., 0., 0.]), \
                 patch.object(continuous, 'habitat_vector', side_effect=lambda x: x), \
                 patch.object(continuous, 'evaluate_goals', return_value=[{'status': 'observed_in_proxy', 'required': True}]):
                if fail_round:
                    with self.assertRaisesRegex(TimeoutError, 'live planning timeout'):
                        continuous.run(args, summary, Provider())
                else:
                    continuous.run(args, summary, Provider())
            return observations, lifecycle, summary

    def test_live_loop_plans_after_previous_execution_without_reset(self):
        observations, lifecycle, summary = self.exercise_live_loop()
        self.assertEqual(lifecycle, {'opened': 1, 'closed': 1})
        self.assertEqual([r[1] for r in observations], [None, None, 67, None, 67])
        self.assertEqual([r[2] for r in observations], [0, 3, 6, 9, 13])
        self.assertEqual(summary['status'], 'passed')
        self.assertTrue(all(r['continuity_with_previous_round'] for r in summary['rounds']))

    def test_planning_failure_executes_no_failed_round_and_skips_later_rounds(self):
        observations, lifecycle, summary = self.exercise_live_loop(fail_round=2)
        self.assertEqual(lifecycle['closed'], 1)
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(summary['rounds']), 2)
        self.assertEqual(summary['rounds'][-1]['actions'], [])
        self.assertEqual(summary['rounds'][-1]['failed_stage'], 'live_llm_planning')


if __name__ == '__main__':
    unittest.main()
