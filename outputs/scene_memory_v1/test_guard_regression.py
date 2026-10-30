import argparse
import gzip
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from update_memory import initialize, apply_events, write, read
from run_memory_guard_regression import run


class GuardRegressionTests(unittest.TestCase):
    def test_real_cli_rejects_deleted_object_with_features_still_present(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base'
            (base / 'map').mkdir(parents=True)
            nodes = [{'id': 67, 'object_tag': 'electric kettle', 'bbox_center': [1, 2, 3]},
                     {'id': 13, 'object_tag': 'sideboard', 'bbox_center': [4, 5, 6]}]
            write(base / 'scene_graph.json', nodes)
            write(base / 'scene_graph_edges.json', [])
            with gzip.open(base / 'map/scene_map_cfslam_pruned.pkl.gz', 'wb') as stream:
                pickle.dump([{'caption_dict': {'id': n['id']}, 'clip_ft': [1, 0]} for n in nodes], stream)
            event = {'event_id': 'move1', 'action_id': 'place', 'operation': 'move',
                'object_id': 67, 'action_status': 'completed', 'observed_at': '2026-09-26T03:00:00Z',
                'coordinate_frame': 'conceptgraphs_world', 'position': [4, 5, 6], 'support_object_id': 13,
                'evidence': {'source': 'simulator_observation', 'verified': True, 'reference': 'fixture'}}
            memory = root / 'memory'
            memory.mkdir()
            write(memory / 'memory_state.json', apply_events(initialize(nodes, [], base), [event]))
            role = {'role_id': 'kettle', 'name': 'electric kettle', 'semantic_query_en': 'electric kettle',
                    'semantic_labels': ['electric kettle', 'kettle'], 'quantity': 1, 'required': True}
            decomposition = root / 'task.json'
            write(decomposition, {'instruction': 'Go to the electric kettle.', 'task_name': 'Navigate',
                'subtasks': [{'subtask_id': 's1', 'description': 'Navigate to kettle',
                              'dependencies': [], 'required_roles': [role]}],
                'initial_scene_constraints': [],
                'goal_state': [{'constraint_id': 'g1', 'subject_role': 'robot', 'relation': 'near',
                                'object_role': 'kettle', 'required': True}]})
            output = root / 'regression'
            self.assertEqual(run(argparse.Namespace(memory_dir=memory, decomposition=decomposition,
                planning_python=sys.executable, output_dir=output)), 0)
            self.assertEqual(read(output / 'regression_summary.json')['status'], 'passed')
