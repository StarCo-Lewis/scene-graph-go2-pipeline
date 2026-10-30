import copy
import tempfile
import unittest
from pathlib import Path
from replay_live_round_planning import validate_checkpoint
from run_navigation_demo import write_json


class CheckpointTests(unittest.TestCase):
    def make_inputs(self, root, mismatch=None):
        nodes = [{'id': 67, 'bbox_center': [-2., 3., 1.]}, {'id': 13, 'bbox_center': [-5., 1., .5]}]
        snapshot = {'held_object_id': None, 'lifted': False, 'prepared_target_id': None,
            'backend': {'memory_version': 4, 'objects': {'67': {'center': [-2., 1., -3.]}, '13': {'center': [-5., .5, -1.]}}}}
        memory = {'version': 4, 'edges': [], 'entities': {str(n['id']): {'active': True, 'node': copy.deepcopy(n)} for n in nodes}}
        if mismatch == 'pose':
            snapshot['backend']['objects']['67']['center'][0] += 1
        if mismatch == 'version':
            snapshot['backend']['memory_version'] = 0
        if mismatch == 'nodes':
            memory['entities']['67']['node']['bbox_center'][0] += 1
        for path, value in [('runtime_state.json', snapshot), ('input_memory/memory_state.json', memory),
                            ('input_memory/scene/scene_graph.json', nodes), ('input_memory/scene/scene_graph_edges.json', []),
                            ('round_input.json', {'round': 4, 'instruction': 'Go to the kettle.'})]:
            write_json(root / path, value)

    def test_saved_state_keeps_original_round_and_digest_without_loading_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            snapshot, memory, descriptor, before = validate_checkpoint(root)
            self.assertEqual(descriptor['round'], 4)
            self.assertEqual(memory['version'], 4)
            self.assertEqual(validate_checkpoint(root)[-1], before)

    def test_pose_version_and_map_mismatch_not_repaired_by_guessing(self):
        for mismatch in ('pose', 'version', 'nodes'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_inputs(root, mismatch)
                with self.assertRaises(ValueError):
                    validate_checkpoint(root)


if __name__ == '__main__':
    unittest.main()
