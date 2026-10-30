import copy
import unittest
from update_memory import initialize, apply_events
from run_visual_repaired_navigation import validate_visual_repair


class VisualNavigationTests(unittest.TestCase):
    def setUp(self):
        self.old = initialize([{'id': 67, 'bbox_center': [1., 2., 3.]},
                               {'id': 13, 'bbox_center': [4., 5., 6.]}], [], 'fixture')
        self.estimate = {'status': 'translation_estimated', 'translation_cg': [.25, 0., 0.]}
        self.evaluation = {'status': 'passed'}
        event = {'event_id': 'visual1', 'action_id': 'reconcile_visual_translation', 'object_id': 67,
            'operation': 'move', 'action_status': 'completed', 'observed_at': '2026-09-26T01:00:00Z',
            'coordinate_frame': 'conceptgraphs_world', 'position': [1.25, 2., 3.],
            'evidence': {'source': 'verified_post_action_observation', 'verified': True, 'reference': 'fixture',
                         'estimated_from': 'VLM_SAM_RGBD_translation_registration'}}
        self.repaired = apply_events(self.old, [event])

    def test_valid_visual_update_is_not_proxy_restore(self):
        result = validate_visual_repair(self.old, self.repaired, self.estimate, self.evaluation)
        self.assertEqual(result['repaired_position'], [1.25, 2., 3.])

    def test_modified_target_rejected(self):
        changed = copy.deepcopy(self.repaired)
        changed['entities']['67']['node']['bbox_center'][0] += .1
        with self.assertRaises(ValueError):
            validate_visual_repair(self.old, changed, self.estimate, self.evaluation)

    def test_unrelated_entity_change_rejected(self):
        changed = copy.deepcopy(self.repaired)
        changed['entities']['13']['node']['bbox_center'][0] += .1
        with self.assertRaises(ValueError):
            validate_visual_repair(self.old, changed, self.estimate, self.evaluation)

    def test_failed_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            validate_visual_repair(self.old, self.repaired, self.estimate, {'status': 'failed'})
