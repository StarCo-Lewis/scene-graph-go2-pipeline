import copy
import unittest
from run_visual_recovery import blocked_commit_safe, category_only_guard, recovery_dropout_arguments
from update_memory import initialize


class RecoveryTests(unittest.TestCase):
    def test_depth_failure_fixture_is_full_not_partial(self):
        self.assertEqual(recovery_dropout_arguments('depth'), ['--test-occlusion', 'full', '--dropout-modality', 'depth'])
        self.assertEqual(recovery_dropout_arguments('full')[-1], 'joint')
        with self.assertRaises(ValueError):
            recovery_dropout_arguments('partial')
    def test_category_only_not_position_commit(self):
        row = {'observation_state': 'category_only_position_unresolved', 'position_update_allowed': False, 'delete_allowed': False}
        audit = {'graph_commit_performed': False, 'observations': [dict(row), dict(row)]}
        self.assertTrue(category_only_guard({'category_only_view_count': 2}, audit))
        audit['observations'][0]['position_update_allowed'] = True
        self.assertFalse(category_only_guard({'category_only_view_count': 2}, audit))
        self.assertFalse(category_only_guard({'category_only_view_count': 0}, {'graph_commit_performed': False, 'observations': []}))
    def setUp(self):
        self.state = initialize([{'id': 67, 'bbox_center': [1, 2, 3]}], [], 'fixture')
        self.failure = {'status': 'failed', 'failed_stage': 'VLM_SAM_after_observation'}
        self.detection = {'status': 'insufficient_visual_evidence', 'frames': [{'status': 'rejected'}]*3}

    def test_evidence_rejection_without_commit_allowed(self):
        self.assertTrue(blocked_commit_safe(self.failure, self.detection, self.state, self.state, False))

    def test_runtime_failure_candidate_commit_or_mutation_rejected(self):
        self.assertFalse(blocked_commit_safe(self.failure, {'status': 'error'}, self.state, self.state, False))
        self.assertFalse(blocked_commit_safe(self.failure, self.detection, self.state, self.state, True))
        accepted = {'status': 'insufficient_visual_evidence', 'frames': [{'status': 'accepted'}]*3}
        self.assertFalse(blocked_commit_safe(self.failure, accepted, self.state, self.state, False))
        changed = copy.deepcopy(self.state)
        changed['entities']['67']['active'] = False
        self.assertFalse(blocked_commit_safe(self.failure, self.detection, self.state, changed, False))
