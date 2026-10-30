import math
import unittest
from estimate_visual_translation import register_translation, consensus
from update_memory import initialize, apply_events


class TranslationTests(unittest.TestCase):
    def test_unique_majority_excludes_one_outlier(self):
        rows = [{'frame': str(i), 'translation_cg': [x, 0, 0], 'symmetric_trimmed_rms_m': .01}
                for i, x in enumerate([.25, .251, .249, .60])]
        result = consensus(rows)
        self.assertEqual(result['accepted_views'], 3)
        self.assertEqual(result['excluded_frames'], ['3'])
        self.assertAlmostEqual(result['translation_cg'][0], .25)

    def test_competing_groups_and_two_view_minority_rejected(self):
        for values in ([.25, .251, .6, .601], [.25, .251, .6], [.20, .24, .28, .32]):
            rows = [{'translation_cg': [x, 0, 0], 'symmetric_trimmed_rms_m': .01} for x in values]
            with self.assertRaises(ValueError):
                consensus(rows)

    def points(self):
        return [[i*.01, j*.011, .02*math.sin(i)] for i in range(8) for j in range(8)]

    def test_estimator_recovers_translation_without_truth_input(self):
        before = self.points()
        after = [[p[0]+.25, p[1]-.12, p[2]+.07] for p in before]
        result = register_translation(before, after)
        for a, b in zip(result['translation_cg'], [.25, -.12, .07]):
            self.assertAlmostEqual(a, b, places=6)
        self.assertLess(result['symmetric_trimmed_rms_m'], 1e-6)

    def test_multiview_consensus(self):
        result = consensus([{'translation_cg': [.25, 0, 0], 'symmetric_trimmed_rms_m': .01},
                            {'translation_cg': [.26, 0, 0], 'symmetric_trimmed_rms_m': .01}])
        self.assertAlmostEqual(result['translation_cg'][0], .255)

    def test_single_or_inconsistent_or_poor_fit_rejected(self):
        valid = {'translation_cg': [.25, 0, 0], 'symmetric_trimmed_rms_m': .01}
        for rows in ([valid], [valid, {'translation_cg': [.5, 0, 0], 'symmetric_trimmed_rms_m': .01}],
                     [valid, {'translation_cg': [.25, 0, 0], 'symmetric_trimmed_rms_m': .1}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                consensus(rows)

    def test_move_invalidates_visual_metadata(self):
        state = initialize([{'id': 67, 'bbox_center': [0, 0, 0], 'visual_memory': {'voxel_count': 100}}], [], 'fixture')
        event = {'event_id': 'translation', 'action_id': 'reconcile_visual_translation', 'object_id': 67,
            'operation': 'move', 'action_status': 'completed', 'observed_at': '2026-09-26T01:00:00Z',
            'coordinate_frame': 'conceptgraphs_world', 'position': [.25, 0, 0],
            'evidence': {'source': 'verified_post_action_observation', 'verified': True, 'reference': 'test-registration'}}
        changed = apply_events(state, [event])
        self.assertNotIn('visual_memory', changed['entities']['67']['node'])
        self.assertIn('visual_memory', state['entities']['67']['node'])

    def test_small_cloud_rejected(self):
        with self.assertRaises(ValueError):
            register_translation([[0, 0, 0]], [[.25, 0, 0]])
