import unittest
from run_image_occlusion import assess_case
from replay_vlm_rgbd import build_parser, occluded_width


class OcclusionTests(unittest.TestCase):
    def test_partial_fraction_explicit_and_bounded(self):
        for fraction, expected in ((.25, 5), (.5, 10), (.75, 15)):
            self.assertEqual(occluded_width(20, fraction), expected)
            parsed = build_parser().parse_args(['--test-occlusion', 'partial', '--occlusion-fraction', str(fraction)])
            self.assertEqual(parsed.occlusion_fraction, fraction)
        for fraction in (0, 1, float('nan')):
            with self.assertRaises(ValueError):
                occluded_width(20, fraction)

    def test_modality_ablation_cli(self):
        for modality in ('rgb', 'depth', 'joint'):
            parsed = build_parser().parse_args(['--test-occlusion', 'partial', '--dropout-modality', modality])
            self.assertEqual(parsed.dropout_modality, modality)

    def test_partial_rejection_is_safe_but_not_recognition_success(self):
        result = {'status': 'insufficient_visual_evidence', 'frames': [], 'source_memory_unchanged': True}
        report = assess_case('partial_dropout_75', result, 2)
        self.assertTrue(report['passed'])
        self.assertFalse(report['partial_visibility_success'])
    def test_empty_evidence_is_expected_only_in_dropout(self):
        report = {'status': 'insufficient_visual_evidence', 'frames': [], 'source_memory_unchanged': True}
        self.assertTrue(assess_case('fully_unobservable', report, 2)['passed'])
        self.assertFalse(assess_case('visible_control', report, 2)['passed'])
        self.assertFalse(assess_case('visibility_restored', report, 2)['passed'])
        self.assertFalse(assess_case('partial_dropout', report, 2)['partial_visibility_success'])

    def test_runtime_failure_not_safe_rejection(self):
        self.assertFalse(assess_case('fully_unobservable', {'status': 'error'}, 2)['passed'])

    def test_full_dropout_must_not_accept_candidate(self):
        report = {'status': 'insufficient_visual_evidence', 'frames': [{'status': 'accepted'}], 'source_memory_unchanged': True}
        self.assertFalse(assess_case('fully_unobservable', report, 2)['passed'])

    def test_parser_supports_both_modes(self):
        for mode in ('partial', 'full'):
            self.assertEqual(build_parser().parse_args(['--test-occlusion', mode]).test_occlusion, mode)
