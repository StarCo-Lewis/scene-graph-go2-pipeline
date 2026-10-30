import unittest
from replay_vlm_rgbd import validate_detection, position_report, build_parser


class VLMRGBDTests(unittest.TestCase):
    def test_five_view_cli_used_by_long_workflow(self):
        args = build_parser().parse_args(['--max-views', '5', '--translation-evidence'])
        self.assertEqual(args.max_views, 5)
        self.assertTrue(args.translation_evidence)
        self.assertEqual(build_parser().parse_args([]).max_views, 3)
        self.assertEqual(build_parser().parse_args(['--max-views', '9']).max_views, 9)

    def answer(self, box):
        return {'visibility': 'visible', 'instances': [{'label_en': 'electric kettle', 'bbox_xyxy_1000': box}]}

    def test_normalized_box_to_original_resolution(self):
        self.assertEqual(validate_detection(self.answer([100, 200, 500, 600]), 256, 256),
                         [25.6, 51.2, 128., 153.6])

    def test_uncertain_is_not_a_detection(self):
        self.assertIsNone(validate_detection({'visibility': 'uncertain', 'instances': []}, 256, 256))

    def test_multiple_instances_rejected(self):
        answer = self.answer([10, 10, 500, 500])
        answer['instances'] *= 2
        with self.assertRaises(ValueError):
            validate_detection(answer, 256, 256)

    def test_wrong_category_rejected(self):
        answer = self.answer([10, 10, 500, 500])
        answer['instances'][0]['label_en'] = 'cup'
        with self.assertRaises(ValueError):
            validate_detection(answer, 256, 256)

    def test_invalid_boxes_rejected(self):
        for box in ([0, 0, 1001, 500], [500, 0, 0, 500], [0, 0, 1, 1], [0, 0, float('nan'), 500]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_detection(self.answer(box), 256, 256)

    def test_consistent_surface_never_overwrites_centroid(self):
        result = position_report([[0, 0, 0], [.1, 0, 0]])
        self.assertEqual(result['status'], 'surface_evidence_consistent')
        self.assertFalse(result['pose_update_allowed'])

    def test_inconsistent_and_single_view_rejected(self):
        self.assertEqual(position_report([[0, 0, 0]])['status'], 'insufficient_multiview_evidence')
        self.assertEqual(position_report([[0, 0, 0], [1, 0, 0]])['status'], 'inconsistent_views')
