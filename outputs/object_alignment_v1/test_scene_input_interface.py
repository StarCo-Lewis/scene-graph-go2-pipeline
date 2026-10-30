import json
import unittest
from scene_input_interface import normalize_scene_input


class InputTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [{'id': 4, 'object_tag': 'coke can', 'caption': 'A red can',
                       'bbox_center': [2.7, .65, .82], 'bbox_extent': [.08, .08, .15]}]

    def test_formats_preserve_full_extent(self):
        for value in (self.nodes, json.dumps(self.nodes), {'nodes': self.nodes}):
            result = normalize_scene_input(value, 'Go to the can.')
            self.assertEqual(result['nodes'], self.nodes)
            self.assertIsNot(result['nodes'], self.nodes)

    def test_invalid(self):
        for field, value in [('bbox_extent', [0, 1, 1]), ('bbox_center', [float('nan'), 0, 0]), ('id', True)]:
            node = {**self.nodes[0], field: value}
            with self.assertRaises(ValueError):
                normalize_scene_input([node], 'Go.')

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            normalize_scene_input(self.nodes * 2, 'Go.')


if __name__ == '__main__':
    unittest.main()
