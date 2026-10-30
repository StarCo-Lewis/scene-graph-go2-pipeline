import unittest
from align_objects import preserve_original_instruction


class InstructionTests(unittest.TestCase):
    def test_explicit_id_preserved(self):
        original = 'Go to the red coke can with object ID 6.'
        result = preserve_original_instruction({'instruction': 'Go to the red coke can.'}, original)
        self.assertEqual(result['instruction'], original)
        self.assertEqual(result['model_instruction'], 'Go to the red coke can.')

    def test_unchanged(self):
        result = preserve_original_instruction({'instruction': 'Go.'}, 'Go.')
        self.assertNotIn('model_instruction', result)
