import unittest
from unittest.mock import patch
from run_visual_move_regression import validate_api_environment


class APIPreflightTests(unittest.TestCase):
    def test_missing_or_blank_key_rejected_before_capture(self):
        for value in ('', '   '):
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': value}), self.assertRaises(ValueError):
                validate_api_environment()

    def test_present_key_and_cached_mode(self):
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-placeholder'}):
            validate_api_environment()
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': ''}):
            validate_api_environment(cached=True)
