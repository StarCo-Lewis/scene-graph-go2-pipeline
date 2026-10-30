import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch
from llm_transport import request_json
from llm_graph_binding import request_bindings
from align_objects import call_llm


class TransportTests(unittest.TestCase):
    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_timeout_recovers_using_the_identical_request(self, open_url, sleep):
        request = object()
        open_url.side_effect = [TimeoutError('read timed out'), io.StringIO('{"ok":true}')]
        self.assertEqual(request_json(request, attempts=2), {'ok': True})
        self.assertEqual([c.args[0] for c in open_url.call_args_list], [request, request])
        sleep.assert_called_once_with(5)

    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen', side_effect=TimeoutError('read timed out'))
    def test_timeout_stops_after_two_attempts(self, open_url, sleep):
        with self.assertRaisesRegex(RuntimeError, 'after 2 attempts: TimeoutError'):
            request_json(object(), attempts=2)
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(5)

    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_auth_denial_is_not_retried(self, open_url, sleep):
        open_url.side_effect = urllib.error.HTTPError('https://example.test', 403, 'Forbidden', {}, io.BytesIO(b'denied'))
        with self.assertRaises(urllib.error.HTTPError):
            request_json(object(), attempts=2)
        self.assertEqual(open_url.call_count, 1)
        sleep.assert_not_called()

    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_transient_http_503_recovers(self, open_url, sleep):
        open_url.side_effect = [urllib.error.HTTPError('https://example.test', 503, 'Unavailable', {}, io.BytesIO()),
                                io.StringIO('{"ok":true}')]
        self.assertTrue(request_json(object(), attempts=2)['ok'])
        self.assertEqual(open_url.call_count, 2)

    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_invalid_json_is_not_transport_retry(self, open_url, sleep):
        open_url.return_value = io.StringIO('not json')
        with self.assertRaises(json.JSONDecodeError):
            request_json(object(), attempts=2)
        self.assertEqual(open_url.call_count, 1)
        sleep.assert_not_called()

    def test_invalid_budget_and_timeout_rejected(self):
        for kwargs in ({'attempts': 4}, {'attempts': True}, {'timeout': float('nan')}, {'timeout': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                request_json(object(), **kwargs)

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-only'})
    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_binding_retries_same_graph_not_new_decomposition(self, open_url, sleep):
        reply = json.dumps({'choices': [{'message': {'content': '{"bindings":[]}'}}]})
        open_url.side_effect = [TimeoutError(), io.StringIO(reply)]
        self.assertEqual(request_bindings([], [], 'test', 'https://example.test', 'DASHSCOPE_API_KEY', request_attempts=2), {'bindings': []})
        self.assertIs(open_url.call_args_list[0].args[0], open_url.call_args_list[1].args[0])

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-only'})
    @patch('align_objects.validate_planning_capability_claims')
    @patch('align_objects.validate_decomposition', side_effect=lambda x: x)
    @patch('llm_transport.time.sleep')
    @patch('llm_transport.urllib.request.urlopen')
    def test_decomposition_retries_transport_within_same_schema_attempt(self, open_url, sleep, validate, claims):
        candidate = {'goal_state': [{'required': True}], 'instruction': 'Visit the kettle.'}
        reply = json.dumps({'choices': [{'message': {'content': json.dumps(candidate)}}]})
        open_url.side_effect = [TimeoutError(), io.StringIO(reply)]
        self.assertEqual(call_llm('Visit the kettle.', 'test', 'https://example.test', 'DASHSCOPE_API_KEY', request_attempts=2), candidate)
        self.assertEqual(validate.call_count, 1)
        self.assertIs(open_url.call_args_list[0].args[0], open_url.call_args_list[1].args[0])


if __name__ == '__main__':
    unittest.main()
