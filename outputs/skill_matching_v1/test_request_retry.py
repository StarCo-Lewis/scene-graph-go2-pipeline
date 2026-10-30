import io
import unittest
import urllib.error
from unittest.mock import patch
from match_skills import request_json_with_retry, PlanError


class RetryTests(unittest.TestCase):
    @patch("match_skills.time.sleep")
    @patch("match_skills.urllib.request.urlopen")
    def test_timeout_retries_and_recovers(self, open_url, sleep):
        open_url.side_effect = [TimeoutError("read timed out"), io.StringIO('{"ok":true}')]
        self.assertEqual(request_json_with_retry(object()), {"ok": True})
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(5)

    @patch("match_skills.time.sleep")
    @patch("match_skills.urllib.request.urlopen", side_effect=TimeoutError("read timed out"))
    def test_retry_is_bounded(self, open_url, sleep):
        with self.assertRaisesRegex(PlanError, "after 3 attempts"):
            request_json_with_retry(object())
        self.assertEqual(open_url.call_count, 3)

    @patch("match_skills.time.sleep")
    @patch("match_skills.urllib.request.urlopen")
    def test_auth_denial_not_retried(self, open_url, sleep):
        open_url.side_effect = urllib.error.HTTPError("url", 403, "Forbidden", {}, io.BytesIO(b"denied"))
        with self.assertRaisesRegex(PlanError, "403"):
            request_json_with_retry(object())
        self.assertEqual(open_url.call_count, 1)
        sleep.assert_not_called()
