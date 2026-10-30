"""Bounded JSON HTTP retries; never log request headers, keys or payloads."""
import json
import logging
import math
import time
import urllib.error
import urllib.request
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'llm_audit_v1'))
from response_audit import response_text

LOG = logging.getLogger('llm_transport')


def request_json(request, timeout=180, attempts=1, audit=None):
    if not math.isfinite(timeout) or timeout <= 0 or type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError('Positive finite timeout and 1..3 request attempts required')
    for attempt in range(1, attempts + 1):
        LOG.info('LLM HTTP attempt=%s/%s timeout=%ss', attempt, attempts, timeout)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if audit is None:
                    return json.load(response)
                raw = response_text(response)
                audit.received(attempt, raw)
                try:
                    return json.loads(raw)
                except (ValueError, TypeError) as exc:
                    audit.parsed(raw, 'rejected', exc)
                    raise
        except urllib.error.HTTPError as exc:
            body = exc.read() if audit is not None else None
            if audit is not None:
                audit.received(attempt, body, exc.code)
            if exc.code not in (408, 429, 500, 502, 503, 504):
                if body is not None:
                    raise urllib.error.HTTPError(exc.url, exc.code, exc.reason, exc.headers, io.BytesIO(body)) from exc
                raise
            reason = f'HTTP {exc.code}'
            exc.close()
        except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            if audit is not None:
                audit.transport_error(attempt, exc)
            # Avoid returning request URLs or credential-bearing HTTP messages in logs.
            reason = type(exc).__name__
        if attempt == attempts:
            raise RuntimeError(f'LLM transport failed after {attempts} attempts: {reason}')
        delay = 5 * attempt
        LOG.warning('Transient LLM failure=%s; retry in %ss on the SAME input. '
                    'The repeated API request may incur additional cost.', reason, delay)
        time.sleep(delay)
