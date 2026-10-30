"""Read provider usage from saved HTTP envelopes; missing usage is NOT zero cost."""
import json
import os
from pathlib import Path


def usage(root):
    root = Path(root)
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root.resolve()))
    totals = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    seen, responses, reported, missing, transports = set(), 0, 0, 0, 0
    stages = {}
    for path in sorted(root.rglob('*.json')):
        if path.parent.name != 'llm_response_audit' or not ('_http_' in path.name or '_transport_' in path.name):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        key = (record.get('request_id_local', str(path)), record.get('http_attempt'), record.get('event'))
        if key in seen:
            continue
        seen.add(key)
        if record.get('event') == 'transport_failure':
            transports += 1
            continue
        if record.get('event') != 'http_response':
            continue
        responses += 1
        try:
            body = record['response_body']
            body = json.loads(body) if isinstance(body, str) else body
            u = body['usage']
            if any(type(u.get(k)) is not int or u[k] < 0 for k in totals):
                raise ValueError('incomplete usage')
        except (ValueError, KeyError, TypeError):
            missing += 1
            continue
        reported += 1
        stage = stages.setdefault(record.get('stage', 'unknown'), dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, responses=0))
        stage['responses'] += 1
        for k in totals:
            totals[k] += u[k]
            stage[k] += u[k]
    return {**totals, 'responses_with_usage': reported, 'http_responses': responses,
            'responses_missing_usage': missing, 'transport_failures_unknown_usage': transports,
            'by_stage': stages,
            'scope': 'Provider-reported tokens for received responses only; not billing. '
                     'Timed-out requests may incur unobserved usage. Reasoning tokens may be included in completion.'}
