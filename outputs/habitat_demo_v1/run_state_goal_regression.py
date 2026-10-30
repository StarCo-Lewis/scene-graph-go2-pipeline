#!/usr/bin/env python3
"""Offline protocol regression and optional unchanged historical decomposition replay."""
import argparse
import hashlib
import io
import json
import unittest
from pathlib import Path

from test_runtime_state_goals import RuntimeStateGoalsTests
from decomposition_wire import from_flat_roles
from align_objects import validate_decomposition
from runtime_goal_protocol import VERSION
from run_navigation_demo import write_json


def run(args):
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeStateGoalsTests))
    summary = {'schema_version': '1.0', 'goal_protocol': VERSION,
               'status': 'offline_protocol_passed' if result.wasSuccessful() else 'failed',
               'unit_test_count': result.testsRun, 'unit_tests_passed': result.wasSuccessful(),
               'failures': [{'test': str(test), 'traceback': error} for test, error in result.failures+result.errors],
               'unit_test_log': stream.getvalue(), 'fresh_llm_calls': False,
               'simulation_executed': False, 'geometry_verified': False,
               'physical_grasp_verified': False, 'training_eligible': False}
    if args.audit_parse:
        source = args.audit_parse.resolve()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        record = json.loads(source.read_text(encoding='utf-8'))
        try:
            # Reuse the exact returned JSON, not the later corrected response.
            original = json.loads(record['response_content'])
            decomposition = validate_decomposition(from_flat_roles(original))
            summary['historical_replay'] = {'source': str(source), 'source_sha256': before,
                'historical_status': record.get('status'), 'historical_error': record.get('error'),
                'current_status': 'decomposition_protocol_accepted',
                'scope': 'Decomposition validation only; no binding, new plan or execution asserted'}
            write_json(args.output_dir / 'replayed_decomposition.json', decomposition)
        except (KeyError, ValueError, RuntimeError) as exc:
            summary['status'] = 'failed'
            summary['historical_replay'] = {'source': str(source), 'current_status': 'rejected', 'error': str(exc)}
        summary['historical_replay']['source_unchanged'] = before == hashlib.sha256(source.read_bytes()).hexdigest()
        if not summary['historical_replay']['source_unchanged']:
            summary['status'] = 'failed'
    write_json(args.output_dir / 'regression_summary.json', summary)
    print(f"status={summary['status']} tests={result.testsRun} summary={args.output_dir/'regression_summary.json'}")
    return 0 if summary['status'] == 'offline_protocol_passed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-parse', type=Path, help='Read-only historical *_parse.json containing response_content')
    raise SystemExit(run(parser.parse_args()))
