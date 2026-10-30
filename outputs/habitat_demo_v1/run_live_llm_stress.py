#!/usr/bin/env python3
"""Bounded 12/18-round live-LLM stress cases; no prewritten skill plans."""
import argparse
import copy
import math
import os
from datetime import datetime
from pathlib import Path

from run_live_llm_sequence import validate_instructions
from run_live_llm_regression import run_suite
from run_continuous_sequence import SCENE
from run_navigation_demo import read_json
from match_skills import DEFAULT_URL
from stage_collision_config import STAGE_MARGIN_M
from visual_audit import add_arguments as add_audit_arguments


def stress_cases():
    def row(index, text, held=False, visits=None):
        value = {'round': index, 'instruction': text,
                 'expected_end': {'held_object_id': 67 if held else None, 'lifted': held}}
        if visits is not None:
            value['expected_navigation_targets'] = visits
        return value

    rows = [
        row(1, 'Visit the sideboard, then the kettle, then the sideboard, then the kettle.',
            visits=[13, 67, 13, 67]),
        row(2, 'Pick up and lift the kettle. Keep holding it for my next instruction.', True),
        row(3, 'Carry the held kettle to the sideboard. Do not place it; keep holding it.', True,
            visits=[13]),
        row(4, 'Place the kettle you are holding on the sideboard.'),
        row(5, 'Visit the moved kettle, then the sideboard, then the moved kettle.',
            visits=[67, 13, 67]),
        row(6, 'Pick up the moved kettle, lift it, carry it to the sideboard and place it there.'),
        row(7, 'Pick up and lift the kettle again. Keep holding it for the next instruction.', True),
        row(8, 'While holding the kettle, visit the sideboard. Keep holding the kettle.', True,
            visits=[13]),
        row(9, 'Place the held kettle on the sideboard.'),
        row(10, 'Visit the sideboard, then the kettle, then the sideboard, then the kettle, then the sideboard.',
            visits=[13, 67, 13, 67, 13]),
        row(11, 'Go to the moved kettle, pick it up and lift it. Keep holding it.', True),
        row(12, 'Carry the held kettle to the sideboard and place it there.'),
    ]
    extra = [
        row(13, 'Pick up the kettle, lift it, carry it to the sideboard and place it there.'),
        row(14, 'Visit the kettle, then the sideboard, then the kettle, then the sideboard.',
            visits=[67, 13, 67, 13]),
        row(15, 'Go to the kettle, pick it up and lift it. Keep holding it.', True),
        row(16, 'Visit the sideboard while carrying the kettle; do not release it.', True,
            visits=[13]),
        row(17, 'Visit the sideboard once more while holding the kettle. Keep holding it.', True,
            visits=[13]),
        row(18, 'Place the kettle you are holding on the sideboard.'),
    ]
    def case(name, seed, instructions, held, placements):
        episode = validate_instructions({'schema_version': '1.0', 'training_eligible': False,
            'planning_origin': 'fresh_llm_each_round_live_memory', 'rounds': copy.deepcopy(instructions)})
        return dict(name=name, seed=seed, instructions=episode, held_continuations=held,
                    expected_placements=len(placements), placement_rounds=placements)
    return [case('mixed_12_seed_202', 202, rows, [3, 4, 8, 9, 12], [4, 6, 9, 12]),
            case('extended_18_seed_203', 203, rows + extra, [3, 4, 8, 9, 12, 16, 17, 18],
                 [4, 6, 9, 12, 13, 18])]


def stress_checks(result, root, case):
    actions = [a for row in result.get('rounds', []) for a in row.get('actions', [])]
    motion = [check for a in actions for check in a.get('motion_safety_checks', [])]
    config = result.get('stage_collision_configuration', {})
    margin = config.get('loaded_stage_margin_m')
    checks = {
        'fresh_llm_execution_not_fixture': result.get('fresh_llm_calls') is True
            and result.get('planning_origin') == 'fresh_llm_each_round_live_memory',
        'static_collision_enabled_and_margin_verified': config.get('status') == 'loaded_parameters_verified'
            and config.get('loaded_stage_collidable') is True and type(margin) in (int, float)
            and math.isfinite(margin) and abs(margin - STAGE_MARGIN_M) <= 1e-7,
        'motion_evidence_all_sampled_clear': bool(motion)
            and all(m.get('status') == 'sampled_clear' for m in motion),
        'motion_evidence_for_each_moving_action': bool(actions) and all(
            bool(a.get('motion_safety_checks')) for a in actions if a.get('skill_id') in
            ('grasp_object', 'lift_held_object', 'place_object_on_surface')),
    }
    totals = {k: 0 for k in ('logical_request_count', 'http_attempt_count', 'http_response_count',
        'transport_failure_count', 'http_error_count', 'program_rejection_count',
        'content_parse_rejection_count', 'requests_without_transport_result_count')}
    reports = 0
    available_reports = 0
    for descriptor in case['instructions']['rounds']:
        directory = root / f"round_{descriptor['round']}" / 'planning'
        try:
            report = read_json(directory / 'llm_audit_report/llm_error_summary.json')
            # Require raw HTTP and request records, not just a count in a report.
            records_complete = all(
                type(report.get(k)) is int and report[k] >= 0 for k in totals)
            if not records_complete:
                continue
            # Keep failure/pending counts even when that round's evidence is incomplete.
            available_reports += 1
            for key in totals:
                totals[key] += report[key]
            if (report['logical_request_count'] < 4
                    or report['requests_without_transport_result_count'] != 0
                    or len(list(directory.rglob('llm_response_audit/*_request.json'))) != report['logical_request_count']
                    or len(list(directory.rglob('llm_response_audit/*_http_*.json'))) != report['http_response_count']):
                continue
            reports += 1
        except (OSError, ValueError, KeyError, TypeError):
            continue
    checks['every_round_llm_raw_audit_present'] = reports == len(case['instructions']['rounds'])
    audit = result.get('visual_audit')
    if audit is not None or case.get('visual_audit_required'):
        audit = audit if isinstance(audit, dict) else {}
        checks['visual_audit_saved_without_render_errors'] = audit.get('status') == 'saved' and (
            type(audit.get('frame_count')) is int and audit['frame_count'] > 0
            and audit.get('render_error_count') == 0 and audit.get('frame_limit_reached') is False)
    return {'checks': checks, 'statistics': dict(totals,
        available_audit_report_count=available_reports, audited_round_count=reports,
        sampled_motion_check_count=len(motion),
        scope='Counts from available valid audit reports including incomplete rounds; finite sampled motion checks, not billing or continuous collision proof')}


def main():
    catalog = stress_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=[c['name'] for c in catalog],
                        help='Default: mixed_12_seed_202 only; repeat to select both')
    parser.add_argument('--preflight-only', action='store_true', help='Input checks only; no API or Habitat')
    add_audit_arguments(parser)
    parser.set_defaults(audit_max_frames=4000)
    parser.add_argument('--scene-dir', type=Path,
        default=SCENE / 'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene')
    parser.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    parser.add_argument('--model', default='qwen3.8-flash')
    parser.add_argument('--base-url', default=DEFAULT_URL)
    parser.add_argument('--output-dir', type=Path,
        default=SCENE / 'live_llm_stress' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    args.case = args.case or ['mixed_12_seed_202']
    for case in catalog:
        case['visual_audit_required'] = args.visual_audit
    if args.audit_stride <= 0 or args.audit_max_frames <= 0:
        parser.error('Audit stride and frame budget must be positive')
    if not args.preflight_only:
        if not os.environ.get('DASHSCOPE_API_KEY', '').strip():
            parser.error('DASHSCOPE_API_KEY required; no fixture fallback')
        if not Path(args.planning_python).is_file():
            parser.error('Planning Python interpreter does not exist')
    return run_suite(args, case_catalog=catalog, extra_case_checks=stress_checks)


if __name__ == '__main__':
    raise SystemExit(main())
