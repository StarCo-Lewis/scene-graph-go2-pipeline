"""Offline scene-scoped snapshot; never change source records or call models."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import shutil

from .batch import read, read_jsonl, write, jsonl, sha
from .quality import validate_record, duplicate_groups, content_hash

VERSION = 'selected_scene_report_v1'
DEFAULT_SCENES = ('scene200', 'scene202', 'd3_619')


def time_window(records):
    """Wall-clock span, not the sum of parallel task durations."""
    starts = [datetime.fromisoformat(r['collection_times']['started_at_utc'])
              for r in records if r['collection_times'].get('started_at_utc')]
    ends = [datetime.fromisoformat(r['collection_times']['finished_at_utc'])
            for r in records if r['collection_times'].get('finished_at_utc')]
    local = timezone(timedelta(hours=8))
    return {'started_at_local': min(starts).astimezone(local).isoformat() if starts else None,
            'last_finished_at_local': max(ends).astimezone(local).isoformat() if ends else None,
            'elapsed_seconds': (max(ends) - min(starts)).total_seconds() if starts and ends else None,
            'finished_records': len(ends), 'records': len(records)}


def summarize_selected(root, scenes=DEFAULT_SCENES, pilot_record_ids=()):
    root = Path(root).resolve()
    config = read(root / 'collection_config.json')
    if not scenes or len(set(scenes)) != len(scenes) or any(
            not re.fullmatch(r'[a-zA-Z0-9_]+', s) or s not in config['scenes'] for s in scenes):
        raise ValueError('Select unique scene IDs present in collection_config.json')
    dest = root / 'reports' / '_'.join(scenes)
    records, refs, checks, issues = [], {}, [], []
    statuses, per_scene, dimensions = Counter(), {}, {}
    integrity = []

    def optional(path):
        if not path.is_file():
            return None
        try:
            result = read(path)
            if not isinstance(result, dict):
                raise ValueError('Expected JSON object')
            return result
        except (OSError, ValueError) as exc:
            issues.append({'path': path.relative_to(root).as_posix(), 'error_type': type(exc).__name__})
            return None

    for scene in scenes:
        base = root / 'cases' / scene
        # Excluded scenes are never opened, even if their records are malformed.
        for name, key in [('tasks.jsonl', 'tasks_sha256'), ('graph/scene_graph.json', 'graph_sha256'),
                          ('graph/scene_graph_edges.json', 'edges_sha256'),
                          ('graph/scene_metadata.json', 'metadata_sha256')]:
            expected = config['scenes'][scene].get(key)
            if expected:
                actual = sha(base / name)
                if actual != expected:
                    raise ValueError(f'Frozen input hash mismatch: {scene}/{name}')
                integrity.append({'path': (base / name).relative_to(root).as_posix(), 'sha256': actual})
        tasks, nodes = read_jsonl(base / 'tasks.jsonl'), read(base / 'graph/scene_graph.json')
        if len(tasks) != config['scenes'][scene]['count']:
            raise ValueError(f'Catalogue count mismatch: {scene}')
        counts = Counter()
        for task in tasks:
            rid = task['record_id']
            if task['scene_id'] != scene or not re.fullmatch(re.escape(scene) + r'_[0-9]{8}', rid) or rid in refs:
                raise ValueError(f'Invalid or duplicate scene record ID: {rid}')
            run = base / 'tasks' / rid / 'model_run'
            summary = optional(run / 'planning_summary.json')
            error = optional(run / 'harness_error.json')
            marker = optional(run / 'attempt_started.json') or {}
            reference = optional(run.parent / 'reference.json')
            refs[rid] = reference
            times = dict(summary or {})
            for key, value in (error or {}).items():
                if key.startswith('collection_finished_at_') or not times.get(key):
                    times[key] = value
            terminal = bool(times.get('collection_finished_at_utc'))
            status = (error or summary or {}).get('status', 'started_outcome_unknown' if run.exists() else 'not_run')
            kwargs = {key: run / name for key, name in [('decomposition', 'task_decomposition.json'),
                      ('skill_plan', 'skill_plan.json')] if terminal and (run / name).is_file()}
            check = validate_record(task, reference, nodes, **kwargs)
            if reference is None:
                check['reference_valid'] = False
                check['reference_issues'].append({'code': 'missing_or_unreadable_reference_artifact'})
            checks.append(check)
            record = dict(task, collection_status=status, model_result=summary, worker_error=error,
                          model_run=run.relative_to(root).as_posix() if run.exists() else None,
                          structural_quality=check, semantic_evaluation='pending', valid_candidate=None,
                          engineering_json_present=(run / 'engineering_plan.json').is_file(),
                          http_response_present=bool(list((run / 'llm_response_audit').glob('*_http_*.json'))))
            record['collection_times'] = {
                'source_scene_capture_time': task.get('source_scene_capture_time'),
                'source_scene_capture_time_status': 'unknown_not_inferred_from_file_mtime' if task.get('source_scene_capture_time') is None else 'as_recorded',
                'instruction_generated_at_utc': task.get('instruction_generated_at_utc'),
                'instruction_generated_at_local': task.get('instruction_generated_at_local'),
                'started_at_utc': times.get('collection_started_at_utc') or marker.get('started_at_utc'),
                'started_at_local': times.get('collection_started_at_local') or marker.get('started_at_local'),
                'finished_at_utc': times.get('collection_finished_at_utc'),
                'finished_at_local': times.get('collection_finished_at_local'),
                'duration_seconds': times.get('duration_seconds'),
                'request_response_time_evidence': (run / 'llm_response_audit').relative_to(root).as_posix() if run.exists() else None}
            records.append(record)
            counts[status] += 1
            statuses[status] += 1
        per_scene[scene] = dict(counts)
        dimensions[scene] = {key: dict(Counter(t.get(key, 'unknown') for t in tasks))
                             for key in ('horizon', 'task_family', 'room_scope')}

    pilot_ids = set(pilot_record_ids)
    if pilot_ids - set(refs):
        raise ValueError('Pilot IDs must belong to the selected catalogue')
    timing = {'timezone': 'UTC+08:00', 'basis': 'Task attempt start through terminal summary finish; concurrent wall time, not summed CPU/API time.',
              'all_attempts_including_gaps': time_window(records), 'pilot_record_ids': sorted(pilot_ids),
              'continuation_excluding_explicit_pilots': time_window([r for r in records if r['record_id'] not in pilot_ids]) if pilot_ids else None,
              'per_scene': {s: {'all_attempts_including_gaps': time_window([r for r in records if r['scene_id'] == s]),
                               'continuation_excluding_explicit_pilots': time_window([r for r in records if r['scene_id'] == s and r['record_id'] not in pilot_ids]) if pilot_ids else None}
                            for s in scenes}}
    comparable = [r for r in records if isinstance(refs[r['record_id']], dict) and 'steps' in refs[r['record_id']]]
    near = duplicate_groups(comparable, refs)
    exact = defaultdict(list)
    for r in records:
        exact[content_hash({k: r.get(k) for k in ('scene_id', 'instruction', 'initial_state', 'constraints', 'goal')})].append(r['record_id'])
    exact_groups = [{'sha256': h, 'record_ids': ids} for h, ids in exact.items() if len(ids) > 1]
    locks = [str(root / 'cases' / s / 'worker.lock') for s in scenes if (root / 'cases' / s / 'worker.lock').exists()]
    finished = sum(bool(r['collection_times']['finished_at_utc']) for r in records)
    report = {'version': VERSION, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_bundle_root': str(root), 'artifact_reference_base': 'source_bundle_root, not this report directory',
              'selected_scenes': list(scenes), 'excluded_scenes': sorted(set(config['scenes']) - set(scenes)),
              'catalogue_records': len(records), 'finished_records': finished,
              'started_records': sum(bool(r['collection_times']['started_at_utc']) for r in records),
              'engineering_json_count_ungraded': sum(r['engineering_json_present'] for r in records),
              'records_with_http_response': sum(r['http_response_present'] for r in records),
              'status_counts': dict(statuses), 'scene_counts': per_scene, 'stratified_counts': dimensions,
              'raw_task_records_before_output_filtering': len(records), 'records_retained_after_output_filtering': len(records),
              'records_removed': 0, 'exact_duplicate_records': sum(len(g['record_ids']) - 1 for g in exact_groups),
              'near_duplicate_same_spec_groups': len(near),
              'structural_check_failed_records': sum(not c['format_valid'] or not c['reference_valid'] for c in checks),
              'semantic_review_pending': len(records), 'valid_candidate_records': None, 'correctness_invalid_records': None,
              'human_review': 'not_performed', 'robot_or_simulator_execution': False,
              'modality_revalidation': 'not_performed_by_this_report; see original resource audit and modality index',
              'input_hash_checks': integrity, 'read_issues': issues, 'active_worker_locks': locks,
              'snapshot_not_final': bool(locks or issues or finished != len(records)), 'timing': timing,
              'generator_sha256': sha(Path(__file__)), 'collection_config_sha256': sha(root / 'collection_config.json')}
    dest.mkdir(parents=True, exist_ok=True)
    jsonl(dest / 'dataset.jsonl', records)
    jsonl(dest / 'timestamps_index.jsonl', [dict(record_id=r['record_id'], scene_id=r['scene_id'], **r['collection_times']) for r in records])
    jsonl(dest / 'structural_quality.jsonl', checks)
    write(dest / 'duplicate_audit.json', {'exact_groups': exact_groups, 'same_spec_rewording_groups': near,
          'records_not_comparable': len(records) - len(comparable), 'automatically_removed': 0})
    write(dest / 'quality_report.json', report)
    shutil.copy2(__file__, dest / 'report_generator.py')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', default=list(DEFAULT_SCENES))
    parser.add_argument('--pilot-record-ids', nargs='*', default=[])
    args = parser.parse_args()
    report = summarize_selected(args.output, args.scenes, args.pilot_record_ids)
    print({k: report[k] for k in ('selected_scenes', 'catalogue_records', 'finished_records', 'engineering_json_count_ungraded', 'snapshot_not_final', 'timing')})


if __name__ == '__main__':
    main()
