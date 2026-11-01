"""Prepare offline, then explicitly launch four independent scene workers.

No model/embedding loading on import or prepare. No robot or semantic evaluator.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from .._legacy import ROOT, load

VERSION = 'four_scene_single_pass_v2_error_isolation'
SCENES = {
    'scene200': ROOT / 'outputs/vlm_full_scene_v1/samples/scene_200/scene',
    'scene201': Path('D:/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201/final'),
    'scene202': ROOT / 'outputs/vlm_full_scene_v1/samples/scene_202/scene',
    'd3_619': ROOT / 'outputs/real_scene_runs/d3_619_graph_20261022_094742_334872/scene',
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    # No secrets/config credentials are accepted by this serializer's callers.
    tmp = p.with_name(p.name + '.pending')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def jsonl(p, rows):
    p = Path(p)
    tmp = p.with_name(p.name + '.pending')
    p.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, p)


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def portable_source(p):
    text = str(p).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', text):
        return Path('/mnt/' + text[0].lower() + text[2:])
    return Path(p)


def old_instructions():
    """Exclude exact normalized instructions from archived tasks, not their outputs."""
    found = set()
    for base in [ROOT / 'outputs/vlm_full_scene_v1/runs', ROOT / 'outputs/real_scene_runs']:
        for p in base.rglob('round_input.json'):
            try:
                text = read(p).get('instruction')
                if isinstance(text, str):
                    found.add(text)
            except (ValueError, OSError):
                pass
    return sorted(found)


def code_snapshot(root):
    files = list((ROOT / 'embodied_pipeline').rglob('*.py'))
    files.extend((ROOT / 'embodied_pipeline/data_collection').glob('*.sh'))
    for directory in ('object_alignment_v1', 'skill_matching_v1', 'llm_audit_v1',
                      'vlm_full_scene_v1', 'habitat_demo_v1', 'scene_memory_v1'):
        files.extend((ROOT / 'outputs' / directory).glob('*.py'))
    files.append(ROOT / 'outputs/quadruped_atomic_skill_library.v0.1.json')
    files.extend((ROOT / 'outputs/skill_matching_v1').glob('*.json'))
    files.extend(ROOT / ('outputs/vlm_full_scene_v1/scenarios/' + name) for name in
                 ('scene200_symbolic_template.json', 'scene202_crossroom_6.json'))
    hashes = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT).as_posix()
        dst = root / 'reproduction' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        hashes[rel] = sha(dst)
    write(root / 'code_manifest.json', hashes)
    return hashes


def require_code_unchanged(root):
    manifest = root / 'code_manifest.json'
    pointer = root / 'runtime_revision.json'
    if pointer.is_file():
        revision = read(pointer)
        manifest = (root / revision['manifest']).resolve()
        manifest.relative_to(root.resolve())
        if sha(manifest) != revision['manifest_sha256']:
            raise ValueError('Runtime revision manifest integrity changed')
    for rel, digest in read(manifest).items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != digest:
            raise ValueError('Code/config changed since preparation; create a new version before calling API: ' + rel)


def register_runtime_revision(root):
    """Explicit offline upgrade, preserving the original snapshot and all attempts."""
    from .delivery import write_file_manifest
    root = Path(root).resolve()
    if list(root.glob('cases/*/worker.lock')):
        raise ValueError('Workers may be running; do not change the runtime revision yet')
    for folder in root.glob('cases/*/tasks/*/model_run'):
        summary = folder / 'planning_summary.json'
        error = folder / 'harness_error.json'
        if not error.is_file() and (not summary.is_file() or not read(summary).get('collection_finished_at_utc')):
            raise ValueError('Attempt outcome is not terminal; inspect before upgrading: ' + str(folder))
    # Check input identities before registering code; never regenerate tasks here.
    config = read(root / 'collection_config.json')
    for scene, info in config['scenes'].items():
        tasks_for(root, scene)
        require_graph_unchanged(root / 'cases' / scene / 'graph', info)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    revision_dir = root / 'runtime_revisions' / stamp
    revision_dir.mkdir(parents=True, exist_ok=False)
    code_snapshot(revision_dir)
    manifest = revision_dir / 'code_manifest.json'
    previous = root / 'runtime_revision.json'
    if previous.is_file():
        shutil.copyfile(previous, revision_dir / 'previous_runtime_revision.json')
    write(revision_dir / 'change.json', {'version': VERSION,
        'purpose': 'Task/scene error isolation only; no task, prompt, model or skill changes.',
        'original_code_manifest_sha256': sha(root / 'code_manifest.json'),
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'existing_attempts_preserved': True, 'automatic_replay': False})
    write(previous, {'version': VERSION, 'manifest': manifest.relative_to(root).as_posix(),
                     'manifest_sha256': sha(manifest), 'registered_at_utc': datetime.now(timezone.utc).isoformat()})
    summarize(root)
    write_file_manifest(root)
    return {'status': 'runtime_revision_registered_existing_attempts_preserved', 'revision': stamp}


def record_error(root, scene, stage, exc, record_id=None):
    """Best-effort evidence outside original raw/summary files; never retry a call."""
    now = datetime.now(timezone.utc)
    event = load('response_audit').redactor()({
        'version': VERSION, 'scene_id': scene, 'record_id': record_id, 'stage': stage,
        'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
        'recorded_at_utc': now.isoformat(),
        'recorded_at_local': now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'automatic_retry': False, 'semantic_correctness_grade': False})
    base = Path(root) / 'cases' / scene if scene is not None else Path(root)
    try:
        write(base / 'errors' / (uuid4().hex + '.json'), event)
    except Exception as logging_error:
        print('ERROR evidence write failed: ' + type(logging_error).__name__, flush=True)
    print(f"ERROR isolated scene={scene} task={record_id} stage={stage}: {event['error_type']}: {event['error']}", flush=True)
    return event


def safe_summarize(root):
    try:
        summarize(root)
        return True
    except Exception as exc:
        record_error(root, None, 'summary_update', exc)
        return False


def prepare(args):
    from .catalog import build_scene
    from .resources import audit_graph, collect_modalities
    from .quality import validate_record
    from .delivery import write_delivery_docs, write_file_manifest
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    previous = old_instructions()
    scene_reports, all_records = {}, []
    for scene_id, original in SCENES.items():
        source = portable_source(original)
        target = root / 'cases' / scene_id
        graph = target / 'graph'
        graph.mkdir(parents=True)
        for name in ('scene_graph.json', 'scene_graph_edges.json', 'scene_metadata.json'):
            if (source / name).is_file():
                shutil.copyfile(source / name, graph / name)
        audit = audit_graph(source)
        write(target / 'input_graph_audit.json', audit)
        if audit.get('blocking_issues'):
            raise ValueError('Missing/invalid graph input must be repaired from source evidence first: ' + scene_id)
        records, diagnostics, references = build_scene(scene_id, source, count=args.count,
                                                       existing_instructions=previous)
        write(target / 'generation_report.json', diagnostics)
        if len(records) != args.count:
            raise ValueError('Candidate shortage; catalogue preserved without padding: ' + scene_id)
        resources = collect_modalities(scene_id, source, target)
        write(target / 'modality_index.json', resources)
        if resources.get('missing_node_ids') or resources.get('source_gaps'):
            raise ValueError('Modality resources need review before paid collection: ' + scene_id)
        ref_by_id = references if isinstance(references, dict) else {r['record_id']: r for r in references}
        nodes = read(graph / 'scene_graph.json')
        for task in records:
            rid = task['record_id']
            if not re.fullmatch(re.escape(scene_id) + r'_[0-9]{8}', rid):
                raise ValueError('Invalid dataset ID: ' + rid)
            task['modality_index'] = f'cases/{scene_id}/modality_index.json'
            task['input_graph'] = f'cases/{scene_id}/graph'
            task['reference_artifact'] = f'cases/{scene_id}/tasks/{rid}/reference.json'
            task['reference']['separate_artifact'] = task['reference_artifact']
            task['available_skills']['bundle_path'] = 'shared/actual_collection_skill_library.json'
            task['source_type'] = 'synthetic_graph_grounded_instruction_not_robot_collection'
            task['evaluation_status'] = 'not_run_not_graded'
            task['valid_candidate'] = None
            folder = target / 'tasks' / rid
            write(folder / 'task.json', task)
            write(folder / 'reference.json', ref_by_id[rid])
            checks = validate_record(task, ref_by_id[rid], nodes)
            write(folder / 'input_quality.json', checks)
            if not checks['format_valid'] or not checks['reference_valid']:
                raise ValueError('Input format/reference check failed: ' + rid)
        jsonl(target / 'tasks.jsonl', records)
        write(target / 'generation_report.json', diagnostics)
        all_records.extend(records)
        scene_reports[scene_id] = {'count': len(records), 'source_graph': str(source),
            'graph_sha256': sha(graph / 'scene_graph.json'), 'edges_sha256': sha(graph / 'scene_graph_edges.json'),
            'metadata_sha256': sha(graph / 'scene_metadata.json') if (graph / 'scene_metadata.json').is_file() else None,
            'tasks_sha256': sha(target / 'tasks.jsonl'), 'audit': audit,
            'generation_diagnostics': diagnostics}
    if len({t['record_id'] for t in all_records}) != len(all_records):
        raise ValueError('Duplicate record IDs')
    code_snapshot(root)
    config = {'version': VERSION, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'requested_per_scene': args.count, 'scenes': scene_reports, 'model': args.model,
        'base_url': args.base_url, 'max_requests_per_task': 3, 'http_attempts_per_stage': 1,
        'validation_attempts_per_stage': 1, 'semantic_evaluation': False,
        'robot_or_simulator_execution': False, 'history_from_prior_tasks': False,
        'room_topology_used': False, 'graph_cropping': False,
        'new_graph_context_in_decomposition': True, 'budget_authorized': False,
        'max_full_collection_requests': 3 * len(all_records),
        'source_assessment': 'Graph evidence is not independently verified physical truth.',
        'room_scope_policy': 'scene201 and d3_619 unknown; scene200/202 existing author annotations only.'}
    write(root / 'collection_config.json', config)
    write_delivery_docs(root, config)
    jsonl(root / 'dataset.jsonl', all_records)
    summarize(root)
    write_file_manifest(root)
    print(json.dumps({'status': 'prepared_no_model_calls', 'output': str(root),
                      'task_count': len(all_records), 'max_new_requests': 3 * len(all_records)}, ensure_ascii=False))


def tasks_for(root, scene):
    expected = read(root / 'collection_config.json')['scenes'][scene]['tasks_sha256']
    path = root / 'cases' / scene / 'tasks.jsonl'
    if sha(path) != expected:
        raise ValueError('Task catalogue changed after preparation')
    return read_jsonl(path)


def require_graph_unchanged(graph, info):
    for name, key in (('scene_graph.json', 'graph_sha256'), ('scene_graph_edges.json', 'edges_sha256'),
                      ('scene_metadata.json', 'metadata_sha256')):
        actual = sha(graph / name) if (graph / name).is_file() else None
        if actual != info.get(key):
            raise ValueError('Frozen graph/metadata changed; no calls allowed: ' + name)


def pending_tasks(root, scene, limit):
    result = []
    for task in tasks_for(root, scene):
        path = root / 'cases' / scene / 'tasks' / task['record_id'] / 'model_run'
        # Even an interrupted/failed task is never sent again.
        if not path.exists():
            result.append(task)
    if limit is None:
        return result
    # A pilot covers different horizons and families instead of testing only
    # the first navigation template. These are still members of the same 800.
    pilot, seen = [], set()
    for horizon in ('short', 'long'):
        match = next((t for t in result if t.get('horizon') == horizon), None)
        if match is not None and len(pilot) < limit:
            pilot.append(match)
            seen.add(match['record_id'])
    for task in result:
        families = {t.get('task_family') for t in pilot}
        if task['record_id'] not in seen and task.get('task_family') not in families and len(pilot) < limit:
            pilot.append(task)
            seen.add(task['record_id'])
    return (pilot + [t for t in result if t['record_id'] not in seen])[:limit]


def stop_path(root, scene):
    return Path(root) / 'cases' / scene / 'stop_requested.json'


def request_stop(root, scenes):
    """Cooperative stop: let the one in-flight task finish, send no next task."""
    for item in scenes:
        scene, pid = (item[0], item[1].pid) if isinstance(item, tuple) else (item, None)
        write(stop_path(root, scene), {'requested_at_utc': datetime.now(timezone.utc).isoformat(),
              'policy': 'finish_current_single_pass_task_then_stop_no_new_task', 'requester_pid': os.getpid(),
              'target_worker_pid': pid})


def stop_requested_for_current(root, scene):
    path = stop_path(root, scene)
    if not path.is_file():
        return False
    try:
        target = read(path).get('target_worker_pid')
    except (ValueError, OSError):
        return True
    return target is None or target == os.getpid()


def lock_owner_state(path):
    """Diagnostic only. Never auto-delete an old lock or guess across hosts."""
    try:
        owner = read(path)
    except (OSError, ValueError):
        return 'unknown_owner_manual_review_required'
    if owner.get('hostname') != socket.gethostname() or owner.get('platform') != sys.platform:
        return 'other_host_or_platform_manual_review_required'
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 0:
        return 'invalid_owner_manual_review_required'
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return 'owner_absent_manual_review_required' if ctypes.get_last_error() == 87 else 'owner_liveness_unknown'
        try:
            state = kernel.WaitForSingleObject(handle, 0)
            return 'owner_pid_alive_or_reused' if state == 258 else 'owner_exited_manual_review_required' if state == 0 else 'owner_liveness_unknown'
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return 'owner_pid_alive_or_reused'
    except ProcessLookupError:
        return 'owner_absent_manual_review_required'
    except OSError:
        return 'owner_liveness_unknown'


def claim_worker_lock(path):
    path = Path(path)
    owner = {'pid': os.getpid(), 'hostname': socket.gethostname(), 'platform': sys.platform,
             'token': uuid4().hex, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'stale_lock_policy': 'manual_review_only_never_auto_remove'}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Scene worker lock exists: ' + str(path) + '; ' + lock_owner_state(path)
                           + '. Confirm no worker/alignment request is still running before manually archiving this lock; never delete task model_run directories.') from exc
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(owner, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return owner


def release_worker_lock(path, owner):
    path = Path(path)
    # Never remove a lock installed by another process after our own lock vanished.
    if path.is_file() and read(path).get('token') == owner['token']:
        path.unlink()


def install_stop_signals(callback):
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda _signum, _frame: callback())
    return previous


def restore_signals(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_scene(args):
    from ..orchestration.task_collection_worker import run_task
    root = args.output.resolve()
    if not args.approve_new_api_calls or not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('Worker needs explicit new-API authorization and an environment key')
    require_code_unchanged(root)
    config = read(root / 'collection_config.json')
    args.model, args.base_url = config['model'], config['base_url']
    info = config['scenes'][args.scene]
    graph = root / 'cases' / args.scene / 'graph'
    require_graph_unchanged(graph, info)
    lock = root / 'cases' / args.scene / 'worker.lock'
    owner = claim_worker_lock(lock)
    encoder = None
    stopped = [False]
    handlers = {}
    try:
        # Stops target a particular worker PID, so a new owner does not erase a
        # newly delivered stop during startup. Untargeted stops require review.
        def stop_current_worker():
            stopped[0] = True
        handlers = install_stop_signals(stop_current_worker)
        tasks = pending_tasks(root, args.scene, args.limit_per_scene)
        if args.max_api_calls is None or args.max_api_calls < 3 * len(tasks):
            raise ValueError('Worker API request ceiling missing or insufficient')
        if not tasks:
            return 0
        environment = {'candidate_policy': 'full_inventory_no_cosine_v1', 'embedding_used': False,
               'max_api_requests_this_worker': 3 * len(tasks),
               'python': sys.executable, 'model': args.model, 'dispatcher_version': VERSION,
               'dispatcher_sha256': sha(Path(__file__)), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
        write(root / 'cases' / args.scene / 'worker_environments' / (owner['token'] + '.json'), environment)
        write(root / 'cases' / args.scene / 'worker_environment.json', environment)
        for i, task in enumerate(tasks, 1):
            if config.get('account_circuit_breaker') and (root / 'account_blocked.json').exists():
                print('Account circuit breaker active; remaining tasks were not started. Recharge and explicitly acknowledge before resuming.', flush=True)
                return 2
            if stopped[0] or stop_requested_for_current(root, args.scene):
                print('Stopped before starting another task. Started task directories remain reserved.', flush=True)
                break
            print(f"START {args.scene} {i}/{len(tasks)} {task['record_id']}", flush=True)
            folder = root / 'cases' / args.scene / 'tasks' / task['record_id'] / 'model_run'
            try:
                require_graph_unchanged(graph, info)
            except Exception as exc:
                record_error(root, args.scene, 'input_integrity_scene_blocked', exc)
                # No valid common scene input: isolate this scene, not other workers.
                return 2
            task_started, task_clock = datetime.now(timezone.utc), time.monotonic()
            try:
                result = run_task(args, task, graph, folder, encoder)
                print(f"END {task['record_id']} {result['status']}", flush=True)
                if config.get('account_circuit_breaker') and 'Arrearage' in (result.get('error') or ''):
                    # Shared account condition: do not burn every remaining task.
                    # Preserve this attempt; other in-flight calls are not cancelled.
                    import uuid
                    event = {'scene_id': args.scene, 'record_id': task['record_id'],
                             'reason': 'Arrearage', 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
                    event_path = root / ('account_blocked_' + uuid.uuid4().hex + '.json')
                    write(event_path, event)
                    try:
                        with (root / 'account_blocked.json').open('x', encoding='utf-8') as marker:
                            json.dump(event, marker, ensure_ascii=False)
                    except FileExistsError:
                        pass
                    print('Account refused this request (Arrearage); stopping further tasks across this retry bundle.', flush=True)
                    return 2
            except FileExistsError as exc:
                # The exclusive claim belongs to an earlier attempt. Preserve it.
                record_error(root, args.scene, 'already_claimed_skipped', exc, task['record_id'])
            except (Exception, SystemExit) as exc:
                event = record_error(root, args.scene, 'task_worker_unhandled', exc, task['record_id'])
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    # Do not replace planning_summary/raw even if finalization failed.
                    error_path = folder / 'harness_error.json'
                    if not error_path.exists():
                        write(error_path, dict(event, status='task_harness_error_ungraded',
                            collection_started_at_utc=task_started.isoformat(),
                            collection_started_at_local=task_started.astimezone(timezone(timedelta(hours=8))).isoformat(),
                            duration_seconds=time.monotonic()-task_clock,
                            collection_finished_at_utc=event['recorded_at_utc'],
                            collection_finished_at_local=event['recorded_at_local']))
                except Exception as recording_error:
                    record_error(root, args.scene, 'task_error_persistence_failed', recording_error, task['record_id'])
                print(f"END {task['record_id']} task_harness_error_ungraded; continue next task", flush=True)
        return 0
    finally:
        try:
            if encoder is not None:
                encoder.close()
        finally:
            restore_signals(handlers)
            release_worker_lock(lock, owner)


def launch(args):
    root = args.output.resolve()
    config = read(root / 'collection_config.json')
    if not args.approve_new_api_calls:
        raise ValueError('Explicit --approve-new-api-calls required; prior budgets are not reused')
    if not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('DASHSCOPE_API_KEY required; no API calls started')
    require_code_unchanged(root)
    args.model, args.base_url = config['model'], config['base_url']
    eligible, scene_errors = {}, []
    for scene, info in config['scenes'].items():
        try:
            require_graph_unchanged(root / 'cases' / scene / 'graph', info)
            eligible[scene] = pending_tasks(root, scene, args.limit_per_scene)
        except Exception as exc:
            scene_errors.append(record_error(root, scene, 'scene_preflight', exc))
    remaining = sum(len(tasks) for tasks in eligible.values())
    maximum = 3 * remaining
    if args.max_api_calls is None or args.max_api_calls < maximum:
        raise ValueError(f'Authorize a request ceiling of at least {maximum}, or use --limit-per-scene')
    if remaining == 0:
        safe_summarize(root)
        return 2 if scene_errors else 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write(root / 'launches' / (stamp + '.json'), {'requested_tasks': remaining, 'max_new_api_requests': maximum,
          'approved_request_ceiling': args.max_api_calls, 'cost_ceiling_cny': None,
          'known_limit': 'Request count is not a monetary budget; provider token prices/limits must be confirmed.',
          'semantic_grading': False, 'repeat_failed_tasks': False})
    children, handles = [], []
    stopped = [False]
    def stop_all_workers():
        # Signal callbacks only set memory flags: filesystem/console writes here
        # could be interrupted by another signal and corrupt a pending write.
        stopped[0] = True
    handlers = install_stop_signals(stop_all_workers)
    try:
        for scene, tasks in eligible.items():
            if stopped[0]:
                break
            if not tasks:
                continue
            command = [sys.executable, '-m', 'embodied_pipeline.data_collection.batch', 'worker',
                '--output', str(root), '--scene', scene, '--model', args.model, '--base-url', args.base_url,
                '--embedding-python', str(args.embedding_python), '--embedding-model', str(args.embedding_model),
                '--request-timeout', str(args.request_timeout), '--approve-new-api-calls',
                '--max-api-calls', str(3 * len(tasks))]
            if args.limit_per_scene is not None:
                command += ['--limit-per-scene', str(args.limit_per_scene)]
            try:
                logpath = root / 'cases' / scene / ('worker_' + stamp + '.log')
                handle = logpath.open('w', encoding='utf-8')
                handles.append(handle)
                children.append((scene, subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)))
            except Exception as exc:
                scene_errors.append(record_error(root, scene, 'scene_launch', exc))
                continue
            if stopped[0]:
                request_stop(root, children)
            print('Worker started: ' + scene + '; log=' + str(logpath), flush=True)
        last = 0.
        stop_sent = False
        while any(p.poll() is None for _, p in children):
            if stopped[0] and not stop_sent:
                request_stop(root, children)
                print('Cooperative stop requested. Current tasks may finish their remaining single-pass stages; no next task. Keep this window open.', flush=True)
                stop_sent = True
            if time.monotonic() - last > 30:
                if safe_summarize(root):
                    print('Progress updated: ' + str(root / 'quality_report.json'), flush=True)
                last = time.monotonic()
            time.sleep(1)
    except BaseException:
        # Do not orphan alignment/embedding grandchildren by killing their owner.
        request_stop(root, children)
        raise
    finally:
        try:
            while any(p.poll() is None for _, p in children):
                time.sleep(.2)
            for h in handles:
                h.close()
            safe_summarize(root)
        finally:
            restore_signals(handlers)
    return 0 if not scene_errors and all(p.returncode == 0 for _, p in children) else 2


def summarize(root):
    from .quality import validate_record, duplicate_groups
    root = Path(root)
    statuses, scene_counts, dataset = Counter(), {}, []
    refs, checks_all, dimensions = {}, [], {}
    from .plan_artifact import has_complete
    raw_records = plans = plan_files = started = 0
    for tasks_path in sorted((root / 'cases').glob('*/tasks.jsonl')):
        scene, counts = tasks_path.parent.name, Counter()
        graph_path = tasks_path.parent / 'graph/scene_graph.json'
        nodes = read(graph_path) if graph_path.is_file() else []
        horizons, families, rooms = Counter(), Counter(), Counter()
        for task in read_jsonl(tasks_path):
            record = dict(task)
            run = tasks_path.parent / 'tasks' / task['record_id'] / 'model_run'
            status = 'not_run'
            summary = None
            if run.exists():
                started += 1
                if (run / 'planning_summary.json').is_file():
                    try:
                        summary = read(run / 'planning_summary.json')
                    except (OSError, ValueError):
                        summary = None
                status = summary.get('status', 'started_outcome_unknown') if summary else 'started_outcome_unknown'
            harness_error = None
            if (run / 'harness_error.json').is_file():
                try:
                    harness_error = read(run / 'harness_error.json')
                except (OSError, ValueError):
                    harness_error = {'status': 'task_harness_error_evidence_unreadable'}
                status = harness_error.get('status', 'task_harness_error_ungraded')
            raw = list((run / 'llm_response_audit').glob('*_http_*.json'))
            raw_records += bool(raw)
            # Every claimed task now writes engineering_plan.json; count complete plans separately.
            plan_files += (run / 'engineering_plan.json').is_file()
            plans += has_complete(run / 'engineering_plan.json')
            ref_path = run.parent / 'reference.json'
            reference = read(ref_path) if ref_path.is_file() else None
            refs[task['record_id']] = reference
            # In-progress files may be partially written. Only check model files
            # after a terminal summary; retain all original files unchanged.
            terminal = summary and summary.get('collection_finished_at_utc')
            kwargs = {}
            if terminal:
                for key, name in [('decomposition', 'task_decomposition.json'), ('skill_plan', 'skill_plan.json')]:
                    if (run / name).is_file():
                        kwargs[key] = run / name
            checks = validate_record(task, reference, nodes, **kwargs)
            if reference is None:
                checks['reference_valid'] = False
                checks['reference_issues'].append({'code': 'missing_reference_artifact'})
            checks_all.append(checks)
            marker = {}
            if (run / 'attempt_started.json').is_file():
                try:
                    marker = read(run / 'attempt_started.json')
                except (OSError, ValueError):
                    pass
            result_times = dict(summary or {})
            for key in ('collection_started_at_utc', 'collection_started_at_local', 'duration_seconds'):
                if harness_error and key in harness_error and not result_times.get(key):
                    result_times[key] = harness_error[key]
            for key in ('collection_finished_at_utc', 'collection_finished_at_local'):
                if harness_error and key in harness_error:
                    result_times[key] = harness_error[key]
            record['collection_times'] = {
                'source_scene_capture_time': task.get('source_scene_capture_time'),
                'source_scene_capture_time_status': 'unknown_not_inferred_from_file_mtime',
                'instruction_generated_at_utc': task.get('instruction_generated_at_utc'),
                'instruction_generated_at_local': task.get('instruction_generated_at_local'),
                'started_at_utc': result_times.get('collection_started_at_utc', marker.get('started_at_utc')),
                'started_at_local': result_times.get('collection_started_at_local', marker.get('started_at_local')),
                'finished_at_utc': result_times.get('collection_finished_at_utc'),
                'finished_at_local': result_times.get('collection_finished_at_local'),
                'duration_seconds': result_times.get('duration_seconds'),
                'request_response_time_evidence': (run / 'llm_response_audit').relative_to(root).as_posix() if run.exists() else None,
                'local_timezone': 'UTC+08:00'}
            record.update(model_run=run.relative_to(root).as_posix() if run.exists() else None,
                          collection_status=status, model_result=summary,
                          worker_error=harness_error,
                          structural_quality=checks,
                          semantic_evaluation='pending', valid_candidate=None)
            dataset.append(record)
            counts[status] += 1
            statuses[status] += 1
            horizons[task.get('horizon', 'unknown')] += 1
            families[task.get('task_family', 'unknown')] += 1
            rooms[task.get('room_scope', 'unknown')] += 1
        scene_counts[scene] = dict(counts)
        dimensions[scene] = {'horizon': dict(horizons), 'task_type': dict(families), 'room_scope': dict(rooms)}
        jsonl(tasks_path.parent / 'dataset.jsonl', [r for r in dataset if r['scene_id'] == scene])
    jsonl(root / 'dataset.jsonl', dataset)
    jsonl(root / 'timestamps_index.jsonl', [dict(record_id=r['record_id'], scene_id=r['scene_id'],
                                               **r['collection_times']) for r in dataset])
    exact = Counter(hashlib.sha256(json.dumps({k: t.get(k) for k in
                       ('scene_id', 'instruction', 'initial_state', 'constraints', 'goal')},
                       sort_keys=True, ensure_ascii=False).encode()).hexdigest() for t in dataset)
    comparable = [t for t in dataset if all(k in t for k in ('goal', 'initial_state', 'constraints'))
                  and isinstance(refs.get(t['record_id']), dict) and 'steps' in refs[t['record_id']]]
    near = duplicate_groups(comparable, refs)
    order_groups = {}
    for task in dataset:
        key = task.get('dedup', {}).get('near_duplicate_group')
        if key:
            order_groups.setdefault(key, []).append(task['record_id'])
    order_variants = [{'signature_sha256': key, 'record_ids': ids} for key, ids in order_groups.items() if len(ids)>1]
    write(root / 'duplicate_audit.json', {'same_spec_rewording_groups': near,
        'order_variant_groups_for_review': order_variants, 'records_not_comparable': len(dataset)-len(comparable),
        'automatically_removed': 0, 'note': 'Order variants retain distinct requested temporal goals; not automatically discarded.'})
    jsonl(root / 'structural_quality.jsonl', checks_all)
    structural_failed = sum(not r['format_valid'] or not r['reference_valid'] for r in checks_all)
    write(root / 'quality_report.json', {'version': VERSION,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'catalogue_records': len(dataset),
        'started_records': started, 'records_with_http_response': raw_records,
        'engineering_json_count_ungraded': plans, 'engineering_plan_file_count_incl_incomplete': plan_files,
        'status_counts': dict(statuses),
        'scene_counts': scene_counts, 'exact_duplicate_records': sum(n-1 for n in exact.values() if n>1),
        'stratified_counts': dimensions, 'raw_task_records_before_output_filtering': len(dataset),
        'records_retained_after_output_filtering': len(dataset), 'records_removed': 0,
        'structural_check_failed_records': structural_failed,
        'structural_check_passed_records_not_semantic_valid': len(dataset)-structural_failed,
        'near_duplicate_same_spec_groups': len(near), 'order_variant_review_groups': len(order_variants),
        'semantic_review_pending': len(dataset), 'valid_candidate_records': None,
        'confirmed_valid_candidate_records': 0,
        'correctness_invalid_records': None, 'records_removed_by_semantic_filter': 0,
        'near_duplicate_audit': 'duplicate_audit.json',
        'human_review': 'not_performed', 'robot_or_simulator_execution': False,
        'counting_unit': 'One independent task record, not frames, objects, steps or stage calls.',
        'not_a_10000_record_delivery': True})
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'summarize', 'verify', 'finalize', 'register-runtime'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--model', default='qwen3.8-flash')
    parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--scene', choices=list(SCENES))
    parser.add_argument('--embedding-python', type=Path, default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model', type=Path, default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    parser.add_argument('--request-timeout', type=float, default=180.)
    parser.add_argument('--limit-per-scene', type=int)
    parser.add_argument('--approve-new-api-calls', action='store_true')
    parser.add_argument('--max-api-calls', type=int)
    args = parser.parse_args(argv)
    if args.count < 1 or (args.limit_per_scene is not None and args.limit_per_scene < 1) or not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error('Counts/timeouts must be positive')
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'summarize':
        summarize(args.output)
    elif args.command == 'register-runtime':
        print(json.dumps(register_runtime_revision(args.output), ensure_ascii=False))
    elif args.command in ('verify', 'finalize'):
        from .delivery import write_delivery_docs, write_file_manifest, verify_file_manifest
        if args.command == 'finalize':
            if list(args.output.glob('cases/*/worker.lock')):
                raise ValueError('Wait for every worker before finalizing delivery')
            write_delivery_docs(args.output, read(args.output / 'collection_config.json'))
            summarize(args.output)
            manifest = write_file_manifest(args.output)
            print(json.dumps({'status': 'finalized_not_semantically_graded', 'file_count': manifest['count'],
                              'output': str(args.output)}, ensure_ascii=False))
        else:
            result = verify_file_manifest(args.output)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result.get('status') == 'verified' else 1
    elif args.command == 'worker':
        if args.scene is None:
            parser.error('--scene required for worker')
        return run_scene(args)
    else:
        return launch(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
