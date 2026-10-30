#!/usr/bin/env python3
"""Frozen-engine batch evaluation; no repair, replay, simulator or fallback plan."""
import argparse
import copy
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from run_json_long_horizon import load_nodes, initial_state, prepare_round, SCENE, OUT
from run_navigation_demo import read_json, write_json
from run_live_llm_sequence import execute_worker
from json_symbolic_review import review_symbolic
from json_stability_cases import catalog, validate_cases, DEFAULT_CASES
from match_skills import DEFAULT_URL
from build_report import build as build_audit_report

CODE_DIRS = ('habitat_demo_v1', 'object_alignment_v1', 'skill_matching_v1', 'scene_memory_v1', 'llm_audit_v1')
AUDIT_KEYS = ('logical_request_count', 'http_attempt_count', 'transport_failure_count',
              'http_error_count', 'program_rejection_count', 'content_parse_rejection_count',
              'requests_without_transport_result_count')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_engine(destination):
    files = {}
    for folder in CODE_DIRS:
        sources = list((OUT / folder).glob('*.py'))
        if folder == 'skill_matching_v1':
            sources += list((OUT / folder).glob('skill_semantic_profiles.json'))
        for source in sorted(sources):
            if source.name.startswith('test'):
                continue
            relative = f'outputs/{folder}/{source.name}'
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[relative] = sha(target)
            if sha(source) != files[relative]:
                raise ValueError('Engine source changed while freezing')
    relative = 'outputs/quadruped_atomic_skill_library.v0.1.json'
    target = destination / relative
    shutil.copyfile(OUT / 'quadruped_atomic_skill_library.v0.1.json', target)
    files[relative] = sha(target)
    manifest = {'files': files, 'source_workspace_outputs': str(OUT),
                'policy': 'Workers execute copied code only; no code/prompt edits during this batch'}
    write_json(destination / 'engine_manifest.json', manifest)
    return manifest


def frozen_unchanged(root, manifest):
    try:
        return all(sha(root / path) == digest for path, digest in manifest['files'].items())
    except OSError:
        return False


def plan_one_round(args, folder):
    # Run the unchanged planner in the case worker, not another detached worker.
    # Its alignment subprocess inherits the case process group, so a case-level
    # timeout/interrupt can terminate the entire group in WSL without orphan API calls.
    from plan_live_round import plan
    handler = logging.FileHandler(folder/'planning.console.log',encoding='utf-8')
    logging.getLogger().addHandler(handler)
    try:
        return plan(SimpleNamespace(output_dir=folder,round_input=folder/'round_input.json',
            model=args.model,base_url=args.base_url,reach=1.5),reviewer=review_symbolic)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def positive_worker(args, case):
    root, nodes = args.output_dir, case['nodes']
    state = initial_state(nodes)
    summary = {'status': 'running', 'kind': 'positive', 'rounds': [],
        'expected_round_count': len(case['rounds']), 'simulation_executed': False,
        'training_eligible': False, 'geometry_verified': False}
    destination = root / 'case_summary.json'
    try:
        for descriptor in case['rounds']:
            i = descriptor['round']
            planning_dir = root / f'round_{i}/planning'
            prepare_round(planning_dir, descriptor, state, nodes, args.model, args.base_url)
            row = {'round': i, 'status': 'planning', 'instruction': descriptor['instruction'],
                   'begin_state': copy.deepcopy(state)}
            summary['rounds'].append(row)
            write_json(destination, summary)
            print(f'START JSON round={i} revision={state["backend"]["memory_version"]}', flush=True)
            if plan_one_round(args,planning_dir) != 0:
                raise ValueError('Frozen round planner failed; later rounds skipped')
            planning = read_json(planning_dir / 'planning_summary.json')
            if (planning.get('status') != 'planning_passed'
                    or planning.get('validation_mode') != 'json_symbolic_test'
                    or planning.get('input_memory_version') != i-1
                    or read_json(planning_dir / 'runtime_state.json') != state):
                raise ValueError('Round planning status/mode/input changed')
            review = review_symbolic(dict(descriptor, alignment=read_json(planning_dir/'object_alignment.json'),
                plan=read_json(planning_dir/'skill_plan_final.json')),
                read_json(planning_dir/'execution_skill_library.json'),
                read_json(planning_dir/'input_memory/scene/scene_graph.json'), state, 1.5)
            if review.get('status') != 'symbolically_valid_not_executed':
                raise ValueError('Unexpected symbolic review status')
            state = review['predicted_end_state']
            state['backend']['memory_version'] = i
            row.update(status='json_symbolic_passed', action_count=len(review['symbolic_trace']),
                       end_state=copy.deepcopy(state))
            write_json(root/f'round_{i}/symbolic_projection_after.json', state)
            write_json(destination, summary)
            print(f'END JSON round={i} status=json_symbolic_passed', flush=True)
        summary['status'] = 'json_symbolic_passed'
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        if summary['rounds']:
            summary['rounds'][-1]['status'] = 'failed'
            folder = root / f'round_{summary["rounds"][-1]["round"]}/planning'
            if (folder/'planning_summary.json').exists():
                failure = read_json(folder/'planning_summary.json')
                summary.update(failure_stage=failure.get('failed_stage'), failure_cause=failure.get('error'))
        logging.exception('Case stopped; later rounds in this case skipped, no replay')
    summary['completed_round_count'] = sum(r['status']=='json_symbolic_passed' for r in summary['rounds'])
    summary['skipped_round_count'] = len(case['rounds']) - len(summary['rounds'])
    summary['action_count'] = sum(r.get('action_count',0) for r in summary['rounds'])
    write_json(destination, summary)
    return 0 if summary['status']=='json_symbolic_passed' else 2


def negative_worker(args, case):
    root = args.output_dir
    scene = root/'planning/input_scene'
    write_json(scene/'scene_graph.json', case['nodes'])
    write_json(scene/'scene_graph_edges.json', case['edges'])
    # No mapped-instance hint or author-written binding is given for these tests.
    contract = {'adapter_mode': 'navigation', 'supported_skill_ids': ['navigate_to_object'],
        'planning_capabilities': {'validate_supplied_on_in_edges': True},
        'planning_context': 'Only navigate to the requested object. Preserve explicitly required '
            'object-to-object initial relations. Do not invent a relation or an alternative object.',
        'simulation_executed': False, 'training_eligible': False}
    write_json(root/'planning/execution_contract.json', contract)
    command = [sys.executable, str(OUT/'object_alignment_v1/align_objects.py'),
        '--scene-dir', str(scene), '--instruction', case['instruction'], '--matching-mode', 'llm_graph',
        '--decomposition-format', 'flat_roles', '--execution-contract', str(root/'planning/execution_contract.json'),
        '--model', args.model, '--base-url', args.base_url, '--request-timeout', '180', '--request-attempts', '2',
        '--output', str(root/'planning/object_alignment.json'),
        '--decomposition-output', str(root/'planning/task_decomposition.json')]
    exit_code, process_error = None, None
    try:
        with (root/'planning/planning.console.log').open('w',encoding='utf-8') as log:
            # No nested detached session; alignment has no further worker children.
            completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            exit_code = completed.returncode
    except Exception as exc:
        process_error = str(exc)
    result_path = root/'planning/object_alignment.json'
    result = read_json(result_path) if result_path.exists() else {}
    actual = result.get('error',{}).get('code')
    passed = exit_code == 2 and result.get('status') == 'error' and actual in case['expected_error_codes']
    summary = {'status': 'expected_refusal' if passed else 'failed', 'kind': 'negative',
        'expected_error_codes': case['expected_error_codes'], 'actual_error_code': actual,
        'process_exit_code': exit_code, 'process_error': process_error,
        'failure_cause': result.get('error',{}).get('message'),
        'simulation_executed': False, 'geometry_verified': False, 'training_eligible': False,
        'scope': 'Binding refusal test only; API failure is NOT an expected semantic refusal'}
    write_json(root/'case_summary.json', summary)
    return 0 if passed else 2


def classify(kind, result, audit):
    accepted = result.get('status') == ('json_symbolic_passed' if kind=='positive' else 'expected_refusal')
    if not accepted:
        return 'final_failure'
    minimum = 4 * result.get('expected_round_count',0) if kind=='positive' else 2
    if (audit.get('logical_request_count',0) < minimum
            or audit.get('requests_without_transport_result_count',0) != 0):
        return 'evidence_incomplete'
    if kind == 'negative':
        # The expected refusal itself is an audit rejection, not a positive-plan failure.
        return 'expected_refusal'
    validation = audit.get('program_rejection_count',0) + audit.get('content_parse_rejection_count',0) > 0
    transport = audit.get('transport_failure_count',0) + audit.get('http_error_count',0) > 0
    if validation and transport:
        return 'passed_after_validation_and_transport_retry'
    if validation:
        return 'passed_after_validation_correction'
    if transport:
        return 'passed_after_transport_retry'
    return 'clean_first_pass'


def analysis(summary):
    rows = [r for r in summary['trials'] if r.get('evaluation_outcome') is not None]
    positive = [r for r in rows if r['kind']=='positive']
    negative = [r for r in rows if r['kind']=='negative']
    outcomes = Counter(r['evaluation_outcome'] for r in positive)
    successes = sum(r['evaluation_outcome'].startswith('passed_after') or
                    r['evaluation_outcome']=='clean_first_pass' for r in positive)
    groups = {}
    for row in rows:
        group = groups.setdefault(row['case'], {'kind': row['kind'], 'attempted_trials': 0,
            'difficulty_rank': row.get('difficulty_rank'), 'complexity':row.get('complexity'),
            'outcomes': Counter(), 'completed_rounds': 0, 'planned_rounds': row.get('expected_round_count',0)})
        group['attempted_trials'] += 1
        group['outcomes'][row['evaluation_outcome']] += 1
        group['completed_rounds'] += row.get('completed_round_count',0)
    for group in groups.values():
        group['outcomes'] = dict(group['outcomes'])
    counts = {key: sum(row.get('audit',{}).get(key,0) for row in rows) for key in AUDIT_KEYS}
    rejection_stages = Counter()
    for row in rows:
        rejection_stages.update(row.get('program_rejections_by_stage',{}))
    valid = summary.get('frozen_engine_unchanged',True) is True and summary['status'] != 'invalid_or_harness_failure'
    return {'mode': 'same_map_multiple_task_scenarios', 'independent_environment_count': 1,
        'evaluation_valid': valid,
        'planned_trial_count': summary['planned_trial_count'], 'evaluated_trial_count': len(rows),
        'positive_trial_count': len(positive), 'positive_success_count': successes,
        'positive_success_rate': successes/len(positive) if positive and valid else None,
        'positive_clean_first_pass_count': outcomes['clean_first_pass'],
        'positive_clean_first_pass_rate': outcomes['clean_first_pass']/len(positive) if positive and valid else None,
        'positive_first_validation_pass_count': outcomes['clean_first_pass']+outcomes['passed_after_transport_retry'],
        'positive_outcomes': dict(outcomes), 'negative_trial_count': len(negative),
        'negative_expected_refusal_count': sum(r['evaluation_outcome']=='expected_refusal' for r in negative),
        'by_case': groups, 'audit_counts': counts, 'program_rejections_by_stage': dict(rejection_stages),
        'positive_failure_stages': dict(Counter(r.get('failure_stage') or 'unknown' for r in positive
                                               if r['evaluation_outcome']=='final_failure')),
        'limitations': ['Small repeated sample; rates are descriptive, not generalization estimates',
            'Same three-node map; room membership/placement effects are hypotheses',
            'Negative tests include a synthetic cloned node and withheld edges',
            'Successful early rounds of failed chains are NOT independent successful episodes',
            'Audit events and negative expected refusals must not be added as failed positive tasks',
            'Existing bounded correction remains enabled; no manual repair or failed-case replay'],
        'simulation_executed': False, 'geometry_verified': False, 'training_eligible': False}


def write_analysis(root, summary):
    result = analysis(summary)
    write_json(root/'stability_analysis.json', result)
    percent = lambda v: '尚无样本' if v is None else f'{v:.1%}'
    lines = ['# 冻结版本 JSON 稳定性测试分析', '',
        '同一 scene201 地图上的多个任务场景与合成负例，不是多个独立物理环境。', '',
        f"批次状态：`{summary['status']}`；已评估 {result['evaluated_trial_count']}/{result['planned_trial_count']} 个案例运行。",
        f"正例完整通过：{result['positive_success_count']}/{result['positive_trial_count']}（{percent(result['positive_success_rate'])}）；"
        f"无校验拒绝且无传输重试的首次通过：{result['positive_clean_first_pass_count']}/{result['positive_trial_count']}（{percent(result['positive_clean_first_pass_rate'])}）。",
        f"负例预期拒绝：{result['negative_expected_refusal_count']}/{result['negative_trial_count']}；负例不混入正例成功率。", '',
        '| 案例 | 类型 | 运行数 | 结果分布 | 完成轮数 / 计划轮数 |', '|---|---|---:|---|---|']
    for name, group in result['by_case'].items():
        lines.append(f"| {name} | {group['kind']} | {group['attempted_trials']} | {group['outcomes']} | "
            f"{group['completed_rounds']} / {group['planned_rounds']*group['attempted_trials']} |")
    lines += ['', '## 请求与失败审计', '', f"统计：`{result['audit_counts']}`。", '',
        'clean_first_pass：无程序/解析拒绝，也无网络或 HTTP 失败；passed_after_validation_correction：校验纠错后通过；'
        'passed_after_transport_retry：仅传输重试后通过；passed_after_validation_and_transport_retry：两者均发生；'
        'final_failure：最终失败；evidence_incomplete：记录不足，不计为完整成功。校验拒绝后通过也包括'
        '拒绝冗余审查候选后保留已验证原计划，不一定全是重新请求技能提案。', '',
        '合法负例拒绝本身会进入程序拒绝审计。负例若先出现格式拒绝再得到预期拒绝，仍保留这些事件；'
        'expected_refusal 不表示负例第一次返回便正确。网络失败不算正确拒绝。', '', '## 失败案例', '']
    for row in summary['trials']:
        if row.get('evaluation_outcome') in ('final_failure','evidence_incomplete'):
            reason = str(row.get('failure_cause') or row.get('launcher_error') or '见原始日志').replace('\n',' ')
            lines.append(f"- {row['case']} / repeat {row['repeat']}：{reason}；目录 `{row['output_dir']}`")
    lines += ['', '## 版本与结论边界', '',
        '本批次子进程使用 frozen_engine 内的代码与技能库副本，逐案例检查 SHA-256。没有自动改代码、补跑或替换计划。',
        '每个案例从独立初态开始；单个长程案例失败后跳过其后续轮次，但继续下一个独立案例。原始 LLM 返回保留。',
        '成功说明所设定指令和符号断言下链路一致；不证明真实跨房间导航、无碰撞、抓取、支撑面或感知成功。',
        '样本量小、temperature=0 且模型服务未固定版本，不宜据此推断所有上游场景的泛化性能。',
        '最终应结合原始请求/返回核对失败类别，分开报告输入缺失、输出错误与网络问题，不把全部失败归因于模型。', '']
    (root/'稳定性测试分析.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def run_batch(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    engine = root/'frozen_engine'
    manifest = freeze_engine(engine)
    input_hashes = {}
    for name, source in [('base_graph.json',args.base_graph), ('verified_graph.json',args.verified_graph)]:
        target = root/'frozen_inputs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copyfile(source, target)
        if sha(source) != before or sha(target) != before:
            raise ValueError('Input changed while freezing')
        input_hashes[name] = {'source': str(source), 'sha256': before}
    frozen_args = copy.copy(args)
    frozen_args.base_graph = root/'frozen_inputs/base_graph.json'
    frozen_args.verified_graph = root/'frozen_inputs/verified_graph.json'
    nodes = load_nodes(frozen_args)
    cases = validate_cases(catalog(nodes))
    selected_names = args.case or DEFAULT_CASES
    selected = [c for c in cases if c['name'] in selected_names]
    if args.case and set(args.case) - {c['name'] for c in cases}:
        raise ValueError('Unknown case name')
    summary = {'status': 'running', 'trials': [], 'planned_trial_count': len(selected)*args.repeats,
        'model': args.model, 'temperature': 0, 'repeats_per_case': args.repeats,
        'python_version': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
        'map_id': 'scene201_three_node_subset', 'upstream_inputs': input_hashes,
        'frozen_engine_manifest_sha256':sha(engine/'engine_manifest.json'),
        'simulation_executed': False, 'training_eligible': False, 'geometry_verified': False,
        'manual_repair': False, 'failed_case_replay': False, 'existing_bounded_correction_enabled': True,
        'no_failure_estimated_logical_requests': args.repeats*sum(4*len(c['rounds']) if c['kind']=='positive' else 2 for c in selected)}
    for case in selected:
        write_json(root/'cases'/f'{case["name"]}.json', case)
    summary['case_input_hashes'] = {c['name']: sha(root/'cases'/f'{c["name"]}.json') for c in selected}
    destination = root/'batch_summary.json'
    write_json(destination, summary)
    if args.preflight_only:
        summary.update(status='input_preflight_passed', frozen_engine_unchanged=frozen_unchanged(engine,manifest), fresh_llm_calls=False)
        write_json(destination, summary)
        print(f'status=input_preflight_passed estimated_requests={summary["no_failure_estimated_logical_requests"]} summary={destination}', flush=True)
        return 0
    worker = engine/'outputs/habitat_demo_v1/run_json_stability_batch.py'
    try:
        # Round-robin repetitions keep every task family in the first sweep.
        for repeat in range(1,args.repeats+1):
            for case in selected:
                case_file = root/'cases'/f'{case["name"]}.json'
                if not frozen_unchanged(engine,manifest) or sha(case_file) != summary['case_input_hashes'][case['name']]:
                    raise ValueError('Frozen engine or case input changed; batch invalid')
                folder = root/'trials'/f'{case["name"]}_repeat_{repeat}'
                folder.mkdir(parents=True, exist_ok=False)
                row = {'case':case['name'], 'kind':case['kind'], 'repeat':repeat,
                    'difficulty_rank':case.get('difficulty_rank'), 'complexity':case.get('complexity'),
                    'output_dir':str(folder), 'status':'running'}
                summary['trials'].append(row)
                write_json(destination,summary)
                print(f'START case={case["name"]} repeat={repeat}/{args.repeats}', flush=True)
                started = time.monotonic()
                try:
                    execute_worker([sys.executable,str(worker),'--worker-case',str(case_file),
                        '--output-dir',str(folder),'--model',args.model,'--base-url',args.base_url],
                        folder/'console.log',timeout=1200*(len(case['rounds']) if case['kind']=='positive' else 1)+90)
                except Exception as exc:
                    row['launcher_error'] = str(exc)
                row['duration_seconds'] = time.monotonic()-started
                result_file = folder/'case_summary.json'
                try:
                    result = read_json(result_file) if result_file.exists() else {'status':'failed'}
                except (OSError,ValueError) as exc:
                    result = {'status':'failed','failure_stage':'case_summary_readback','failure_cause':str(exc)}
                report = build_audit_report(folder,folder/'llm_audit_aggregate')
                row.update(result)
                row['audit'] = {k:report[k] for k in AUDIT_KEYS}
                row['program_rejections_by_stage'] = report['program_rejections_by_stage']
                row['evaluation_outcome'] = classify(case['kind'],result,report)
                if row.get('launcher_error') and row['evaluation_outcome'] not in ('final_failure','evidence_incomplete'):
                    row.update(evaluation_outcome='final_failure', failure_stage='case_worker',
                               failure_cause='Worker failed despite reporting an accepted case result')
                row['frozen_engine_unchanged'] = frozen_unchanged(engine,manifest)
                if not row['frozen_engine_unchanged']:
                    raise ValueError('Frozen engine changed during case; batch invalid')
                write_json(destination,summary)
                print(f'END case={case["name"]} repeat={repeat} outcome={row["evaluation_outcome"]}', flush=True)
        summary['status'] = 'completed'
    except (Exception,KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'invalid_or_harness_failure', error=str(exc))
        logging.exception('Batch stopped without repairs or replay')
    finally:
        summary['frozen_engine_unchanged'] = frozen_unchanged(engine,manifest)
        try:
            summary['upstream_sources_unchanged'] = all(sha(Path(v['source']))==v['sha256'] for v in input_hashes.values())
            summary['frozen_inputs_unchanged'] = all(sha(root/'frozen_inputs'/name)==v['sha256'] for name,v in input_hashes.items())
            summary['case_inputs_unchanged'] = all(sha(root/'cases'/f'{name}.json')==digest
                                                 for name,digest in summary['case_input_hashes'].items())
        except OSError as exc:
            summary.update(frozen_inputs_unchanged=False,case_inputs_unchanged=False,provenance_readback_error=str(exc))
        if (not summary['frozen_engine_unchanged'] or not summary['frozen_inputs_unchanged']
                or not summary['case_inputs_unchanged']):
            summary.update(status='invalid_or_harness_failure',error='Frozen code or input changed; statistics are invalid')
        write_json(destination,summary)
        report = write_analysis(root,summary)
        summary['all_expected_checks_passed'] = len(summary['trials'])==summary['planned_trial_count'] and all(
            r.get('evaluation_outcome') not in (None,'final_failure','evidence_incomplete') for r in summary['trials'])
        write_json(destination,summary)
    print(f'status={summary["status"]} all_expected_checks_passed={summary["all_expected_checks_passed"]} summary={destination}', flush=True)
    print(f'analysis={root / "稳定性测试分析.md"}', flush=True)
    return 0 if summary['status']=='completed' and summary['all_expected_checks_passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-case',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--case',action='append',help='Repeat to select families; default six graded positives and three negatives')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default=DEFAULT_URL)
    parser.add_argument('--base-graph',type=Path,default=SCENE/'final/scene_graph.json')
    parser.add_argument('--verified-graph',type=Path,
        default=SCENE/'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene/scene_graph.json')
    parser.add_argument('--output-dir',type=Path,
        default=SCENE/'json_stability_batch'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be in [1,5]')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY','').strip():
        parser.error('DASHSCOPE_API_KEY required')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    if args.worker_case:
        case = read_json(args.worker_case)
        validate_cases([case])
        return positive_worker(args,case) if case['kind']=='positive' else negative_worker(args,case)
    return run_batch(args)


if __name__=='__main__':
    raise SystemExit(main())
