#!/usr/bin/env python3
"""Fresh paired baseline/hierarchical trials using unchanged historical case files."""
import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import run_json_stability_batch as baseline
from hierarchical_json_planner import plan_hierarchical
from run_navigation_demo import read_json, write_json
from run_json_long_horizon import SCENE
from json_stability_cases import DEFAULT_CASES, validate_cases
from build_report import build as build_audit

SOURCE = SCENE/'json_stability_batch/scene201_20261003_163457_100107'


def comparison(summary):
    results = {}
    arms = summary.get('comparison_arms', ('baseline', 'hierarchical'))
    for strategy in arms:
        rows = [r for r in summary['trials'] if r['strategy'] == strategy and r['kind'] == 'positive']
        evaluated = [r for r in rows if r.get('evaluation_outcome') is not None]
        passed = [r for r in evaluated if r['evaluation_outcome'].startswith(('clean_', 'passed_'))]
        results[strategy] = {'planned_positive_trials': summary['positive_trials_per_strategy'],
            'evaluated_positive_trials': len(evaluated), 'passed_positive_trials': len(passed),
            'clean_first_pass_trials': sum(r['evaluation_outcome']=='clean_first_pass' for r in evaluated),
            'success_rate_among_evaluated': len(passed)/len(evaluated) if evaluated else None,
            'logical_requests': sum(r.get('audit', {}).get('logical_request_count', 0) for r in evaluated),
            'http_attempts': sum(r.get('audit', {}).get('http_attempt_count', 0) for r in evaluated),
            'transport_failures': sum(r.get('audit', {}).get('transport_failure_count', 0) for r in evaluated),
            'program_rejections': sum(r.get('audit', {}).get('program_rejection_count', 0) for r in evaluated),
            'seconds': round(sum(r.get('duration_seconds', 0) for r in evaluated), 2)}
    pairs = []
    for key in sorted({(r['case'], r['repeat']) for r in summary['trials'] if r['kind']=='positive'}):
        rows = {r['strategy']: r for r in summary['trials'] if (r['case'], r['repeat']) == key}
        if all(s in rows and rows[s].get('evaluation_outcome') for s in arms):
            pairs.append({'case': key[0], 'repeat': key[1],
                          **{s: rows[s]['evaluation_outcome'] for s in rows}})
    return {'strategies': results, 'completed_pairs': pairs,
        'experiment': summary.get('experiment', 'planner_baseline_hierarchical'),
        'fixed_planner': summary.get('fixed_planner'),
        'evaluation_valid': summary.get('frozen_engine_unchanged', False) and summary.get('case_inputs_unchanged', False),
        'scope': 'Same map and unchanged original case assertions; negative controls use the same legacy binding '
                 'algorithm in both arms. Not equal API budget: report cost alongside robustness.',
        'simulation_executed': False, 'training_eligible': False}


def write_report(root, summary):
    data = comparison(summary)
    write_json(root/'comparison_summary.json', data)
    lines = ['# 分层递归规划与原方案：配对测试', '',
        f"批次状态：{summary['status']}；冻结输入与代码有效：{data['evaluation_valid']}。", '',
        '两组读取同一份历史案例，交替执行；没有人工修复、失败补跑或固定计划替代。',
        '这里只测试 JSON 和符号状态，不运行 Habitat。原方案逐原始轮规划；实验方案在轮内继续分段。',
        '原始访问序列、最终持有/抬起和放置断言未放宽。叶段目标按段末检查，另作整轮重放和 LLM 语义覆盖复核。',
        '负例两组仍调用相同的原绑定路径，作为共享控制条件，不据此声称分层改善了歧义处理。', '',
        '| 方案 | 已评估/计划正例 | 完整通过 | 无重试通过 | 逻辑请求 | HTTP尝试 | 超时/传输失败 | 总耗时秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in data['strategies'].items():
        lines.append(f"| {name} | {r['evaluated_positive_trials']}/{r['planned_positive_trials']} | "
            f"{r['passed_positive_trials']} | {r['clean_first_pass_trials']} | {r['logical_requests']} | "
            f"{r['http_attempts']} | {r['transport_failures']} | {r['seconds']} |")
    lines += ['', '## 每个试次', '', '| 案例 | 重复 | 方案 | 结果 | 已完成/总轮数 | 原因 |', '|---|---:|---|---|---|---|']
    for r in summary['trials']:
        reason = str(r.get('failure_cause') or r.get('error') or r.get('launcher_error') or '').replace('|', '/').replace('\n',' ')
        lines.append(f"| {r['case']} | {r['repeat']} | {r['strategy']} | {r.get('evaluation_outcome','未完成')} | "
            f"{r.get('completed_round_count','—')}/{r.get('expected_round_count','—')} | {reason} |")
    lines += ['', '## 解读边界', '',
        '- 已知的历史歧义负例和严格导航断言原样保留；原始失败评分不等同于已确认的模型语义错误。',
        '- 样本小，同一个地图子集；不能直接推断跨环境泛化，也不应仅凭通过率忽略请求量和耗时。',
        '- 所有未完成或被预算终止的正例均不算通过。若批次中断，需先核对已评估分母与完成配对数。',
        '- 分层引入更多 LLM 调用与一次完整语义复核，因此比较的是整体策略，不是等请求数的单模型能力。',
        '- 整轮最终状态断言不下发给每个叶段；叶段使用空的 expected_end，并依靠原始上下文与当前状态规划。',
        '- 切分文本完整不保证语义完整，整轮 LLM 复核也不是独立真值；保留树、叶计划和原始返回供人工审计。', '']
    if summary.get('experiment') == 'prompt_v1_v2':
        lines[0] = '# 同规划方式下的 prompt v1 / v2 配对测试'
        lines[5] = f"两组固定使用 {summary['fixed_planner']} 规划方式，仅 v2 加入严格提示词附录；校验器、重试、预算不变。"
        lines = [line.replace('原方案逐原始轮规划；实验方案在轮内继续分段。', '本批次不是 baseline 与 hierarchical 的比较。')
                 for line in lines]
        lines = [('两组保留相同的原始访问、持有/抬起和放置断言；仅提示词不同，不运行 Habitat。'
                  if line.startswith('原始访问序列') else line) for line in lines]
        lines = [('- 同一规划方式、相同调用流程与预算上限，但纠错次数和切分结果可能改变实际请求量；仍需报告耗时与调用量。'
                  if line.startswith('- 分层引入更多') else line) for line in lines]
        if summary['fixed_planner'] == 'baseline':
            lines = [line for line in lines if not line.startswith(('- 整轮最终状态断言不下发', '- 切分文本完整'))]
        lines += ['## 提示词版本', '',
            'v1 沿用原提示词；v2 向分解/技能的 execution_contract 增加附录，并加强分层切分提示词。',
            '绑定、冗余审查和整轮语义复核的提示词保持原样；不增加额外 API 阶段。',
            '负例是未改提示词的共享控制，不能据此声称 v2 改善绑定拒绝。',
            '记录的 SHA-256 标识 v2 附录；完整动态请求请查 llm_response_audit。', '']
    (root/'分层规划对照分析.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    source = args.source_batch
    historical = read_json(source/'batch_summary.json')
    selected = args.case or list(DEFAULT_CASES)
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate --case')
    cases = []
    for name in selected:
        if name not in historical['case_input_hashes']:
            raise ValueError(f'Case not present in source batch: {name}')
        path = source/'cases'/f'{name}.json'
        if baseline.sha(path) != historical['case_input_hashes'][name]:
            raise ValueError('Historical case hash mismatch')
        cases.append(read_json(path))
    validate_cases(cases)
    prompt_ab = getattr(args, 'prompt_ab', False)
    arms = ('v1', 'v2') if prompt_ab else ('baseline', 'hierarchical')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    manifest = baseline.freeze_engine(root/'frozen_engine')
    hashes = {}
    for case in cases:
        path = root/'cases'/f'{case["name"]}.json'
        # Copy bytes, not re-serialization, for exact pairing with historical inputs.
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/'cases'/path.name, path)
        hashes[case['name']] = baseline.sha(path)
    summary = {'status': 'running', 'source_batch': str(source), 'model': args.model, 'temperature': 0,
        'experiment': 'prompt_v1_v2' if prompt_ab else 'planner_baseline_hierarchical',
        'comparison_arms': arms, 'fixed_planner': getattr(args, 'planner', 'hierarchical') if prompt_ab else None,
        'trials': [], 'planned_trial_count': len(cases)*args.repeats*2,
        'positive_trials_per_strategy': sum(c['kind']=='positive' for c in cases)*args.repeats,
        'case_input_hashes': hashes, 'hierarchy_limits': {'max_depth': args.max_depth,
            'max_nodes': args.max_nodes, 'max_leaves': args.max_leaves},
        'seconds_per_original_round_for_case_budget': args.seconds_per_round,
        'manual_repair': False, 'failed_case_replay': False, 'simulation_executed': False,
        'geometry_verified': False, 'training_eligible': False,
        'source_vs_current_engine_changed_files': [k for k,v in read_json(source/'frozen_engine/engine_manifest.json')['files'].items()
                if manifest['files'].get(k) != v]}
    if prompt_ab:
        from strict_prompt_v2 import fingerprints
        summary['prompt_v2_sha256'] = fingerprints()
    destination = root/'batch_summary.json'
    write_json(destination, summary)
    try:
        if args.preflight_only:
            summary['status'] = 'input_preflight_passed'
        else:
            worker = root/'frozen_engine/outputs/habitat_demo_v1/run_hierarchical_json_ab.py'
            for repeat in range(1, args.repeats+1):
                for case_index, case in enumerate(cases):
                    strategies = arms if (repeat+case_index)%2 else tuple(reversed(arms))
                    for strategy in strategies:
                        if not baseline.frozen_unchanged(root/'frozen_engine', manifest):
                            raise ValueError('Frozen engine changed')
                        case_file = root/'cases'/f'{case["name"]}.json'
                        if baseline.sha(case_file) != hashes[case['name']]:
                            raise ValueError('Frozen case changed')
                        folder = root/'trials'/f'{case["name"]}_repeat_{repeat}_{strategy}'
                        folder.mkdir(parents=True, exist_ok=False)
                        row = {'case': case['name'], 'kind': case['kind'], 'repeat': repeat,
                               'strategy': strategy, 'output_dir': str(folder),
                               'planner': args.planner if prompt_ab else strategy,
                               'prompt_profile': strategy if prompt_ab else 'v1'}
                        summary['trials'].append(row)
                        write_json(destination, summary)
                        print(f'START {case["name"]} repeat={repeat} strategy={strategy}', flush=True)
                        started = time.monotonic()
                        try:
                            baseline.execute_worker([sys.executable, str(worker), '--worker-case', str(case_file),
                                '--strategy', args.planner if prompt_ab else strategy,
                                '--prompt-version', strategy if prompt_ab else 'v1',
                                '--output-dir', str(folder), '--model', args.model,
                                '--base-url', args.base_url, '--max-depth', str(args.max_depth),
                                '--max-nodes', str(args.max_nodes), '--max-leaves', str(args.max_leaves)],
                                folder/'console.log', timeout=args.seconds_per_round*max(1,len(case.get('rounds',[])))+90)
                        except Exception as exc:
                            row['launcher_error'] = str(exc)
                        row['duration_seconds'] = time.monotonic()-started
                        try:
                            result = read_json(folder/'case_summary.json')
                        except (OSError, ValueError):
                            result = {'status': 'failed', 'error': 'No readable final case summary'}
                        row.update(result)
                        audit = build_audit(folder, folder/'llm_audit_aggregate')
                        row['audit'] = {k: audit[k] for k in baseline.AUDIT_KEYS}
                        row['evaluation_outcome'] = baseline.classify(case['kind'], result, audit)
                        if row.get('launcher_error') and row['evaluation_outcome'] not in ('final_failure','evidence_incomplete'):
                            row['evaluation_outcome'] = 'final_failure'
                        write_json(destination, summary)
                        print(f'END {case["name"]} {strategy} outcome={row["evaluation_outcome"]}', flush=True)
            summary['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted_or_invalid', error=str(exc))
        logging.exception('Paired batch stopped; no repair/replay')
    finally:
        summary['frozen_engine_unchanged'] = baseline.frozen_unchanged(root/'frozen_engine', manifest)
        try:
            summary['case_inputs_unchanged'] = all(baseline.sha(root/'cases'/f'{k}.json') == v for k,v in hashes.items())
        except OSError:
            summary['case_inputs_unchanged'] = False
        if not summary['frozen_engine_unchanged'] or not summary['case_inputs_unchanged']:
            summary['status'] = 'invalid_inputs_or_engine'
        write_json(destination, summary)
        write_report(root, summary)
    print(f'status={summary["status"]} summary={destination}', flush=True)
    print(f'analysis={root/"分层规划对照分析.md"}', flush=True)
    return 0 if summary['status']=='input_preflight_passed' or (summary['status']=='completed' and
        all(r.get('evaluation_outcome') not in (None,'final_failure','evidence_incomplete') for r in summary['trials'])) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch', type=Path, default=SOURCE)
    p.add_argument('--case', action='append')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--prompt-ab', action='store_true', help='Compare prompt v1/v2 with fixed planner')
    p.add_argument('--planner', choices=('baseline','hierarchical'), default='hierarchical')
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--max-nodes', type=int, default=23)
    p.add_argument('--max-leaves', type=int, default=12)
    p.add_argument('--seconds-per-round', type=int, default=1800)
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=baseline.DEFAULT_URL)
    p.add_argument('--output-dir', type=Path, default=SCENE/'hierarchical_json_ab'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--worker-case', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--strategy', choices=('baseline','hierarchical'), default='baseline', help=argparse.SUPPRESS)
    p.add_argument('--prompt-version', choices=('v1','v2'), default='v1', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (1 <= args.repeats <= 5 and 1 <= args.max_depth <= 5 and 1 <= args.max_nodes <= 63
            and 1 <= args.max_leaves <= 24 and 60 <= args.seconds_per_round <= 7200):
        p.error('Invalid repeat/tree/time budget bounds')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY','').strip():
        p.error('DASHSCOPE_API_KEY required')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.worker_case:
        case = read_json(args.worker_case)
        validate_cases([case])
        if case['kind'] == 'negative':
            return baseline.negative_worker(args, case)
        if args.strategy == 'hierarchical':
            # Worker-local dispatch only; no production source or legacy default is modified.
            planner = plan_hierarchical
            if args.prompt_version == 'v2':
                from strict_prompt_v2 import hierarchical_v2
                planner = hierarchical_v2
            baseline.plan_one_round = lambda a, folder: planner(a, folder,
                {'max_depth': args.max_depth, 'max_nodes': args.max_nodes, 'max_leaves': args.max_leaves})
        elif args.prompt_version == 'v2':
            from strict_prompt_v2 import apply_contract
            old_round = baseline.plan_one_round
            def strict_round(a, folder):
                apply_contract(folder)
                return old_round(a, folder)
            baseline.plan_one_round = strict_round
        return baseline.positive_worker(args, case)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
