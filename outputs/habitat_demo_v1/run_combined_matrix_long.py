"""Single-arm cosine + bounded hierarchy + matrix test; historical controls only."""
import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import run_skill_cosine_ab as experiment
from run_navigation_demo import read_json, write_json
from run_json_long_horizon import SCENE

SOURCE = SCENE/'long_matrix_json_ab/scene201_20261006_112644_792021'
ARM = 'bounded_cosine_matrix'
DEFAULT_CASES = ('baseline_12', 'extended_mixed_18')


def hierarchy_metrics(root):
    hierarchies = [read_json(p) for p in root.glob('round_*/planning/hierarchy_summary.json')]
    leaves = [read_json(p) for p in root.glob('round_*/planning/hierarchy/node_*/planning/planning_summary.json')]
    normalizations = sum(n.get('changed', False) for s in leaves
                         for n in s.get('skill_format_normalizations', []))
    guards = [read_json(p) for p in root.rglob('state_guard_report.json')]
    integrity = [read_json(p) for p in root.glob('round_*/planning/round_input_integrity.json')]
    return {
        'attempted_rounds': len(hierarchies),
        'validated_leaves_including_failed_round_prefixes': sum(h.get('leaf_count', 0) for h in hierarchies),
        'split_rounds': sum(any(n.get('decision', {}).get('decision') == 'split'
                              for n in h.get('nodes', [])) for h in hierarchies),
        'planned_leaf_count': len(leaves),
        'all_planned_leaves_matrix': bool(leaves) and all(s.get('skill_output_format') == 'matrix' for s in leaves),
        'semantic_redecompositions': sum(s.get('semantic_redecomposition_count', 0) for s in leaves),
        'format_normalization_attempts': normalizations,
        'deterministic_state_guard_changes': sum(g.get('changed', False) for g in guards),
        'round_inputs_unchanged': len(integrity) == len(hierarchies) and all(i['unchanged'] for i in integrity),
        'failed_rounds': [{'round_folder': str(p.parent.parent.name), 'stage': h.get('failed_stage'),
                          'cause': h.get('error')} for p in root.glob('round_*/planning/hierarchy_summary.json')
                         if (h := read_json(p)).get('status') == 'failed'],
    }


def aggregate(rows):
    evaluated = [r for r in rows if r.get('evaluation_outcome')]
    successes = [r for r in evaluated if r['evaluation_outcome'].startswith(('clean_', 'passed_'))]
    usage = [r.get('token_usage', {}) for r in evaluated]
    prompt = sum(u.get('prompt_tokens', 0) for u in usage)
    completion = sum(u.get('completion_tokens', 0) for u in usage)
    seconds = sum(r.get('duration_seconds', 0) for r in evaluated)
    tokens = sum(u.get('total_tokens', 0) for u in usage)
    return {
        'evaluated_trials': len(evaluated), 'passed_trials': len(successes),
        'clean_first_pass_trials': sum(r['evaluation_outcome'] == 'clean_first_pass' for r in evaluated),
        'pass_rate': len(successes)/len(evaluated) if evaluated else None,
        'outcomes': dict(Counter(r['evaluation_outcome'] for r in evaluated)),
        'completed_rounds': sum(r.get('completed_round_count', 0) for r in evaluated),
        'expected_rounds': sum(r.get('expected_round_count', 0) for r in evaluated),
        'known_prompt_tokens': prompt, 'known_completion_tokens': completion, 'known_total_tokens': tokens,
        'seconds_including_failures': seconds,
        'known_fee_cny_no_discount': (prompt*.8 + completion*2.7)/1_000_000,
        'price_basis': 'User-provided fixed rates: input 0.8, output 2.7 CNY per million tokens; no discount',
        'responses_missing_usage': sum(u.get('responses_missing_usage', 0) for u in usage),
        'transport_failures_unknown_usage': sum(u.get('transport_failures_unknown_usage', 0) for u in usage),
        'tokens_per_success_including_failures': tokens/len(successes) if successes else None,
        'seconds_per_success_including_failures': seconds/len(successes) if successes else None,
        'logical_requests': sum(r.get('audit', {}).get('logical_request_count', 0) for r in evaluated),
        'transport_failures': sum(r.get('audit', {}).get('transport_failure_count', 0) for r in evaluated),
        'validation_rejections': sum(r.get('audit', {}).get('program_rejection_count', 0) +
                                     r.get('audit', {}).get('content_parse_rejection_count', 0) for r in evaluated),
    }


def results(summary, history):
    hashes = summary['case_input_hashes']
    historical_hashes = history.get('case_input_hashes', {})
    if any(historical_hashes.get(name) != digest for name, digest in hashes.items()):
        raise ValueError('Historical comparison inputs do not match current frozen cases')
    comparison = {'new_combined': aggregate(summary['trials'])}
    for arm in ('matrix', 'nested_json'):
        rows = [r for r in history.get('trials', []) if r.get('case') in hashes and r.get('arm') == arm]
        comparison['historical_'+arm] = aggregate(rows)
    by_case = {name: {
        'new_combined': aggregate([r for r in summary['trials'] if r['case'] == name]),
        **{'historical_'+arm: aggregate([r for r in history.get('trials', [])
             if r.get('case') == name and r.get('arm') == arm]) for arm in ('matrix', 'nested_json')}
    } for name in hashes}
    return {'methods': comparison, 'by_case': by_case,
            'comparison_type': 'historical_descriptive_not_contemporaneous_ablation',
            'case_inputs_identical': True, 'historical_batch_status': history.get('status'),
            'simulation_executed': False, 'training_eligible': False}


def make_report(history, historical_summary_sha):
    def report(root, summary):
        for row in summary['trials']:
            if not row.get('evaluation_outcome'):
                continue
            metrics = hierarchy_metrics(Path(row['output_dir']))
            row['hierarchy_metrics'] = metrics
            if row['evaluation_outcome'].startswith(('clean_', 'passed_')):
                if not metrics['all_planned_leaves_matrix'] or not metrics['round_inputs_unchanged']:
                    row.update(evaluation_outcome='final_failure', harness_error='Matrix route or input integrity missing')
                elif row.get('audit', {}).get('logical_request_count', 0) < (
                        2*row.get('expected_round_count', 0)+4*metrics['planned_leaf_count']):
                    row.update(evaluation_outcome='evidence_incomplete',
                               harness_error='Missing mandatory partition/leaf/coverage request audit')
                elif row['evaluation_outcome'] == 'clean_first_pass':
                    if metrics['format_normalization_attempts']:
                        row['evaluation_outcome'] = 'passed_after_format_normalization'
                    elif metrics['deterministic_state_guard_changes']:
                        row['evaluation_outcome'] = 'passed_after_state_guard'
        unchanged = experiment.baseline.sha(Path(summary['source_batch'])/'batch_summary.json') == historical_summary_sha
        summary.update(experiment='cosine_validation_bounded_hierarchy_matrix',
                       historical_summary_unchanged=unchanged, compile_navigation=False,
                       new_nested_json_trials=0, recursive_split_budget=1,
                       no_failure_request_range_per_round=[6, 2+4*summary['max_parts']],
                       historical_comparison='descriptive only; current goal protocol/prompt/engine may differ')
        cases = [read_json(root/'cases'/f'{name}.json') for name in summary['case_input_hashes']]
        rounds = sum(len(c.get('rounds', [])) for c in cases)*summary['repeats']
        summary['no_failure_logical_request_range'] = [6*rounds, (2+4*summary['max_parts'])*rounds]
        if not unchanged:
            summary['status'] = 'invalid_inputs_or_engine'
        data = results(summary, history)
        data['evaluation_valid'] = (unchanged and summary.get('frozen_engine_unchanged') is True
                                   and summary.get('case_inputs_unchanged') is True
                                   and summary.get('historical_sources_unchanged') is True
                                   and summary.get('embedding_model_unchanged', True) is True)
        write_json(root/'comparison_summary.json', data)
        write_json(root/'batch_summary.json', summary)
        lines = ['# 余弦筛选＋校验＋有限递推＋矩阵：长程测试', '',
            f"批次状态：{summary['status']}；冻结输入与代码检查：{data['evaluation_valid']}。", '',
            '本批只运行新组合，不重新请求旧 JSON。历史两组读取相同案例，但不是同期消融实验。',
            '当前目标状态协议、代码与提示可能与旧批次不同，不能把差异单独归因于递推或矩阵。', '',
            '| 方法 | 完整通过/评估 | 无纠错通过 | 完成/总轮数 | 已知 Token（含失败） | 耗时秒 | 已知费用元 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, a in data['methods'].items():
            lines.append(f"| {name} | {a['passed_trials']}/{a['evaluated_trials']} | {a['clean_first_pass_trials']} | "
                f"{a['completed_rounds']}/{a['expected_rounds']} | {a['known_total_tokens']} | "
                f"{a['seconds_including_failures']:.1f} | {a['known_fee_cny_no_discount']:.6f} |")
        lines += ['', '## 新组合逐次结果', '',
            '| 案例 | 重复 | 结果 | 完成/总轮数 | 切分轮数 | 已验证子段 | 语义重分解 | 格式规范化 | 扩展候选 | 原因 |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
        for row in summary['trials']:
            m = row.get('hierarchy_metrics', {})
            reason = str(row.get('failure_cause') or row.get('harness_error') or row.get('launcher_error') or
                         row.get('error') or '').replace('|', '/').replace('\n', ' ')
            lines.append(f"| {row['case']} | {row['repeat']} | {row.get('evaluation_outcome', '待执行')} | "
                f"{row.get('completed_round_count', 0)}/{row.get('expected_round_count', '—')} | "
                f"{m.get('split_rounds', 0)} | {m.get('validated_leaves_including_failed_round_prefixes', 0)} | "
                f"{m.get('semantic_redecompositions', 0)} | {m.get('format_normalization_attempts', 0)} | "
                f"{row.get('retrieval_metrics', {}).get('widened_accepted_proposals', 0)} | {reason} |")
        lines += ['', '## 判读与审计边界', '',
            '- 每条原始指令由 LLM 判断是否切分；最多一层、四段。简单指令可以不切分。每段可包含多个动作。',
            '- 子段继承上一段已校验的符号状态；每轮末再检查原始断言、完整重放和 LLM 语义覆盖。状态是预测，不是机器人实测。',
            '- 五技能执行子库及余弦 top_k=2、阈值0.35、间隔0.05保持原策略；第二次提案可扩展到完整支持库并记录。未另行修复已知候选误筛。',
            '- 不启用纯导航确定性编译；每段技能选择使用矩阵。程序解码、有限格式规范化、前置条件和目标校验全部保留。',
            '- 保留原有状态守卫：只删除满足严格条件且段末状态等价的隐式重复接近；不补动作、不删显式访问。状态守卫修改单独计数。',
            '- 不做人工修复或失败补跑；当前案例失败跳过剩余轮，继续下一个案例/重复。预设的有限自动重试仍保留。',
            '- 无纠错通过要求无传输/解析/程序拒绝、无格式规范化及无状态守卫修改；该指标仍不是独立真值准确率。',
            '- 同一场景两个长程案例的小样本，仅测试计划和符号状态；没有 Habitat、navmesh、碰撞或真实抓取检验。',
            '- 总 Token/耗时包含失败前缀。失败后提前停止不是节省；对比不同重复数应看每案结果和每成功案例成本。',
            '- 原价采用用户给定的输入0.8、输出2.7元/百万Token，不折扣；超时未知用量和缺失usage不记作零，不等于账单。',
            '- 原始返回位于各子段 llm_response_audit；hierarchy_summary.json记录树与失败阶段；批次包含冻结引擎和输入SHA256。', '']
        (root/'组合Pipeline长程测试分析.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch', type=Path, default=SOURCE)
    p.add_argument('--output-dir', type=Path, default=SCENE/'combined_matrix_long'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--case', action='append')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--max-parts', type=int, default=4)
    p.add_argument('--seconds-per-round', type=int, default=3600)
    p.add_argument('--embedding-python', type=Path, default=Path(sys.executable))
    p.add_argument('--embedding-model-dir', type=Path)
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=experiment.baseline.DEFAULT_URL)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.repeats <= 5 or not 2 <= args.max_parts <= 4 or not 60 <= args.seconds_per_round <= 7200:
        p.error('Invalid experiment bounds')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        p.error('DASHSCOPE_API_KEY required')
    args.case = args.case or list(DEFAULT_CASES)
    args.arms = [ARM]
    args.top_k, args.cosine_threshold, args.cosine_margin = 2, .35, .05
    args.compile_navigation = False
    history = read_json(args.source_batch/'batch_summary.json')
    for name in args.case:
        if not any(r.get('case') == name and r.get('arm') in ('matrix', 'nested_json') for r in history.get('trials', [])):
            p.error('Historical JSON/matrix results missing for case: '+name)
    experiment.report = make_report(history, experiment.baseline.sha(args.source_batch/'batch_summary.json'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    return experiment.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
