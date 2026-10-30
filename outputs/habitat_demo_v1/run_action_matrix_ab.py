"""Frozen binding/state/cosine candidates: nested JSON vs discrete action matrix.

Skill-stage-only comparison. No decomposition, simulation, chain resumption,
LLM redundancy review, automatic action filling or historical outcome rewriting.
"""
import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from run_navigation_demo import read_json, write_json
from run_json_long_horizon import SCENE
from replay_json_round_planning import INPUT_FILES, validate_checkpoint
from run_json_stability_batch import freeze_engine, frozen_unchanged
from run_live_llm_sequence import execute_worker
from match_skills import request_plan, build_result, DEFAULT_URL
from action_matrix import make_codebook, request_matrix, decode_matrix
from skill_retrieval import check_membership
from json_symbolic_review import review_symbolic
from verify_plan import flatten_steps, review_redundancy
from response_audit import validation
from build_report import build as build_audit
from token_usage_audit import usage
from adapter_parameter_contract import assert_contract
from skill_output_normalization import normalize_and_record, VERSION as NORMALIZATION_VERSION

SOURCE = SCENE/'skill_cosine_ab/scene201_20261005_200854_623291/trials/extended_mixed_18_1_bounded_cosine/round_11/planning/hierarchy/node_003/planning'
FILES = (*INPUT_FILES, 'object_alignment.json', 'skill_retrieval_semantic_1.json')


def long_path(path):
    path = path.resolve()
    return Path('\\\\?\\'+str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def input_hashes(folder):
    return {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in FILES}


def load_inputs(folder):
    state, descriptor, library, nodes, _ = validate_checkpoint(folder)
    if library.get('execution_contract', {}).get('parameter_contract_version'):
        assert_contract(library)
    alignment = read_json(folder/'object_alignment.json')
    retrieval = read_json(folder/'skill_retrieval_semantic_1.json')
    if alignment.get('status') != 'success' or alignment['task_decomposition']['instruction'] != descriptor['instruction']:
        raise ValueError('Saved binding status/instruction mismatch')
    if {s['subtask_id'] for s in retrieval['subtasks']} != {t['subtask_id'] for t in alignment['task_decomposition']['subtasks']}:
        raise ValueError('Saved cosine subtask IDs mismatch')
    return state, descriptor, library, nodes, alignment, retrieval


def worker(args):
    root, source = args.output_dir, long_path(args.worker_input)
    before = input_hashes(source)
    state, descriptor, library, nodes, alignment, retrieval = load_inputs(source)
    book = make_codebook(alignment, library)
    write_json(root/'codebook.json', book)
    summary = {'status': 'running', 'arm': args.arm, 'attempts': [],
        'input_memory_version': state['backend']['memory_version'],
        'state_source': state['state_source'], 'decomposition_and_binding': 'saved_not_fresh',
        'cosine_candidates': retrieval.get('source', 'same_saved_real_embedding_retrieval'),
        'simulation_executed': False, 'training_eligible': False,
        'scope': 'Skill proposal format and existing symbolic checks only; not full round or long-chain success'}
    summary['format_normalization_version'] = NORMALIZATION_VERSION
    error = None
    for attempt in range(1, 3):
        proposal, raw, evidence = None, None, None
        started = time.monotonic()
        row = {'attempt': attempt, 'widened_inventory': attempt > 1}
        phase = 'request'
        summary['attempts'].append(row)
        try:
            options = dict(timeout=180, attempts=2, validation_error=error,
                retrieval=retrieval, widen_inventory=attempt > 1, diagnostics_dir=root,
                audit_context={'format_ab_arm': args.arm, 'validation_attempt': attempt})
            if args.arm == 'nested_json':
                raw = request_plan(alignment, library, args.model, args.base_url, 'DASHSCOPE_API_KEY', **options)
                proposal = raw
            else:
                raw = request_matrix(alignment, library, book, args.model, args.base_url, 'DASHSCOPE_API_KEY', **options)
            evidence = f'raw_proposal_attempt_{attempt}.json'
            write_json(root/evidence, raw)
            phase = 'format_normalization'
            wire, normalization = normalize_and_record(raw, args.arm, library, root,
                f'proposal_attempt_{attempt}', evidence)
            row['format_normalization'] = normalization
            if args.arm == 'matrix':
                phase = 'decode'
                proposal = decode_matrix(wire, book, alignment, library)
                write_json(root/f'decoded_proposal_attempt_{attempt}.json', proposal)
            else:
                proposal = wire
            phase = 'schema'
            plan = build_result(alignment, library, proposal)
            phase = 'candidates'
            membership = check_membership(proposal, retrieval, widened=attempt > 1)
            phase = 'symbolic'
            review = review_symbolic(dict(descriptor, alignment=alignment, plan=plan), library, nodes, state,
                                     library.get('execution_contract', {}).get('proxy_reach_m', 1.5))
            plan['proposal_origin'] = 'llm_matrix_decoded' if args.arm == 'matrix' else 'fresh_llm_nested_json'
            write_json(root/'skill_plan.json', plan)
            write_json(root/'symbolic_review.json', review)
            write_json(root/'candidate_membership.json', membership)
            write_json(root/'program_redundancy_review.json', review_redundancy(flatten_steps(plan), library))
            validation(root, 'skill_format_ab', attempt, 'accepted',
                       context={'arm': args.arm, 'format_normalized': normalization['changed']}, evidence=evidence)
            row.update(status='accepted', action_count=len(flatten_steps(plan)))
            summary.update(status='skill_stage_symbolic_passed', navigation_targets=review['navigation_targets'])
            break
        except Exception as exc:
            error = str(exc)
            normalization_path = root/f'proposal_attempt_{attempt}.normalization.json'
            if normalization_path.exists():
                row['format_normalization'] = read_json(normalization_path)
            row.update(status='rejected', error=error, failure_phase=phase)
            validation(root, 'skill_format_ab', attempt, 'rejected', exc,
                       context={'arm': args.arm}, evidence=evidence)
            logging.warning('Rejected %s attempt=%s: %s', args.arm, attempt, exc)
            # Match the existing proposal boundary: only completed but invalid
            # outputs get schema correction. Exhausted transport errors stop.
            if raw is None:
                break
        finally:
            row['seconds'] = time.monotonic()-started
            write_json(root/'trial_summary.json', summary)
    if summary['status'] != 'skill_stage_symbolic_passed':
        summary.update(status='failed', failure_cause=error)
    summary['input_unchanged'] = input_hashes(source) == before
    if not summary['input_unchanged']:
        summary.update(status='invalid_inputs', failure_cause='Shared input snapshot changed')
    audit = build_audit(root, root/'llm_audit_report')
    summary['audit'] = {k: audit[k] for k in ('logical_request_count', 'http_attempt_count',
        'transport_failure_count', 'program_rejection_count', 'content_parse_rejection_count')}
    summary['token_usage'] = usage(root)
    summary['first_proposal_valid'] = summary['attempts'][0].get('status') == 'accepted'
    summary['first_proposal_strict_valid'] = (summary['first_proposal_valid'] and
        not summary['attempts'][0].get('format_normalization', {}).get('changed', False))
    summary['first_proposal_normalized_valid'] = (summary['first_proposal_valid'] and
        summary['attempts'][0].get('format_normalization', {}).get('changed', False))
    summary['accepted_after_normalization'] = any(a.get('status') == 'accepted' and
        a.get('format_normalization', {}).get('changed', False) for a in summary['attempts'])
    summary['normalization_applied_attempts'] = sum(
        a.get('format_normalization', {}).get('changed', False) for a in summary['attempts'])
    write_json(root/'trial_summary.json', summary)
    return 0 if summary['status'] == 'skill_stage_symbolic_passed' else 2


def report(root, summary):
    lines = ['# 动作JSON与编码矩阵对照', '', f"批次状态：{summary['status']}。", '',
        '固定同一失败叶段的分解、物体绑定、状态与真实余弦候选。重新请求技能输出，不重新分解、不接着执行后续轮次。', '',
        '首提案有效包含确定性归一化后通过；原格式直接通过和归一化通过分别列出。', '',
        '| 重复 | 格式 | 结果 | 首提案有效 | 首次原格式直接通过 | 首次归一化后通过 | 收到响应Token | 耗时秒 |',
        '|---|---|---|---|---|---|---:|---:|']
    for r in summary['trials']:
        lines.append(f"| {r['repeat']} | {r['arm']} | {r.get('status','未完成')} | "
            f"{r.get('first_proposal_valid','—')} | {r.get('first_proposal_strict_valid','—')} | "
            f"{r.get('first_proposal_normalized_valid','—')} | "
            f"{r.get('token_usage',{}).get('total_tokens','—')} | {r.get('seconds',0):.1f} |")
        if r.get('failure_cause') or r.get('launcher_error'):
            lines += ['', f"失败：{str(r.get('failure_cause') or r['launcher_error']).replace(chr(10),' ')}", '']
    lines += ['', '## 边界', '',
        '- 允许裸矩阵包装、明确且库内的s→skill_id别名；原始与归一化文件分别保存，不填技能/参数，不删除动作；归一化后仍跑原验证器。',
        '- 两组均最多2次技能提案，第二次扩大到完整已支持技能库；每次HTTP超时180秒、最多2次传输尝试。',
        '- 矩阵组改变输出格式、编码表与相应格式提示；不能将结果归因于单一字段长度。状态、绑定、候选、原技能库和验证器一致。',
        '- 本地余弦不重新计算，复用已保存的真实排名；本批成本只覆盖技能请求与程序校验。',
        '- 矩阵严格解码，不删除空行，不推测参数，不补动作。JSON组使用原请求路径，不修正旧提示。',
        '- v1仅覆盖当前五技能、最多两个参数槽；对象用绑定代码，可选数值仅可省略或选择库中默认值的预设代码，不支持任意连续参数。',
        '- 没有LLM冗余删减或整条指令覆盖复核，没有仿真/物理验证；仅是技能格式＋原符号校验结果，training_eligible=false。',
        '- 原始API响应、失败输出、编码表、解码JSON分别保留；旧长程失败不改写。',
        '- 小样本不能证明稳定正确率；超时的未知计费不计为零。', '']
    (root/'动作矩阵对照分析.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    source = long_path(args.source_planning_dir)
    original = input_hashes(source)
    state, descriptor, library, nodes, alignment, retrieval = load_inputs(source)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    shared = root/'shared_input'
    for name in FILES:
        (shared/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, shared/name)
    write_json(root/'codebook.json', make_codebook(alignment, library))
    manifest = freeze_engine(root/'frozen_engine')
    summary = {'status': 'running', 'trials': [], 'source_planning_dir': str(args.source_planning_dir),
        'format_normalization_version': NORMALIZATION_VERSION,
        'shared_input_hashes': original, 'instruction': descriptor['instruction'], 'model': args.model,
        'repeats': args.repeats, 'skill_proposal_limit': 2,
        'request_timeout_seconds': 180, 'http_attempt_limit': 2,
        'manual_repair': False, 'failed_trial_replay': False,
        'simulation_executed': False, 'training_eligible': False,
        'scope': 'Frozen leaf skill-format comparison, not full pipeline success'}
    try:
        if args.preflight_only:
            summary['status'] = 'input_preflight_passed'
        else:
            for repeat in range(1, args.repeats+1):
                arms = ('nested_json', 'matrix') if repeat % 2 else ('matrix', 'nested_json')
                for arm in arms:
                    folder = root/'trials'/f'{repeat}_{arm}'
                    folder.mkdir(parents=True, exist_ok=False)
                    trial = {'repeat': repeat, 'arm': arm, 'output_dir': str(folder)}
                    summary['trials'].append(trial)
                    write_json(root/'batch_summary.json', summary)
                    print(f'START repeat={repeat} arm={arm}', flush=True)
                    started = time.monotonic()
                    command = [sys.executable, str(root/'frozen_engine/outputs/habitat_demo_v1/run_action_matrix_ab.py'),
                        '--worker-input', str(shared.resolve()), '--arm', arm, '--output-dir', str(folder),
                        '--model', args.model, '--base-url', args.base_url]
                    try:
                        execute_worker(command, folder/'console.log', timeout=1000)
                    except Exception as exc:
                        trial['launcher_error'] = str(exc)
                    result = read_json(folder/'trial_summary.json') if (folder/'trial_summary.json').exists() else {'status':'failed','failure_cause':'No trial summary'}
                    trial.update(result, seconds=time.monotonic()-started)
                    if trial.get('launcher_error'):
                        trial['status'] = 'failed'
                    write_json(root/'batch_summary.json', summary)
                    report(root, summary)
                    print(f'END repeat={repeat} arm={arm} status={trial["status"]}', flush=True)
            summary['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted_or_invalid', error=str(exc))
    finally:
        summary['frozen_engine_unchanged'] = frozen_unchanged(root/'frozen_engine', manifest)
        summary['shared_input_unchanged'] = input_hashes(shared) == original
        summary['source_input_unchanged'] = input_hashes(source) == original
        if not all(summary[k] for k in ('frozen_engine_unchanged','shared_input_unchanged','source_input_unchanged')):
            summary['status'] = 'invalid_inputs_or_engine'
        summary['all_skill_stage_checks_passed'] = (all(t.get('status') == 'skill_stage_symbolic_passed'
            for t in summary['trials']) if len(summary['trials']) == 2*args.repeats else None)
        write_json(root/'batch_summary.json', summary)
        report(root, summary)
    print(f'status={summary["status"]} summary={root/"batch_summary.json"}', flush=True)
    return 0 if summary['status'] == 'input_preflight_passed' or (summary['status'] == 'completed' and
        summary['all_skill_stage_checks_passed']) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-planning-dir', type=Path, default=SOURCE)
    p.add_argument('--output-dir', type=Path, default=SCENE/'action_matrix_ab'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--worker-input', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--arm', choices=('nested_json','matrix'), help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.repeats <= 5:
        p.error('repeats must be in [1,5]')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY','').strip():
        p.error('DASHSCOPE_API_KEY required')
    if args.worker_input and not args.arm:
        p.error('--arm required for worker')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    return worker(args) if args.worker_input else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
