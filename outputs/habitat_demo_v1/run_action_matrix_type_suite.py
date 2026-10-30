"""Paired task-type format comparison on saved Scene201 symbolic checkpoints.

No fresh decomposition/binding, recursive planning or simulator.
Each case is independent; both arms share one real local cosine retrieval.
"""
import argparse
import hashlib
import logging
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from run_action_matrix_ab import (FILES, long_path, input_hashes, load_inputs,
    make_codebook, build_result, review_symbolic, DEFAULT_URL)
from replay_json_round_planning import INPUT_FILES, validate_checkpoint
from run_json_long_horizon import SCENE
from run_json_stability_batch import freeze_engine, frozen_unchanged
from run_live_llm_sequence import execute_worker
from run_navigation_demo import read_json, write_json
from verify_plan import flatten_steps
from skill_retrieval import LocalEncoder, model_fingerprint, enrich, candidates, check_membership
from adapter_parameter_contract import project_library, VERSION as PARAMETER_VERSION
from skill_output_normalization import VERSION as NORMALIZATION_VERSION

SOURCE_FILES = (*INPUT_FILES, 'object_alignment.json', 'skill_plan_final.json')
SAVED = 'skill_cosine_ab/scene201_20261005_155607_698406/trials'
MIXED = 'skill_cosine_ab/scene201_20261005_200854_623291/trials/extended_mixed_18_1_bounded_cosine'


def catalog(scene):
    def case(key, label, difficulty, path):
        return dict(case_id=key, label=label, difficulty=difficulty, source=scene/path)
    return [
        case('single_navigation', '单目标导航', 1,
             SAVED+'/short_navigation_1_1_bounded_cosine/round_1/planning'),
        case('ordered_navigation', '顺序访问两个目标', 2,
             SAVED+'/compound_transfer_2_1_v2_state_guards/round_1/planning'),
        case('pickup_lift', '抓取、抬起并保持持有', 3,
             MIXED+'/round_2/planning/hierarchy/node_001/planning'),
        case('carry_keep_holding', '搬运且禁止放置', 3,
             MIXED+'/round_3/planning/hierarchy/node_001/planning'),
        case('place_held', '放置已经持有的物体', 3,
             MIXED+'/round_4/planning/hierarchy/node_001/planning'),
        case('carry_place', '搬运并放置', 4,
             MIXED+'/round_8/planning/hierarchy/node_001/planning'),
        case('mixed_transfer_revisit', '抓取搬运放置后访问并返回移动物体', 5,
             SAVED+'/compound_transfer_2_1_v2_state_guards/round_2/planning'),
    ]


def source_hashes(source):
    return {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def normalize_assertions(descriptor):
    # Historical round writers used null to mean "no support assertion".
    # Preserve all actual goals and non-null destination assertions.
    descriptor = dict(descriptor)
    if descriptor.get('expected_support_id', 'absent') is None:
        descriptor.pop('expected_support_id')
    return descriptor


def prepare_case(case, destination):
    source = long_path(case['source'])
    before = source_hashes(source)
    state, descriptor, library, nodes, _ = validate_checkpoint(source)
    library = project_library(enrich(library))
    alignment = read_json(source/'object_alignment.json')
    if alignment.get('status') != 'success' or alignment['task_decomposition']['instruction'] != descriptor['instruction']:
        raise ValueError(f"Saved binding/instruction mismatch: {case['case_id']}")
    descriptor = normalize_assertions(descriptor)
    # Check the historically accepted plan independently; it is not sent to LLM.
    reference = build_result(alignment, library, read_json(source/'skill_plan_final.json'))
    review = review_symbolic(dict(descriptor, alignment=alignment, plan=reference),
        library, nodes, state, library['execution_contract']['proxy_reach_m'])
    book = make_codebook(alignment, library)
    skills = [s['skill_id'] for s in library['skills']]
    if len(skills) != 5 or len(set(skills)) != 5:
        raise ValueError('Only the full five-skill inventory is supported in this suite')
    for name in (*INPUT_FILES, 'object_alignment.json'):
        (destination/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, destination/name)
    write_json(destination/'round_input.json', descriptor)
    write_json(destination/'execution_contract.json', library['execution_contract'])
    write_json(destination/'execution_skill_library.json', library)
    retrieval = {'source': 'structural_preflight_placeholder_not_real_retrieval',
        'scope': 'Not permitted for live trials; must be replaced by real frozen cosine preparation',
        'subtasks': [{'subtask_id': task['subtask_id'], 'candidate_skill_ids': skills}
                     for task in alignment['task_decomposition']['subtasks']]}
    write_json(destination/'skill_retrieval_semantic_1.json', retrieval)
    load_inputs(destination)
    if source_hashes(source) != before:
        raise ValueError('Source checkpoint changed during snapshotting')
    result = {k: case[k] for k in ('case_id', 'label', 'difficulty')}
    result.update(source_planning_dir=str(case['source']), source_hashes=before,
        shared_input_hashes=input_hashes(destination), instruction=descriptor['instruction'],
        memory_version=state['backend']['memory_version'],
        input_held_object_id=state['held_object_id'], input_lifted=state['lifted'],
        subtask_count=len(alignment['task_decomposition']['subtasks']),
        reference_action_count=len(flatten_steps(reference)),
        reference_navigation_targets=review['navigation_targets'],
        reference_validation='existing_saved_plan_symbolically_valid_not_executed',
        parameter_contract_version=PARAMETER_VERSION,
        parameter_projection=library['adapter_parameter_projection'],
        nullable_support_assertion_removed='expected_support_id' not in descriptor)
    write_json(destination.parent/'reference_review.json', review)
    write_json(destination.parent/'codebook.json', book)
    write_json(destination.parent/'case_manifest.json', result)
    return result


def retrieval_worker(args):
    """Runs frozen code and one local CPU encoder; never calls LLM."""
    folder = long_path(args.worker_shared_input)
    state, descriptor, library, nodes, alignment, _ = load_inputs(folder)
    before = input_hashes(folder)
    if model_fingerprint(args.embedding_model_dir) != args.embedding_model_sha:
        raise ValueError('Embedding model fingerprint mismatch before encoding')
    encoder = LocalEncoder(args.embedding_python, args.embedding_model_dir, folder.parent/'embedding')
    try:
        retrieval = candidates(alignment, library, encoder, top_k=2, threshold=.35, margin=.05)
        retrieval.update(source='same_frozen_real_embedding_retrieval_for_both_formats',
            embedding_model_sha256=args.embedding_model_sha)
        stats = encoder.stats()
    finally:
        encoder.close()
    if model_fingerprint(args.embedding_model_dir) != args.embedding_model_sha:
        raise ValueError('Embedding model changed during encoding')
    if input_hashes(folder) != before:
        raise ValueError('Shared inputs changed during encoding')
    write_json(folder/'skill_retrieval_semantic_1.json', retrieval)
    write_json(folder.parent/'embedding_metrics.json', stats)
    return 0


def finalize_retrieval(case, shared):
    retrieval = read_json(shared/'skill_retrieval_semantic_1.json')
    if retrieval.get('source') != 'same_frozen_real_embedding_retrieval_for_both_formats':
        raise ValueError('Real cosine preparation missing; no full-inventory substitution')
    load_inputs(shared)
    tasks = retrieval['subtasks']
    case['retrieval'] = dict(subtasks=len(tasks), confident_subtasks=sum(t['confident'] for t in tasks),
        ambiguous_subtasks=sum(not t['confident'] for t in tasks),
        average_candidate_count=sum(len(t['candidate_skill_ids']) for t in tasks)/len(tasks))
    reference = read_json(long_path(Path(case['source_planning_dir']))/'skill_plan_final.json')
    try:
        reference_membership = check_membership(reference, retrieval)
    except ValueError as exc:
        reference_membership = dict(candidate_membership=False, error=str(exc))
    # Diagnostic only: reference actions never added to candidates or LLM input.
    write_json(shared.parent/'reference_candidate_membership.json', reference_membership)
    case['reference_in_initial_candidates'] = reference_membership['candidate_membership']
    case['embedding_metrics'] = read_json(shared.parent/'embedding_metrics.json')
    case['shared_input_hashes'] = input_hashes(shared)
    write_json(shared.parent/'case_manifest.json', case)


def aggregate(trials, cases):
    rows = []
    for case in cases:
        for arm in ('nested_json', 'matrix'):
            selected = [t for t in trials if t['case_id'] == case['case_id'] and t['arm'] == arm]
            if not selected:
                continue
            times = [t['seconds'] for t in selected]
            no_transport = [t['seconds'] for t in selected
                            if not t.get('audit', {}).get('transport_failure_count', 0)
                            and 'token_usage' in t]
            rows.append(dict(case_id=case['case_id'], label=case['label'], arm=arm,
                trials=len(selected), final_passes=sum(t.get('status') == 'skill_stage_symbolic_passed' for t in selected),
                first_proposal_valid=sum(t.get('first_proposal_valid') is True for t in selected),
                first_proposal_strict_valid=sum(t.get('first_proposal_strict_valid') is True for t in selected),
                first_proposal_normalized_valid=sum(t.get('first_proposal_normalized_valid') is True for t in selected),
                normalized_accepted_trials=sum(t.get('accepted_after_normalization') is True for t in selected),
                normalization_applied_attempts=sum(t.get('normalization_applied_attempts', 0) for t in selected),
                clean_first_passes=sum(t.get('status') == 'skill_stage_symbolic_passed' and t.get('first_proposal_valid') is True and
                    t.get('audit', {}).get('http_attempt_count') == 1 for t in selected),
                program_rejections=sum(t.get('audit', {}).get('program_rejection_count', 0) for t in selected),
                transport_failures=sum(t.get('audit', {}).get('transport_failure_count', 0) for t in selected),
                known_total_tokens=sum(t.get('token_usage', {}).get('total_tokens', 0) for t in selected),
                unknown_usage_transport_failures=sum(t.get('token_usage', {}).get('transport_failures_unknown_usage', 0) for t in selected),
                responses_missing_usage=sum(t.get('token_usage', {}).get('responses_missing_usage', 0) for t in selected),
                elapsed_seconds=sum(times), median_seconds=statistics.median(times),
                median_seconds_without_transport_failure=statistics.median(no_transport) if no_transport else None,
                unreported_trials=sum('token_usage' not in t for t in selected)))
    return rows


def report(root, summary):
    lines = ['# 不同任务类型：嵌套JSON与动作矩阵对照', '',
        f"批次状态：{summary['status']}；每类重复 {summary['repeats']} 次。", '',
        '## 固定输入与实验范围', '',
        '每个案例复用历史真实LLM分解/绑定和同一初始符号状态；两组共用一次真实本地余弦筛选结果。'
        '本批不重新分解、不递推、不仿真。复杂案例的历史分解曾由分层规划合并，'
        '本批一次性请求整组技能动作，不将其宣传为新端到端规划。', '',
        f"参数契约版本：{summary.get('parameter_contract_version', '未报告')}。模型请求前将库参数投影到执行器实际接口；"
        '导航只提供target，矩阵第4列必须为0。不删除模型返回参数，也不放宽执行器校验。', '',
        '| 类型 | 子任务数 | 历史参考动作数 | 指令 |', '|---|---:|---:|---|']
    for c in summary['cases']:
        lines.append(f"| {c['label']} | {c['subtask_count']} | {c['reference_action_count']} | {c['instruction']} |")
    lines += ['', '## 余弦筛选与共用准备成本', '',
        '每类只计算一次并冻结候选供两组和所有重复使用；独立记录准备成本，不把共用模型启动时间重复计给每个trial。'
        'top_k=2、threshold=0.35、margin=0.05；保留前置技能闭包，歧义保留完整库，第二次提案允许有界扩大候选。', '',
        '| 类型 | 高置信子任务 | 歧义子任务 | 平均候选数 | 本地编码含启动秒 |',
        '|---|---:|---:|---:|---:|']
    for c in summary['cases']:
        r, m = c.get('retrieval', {}), c.get('embedding_metrics', {})
        avg = f"{r['average_candidate_count']:.2f}" if 'average_candidate_count' in r else '待计算'
        lines.append(f"| {c['label']} | {r.get('confident_subtasks','待计算')} | {r.get('ambiguous_subtasks','待计算')} | {avg} | {m.get('seconds_including_startup','待计算')} |")
    lines += ['', '## 分类型统计', '',
        '“首提案有效”包含确定性格式归一化后通过，原始格式直接有效与归一化通过另表记录。', '',
        '| 类型 | 格式 | 最终通过 | 首提案有效 | 无重试首次通过 | 格式/程序拒绝 | 传输失败 | 已知Token | 耗时中位数秒 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['aggregates']:
        lines.append(f"| {r['label']} | {r['arm']} | {r['final_passes']}/{r['trials']} | "
            f"{r['first_proposal_valid']}/{r['trials']} | {r['clean_first_passes']}/{r['trials']} | "
            f"{r['program_rejections']} | {r['transport_failures']} | {r['known_total_tokens']} | {r['median_seconds']:.1f} |")
    lines += ['', '## 格式归一化统计', '',
        f"归一化版本：{summary.get('format_normalization_version', '未报告')}。只包装裸矩阵或重命名明确技能别名，不补决策。", '',
        '| 类型 | 格式 | 首次原始格式直接通过 | 首次归一化后通过 | 最终归一化后通过 | 归一化提案次数 |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary['aggregates']:
        lines.append(f"| {r['label']} | {r['arm']} | {r['first_proposal_strict_valid']}/{r['trials']} | "
            f"{r['first_proposal_normalized_valid']}/{r['trials']} | {r['normalized_accepted_trials']} | {r['normalization_applied_attempts']} |")
    lines += ['', '## 每次结果与失败证据', '',
        '| 类型 | 重复 | 格式 | 状态 | 已知Token | 未知用量超时数 | 耗时秒 |',
        '|---|---:|---|---|---:|---:|---:|']
    for t in summary['trials']:
        lines.append(f"| {t['case_id']} | {t['repeat']} | {t['arm']} | {t.get('status')} | "
            f"{t.get('token_usage',{}).get('total_tokens','未报告')} | "
            f"{t.get('token_usage',{}).get('transport_failures_unknown_usage','未报告')} | {t['seconds']:.1f} |")
        errors = [f"{a.get('failure_phase','unknown')}: {a['error']}" for a in t.get('attempts', []) if a.get('error')]
        if t.get('launcher_error'):
            errors.append('launcher: '+t['launcher_error'])
        for error in errors:
            lines += ['', f"- {t['case_id']} / {t['repeat']} / {t['arm']}：{error.replace(chr(10),' ')}", '']
    lines += ['', '## 解读边界', '',
        '- 每次最多2次提案，每次HTTP最多2次；第二次可扩大到五技能库，扩大与候选冲突记录在trial内；不自动补动作。',
        '- 先分解/绑定、再余弦候选、再LLM技能选择；相似度不等于成功概率，两者并非独立双重认证。',
        '- 递推预算为0，分解和绑定固定；本批不能直接证明递推次数减少或完整组合端到端提升。',
        '- 首提案有效不等于无超时；单独报告无重试首次通过和传输失败。',
        '- 原始返回不改写；proposal_attempt_N.normalization.json保存转换规则/哈希，发生转换时另存.normalized.json；无歧义别名和包装不触发API重试。',
        '- 原始返回、失败输出、编码表、解码JSON、符号审查与Token审计均保留在各trial目录。',
        '- 已知Token仅统计收到响应的usage，未知超时计费及缺失usage不能当作零；小样本不作显著性结论。',
        '- 来源快照不修改；副本将历史expected_support_id:null解释为无支撑面断言，保留全部实际目标和非空断言。',
        '- 同一Scene201三节点测试世界，不是多个真实场景；房间归属和动作效果均为符号假设。',
        '- 完整状态、显式访问顺序、支撑面与目标仍由原验证器检查；未增加LLM整指令语义复核。',
        '- 本批接口限五技能、两个对象参数槽；没有数值预设。通用库仍保留其原可选参数；training_eligible=false。', '']
    (root/'任务类型矩阵对照分析.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    cases = catalog(args.scene_dir)
    if args.cases:
        requested = args.cases.split(',')
        if len(set(requested)) != len(requested) or set(requested)-{c['case_id'] for c in cases}:
            raise ValueError('Unknown or repeated case ID')
        cases = [c for c in cases if c['case_id'] in requested]
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    summary = dict(status='preparing', cases=[], trials=[], repeats=args.repeats, model=args.model,
        inventory_policy='identical_real_cosine_candidates_then_bounded_widening',
        simulation_executed=False, training_eligible=False, fresh_decomposition=False,
        manual_repair=False, frozen_code=True, proposal_limit=2, http_attempt_limit=2,
        scope='Paired saved-checkpoint cosine+skill-output format comparison; not a long-horizon chain',
        recursive_planning_budget=0, retrieval_parameters=dict(top_k=2, threshold=.35, margin=.05),
        embedding_prepared=False)
    summary['parameter_contract_version'] = PARAMETER_VERSION
    summary['format_normalization_version'] = NORMALIZATION_VERSION
    manifest = None
    try:
        # All cases must preflight before spending any API calls.
        for case in cases:
            summary['cases'].append(prepare_case(case, root/'cases'/case['case_id']/'shared_input'))
        manifest = freeze_engine(root/'frozen_engine')
        if not args.preflight_only:
            started = time.monotonic()
            digest = model_fingerprint(args.embedding_model_dir)
            summary['embedding_model_sha256'] = digest
            summary['embedding_model_dir'] = str(args.embedding_model_dir)
            for case in summary['cases']:
                shared = root/'cases'/case['case_id']/'shared_input'
                print(f"PREPARE cosine case={case['case_id']}", flush=True)
                execute_worker([sys.executable,
                    str(root/'frozen_engine/outputs/habitat_demo_v1/run_action_matrix_type_suite.py'),
                    '--worker-shared-input', str(shared.resolve()),
                    '--embedding-python', str(args.embedding_python),
                    '--embedding-model-dir', str(args.embedding_model_dir),
                    '--embedding-model-sha', digest], shared.parent/'retrieval.console.log', timeout=600)
                finalize_retrieval(case, shared)
            summary['embedding_preparation_seconds'] = time.monotonic()-started
            summary['embedding_prepared'] = True
        summary['status'] = 'input_preflight_passed' if args.preflight_only else 'running'
        write_json(root/'batch_summary.json', summary)
        if not args.preflight_only:
            for repeat in range(1, args.repeats+1):
                for index, case in enumerate(summary['cases']):
                    arms = ('nested_json', 'matrix') if (repeat+index) % 2 else ('matrix', 'nested_json')
                    for arm in arms:
                        folder = root/'trials'/case['case_id']/f'{repeat}_{arm}'
                        folder.mkdir(parents=True, exist_ok=False)
                        trial = dict(case_id=case['case_id'], repeat=repeat, arm=arm, output_dir=str(folder))
                        summary['trials'].append(trial)
                        write_json(root/'batch_summary.json', summary)
                        print(f"START case={case['case_id']} repeat={repeat}/{args.repeats} arm={arm}", flush=True)
                        started = time.monotonic()
                        try:
                            execute_worker([sys.executable,
                                str(root/'frozen_engine/outputs/habitat_demo_v1/run_action_matrix_ab.py'),
                                '--worker-input', str((root/'cases'/case['case_id']/'shared_input').resolve()),
                                '--arm', arm, '--output-dir', str(folder), '--model', args.model,
                                '--base-url', args.base_url], folder/'console.log', timeout=1000)
                        except Exception as exc:
                            trial['launcher_error'] = str(exc)
                        result = read_json(folder/'trial_summary.json') if (folder/'trial_summary.json').exists() else dict(status='failed', failure_cause='No trial summary')
                        trial.update(result, seconds=time.monotonic()-started)
                        if trial.get('launcher_error'):
                            trial['status'] = 'failed'
                        summary['aggregates'] = aggregate(summary['trials'], summary['cases'])
                        write_json(root/'batch_summary.json', summary)
                        report(root, summary)
                        print(f"END case={case['case_id']} repeat={repeat} arm={arm} status={trial['status']}", flush=True)
            summary['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'preflight_or_batch_failed', error=str(exc))
        logging.exception('Task-type suite stopped')
    finally:
        # Incomplete retrieval preparation is not an evaluated LLM trial.
        for case in summary['cases']:
            shared = root/'cases'/case['case_id']/'shared_input'
            if not summary['trials'] and not summary['embedding_prepared']:
                case['shared_input_hashes'] = input_hashes(shared)
        summary['source_unchanged'] = all(source_hashes(long_path(Path(c['source_planning_dir']))) == c['source_hashes'] for c in summary['cases'])
        summary['shared_input_unchanged'] = all(input_hashes(root/'cases'/c['case_id']/'shared_input') == c['shared_input_hashes'] for c in summary['cases'])
        summary['frozen_engine_unchanged'] = frozen_unchanged(root/'frozen_engine', manifest) if manifest else None
        if summary.get('embedding_model_sha256'):
            try:
                summary['embedding_model_unchanged'] = model_fingerprint(args.embedding_model_dir) == summary['embedding_model_sha256']
            except (OSError, ValueError):
                summary['embedding_model_unchanged'] = False
        if not summary['source_unchanged'] or not summary['shared_input_unchanged'] or summary['frozen_engine_unchanged'] is False or summary.get('embedding_model_unchanged') is False:
            summary['status'] = 'invalid_inputs_or_engine'
        summary['aggregates'] = aggregate(summary['trials'], summary['cases'])
        expected = len(cases)*args.repeats*2
        summary['all_skill_stage_checks_passed'] = (all(t.get('status') == 'skill_stage_symbolic_passed' for t in summary['trials']) if len(summary['trials']) == expected else None)
        write_json(root/'batch_summary.json', summary)
        report(root, summary)
    print(f"status={summary['status']} summary={root/'batch_summary.json'}", flush=True)
    print(f"analysis={root/'任务类型矩阵对照分析.md'}", flush=True)
    return 0 if summary['status'] == 'input_preflight_passed' or (summary['status'] == 'completed' and summary['all_skill_stage_checks_passed']) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    scene = Path('D:/'+str(SCENE)[7:]) if os.name == 'nt' else SCENE
    p.add_argument('--scene-dir', type=Path, default=scene)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--cases', help='Comma-separated case IDs; default all seven')
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--embedding-python', type=Path, default=Path(sys.executable))
    p.add_argument('--embedding-model-dir', type=Path)
    p.add_argument('--worker-shared-input', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--embedding-model-sha', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.repeats <= 3:
        p.error('repeats must be in [1,3]')
    if args.worker_shared_input:
        if not args.embedding_model_dir or not args.embedding_model_sha:
            p.error('Embedding model directory/fingerprint required')
        return retrieval_worker(args)
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        p.error('DASHSCOPE_API_KEY required')
    if not args.preflight_only and not args.embedding_model_dir:
        p.error('--embedding-model-dir required; no automatic download')
    args.output_dir = args.output_dir or args.scene_dir/'action_matrix_type_ab'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
