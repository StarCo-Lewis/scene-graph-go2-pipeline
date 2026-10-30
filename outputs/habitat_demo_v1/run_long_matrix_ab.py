"""Fresh full-round JSON/matrix paired evaluation, without Habitat or repairs."""
import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_json_stability_batch as batch
from run_hierarchical_json_ab import SOURCE
from run_json_long_horizon import SCENE
from json_stability_cases import catalog, validate_cases, DEFAULT_CASES
from run_navigation_demo import read_json, write_json
from json_symbolic_review import review_symbolic, SYMBOLIC_GOAL_POLICY
from skill_retrieval import enrich, LocalEncoder, model_fingerprint
from token_usage_audit import usage

ARMS = ('nested_json', 'matrix')
DEFAULT_SELECTION = (*DEFAULT_CASES[:5], 'baseline_12', *DEFAULT_CASES[5:])


def select_cases(source, names):
    source_summary = read_json(source/'batch_summary.json')
    hashes = source_summary['case_input_hashes']
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate case selection')
    rows, provenance = [], {}
    for name in names:
        if name == 'baseline_12' and name not in hashes:
            node_source = source/'cases/extended_mixed_18.json'
            if batch.sha(node_source) != hashes.get('extended_mixed_18'):
                raise ValueError('Historical node source hash mismatch')
            rows.append(next(c for c in catalog(read_json(node_source)['nodes']) if c['name']==name))
            provenance[name] = {'source':str(node_source), 'sha256':batch.sha(node_source),
                'method':'original twelve instructions on historical nodes; no saved plan'}
        else:
            path = source/'cases'/f'{name}.json'
            if name not in hashes or batch.sha(path) != hashes[name]:
                raise ValueError('Unknown case or historical hash mismatch: '+name)
            rows.append(read_json(path))
            provenance[name] = {'source':str(path), 'sha256':hashes[name], 'method':'unchanged historical case'}
    return validate_cases(rows), provenance


def request_estimate(cases, repeats):
    # One decomposition, binding, skills and redundancy request per positive round.
    return repeats * sum(8*len(c['rounds']) if c['kind']=='positive' else 2 for c in cases)


def plan_round(args, folder, encoder):
    from plan_live_round import plan
    library = enrich(read_json(folder/'execution_skill_library.json'))
    write_json(folder/'execution_skill_library.json', library)
    files = ('round_input.json', 'runtime_state.json', 'execution_skill_library.json',
        'execution_contract.json', 'input_memory/scene/scene_graph.json', 'input_memory/scene/scene_graph_edges.json')
    before = {name:batch.sha(folder/name) for name in files}
    handler = logging.FileHandler(folder/'planning.console.log', encoding='utf-8')
    logging.getLogger().addHandler(handler)
    try:
        result = plan(SimpleNamespace(output_dir=folder, round_input=folder/'round_input.json',
            model=args.model, base_url=args.base_url, reach=1.5, skill_output_format=args.arm,
            skill_proposal_limit=2, compile_navigation=False, skill_retriever=encoder,
            retrieval_top_k=args.top_k, retrieval_threshold=args.cosine_threshold,
            retrieval_margin=args.cosine_margin), reviewer=review_symbolic)
        unchanged = all(batch.sha(folder/name)==digest for name,digest in before.items())
        write_json(folder/'round_input_integrity.json', {'unchanged':unchanged, 'before_sha256':before})
        if not unchanged:
            summary = read_json(folder/'planning_summary.json')
            summary.update(status='failed', failed_stage='round_input_integrity', error='Planning input mutated')
            write_json(folder/'planning_summary.json', summary)
            return 2
        return result
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def worker(args):
    case = read_json(args.worker_case)
    validate_cases([case])
    if case['kind']=='negative':
        if args.arm != 'shared_control':
            raise ValueError('Negative cases are shared binding controls, not format arms')
        return batch.negative_worker(args, case)
    encoder = None
    original = batch.plan_one_round
    try:
        if model_fingerprint(args.embedding_model_dir) != args.embedding_model_sha:
            raise ValueError('Embedding model changed')
        encoder = LocalEncoder(args.embedding_python, args.embedding_model_dir, args.output_dir/'embedding')
        batch.plan_one_round = lambda a,folder: plan_round(a,folder,encoder)
        return batch.positive_worker(args,case)
    finally:
        batch.plan_one_round = original
        if encoder is not None:
            write_json(args.output_dir/'embedding_metrics.json',encoder.stats())
            encoder.close()


def format_metrics(root):
    summaries = [read_json(p) for p in root.glob('round_*/planning/planning_summary.json')]
    finals = [read_json(p) for p in root.glob('round_*/planning/skill_plan_final.json')]
    return {'planning_summaries':len(summaries),
        'semantic_redecompositions':sum(s.get('semantic_redecomposition_count',0) for s in summaries),
        'format_normalization_attempts':sum(n.get('changed',False) for s in summaries
            for n in s.get('skill_format_normalizations',[])),
        'accepted_normalized_rounds':sum(p.get('format_normalization',{}).get('changed',False) for p in finals),
        'rounds_with_skill_correction':sum(any(stage.get('schema_attempts',1)>1 for stage in s.get('stages',[]))
            for s in summaries),
        'round_inputs_unchanged':all(read_json(p)['unchanged'] for p in root.glob('round_*/planning/round_input_integrity.json'))}


def passed(trial):
    return trial.get('evaluation_outcome','').startswith(('clean_', 'passed_'))


def comparison(summary):
    valid = all(summary.get(k) is True for k in ('frozen_engine_unchanged','case_inputs_unchanged','historical_sources_unchanged'))
    valid = valid and summary.get('embedding_model_unchanged',True) is True
    valid = valid and summary['status'] not in ('invalid_or_harness_failure','invalid_inputs_or_engine')
    valid = valid and all(r.get('format_metrics',{}).get('round_inputs_unchanged',True) for r in summary['trials'])
    totals = {}
    for arm in ARMS:
        rows = [r for r in summary['trials'] if r['arm']==arm and r.get('evaluation_outcome')]
        successes = sum(passed(r) for r in rows)
        tokens = sum(r.get('token_usage',{}).get('total_tokens',0) for r in rows)
        seconds = sum(r.get('duration_seconds',0) for r in rows)
        totals[arm] = {'trials':len(rows), 'passes':successes,
            'clean_first_pass':sum(r['evaluation_outcome']=='clean_first_pass' for r in rows),
            'normalization_only_passes':sum(r['evaluation_outcome']=='passed_after_format_normalization' for r in rows),
            'completed_rounds':sum(r.get('completed_round_count',0) for r in rows),
            'expected_rounds':sum(r.get('expected_round_count',0) for r in rows),
            'known_tokens_including_failures':tokens, 'seconds_including_failures':seconds,
            'tokens_per_success_including_failures':tokens/successes if successes else None,
            'seconds_per_success_including_failures':seconds/successes if successes else None,
            'transport_failures':sum(r.get('audit',{}).get('transport_failure_count',0) for r in rows),
            'program_rejections':sum(r.get('audit',{}).get('program_rejection_count',0) for r in rows),
            'unknown_usage_transport_failures':sum(r.get('token_usage',{}).get('transport_failures_unknown_usage',0) for r in rows),
            'responses_missing_usage':sum(r.get('token_usage',{}).get('responses_missing_usage',0) for r in rows)}
    pairs = []
    for name,repeat in sorted({(r['case'],r['repeat']) for r in summary['trials'] if r['kind']=='positive'}):
        rows = {r['arm']:r for r in summary['trials'] if (r['case'],r['repeat'])==(name,repeat)}
        if all(arm in rows and rows[arm].get('evaluation_outcome') for arm in ARMS):
            pairs.append({'case':name,'repeat':repeat,'both_complete':all(passed(rows[a]) for a in ARMS),
                'both_no_transport_failure':all(not rows[a].get('audit',{}).get('transport_failure_count',0)
                    and not rows[a].get('audit',{}).get('http_error_count',0) for a in ARMS),
                'arms':{a:{'outcome':rows[a]['evaluation_outcome'], 'tokens':rows[a]['token_usage']['total_tokens'],
                    'seconds':rows[a]['duration_seconds']} for a in ARMS}})
    pair_costs = {}
    for label, selected in (
        ('both_complete',[p for p in pairs if p['both_complete']]),
        ('both_complete_no_transport_failure',[p for p in pairs if p['both_complete'] and p['both_no_transport_failure']])):
        costs = {a:{'tokens':sum(p['arms'][a]['tokens'] for p in selected),
            'seconds':sum(p['arms'][a]['seconds'] for p in selected)} for a in ARMS}
        pair_costs[label] = {'pair_count':len(selected),'costs':costs,
            'matrix_savings':{k:1-costs['matrix'][k]/costs['nested_json'][k] if costs['nested_json'][k] else None
                for k in ('tokens','seconds')}}
    by_case = []
    for name in sorted({r['case'] for r in summary['trials'] if r['kind']=='positive'}):
        for arm in ARMS:
            rows = [r for r in summary['trials'] if r['case']==name and r['arm']==arm and r.get('evaluation_outcome')]
            if rows:
                by_case.append({'case':name,'arm':arm,'trials':len(rows),'passes':sum(passed(r) for r in rows),
                    'clean_first_pass':sum(r['evaluation_outcome']=='clean_first_pass' for r in rows),
                    'known_tokens':sum(r['token_usage']['total_tokens'] for r in rows),
                    'seconds':sum(r['duration_seconds'] for r in rows)})
    return {'evaluation_valid':valid,'arms':totals,'by_case':by_case,'pairs':pairs,'paired_costs':pair_costs}


def report(root, summary):
    result = comparison(summary)
    write_json(root/'comparison_summary.json',result)
    lines = ['# 完整多轮JSON／矩阵对照', '',
        f"批次状态：{summary['status']}；冻结输入和代码有效：{result['evaluation_valid']}。", '',
        f"符号目标校验版本：`{summary.get('symbolic_goal_policy', 'legacy_robot_near_only')}`。", '',
        '两组都从原始指令开始，每轮重新分解、绑定、余弦筛选、技能请求和冗余复核；仅输出表示不同。'
        '不是复用已通过的分解，也不是对旧递推方法整体重跑。两组不递推、不编译导航，技能提案最多两次。', '',
        '| 方法 | 完整通过/已测案例 | 原格式无拒绝首次通过 | 仅归一化后通过 | 完成/请求轮数 | 已知Token（含失败） | 耗时秒（含失败） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,r in result['arms'].items():
        lines.append(f"| {a} | {r['passes']}/{r['trials']} | {r['clean_first_pass']} | {r['normalization_only_passes']} | "
            f"{r['completed_rounds']}/{r['expected_rounds']} | {r['known_tokens_including_failures']} | {r['seconds_including_failures']:.1f} |")
    lines += ['', '## 按任务类型统计', '',
        '| 案例 | 方法 | 完整通过/已测 | 原格式首次通过 | 已知Token（含失败） | 耗时秒 |', '|---|---|---:|---:|---:|---:|']
    for r in result['by_case']:
        lines.append(f"| {r['case']} | {r['arm']} | {r['passes']}/{r['trials']} | {r['clean_first_pass']} | {r['known_tokens']} | {r['seconds']:.1f} |")
    lines += ['', '## 成功配对成本', '', '只对两组完整通过的同案例／同重复比较成本，避免提前失败被误记为节省。', '',
        '| 子集 | 配对数 | 矩阵Token减少 | 矩阵耗时减少 |', '|---|---:|---:|---:|']
    for label,r in result['paired_costs'].items():
        def percentage(value): return f'{100*value:.1f}%' if value is not None else '无可比样本'
        lines.append(f"| {label} | {r['pair_count']} | {percentage(r['matrix_savings']['tokens'])} | {percentage(r['matrix_savings']['seconds'])} |")
    lines += ['', '## 每个案例与失败原因', '', '| 案例 | 重复 | 方法 | 结果 | 完成轮数 | 失败阶段／原因 |', '|---|---:|---|---|---:|---|']
    for r in summary['trials']:
        reason = str(r.get('failure_cause') or r.get('launcher_error') or r.get('error') or '').replace('|','/').replace('\n',' ')
        lines.append(f"| {r['case']} | {r['repeat']} | {r['arm']} | {r.get('evaluation_outcome','待完成')} | "
            f"{r.get('completed_round_count','—')} | {r.get('failure_stage','')} {reason} |")
    lines += ['', '## 限制与审计', '',
        '- 原格式无拒绝首次通过要求整条链无验证拒绝、解析错误或传输失败；归一化和有界纠错分别保留记录。验证拒绝也可能是未采纳的冗余复核，不一定是重新生成技能。',
        '- 所有失败消耗计入；超时未知计费及缺失usage不当作零。成本包含新分解／绑定、技能、复核、本地编码启动和推理，不直接套用先前技能单阶段A/B降幅。',
        '- 每个方法每个案例启动一次本地编码器，跨该案例所有轮缓存；初始化探测是共用准备成本，单独记录，不属于LLM Token。',
        '- 独立案例从相同初始状态开始，之后各组按自己的已校验计划更新符号状态；状态、分解和候选可能随输出不同，不强称每轮候选集合相同。',
        '- 缺物体／歧义／未知关系是共享绑定负例，每重复仅运行一次，不重复计给两个方法，不用来宣称矩阵改善。',
        '- 沿用历史案例和节点；空expected_support_id沿用旧案例语义。每轮输入哈希不变，冻结代码／模型／案例不变；失败跳过该案例后续轮次，但继续其他独立试次。',
        '- 不人工修复、不失败补跑、不退回另一方法。内部原有有界纠错仍允许，必须纳入结果和成本。',
        '- 只有同一个Scene201三节点符号世界，房间与动作效果是假设。没有Habitat、导航路径、碰撞、真实抓取或支撑验证；training_eligible=false。',
        '- 小样本描述性比较，不证明新场景泛化或显著性；旧递推日志只作背景，不与本批直接当作严格同条件对照。', '']
    (root/'长程矩阵对照分析.md').write_text('\n'.join(lines),encoding='utf-8')


def run(args):
    started = time.monotonic()
    cases, provenance = select_cases(args.source_batch, args.case or list(DEFAULT_SELECTION))
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest = batch.freeze_engine(root/'frozen_engine')
    hashes = {}
    for case in cases:
        path = root/'cases'/f'{case["name"]}.json'
        write_json(path,case)
        hashes[case['name']] = batch.sha(path)
    summary = {'status':'running','trials':[],'arms':list(ARMS),'case_input_hashes':hashes,
        'historical_sources':provenance,'repeats':args.repeats,'model':args.model,
        'planned_trial_count':args.repeats*sum(2 if c['kind']=='positive' else 1 for c in cases),
        'no_failure_estimated_logical_requests':request_estimate(cases,args.repeats),
        'skill_proposal_limit':2,'recursive_split_budget':0,'compile_navigation':False,
        'symbolic_goal_policy':SYMBOLIC_GOAL_POLICY,
        'retrieval':{'top_k':args.top_k,'threshold':args.cosine_threshold,'margin':args.cosine_margin},
        'embedding_model_sha256':None,'manual_repair':False,'failed_case_replay':False,
        'simulation_executed':False,'geometry_verified':False,'training_eligible':False}
    destination = root/'batch_summary.json'
    write_json(destination,summary)
    try:
        if args.preflight_only:
            summary['status'] = 'input_preflight_passed'
        else:
            probe_started = time.monotonic()
            digest = model_fingerprint(args.embedding_model_dir)
            summary['embedding_model_sha256'] = digest
            probe = LocalEncoder(args.embedding_python,args.embedding_model_dir,root/'embedding_preflight')
            try: probe.encode(['grasp the kettle','place it on a cabinet'])
            finally: probe.close()
            summary['embedding_probe_and_fingerprint_seconds'] = time.monotonic()-probe_started
            script = root/'frozen_engine/outputs/habitat_demo_v1/run_long_matrix_ab.py'
            for repeat in range(1,args.repeats+1):
                for index,case in enumerate(cases):
                    order = ARMS if (index+repeat)%2 else tuple(reversed(ARMS))
                    for arm in order if case['kind']=='positive' else ('shared_control',):
                        if not batch.frozen_unchanged(root/'frozen_engine',manifest) or batch.sha(root/'cases'/f'{case["name"]}.json') != hashes[case['name']]:
                            raise ValueError('Frozen engine or case changed')
                        folder = root/'trials'/f'{case["name"]}_{repeat}_{arm}'
                        folder.mkdir(parents=True,exist_ok=False)
                        row = {'case':case['name'],'kind':case['kind'],'repeat':repeat,'arm':arm,'output_dir':str(folder),
                            'expected_round_count':len(case.get('rounds',[])), 'difficulty_rank':case.get('difficulty_rank'),
                            'complexity':case.get('complexity')}
                        summary['trials'].append(row)
                        write_json(destination,summary)
                        command = [sys.executable,str(script),'--worker-case',str(root/'cases'/f'{case["name"]}.json'),
                            '--arm',arm,'--output-dir',str(folder),'--model',args.model,'--base-url',args.base_url,
                            '--embedding-python',str(args.embedding_python),'--embedding-model-dir',str(args.embedding_model_dir),
                            '--embedding-model-sha',digest,'--top-k',str(args.top_k),
                            '--cosine-threshold',str(args.cosine_threshold),'--cosine-margin',str(args.cosine_margin)]
                        print(f'START case={case["name"]} repeat={repeat}/{args.repeats} arm={arm}',flush=True)
                        trial_started = time.monotonic()
                        try: batch.execute_worker(command,folder/'console.log',timeout=1200*max(1,len(case.get('rounds',[])))+90)
                        except Exception as exc: row['launcher_error'] = str(exc)
                        row['duration_seconds'] = time.monotonic()-trial_started
                        result = read_json(folder/'case_summary.json') if (folder/'case_summary.json').exists() else {'status':'failed','error':'Missing case summary'}
                        row.update(result)
                        audit = batch.build_audit_report(folder,folder/'llm_audit_aggregate')
                        row['audit'] = {k:audit[k] for k in batch.AUDIT_KEYS}
                        row['program_rejections_by_stage'] = audit['program_rejections_by_stage']
                        row['evaluation_outcome'] = batch.classify(case['kind'],result,audit)
                        row['format_metrics'] = format_metrics(folder)
                        if row['evaluation_outcome']=='clean_first_pass' and row['format_metrics']['format_normalization_attempts']:
                            row['evaluation_outcome'] = 'passed_after_format_normalization'
                        if row.get('launcher_error') or not row['format_metrics']['round_inputs_unchanged']:
                            row['evaluation_outcome'] = 'final_failure'
                        row['token_usage'] = usage(folder)
                        if (folder/'embedding_metrics.json').exists(): row['embedding_metrics'] = read_json(folder/'embedding_metrics.json')
                        write_json(destination,summary)
                        report(root,summary)
                        print(f'END case={case["name"]} repeat={repeat} arm={arm} outcome={row["evaluation_outcome"]}',flush=True)
            summary['status'] = 'completed'
    except (Exception,KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'invalid_or_harness_failure',error=str(exc))
        logging.exception('Batch stopped; no repair or replay')
    finally:
        summary['frozen_engine_unchanged'] = batch.frozen_unchanged(root/'frozen_engine',manifest)
        try:
            summary['case_inputs_unchanged'] = all(batch.sha(root/'cases'/f'{name}.json')==h for name,h in hashes.items())
            summary['historical_sources_unchanged'] = all(batch.sha(Path(p['source']))==p['sha256'] for p in provenance.values())
            if summary['embedding_model_sha256'] is not None:
                summary['embedding_model_unchanged'] = model_fingerprint(args.embedding_model_dir)==summary['embedding_model_sha256']
        except (OSError,ValueError) as exc:
            summary.update(case_inputs_unchanged=False,historical_sources_unchanged=False,integrity_readback_error=str(exc))
        summary['batch_wall_seconds'] = time.monotonic()-started
        summary['all_expected_checks_passed'] = len(summary['trials'])==summary['planned_trial_count'] and all(
            r.get('evaluation_outcome') not in (None,'final_failure','evidence_incomplete') for r in summary['trials'])
        if not comparison(summary)['evaluation_valid']: summary['status'] = 'invalid_inputs_or_engine'
        write_json(destination,summary)
        report(root,summary)
    print(f'status={summary["status"]} estimated_requests={summary["no_failure_estimated_logical_requests"]} summary={destination}',flush=True)
    print(f'analysis={root/"长程矩阵对照分析.md"}',flush=True)
    return 0 if summary['status']=='input_preflight_passed' or (summary['status']=='completed' and summary['all_expected_checks_passed']) else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_source = Path('D:/'+str(SOURCE)[7:]) if os.name=='nt' else SOURCE
    default_scene = Path('D:/'+str(SCENE)[7:]) if os.name=='nt' else SCENE
    p.add_argument('--source-batch',type=Path,default=default_source)
    p.add_argument('--output-dir',type=Path,default=default_scene/'long_matrix_json_ab'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--case',action='append')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--model',default='qwen3.8-flash')
    p.add_argument('--base-url',default=batch.DEFAULT_URL)
    p.add_argument('--embedding-python',type=Path,default=Path(sys.executable))
    p.add_argument('--embedding-model-dir',type=Path)
    p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--cosine-threshold',type=float,default=.35)
    p.add_argument('--cosine-margin',type=float,default=.05)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--worker-case',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--arm',choices=(*ARMS,'shared_control'),help=argparse.SUPPRESS)
    p.add_argument('--embedding-model-sha',help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1<=args.repeats<=3 or not 1<=args.top_k<=5 or not 0<=args.cosine_threshold<=1 or not 0<=args.cosine_margin<=1:
        p.error('Invalid experiment bounds')
    if not args.preflight_only and (not os.environ.get('DASHSCOPE_API_KEY','').strip() or args.embedding_model_dir is None):
        p.error('DASHSCOPE_API_KEY and local embedding model required')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    if args.worker_case:
        if args.arm is None or args.embedding_model_sha is None:
            p.error('Worker arm and frozen embedding model fingerprint required')
        return worker(args)
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
