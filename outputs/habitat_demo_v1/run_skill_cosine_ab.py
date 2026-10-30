"""Frozen, paired cost/robustness experiment on unchanged historical task cases."""
import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import run_json_stability_batch as baseline
from run_hierarchical_json_ab import SOURCE
from run_json_long_horizon import SCENE
from run_navigation_demo import read_json, write_json
from strict_prompt_v2 import hierarchical_v2
from bounded_skill_planner import plan_bounded
from skill_retrieval import LocalEncoder, model_fingerprint
from json_stability_cases import validate_cases
from token_usage_audit import usage

ARMS = ('v2_state_guards', 'bounded_descriptions', 'bounded_cosine', 'bounded_cosine_matrix')
PILOT_CASES = ('short_navigation_1', 'compound_transfer_2', 'long_navigation_8')


def retrieval_metrics(folder):
    requests = [read_json(p) for p in folder.rglob('skill_retrieval_semantic_*.json')]
    tasks = [s for r in requests for s in r['subtasks']]
    memberships = [read_json(p) for p in folder.rglob('skill_retrieval_membership_*.json')]
    routes = [read_json(p) for p in folder.rglob('navigation_compiler_semantic_*.json')]
    return {'retrieval_subtasks_including_replans':len(tasks),
        'confident_subtasks':sum(s['confident'] for s in tasks),
        'ambiguous_full_inventory_subtasks':sum(not s['confident'] for s in tasks),
        'average_candidate_count':sum(len(s['candidate_skill_ids']) for s in tasks)/len(tasks) if tasks else None,
        'accepted_membership_records':len(memberships),
        'widened_accepted_proposals':sum(m['widened_inventory'] for m in memberships),
        'candidate_conflicts_in_accepted_proposals':sum(not m['candidate_membership'] for m in memberships),
        'compiled_navigation_segments':sum(r['route']=='compiled' for r in routes),
        'compiler_deferred_segments':sum(r['route']=='llm_required' for r in routes),
        'scope':'Retrieval events, not independent task accuracy or independent LLM agreement'}


def report(root, summary):
    comparison = {}
    valid = (summary.get('frozen_engine_unchanged') is True and summary.get('case_inputs_unchanged') is True
             and summary.get('embedding_model_unchanged', True) is True)
    for arm in summary['arms']:
        rows = [r for r in summary['trials'] if r['arm']==arm and r.get('evaluation_outcome')]
        positives = [r for r in rows if r['kind']=='positive']
        passes = [r for r in positives if r['evaluation_outcome'].startswith(('clean_', 'passed_'))]
        rounds = sum(r.get('completed_round_count', 0) for r in positives)
        tokens = sum(r.get('token_usage',{}).get('total_tokens',0) for r in positives)
        seconds = sum(r.get('duration_seconds',0) for r in positives)
        comparison[arm] = {'evaluated_positive_trials': len(positives), 'passed_trials': len(passes),
            'success_rate': len(passes)/len(positives) if positives and valid else None,
            'clean_first_pass_trials': sum(r['evaluation_outcome']=='clean_first_pass' for r in positives),
            'completed_rounds': rounds, 'provider_tokens': tokens, 'seconds': seconds,
            'tokens_per_completed_round_descriptive': tokens/rounds if rounds else None,
            'tokens_per_successful_case_including_failures': tokens/len(passes) if passes else None,
            'seconds_per_successful_case_including_failures': seconds/len(passes) if passes else None,
            'embedding_seconds': sum(r.get('embedding_metrics',{}).get('seconds_including_startup',0) for r in positives),
            'missing_usage_responses': sum(r.get('token_usage',{}).get('responses_missing_usage',0) for r in positives),
            'transport_failures_unknown_tokens': sum(r.get('token_usage',{}).get('transport_failures_unknown_usage',0) for r in positives),
            'negative_expected_refusals': sum(r['evaluation_outcome']=='expected_refusal' for r in rows if r['kind']=='negative')}
    pairs = []
    for name, repeat in sorted({(r['case'],r['repeat']) for r in summary['trials']}):
        rows = {r['arm']: r for r in summary['trials'] if (r['case'],r['repeat'])==(name,repeat)}
        if not all(a in rows and rows[a].get('evaluation_outcome') for a in summary['arms']):
            continue
        all_pass = all(rows[a]['evaluation_outcome'].startswith(('clean_', 'passed_')) for a in summary['arms'])
        pairs.append({'case':name, 'repeat':repeat, 'all_positive_arms_passed':all_pass,
            'arms': {a: {'outcome':rows[a]['evaluation_outcome'], 'tokens':rows[a]['token_usage']['total_tokens'],
                         'seconds':rows[a]['duration_seconds']} for a in summary['arms']}})
    write_json(root/'comparison_summary.json', {'arms':comparison, 'pairs':pairs, 'evaluation_valid':valid,
        'compile_navigation': summary.get('compile_navigation', False),
        'scope':'Full-case paired outcomes. Cost per completed round is descriptive; failed prefixes are not equivalent workload.'})
    lines = ['# 技能描述、有限细分与余弦匹配对照', '',
        f"批次状态：{summary['status']}；输入与冻结代码有效：{valid}。", '',
        f"明确 ID 纯导航程序组装：{summary.get('compile_navigation', False)}（独立新版本；不回写旧批次）。", '',
        '| 方法 | 完整通过/已评估正例 | 首次无纠错通过 | Token（含失败） | 耗时秒 | 完成轮数 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, r in comparison.items():
        lines.append(f"| {name} | {r['passed_trials']}/{r['evaluated_positive_trials']} | "
            f"{r['clean_first_pass_trials']} | {r['provider_tokens']} | {r['seconds']:.1f} | {r['completed_rounds']} |")
    lines += ['', '## 每个试次', '', '| 案例 | 重复 | 方法 | 结果 | 完成/总轮数 | 原因 |', '|---|---:|---|---|---|---|']
    for r in summary['trials']:
        reason = str(r.get('failure_cause') or r.get('launcher_error') or r.get('error') or '').replace('|','/').replace('\n',' ')
        lines.append(f"| {r['case']} | {r['repeat']} | {r['arm']} | {r.get('evaluation_outcome',r.get('status','待执行'))} | "
            f"{r.get('completed_round_count','—')}/{r.get('expected_round_count','—')} | {reason} |")
    lines += ['', '## 解读边界', '',
        '- v2_state_guards 使用原始描述与原分层预算；bounded_descriptions 补描述、最多一次细分、最多四段；bounded_cosine 在后者上增加向量检索。',
        '- 两个 bounded 组技能提案最多两次：第二次可扩大到完整已支持技能库；原组保持原有三次。传输和语义重规划仍有原有限额。',
        '- 相似度不等于动作成功概率。候选保留依赖闭包；低分/低间隔保留完整适配器库；扩展候选必须记录，不自动补动作。',
        '- 向量候选已提供给 LLM，因此候选一致不是两名独立判断者的一致，不能声称独立双重认证。',
        '- 本地 CPU 模型无 embedding API Token；推理启动/编码时间计入试次耗时；首次下载、安装单独作为准备成本。',
        '- 完整成功率与配对全部通过的案例成本需一起看。失败后提前停止的总成本低不能解释为节省。',
        '- 仅一个地图子集、重复小样本；没有 Habitat、导航/碰撞/真实抓取验证，training_eligible=false。',
        '- 负例是三组共享的原绑定控制，不据此评估技能余弦的提升。没有人工修复或失败补跑。', '']
    if summary.get('compile_navigation', False):
        lines += ['- 本批启用明确ID纯导航组装：命中后不请求技能JSON或LLM删减，不做向量检索；物体分解/绑定与全指令覆盖审查仍调用LLM。',
                  '- 纯导航的两个bounded组可能走相同程序路径，不能把差异归因于余弦。需另跑混合动作任务评估余弦增量效果。', '']
    (root/'技能匹配成本对照分析.md').write_text('\n'.join(lines),encoding='utf-8')


def worker(args):
    case = read_json(args.worker_case)
    validate_cases([case])
    if case['kind']=='negative':
        return baseline.negative_worker(args, case)
    encoder = None
    original = baseline.plan_one_round
    try:
        if args.arm in ('bounded_cosine', 'bounded_cosine_matrix'):
            if model_fingerprint(args.embedding_model_dir) != args.embedding_model_sha:
                raise ValueError('Embedding model differs from frozen fingerprint')
            encoder = LocalEncoder(args.embedding_python, args.embedding_model_dir, args.output_dir)
        args.skill_output_format = 'matrix' if args.arm == 'bounded_cosine_matrix' else 'nested_json'
        if args.arm=='v2_state_guards':
            baseline.plan_one_round = lambda a, folder: hierarchical_v2(a,folder,
                {'max_depth':3,'max_nodes':23,'max_leaves':12},state_guards=True)
        else:
            baseline.plan_one_round = lambda a, folder: plan_bounded(a,folder,encoder,args.max_parts)
        return baseline.positive_worker(args,case)
    finally:
        baseline.plan_one_round = original
        if encoder is not None:
            write_json(args.output_dir/'embedding_metrics.json',encoder.stats())
            encoder.close()


def run(args):
    batch_started = time.monotonic()
    source_summary = read_json(args.source_batch/'batch_summary.json')
    selected = args.case or list(PILOT_CASES)
    if len(selected)!=len(set(selected)):
        raise ValueError('Duplicate case selection')
    cases = []
    for name in selected:
        if name not in source_summary['case_input_hashes']:
            raise ValueError('Unknown historical case: '+name)
        path = args.source_batch/'cases'/f'{name}.json'
        if baseline.sha(path) != source_summary['case_input_hashes'][name]:
            raise ValueError('Historical input hash mismatch')
        cases.append(read_json(path))
    validate_cases(cases)
    root = args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    digest = None
    setup_seconds = 0.
    if not args.preflight_only and any(a in args.arms for a in ('bounded_cosine', 'bounded_cosine_matrix')):
        setup_started = time.monotonic()
        if args.embedding_model_dir is None:
            raise ValueError('--embedding-model-dir required; no auto-download during test')
        digest = model_fingerprint(args.embedding_model_dir)
        # Fail dependency/model loading before starting any expensive LLM arm.
        probe = LocalEncoder(args.embedding_python,args.embedding_model_dir,root/'embedding_preflight')
        try:
            probe.encode(['grasp the kettle','place the kettle on a cabinet'])
        finally:
            probe.close()
        setup_seconds = time.monotonic()-setup_started
    manifest = baseline.freeze_engine(root/'frozen_engine')
    hashes = {}
    for case in cases:
        target = root/'cases'/f'{case["name"]}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.source_batch/'cases'/target.name,target)
        hashes[case['name']] = baseline.sha(target)
    summary = {'status':'running','arms':args.arms,'trials':[], 'case_input_hashes':hashes,
        'source_batch':str(args.source_batch),'repeats':args.repeats,'model':args.model,
        'skill_output_formats':{a: 'matrix' if a=='bounded_cosine_matrix' else 'nested_json' for a in args.arms},
        'embedding_model_dir':str(args.embedding_model_dir),'embedding_model_sha256':digest,
        'embedding_python':str(args.embedding_python),'retrieval':{'top_k':args.top_k,
            'threshold':args.cosine_threshold,'margin':args.cosine_margin,'calibrated':False},
        'max_parts':args.max_parts,'embedding_probe_and_fingerprint_seconds':setup_seconds,
        'compile_navigation':getattr(args, 'compile_navigation', False),
        'manual_repair':False,'failed_case_replay':False,
        'simulation_executed':False,'geometry_verified':False,'training_eligible':False}
    write_json(root/'batch_summary.json',summary)
    try:
        if args.preflight_only:
            summary.update(status='input_preflight_passed',
                embedding_model_verified=False, note='Structural preflight only; live run probes local model before LLM calls.')
        else:
            script = root/'frozen_engine/outputs/habitat_demo_v1/run_skill_cosine_ab.py'
            for repeat in range(1,args.repeats+1):
                for index,case in enumerate(cases):
                    arms = args.arms[(index+repeat-1)%len(args.arms):]+args.arms[:(index+repeat-1)%len(args.arms)]
                    for arm in arms:
                        if not baseline.frozen_unchanged(root/'frozen_engine',manifest) or baseline.sha(root/'cases'/f'{case["name"]}.json') != hashes[case['name']]:
                            raise ValueError('Frozen engine or case changed')
                        folder = root/'trials'/f'{case["name"]}_{repeat}_{arm}'
                        folder.mkdir(parents=True,exist_ok=False)
                        row = {'case':case['name'],'repeat':repeat,'arm':arm,'kind':case['kind'],
                               'expected_round_count':len(case.get('rounds', [])), 'output_dir':str(folder)}
                        summary['trials'].append(row)
                        write_json(root/'batch_summary.json',summary)
                        command = [sys.executable,str(script),'--worker-case',str(root/'cases'/f'{case["name"]}.json'),
                            '--arm',arm,'--output-dir',str(folder),'--model',args.model,'--base-url',args.base_url,
                            '--max-parts',str(args.max_parts),'--top-k',str(args.top_k),
                            '--cosine-threshold',str(args.cosine_threshold),'--cosine-margin',str(args.cosine_margin)]
                        if getattr(args, 'compile_navigation', False):
                            command += ['--compile-navigation']
                        if digest is not None:
                            command += ['--embedding-model-dir',str(args.embedding_model_dir),
                                '--embedding-model-sha',digest,'--embedding-python',str(args.embedding_python)]
                        print(f'START case={case["name"]} repeat={repeat} arm={arm}',flush=True)
                        started=time.monotonic()
                        try:
                            baseline.execute_worker(command,folder/'console.log',
                                timeout=args.seconds_per_round*max(1,len(case.get('rounds',[])))+90)
                        except Exception as exc:
                            row['launcher_error']=str(exc)
                        row['duration_seconds']=time.monotonic()-started
                        path=folder/'case_summary.json'
                        result=read_json(path) if path.exists() else {'status':'failed','error':'No case summary'}
                        row.update(result)
                        audit=baseline.build_audit_report(folder,folder/'llm_audit_aggregate')
                        row['audit']={k:audit[k] for k in baseline.AUDIT_KEYS}
                        row['evaluation_outcome']=baseline.classify(case['kind'],result,audit)
                        if row.get('launcher_error'):
                            row['evaluation_outcome']='final_failure'
                        row['token_usage']=usage(folder)
                        row['retrieval_metrics']=retrieval_metrics(folder)
                        if (folder/'embedding_metrics.json').exists():
                            row['embedding_metrics']=read_json(folder/'embedding_metrics.json')
                        write_json(root/'batch_summary.json',summary)
                        report(root,summary)
                        print(f'END case={case["name"]} arm={arm} outcome={row["evaluation_outcome"]}',flush=True)
            summary['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:
        summary.update(status='interrupted_or_invalid',error=str(exc))
        logging.exception('Comparison stopped; no repair or replay')
    finally:
        summary['frozen_engine_unchanged']=baseline.frozen_unchanged(root/'frozen_engine',manifest)
        summary['case_inputs_unchanged']=all(baseline.sha(root/'cases'/f'{name}.json')==h for name,h in hashes.items())
        summary['historical_sources_unchanged']=all(baseline.sha(args.source_batch/'cases'/f'{name}.json')==h for name,h in hashes.items())
        if digest is not None:
            summary['embedding_model_unchanged']=model_fingerprint(args.embedding_model_dir)==digest
        if not summary['frozen_engine_unchanged'] or not summary['case_inputs_unchanged'] or not summary['historical_sources_unchanged'] or summary.get('embedding_model_unchanged') is False:
            summary['status']='invalid_inputs_or_engine'
        summary['batch_wall_seconds_including_preflight_and_freeze']=time.monotonic()-batch_started
        write_json(root/'batch_summary.json',summary)
        report(root,summary)
    print(f'status={summary["status"]} summary={root/"batch_summary.json"}',flush=True)
    return 0 if summary['status']=='input_preflight_passed' or (summary['status']=='completed' and
        all(r['evaluation_outcome'] not in ('final_failure','evidence_incomplete') for r in summary['trials'])) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch',type=Path,default=SOURCE)
    p.add_argument('--output-dir',type=Path,default=SCENE/'skill_cosine_ab'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--case',action='append')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS[:3]))
    p.add_argument('--max-parts',type=int,default=4)
    p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--cosine-threshold',type=float,default=.35)
    p.add_argument('--cosine-margin',type=float,default=.05)
    p.add_argument('--seconds-per-round',type=int,default=1800)
    p.add_argument('--embedding-python',type=Path,default=Path(sys.executable))
    p.add_argument('--embedding-model-dir',type=Path)
    p.add_argument('--model',default='qwen3.8-flash')
    p.add_argument('--base-url',default=baseline.DEFAULT_URL)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--compile-navigation',action='store_true',help='Opt-in exact-ID navigation assembly in all selected arms')
    p.add_argument('--worker-case',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    p.add_argument('--embedding-model-sha',help=argparse.SUPPRESS)
    args=p.parse_args()
    if (not 1<=args.repeats<=5 or not 2<=args.max_parts<=4 or not 1<=args.top_k<=5 or
            not 0<=args.cosine_threshold<=1 or not 0<=args.cosine_margin<=1 or
            not 60<=args.seconds_per_round<=7200 or len(args.arms)!=len(set(args.arms))):
        p.error('Invalid experiment bounds')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY','').strip():
        p.error('DASHSCOPE_API_KEY required')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    if args.worker_case:
        if args.arm is None:
            p.error('--arm required for worker')
        return worker(args)
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
