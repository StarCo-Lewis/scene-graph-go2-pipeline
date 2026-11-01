"""Read-only evidence audit of the completed 12+18 task pilots; writes separate reports."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/symbolic30_review_20261031_v1'
BASES = [ROOT/'outputs/symbolic_pilot_12_20261031_ready', ROOT/'outputs/symbolic_extension18_20261031_ready_v2']


def local(p):
    p = Path(p).resolve()
    return Path('\\\\?\\'+str(p)) if os.name == 'nt' else p


def read(p):
    return json.loads(local(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(local(p).read_bytes()).hexdigest()


def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fee(u, cache=True):
    return (u['prompt_tokens']*.8 + u['completion_tokens']*2.7 - (u['cached_tokens']*.7 if cache else 0))/1e6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, evidence, batches = [], [], []
    totals = Counter(); stages = defaultdict(Counter); scene_usage = defaultdict(Counter)
    frozen = [read(b/('PILOT_MANIFEST.json' if i == 0 else 'EXTENSION_MANIFEST.json')) for i,b in enumerate(BASES)]
    assert frozen[0]['source_revision_sha256'] == frozen[1]['source_revision_sha256']
    for base in BASES:
        summary = read(base/'results/pilot_summary.json')
        tasks = {t['record_id']:t for t in read(base/'tasks.json')}
        refs = {t['record_id']:t for t in read(base/'references_not_model_input.json')}
        assert summary['status'] == 'completed_unreviewed'
        assert len(summary['rows']) == len(tasks) == summary['finished_tasks']
        batch_usage = Counter()
        batches.append({'root':str(base), 'tasks':len(tasks), 'duration_seconds_monotonic':summary['duration_seconds'],
                       'started_at_utc':summary.get('started_at_utc'), 'updated_at_utc':summary.get('updated_at_utc')})
        for run in summary['rows']:
            rid = run['record_id']; task = tasks[rid]; ref = refs[rid]; folder = base/'results'/rid
            d = read(folder/'task_decomposition.json'); binding = read(folder/'object_alignment.json')
            plan = read(folder/'engineering_plan.json') if (folder/'engineering_plan.json').exists() else None
            assert d['instruction'] == task['instruction'] == run['instruction']
            assert run['input_files_unchanged'] is True
            for name,h in task['prepared_scene_sha256'].items():
                assert sha(folder/'input_memory/scene'/name) == h
            raw = {}; requests = []; usage = Counter()
            for f in sorted((folder/'llm_response_audit').glob('*.json')):
                if f.name.endswith('_request.json') or '_http_' in f.name:
                    evidence.append({'record_id':rid, 'path':str(f.relative_to(ROOT)), 'sha256':sha(f)})
                if f.name.endswith('_request.json'):
                    requests.append(read(f))
                elif '_http_' in f.name:
                    response = read(f)
                    assert response['http_status'] == 200
                    body = json.loads(response['response_body']); assert body['model'] == 'qwen3.8-flash'
                    raw[response['stage']] = json.loads(body['choices'][0]['message']['content'])
                    u = Counter({k:body['usage'][k] for k in ['prompt_tokens','completion_tokens','total_tokens']})
                    u['cached_tokens'] = body['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
                    u['requests'] = 1
                    usage.update(u); stages[response['stage']].update(u)
            assert len(requests) == len(raw) == usage['requests']
            assert run['response_evidence']['transport_error_records'] == 0 and run['response_evidence']['usage_complete']
            assert all(usage[k] == run['response_evidence']['reported_token_usage'][k] for k in ['prompt_tokens','completion_tokens','total_tokens'])
            assert all('references_not_model_input' not in json.dumps(r['model_input']) for r in requests)
            totals.update(usage); batch_usage.update(usage); scene_usage[task['scene_id']].update(usage)
            mapped = {k:v['element_ids'] for k,v in (binding.get('context_bindings') or {}).items()}
            goals = d['goal_state']; terminal = [i for g in goals for i in mapped.get(g['object_role'],[])]
            raw_final = [r for r in raw['decomposition']['constraints'] if r[1] == 2]
            assert len(raw_final) == len(goals)
            terminal_bad = len(goals) != 1  # Every visit task has one final destination, including unresolved roles.
            issues = []; warning = []; checks = {}
            if terminal_bad:
                issues.append({'code':'intermediate_visit_as_terminal_goal', 'stage':'decomposition',
                    'cause':'model_output', 'observed':goals,
                    'expected':ref.get('terminal_target_id') if ref['ordered_target_ids'] else 'Only the final unresolved cabinet role',
                    'explanation':'将中间到访与最终状态混淆。原始模型 constraints 已含多个 final 行，解析器不是起因。',
                    'likely_mechanism':'可能把各子任务的到达条件合并为全局终态；不能据输出确定内部原因。',
                    'engineering_effect':'采集入口保留未经语义审计的提案，未阻止错误 goal_state 进入工程文件。' if plan else '后续绑定合理澄清，未生成工程计划；中间分解错误仍存在。'})
            preservation_gap = bool(ref.get('preserve_objects')) and d.get('process_constraints') == []
            if preservation_gap:
                assert all('without moving any objects' in s['description'] for s in d['subtasks'])
                issues.append({'code':'protection_text_only', 'stage':'decomposition_and_export', 'cause':'model_output_and_contract_scope',
                    'explanation':'不移动物体的要求保留在每步文字，但 process_constraints 为空，工程计划缺少结构化保护。当前纯导航动作没有违反保护的操作。',
                    'expected':'至少对已有目标角色输出 do_not_manipulate；对 all objects 的全场景量化范围另行明确契约，不能宣称运行时保护已建立。',
                    'severity':'constraint_enforcement_gap_no_observed_bad_action'})
            if plan:
                steps = [s for sub in plan['plan']['subtask_plans'] for s in sub['steps']]
                actual = [s['arguments']['target']['element_id'] for s in steps]
                subs = d['subtasks']; seq = plan['plan']['action_sequence']
                checks = {'order':actual == ref['ordered_target_ids'],
                    'navigation_only':all(s['skill_id']=='navigate_to_object' for s in steps),
                    'dependencies':all(s['dependencies']==([] if i==0 else [subs[i-1]['subtask_id']]) for i,s in enumerate(subs)),
                    'action_dependencies':all(s['depends_on']==([] if i==0 else [seq[i-1]['step_id']]) for i,s in enumerate(seq)),
                    'terminal_goal':terminal==[ref['terminal_target_id']],
                    'predicates':all(g['subject_role']=='robot' and g['relation']=='near' for g in goals),
                    'unknown_initial':d['initial_scene_constraints']==[],
                    'frame':plan['coordinate_frame']==read(folder/'input_memory/scene/scene_metadata.json')['coordinate_frame'] and plan['coordinate_conversion_applied'] is False,
                    'not_executed':plan['execution_ready'] is False and plan['robot_executed'] is False and plan['robot_goal_pose'] is None,
                    'feasibility_unknown':all(s['feasibility']=='unknown' for s in steps)}
                assert all(v for k,v in checks.items() if k!='terminal_goal'), (rid,checks)
                assert plan['plan']['deferred']['goal_state'] == goals
                outcome = 'terminal_goal_error' if terminal_bad else 'protection_gap' if preservation_gap else 'pass'
                scores = {'reliability':[1,1,1,1], 'honesty':[1,2,2,1] if issues else [2,2,2,2]}
                score = sum(scores['reliability'])+sum(scores['honesty'])
                explanation = '访问对象、顺序、依赖及终态符合参考；可达性仍未知。' if not issues else issues[0]['explanation']
                if terminal_bad:
                    explanation += f' 实际访问 {actual} 正确，最终只应 near({ref["terminal_target_id"]})；实际终态目标 {terminal}。'
            else:
                actual = []
                expected_error = 'NO_OBJECT_CAN_BE_USED_IN_THIS_TASK' if ref['task_type'] in ('missing_id','missing_category') else 'OBJECT_BINDING_AMBIGUOUS'
                checks = {'expected_stop':binding['error']['code']==expected_error, 'no_engineering_json':True,
                    'no_skill_request':'skill_matching_matrix' not in raw,
                    'no_accepted_full_binding':not mapped}
                assert all(checks.values()), (rid, checks)
                decisions = raw['object_binding']['decisions']; unresolved = {r[0] for r in decisions if r[1]!=1}
                assert unresolved and not any(r[0] in unresolved for r in raw['object_binding']['matches'])
                outcome = 'clarification_with_decomposition_issue' if terminal_bad else 'expected_clarification'
                scores = {'reliability':[1,0,None,None], 'honesty':[1 if terminal_bad else 2,2,None,None]}; score=None
                reason_text = ' '.join(r[1] for r in raw['object_binding']['reasons'])
                explanation = '按预期停止，未替未知目标作决定。原始绑定理由：'+reason_text
                if terminal_bad:
                    explanation += ' 但分解把先访问的冰箱也列为终态；合理澄清不能掩盖这个中间错误。'
            if ref['task_type'] in ('ambiguous_intermediate','ambiguous_final'):
                warning.append('a chair/a kitchen cabinet 在自然语言中也可表示任意一个；本测试采用预先冻结的保守澄清政策。未来应加入 any chair 对照，不能将该规则当成普遍语言真值。')
            row = {'record_id':rid, 'scene_id':task['scene_id'], 'instruction':task['instruction'], 'task_type':ref['task_type'],
                'review_result':outcome, 'run_status':run['status'], 'expected_order':ref['ordered_target_ids'], 'actual_order':actual,
                'actual_terminal_targets':terminal, 'scores':scores, 'score_out_of_12':score, 'checks':checks, 'issues':issues,
                'explanation':explanation, 'review_caveats':warning, 'raw_responses':raw,
                'source_folder':str(folder), 'usage':dict(usage), 'estimated_list_cost_cny':fee(usage),
                'duration_seconds':run['duration_seconds'], 'batch':base.name,
                'terminal_rule_present_in_actual_input':any('only the final destination' in json.dumps(r['model_input']) for r in requests)}
            if terminal_bad:
                assert row['terminal_rule_present_in_actual_input']
            rows.append(row)
        batches[-1]['usage'] = dict(batch_usage)
        batches[-1]['list_cost_with_cache_cny'] = fee(batch_usage)
        batches[-1]['list_cost_no_cache_cny'] = fee(batch_usage,False)
    assert len(rows) == len({r['record_id'] for r in rows}) == 30
    counts = Counter(r['review_result'] for r in rows)
    byscene = {s:dict(Counter(r['review_result'] for r in rows if r['scene_id']==s)) for s in scene_usage}
    full = [r for r in rows if r['score_out_of_12'] is not None]
    budget_by_scene = {}
    for s in scene_usage:
        subset = [r for r in full if r['scene_id']==s]
        n=len(subset); u=sum((Counter(r['usage']) for r in subset),Counter())
        budget_by_scene[s] = {'completed_three_stage_tasks':n, 'mean_tokens':u['total_tokens']/n,
            'mean_list_cost_with_cache_cny':fee(u)/n, 'mean_list_cost_no_cache_cny':fee(u,False)/n,
            'mean_seconds':sum(r['duration_seconds'] for r in subset)/n}
    rates=[v['mean_list_cost_with_cache_cny'] for v in budget_by_scene.values()]
    highrates=[v['mean_list_cost_no_cache_cny'] for v in budget_by_scene.values()]
    scenarios = [{'task_attempts':n, 'base_cost_cny':[min(rates)*n,max(highrates)*n],
        'with_debug_60_and_retry_10_to_30_percent_cny':[(n*1.1+60)*min(rates),(n*1.3+60)*max(highrates)],
        'condition':'Only if future graphs/prompts/task lengths match these two small scenes; no guarantee of unique/accepted data'} for n in (200,9400,10000,20000)]
    stats = {'reviewed_at_utc':datetime.now(timezone.utc).isoformat(), 'reviewer':'Offline AI review + deterministic evidence checks; not independent human review',
        'source_versions_identical':True, 'new_api_calls':0, 'raw_modified':False, 'robot_executed':False,
        'counts':dict(counts), 'by_scene':byscene, 'usage':dict(totals), 'stage_usage':{k:dict(v) for k,v in stages.items()},
        'scene_usage':{k:dict(v) for k,v in scene_usage.items()}, 'batches':batches,
        'duration_seconds_monotonic_total':sum(b['duration_seconds_monotonic'] for b in batches),
        'billing':{'actual_new18_cost_cny':None,'actual_30_cost_cny':None,
            'earlier_1_73_cny_batch_attribution':'unverified; not used as measured per-task price',
            'pricing_source':'https://help.aliyun.com/zh/model-studio/qwen3-8-flash','verified_date':'2026-10-31',
            'cny_per_million':{'input':.8,'cached_input':.1,'output':2.7},
            'list_cost_with_cache_cny':fee(totals), 'list_cost_no_cache_cny':fee(totals,False)},
        'full_pipeline_cost_by_scene':budget_by_scene, 'conditional_scenarios':scenarios,
        'observed_cost_per_accepted_record':{
            'strict_complete_plans':16, 'strict_complete_plans_plus_clean_expected_stops':24,
            'all_30_attempt_cost_divided_by_16_cny':[fee(totals)/16,fee(totals,False)/16],
            'all_30_attempt_cost_divided_by_24_cny':[fee(totals)/24,fee(totals,False)/24],
            'note':'Descriptive accounting for this purposive sample only; not a general acceptance rate or geometric retry model'},
        'score_policy':'R: four observed artifact stages 0/1; H: compliance four stages 0/1/2. Clarification stops have unobserved downstream=null and no /12 total. Correct stop and earlier artifact defects are separate. Error propagation is not counted as multiple independent failures.',
        'scope':'Purposive 30 tasks, 2 small scenes. Text graph consumed; no image inference, robot navigation, room-topology validation or 100-scene generalization verified.',
        'rows':rows}
    save('review.json',stats); save('evidence_sha256.json',evidence)
    save('budget.json',{k:stats[k] for k in ['billing','usage','stage_usage','batches','full_pipeline_cost_by_scene','conditional_scenarios','observed_cost_per_accepted_record']})
    write_html(stats)
    print(json.dumps({k:stats[k] for k in ['counts','by_scene','usage','batches','billing','full_pipeline_cost_by_scene']},ensure_ascii=False,indent=2))


def write_html(s):
    e=html.escape
    labels={'pass':'符合参考','terminal_goal_error':'终态错误','protection_gap':'保护约束缺口','expected_clarification':'合理澄清／拒绝','clarification_with_decomposition_issue':'合理澄清，但分解有错'}
    table=''.join('<tr><td>'+e(r['scene_id'].split('_')[0]+'/'+r['record_id'][-8:])+'</td><td>'+e(r['instruction'])+'</td><td>'+labels[r['review_result']]+'</td><td>'+str(r['score_out_of_12'] if r['score_out_of_12'] is not None else 'N/A')+'</td><td>'+e(r['explanation'])+'</td></tr>' for r in s['rows'])
    details=''
    for r in s['rows']:
        folder=Path(r['source_folder'])
        links=' · '.join('<a href="'+e(os.path.relpath(folder/f,OUT).replace('\\','/'))+'">'+name+'</a>' for f,name in [('task_decomposition.json','分解'),('object_alignment.json','绑定'),('engineering_plan.json','工程JSON')] if (folder/f).exists())
        details+='<details><summary>'+e(r['record_id'])+'</summary>'+links+'<pre>'+e(json.dumps({k:r[k] for k in ['scores','checks','issues','review_caveats','raw_responses']},ensure_ascii=False,indent=2))+'</pre></details>'
    costs=''.join(f'<tr><td>{b["tasks"]}条</td><td>{b["usage"]["requests"]}</td><td>{b["usage"]["total_tokens"]:,}</td><td>{b["duration_seconds_monotonic"]:.1f}秒</td><td>¥{b["list_cost_with_cache_cny"]:.4f}–{b["list_cost_no_cache_cny"]:.4f}</td></tr>' for b in s['batches'])
    estimates=''.join(f'<tr><td>{b["task_attempts"]:,}</td><td>¥{b["base_cost_cny"][0]:.1f}–{b["base_cost_cny"][1]:.1f}</td><td>¥{b["with_debug_60_and_retry_10_to_30_percent_cny"][0]:.1f}–{b["with_debug_60_and_retry_10_to_30_percent_cny"][1]:.1f}</td></tr>' for b in s['conditional_scenarios'])
    feedback='''两份新增输入可以用于小批量的符号访问、访问顺序与歧义澄清任务。我们已使用同一规划代码和 Qwen 3.8 Flash 完成30个唯一任务：TUM 12条、ScanNet scene0488 18条。21条明确目标任务均输出工程JSON，访问对象、顺序及依赖均正确；其中4条的终态额外保留中间访问目标，1条将“不移动物体”保留在文本但未形成结构化保护，16条在本次范围内符合参考。其余9条按预期澄清或拒绝；其中1条的中间分解仍有终态错误。无传输错误。结果是离线AI及脚本审阅，不是真机成功率或独立人工验收。
当前适用边界仍是图中给定ID/类别的符号任务。未标注的房间、邻接、说话人位置保持未知；相机/数据集局部坐标未转成机器人世界坐标，也没有验证可达性。两图的edges为空，不能声称关系推理已验证。
请继续按相同格式提供其余场景的实际图、metadata、模态索引、来源/权限与哈希，并明确坐标、未知信息和目标适用范围。现有两个小样不能证明全部100个场景已就绪。我们先处理终态审计与保护契约问题，再做定向回归；收到实际图清单后按输入规模与任务长度分层核预算。30条的真实token已统计，账单归集仍待核对，不把估算当报销金额。'''
    (OUT/'feedback_zh.txt').write_text(feedback+'\n',encoding='utf-8')
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>30条符号任务联合审阅</title><style>body{font:16px/1.65 system-ui;max-width:1300px;margin:35px auto;padding:20px;color:#183047}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd6df;padding:9px;vertical-align:top;text-align:left}th{background:#eaf2f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px}details{margin:16px 0}.note{background:#fff3dc;padding:16px}</style>
<h1>30条符号任务：联合审阅、预算与输入适用性反馈</h1><p>同一源码/配置指纹，Qwen 3.8 Flash；原12条+新增18条，每个任务只计一次。原始输入输出未改写；本次审阅无模型调用、无WSL或机器人操作。AI审阅＋确定性检查，不是独立人工审核。</p>
<h2>结论</h2><p>21份工程JSON：16份在本次范围内符合参考，4份终态错误，1份保护约束结构化不足。9条合理澄清/拒绝，其中1条同时有中间分解终态错误。终态错误共5条，不应因合理澄清而漏计中间错误。无超时/HTTP失败。</p>
<p>21/21明确目标任务的对象、动作顺序与依赖正确。严格完整产物符合16/21；9/9信息不足任务按预期停止，其中8条无已发现中间问题。这些是定向样本内比例，不是100场景总体准确率。</p>
<h2>问题在哪里</h2><ol><li>终态混淆：TUM 91000003、91000008、91000009，ScanNet 91000009、91000015。原始decomposition已把中间到访编码成final constraints，实际提示已有“只有最后目的地是终态”的规则，主要归因模型遵循失败。工程采集入口未作语义评判而透传；应新增审计门槛并保留raw，不可静默改正后算模型正确。</li><li>保护契约：ScanNet 91000014的五目标顺序正确，每步文字均保留不移动物体，但process_constraints=[]。属于模型结构化遗漏与全场景量化契约需明确的问题；当前没有抓放动作，不能说已发生物理违规。应区分动作符合与运行时保护保证。</li><li>合理澄清：最后的91000018是“隔壁房间的柜子”，图无房间归属与邻接；原绑定理由明确缺少定位该柜子的证据。因此collection_stopped_at_stage_ungraded是合理停止，不是系统崩溃。91000017同理缺少说话人位姿。</li></ol>
<p>注：relations里的expected=-1用于on/in/near是合法协议占位，不能误判成held/lifted状态取值错误。工程文件中的0.6米仅为既有符号技能默认值，不是验证过的物理停靠点。</p>
<h2>30条逐任务评分</h2><p>可靠性R：分解/绑定/技能/工程产物各1分；诚实性H：四阶段各0–2分。有产物不等于语义通过。终态/保护缺口样本R=4、H=1+2+2+1，共10/12；同一问题向工程输出传播不算两起根因。澄清任务R=[1,0,null,null]，H通常[2,2,null,null]；ScanNet15的H=[1,2,null,null]，不强行给12分总分。</p><table><tr><th>场景/ID</th><th>原指令</th><th>结论</th><th>评分</th><th>原因</th></tr>'''+table+'''</table>
<h2>任务设计自身的限制</h2><p>“a chair / a kitchen cabinet”也可表示任意一个；本批独立参考采取保守澄清策略，不能据此证明任意选择必然错误。后续建议添加“any chair is fine”与“the chair”的区分测试。小图、明确ID占比较高、最长仅5次访问；这不是复杂跨房间长程或物体操作测试。无新增视觉识别验证。</p>
<h2>费用与耗时</h2><table><tr><th>批次</th><th>请求</th><th>tokens</th><th>计时器耗时</th><th>目录价估算</th></tr>'''+costs+'''</table><p>单阶段原始usage已重新求和并与各task summary核对。累计 '''+f'{s["usage"]["total_tokens"]:,} tokens，{s["usage"]["requests"]} 次请求；计时器耗时 {s["duration_seconds_monotonic_total"]:.1f} 秒。'+'''新增批次UTC时间戳的起止差与monotonic耗时有约20秒差异，原因未核实；预算使用记录的monotonic，不能把两批之间等待计为执行时间。</p>
<p>按<a href="https://help.aliyun.com/zh/model-studio/qwen3-8-flash">官方北京价格页（2026-10-31核查）</a>：输入0.8、缓存输入0.1、输出2.7元/百万tokens。累计目录价 '''+f'¥{s["billing"]["list_cost_with_cache_cny"]:.4f}–{s["billing"]["list_cost_no_cache_cny"]:.4f}。'+'''这不是账单实付；此前¥1.73的批次归属未核实，不能乘成确定单价。新增18条实付及30条实付均待用户对账。</p>
<h2>条件预算，不是对未交付场景的固定报价</h2><p>以下按两图中完整三阶段任务的每条均值范围外推，不使用提前澄清任务拉低完整规划单价；含调试列假设额外60条调试及10%–30%补跑。只在图规模、提示长度和任务难度相近时成立；不含建图/VLM、人工审核或真机。一次尝试不等于一条合格数据，不能把失败重试算新增样本。</p><table><tr><th>任务尝试数</th><th>基础目录价</th><th>含调试与补跑</th></tr>'''+estimates+'''</table><p>实际预算可按计费明细拟合实际系数：归属于本批的实付÷同批目录价估算。未对账前不应用此前截图构造确定倍率。更大的场景需先统计图输入token并分层；并发并不降低token费用，理想耗时约总串行耗时/并发数，但限流与资源竞争尚未实测。</p>
<h2>可发给提供方的反馈</h2><pre>'''+e(feedback)+'''</pre><p><a href="feedback_zh.txt">反馈文本</a> · <a href="review.json">完整评分JSON</a> · <a href="budget.json">预算与用量</a> · <a href="evidence_sha256.json">原请求/响应哈希索引</a></p><h2>逐条原响应与检查证据</h2>'''+details+'</html>'
    doc = doc.replace('<h2>条件预算，不是对未交付场景的固定报价</h2>',
        '<p>本批全部尝试成本÷16份严格符合参考的完整计划：约¥'+f'{s["billing"]["list_cost_with_cache_cny"]/16:.4f}–{s["billing"]["list_cost_no_cache_cny"]/16:.4f}'+
        '/份；若把8条无中间问题的合理澄清也视作合格推理记录，则分母24，约¥'+f'{s["billing"]["list_cost_with_cache_cny"]/24:.4f}–{s["billing"]["list_cost_no_cache_cny"]/24:.4f}'+
        '/条。仅为本批会计口径，不用它假设未来独立重试成功率。</p><h2>条件预算，不是对未交付场景的固定报价</h2>')
    (OUT/'report.html').write_text(doc,encoding='utf-8')


if __name__ == '__main__':
    main()
