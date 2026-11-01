"""Explain every scored deviation and separately every ungenerated task.

Read-only with respect to task/model artifacts. No new model or robot calls.
Direct output mechanisms are distinguished from unproven internal model causes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import shutil

from .batch import read, read_jsonl, write, jsonl, sha, require_code_unchanged
from .qc_report_v1 import STAGES
from .qc_rules_v1 import SCENES

CAUSES = {
 'missing_final_release': ('释放终态漏编码', '将“放到目标上并释放”压缩成 on 关系，漏掉与空间关系独立的 held=false。on 不能代替释放状态。', '提示词已明确要求 placement 同时编码 on 与 held=0；现有格式门槛并不逐项证明任务的全部终态已覆盖。', '同时表达 on(搬运物,目标面) 与 held(搬运物)=false；动作应完成放置和释放。'),
 'missing_final_destination': ('持有目标缺目的地', '保留了“继续持有”，但没有在最终条件中保留“到指定目的地”。held=true 在原地也可能成立，不能单独表达送达。', '格式合法、至少有一个 required goal 即可通过这一层；该采集路径没有把全部自然语言要求与终态逐项对比。', 'held(payload)=true 加最终靠近目的地；持有物 near 目的地也可作为等效表达。'),
 'intermediate_as_final': ('中间访问与最终状态混淆', '将按时间先后访问多个地点，错误编码为最终同时 near 所有地点。问题是语义/时间范围错位，不必假定这些地点在几何上绝对不可能同时接近。', '现有校验检查引用与部分持有状态冲突，并未证明 near 目标是最后一站而不是全过程访问清单；deferred 原样保留这些目标。', '各次访问和先后顺序放在子任务/依赖中；最终 near 只要求最后一站，除非原指令另有最终要求。'),
 'unsupported_initial_assertion': ('新增无依据或冲突的初始条件', '把目标、动作需要的条件或未观测状态写进 initial_scene_constraints。初始断言只能检查输入事实，不能把期望状态变成事实。', '这些已生成文件中的错误条件可被标为 optional；关系检查能够记下 unknown/conflict，但只阻断 required 的错误条件。required=false 表示非必需，不是否定该关系。', '删除无依据的初始断言；若确有来源对象/支撑关系，应按原图和角色引用准确表达，未知保持未知。'),
 'protection_text_omission': ('结构保护与任务文本不一致', 'process_constraints 中仍有保护，但受影响的子任务描述没有按固定提示同步保留“不要动该对象”。这是跨表达形式的信息遗漏。', '结构保护存在，动作也未触碰保护物，因此本轮未把它解释为实际保护失效；程序未强制每段自然描述复述保护。', '保留已有过程限制，并把保护要求写入对应的完整任务描述；不需要增加操作保护物的动作。'),
 'atomic_overfragmentation': ('完整意图被拆成技能阶段', '将普通搬运按抓取、携带、放置拆成子任务，偏离本次“完整意图＋约束”的组织规则；并不意味着这些动作本身错。', '每个子任务都有合法引用和动作，结构门槛未判定粒度；未改变最终动作时按轻微组织偏差处理。', '同一普通搬运意图包含对象、来源、目的地和释放/持有要求，把原子技能选择留给下一阶段。'),
 'explicit_instance_ids_missing_from_queries': ('角色查询漏掉显式实例ID', '原指令有明确实例ID，但角色查询只保留类别/描述，没有按协议保留ID，削弱后续确定性绑定依据。', '原始指令仍提供ID且绑定结果正确，遗漏未在本样本造成错误对象；结构校验没有检验查询文本必须包含原ID。', '在每个相关角色query中保留相应实例ID，仍检查类别与属性是否匹配。'),
 'pickup_prerequisites': ('抓取前置动作缺失', '在没有可继承的已接近/准备状态证据时，直接准备或抓取。技能语义要求先接近，再针对抓取对象准备机械臂。', '采集路径检查动作类型、参数、绑定和形状；动态前置以 unknown 保留，本轮没有运行完整的动作链可行性审查。这不是证明物理上一定失败。', '补足缺少的接近与机械臂准备；允许导航到有图依据的来源支撑处，不强制导航到物体中心。'),
 'carry_without_lift': ('携带前缺抬升', '序列抓住对象后就导航携带，省略了技能库所定义的“抬离原支撑再携带”。抓住与抬起是不同状态变化。', 'grasp 和 navigate 都是合法技能行，但行合法不证明它们之间的前置状态链完整；未做本轮顺序状态校验。', '抓取后先 lift_held_object，再携带到目标；不得为已明确抬起的对象无意义重复抬升。'),
 'missing_grasp_and_lift': ('只接近却没有取得物体', '要求把物体带到终点并持有，但动作只有导航/准备，没有 grasp 或 lift；靠近/准备不会产生持有状态。', '动作行和参数合法且覆盖了子任务，采集阶段没有对原任务 held=true 做最终达成检查。', '加入合法的接近、准备、抓取、抬升与携带动作；最终不放下并保持持有。'),
 'passive_holding_as_action': ('保持持有误拆成独立动作任务', '把“到达后继续拿着”这个终态要求拆成新的行动子任务，下一阶段为满足每个子任务必须有动作而又生成导航。', '分解格式允许额外任务；技能提示要求每个子任务至少一行动作，错误分解由此向下游传播。', '把保持持有合并进抓取/携带意图和 held=true 终态，不单独建一个无新行为的任务。'),
 'unsupported_source_room': ('把目标房间写成当前来源房间', '角色查询把对象当前位置写成目标所在房间，超出了该对象已有位置证据，混淆了当前位置和目标位置。', '显式ID仍使绑定选中正确对象，错误来源描述没有改变本次动作参数，但分解文本已不忠于输入。', '使用有依据的实际来源或不声明房间；不可因为目的地在某房间就把对象初态放进去。'),
 'global_prohibition_narrowed_in_text': ('全局禁止被缩成局部禁止', '原指令禁止操作任何对象，分解文本却只说不要操作目标对象，约束范围缩小。', '原始指令仍在、动作只导航，未出现实际额外操纵，故本次记录为轻微文本偏离。', '保留“任何对象均不抓取/移动”的全局范围。'),
 'no_manipulation_constraint_omitted': ('禁止操作要求从分解中遗漏', '行动任务文本和过程表达未保留原指令的禁止抓取/移动要求；不是为了简洁就能删除的限定。', '实际技能本次仍只导航，结构校验未检查自然语言禁令是否完整传递；遗漏与真实发生操纵是两回事。', '在导航意图中明确保留禁止操纵任何对象；无需创建虚构的保护动作。'),
 'propagated_redundant_navigation': ('错误保持任务诱发冗余导航', '上游把保持持有拆为新任务，技能阶段为了覆盖它又导航至同一目的地。最终目标可能仍达到，但多了一次未请求的控制调用。', '这是前序分解诱发的传播，不另计技能模型新增根因；自动校验未删除跨子任务的重复导航。', '合并被动保持任务后去掉多余调用；保留原有最终持有和终点要求。'),
}
ALIASES = {
 'missing_release_goal':'missing_final_release', 'missing_final_release_predicate':'missing_final_release', 'missing_required_final_held':'missing_final_release',
 'missing_hold_destination_goal':'missing_final_destination', 'missing_final_destination_predicate':'missing_final_destination', 'missing_required_final_near':'missing_final_destination',
 'intermediate_visits_as_final_goals':'intermediate_as_final', 'intermediate_visits_as_conjunctive_final_goals':'intermediate_as_final', 'intermediate_visits_encoded_as_final':'intermediate_as_final', 'intermediate_final_goals':'intermediate_as_final',
 'protection_not_repeated_in_task_text':'protection_text_omission', 'protection_not_repeated_in_intent_text':'protection_text_omission',
 'transport_split_into_atomic_phases':'atomic_overfragmentation',
 'missing_pickup_prerequisite_actions':'pickup_prerequisites', 'missing_source_approach_and_preparation':'pickup_prerequisites',
 'missing_pickup_prerequisites':'pickup_prerequisites', 'missing_source_approach':'pickup_prerequisites', 'grasp_without_approach_and_preparation':'pickup_prerequisites',
}


def category(issue):
    code = issue['code']
    if code == 'propagated_redundant_navigation':
        return code
    code = re.sub(r'^(propagated_|inherited_)', '', code)
    return ALIASES.get(code, code)


def optional(p):
    return read(p) if p.is_file() else None


def model_content(p):
    value = optional(p)
    if value and 'response_content' in value:
        try:
            return json.loads(value['response_content'])
        except (ValueError, TypeError):
            return {'raw_content_unparsed': value['response_content']}
    return value


def task_explanation(task, nodes):
    def obj(oid):
        return f'对象{oid}（{nodes.get(oid, {}).get("object_tag", "未命名")}）'
    descriptions = []
    for m in task['goal'].get('ordered_milestones', []):
        p, o = m['predicate'], obj(m['object_id'])
        if p == 'visited_near':
            descriptions.append('靠近' + o)
        elif p == 'delivered_and_released':
            descriptions.append(f'把{o}搬到{obj(m["destination_id"])}上，放下并释放')
        elif p == 'transported_while_held':
            descriptions.append(f'取起{o}并带到{obj(m["destination_id"])}附近，到达后继续持有，不放下')
        else:
            descriptions.append(json.dumps(m, ensure_ascii=False))
    if len(descriptions) > 1:
        text = '按顺序：' + '；然后'.join(descriptions) + '。'
    else:
        text = '；'.join(descriptions) + '。'
    constraints = task.get('constraints', {})
    if constraints.get('preserve_objects'):
        text += '全过程不得操作或移动：' + '、'.join(obj(o) for o in constraints['preserve_objects']) + '。'
    if constraints.get('no_object_manipulation'):
        text += '禁止抓取或移动任何对象。'
    return text


def compact_sequence(run):
    e = optional(run / 'engineering_plan.json')
    if not e or not isinstance(e.get('plan'), dict):  # incomplete envelope: plan is null
        return []
    names = {'navigate_to_object':'接近/导航', 'position_arm_for_object':'机械臂准备', 'grasp_object':'抓取',
             'lift_held_object':'抬升', 'place_object_on_surface':'放置并释放'}
    sequence = []
    for st in e['plan']['subtask_plans']:
        for step in st['steps']:
            args = {k: v.get('element_id') if isinstance(v, dict) else v for k, v in step['arguments'].items()}
            sequence.append({'ordinal': len(sequence)+1, 'subtask_id': st['subtask_id'], 'step_id': step['step_id'],
                             'action': names.get(step['skill_id'], step['skill_id']), 'skill_id': step['skill_id'], 'arguments': args})
    return sequence


def blocked_analysis(row, root):
    rid, scene = row['record_id'], row['scene_id']
    run = root / 'cases' / scene / 'tasks' / rid / 'model_run'
    error = row.get('error') or ''
    raw = model_content(run / 'llm_decomposition_attempt_1.json') or {}
    binding = optional(run / 'llm_binding_proposal.json')
    wire = optional(run / 'llm_binding_matrix_proposal_attempt_1.json')
    evidence = {'planning_summary': row['summary_path'], 'error_message': error}
    status, result, expected = '', '', ''
    if 'TimeoutError' in error:
        kind, stage = 'transport_timeout', 'API传输'
        result = '请求在限定时间内未得到可继续处理的结果，单次采集策略没有自动重试该任务。没有足够输出评价失败阶段的语义质量。'
        status = '基础设施/传输失败；网络、服务端延迟及超时设置的具体责任未由这条日志确定。不能算模型幻觉。'
        expected = '保留超时证据；如之后补采需标为新的尝试，不能把无响应补成模型答案。'
    elif 'TYPED_ID_KIND_MISMATCH' in error:
        kind, stage = 'state_predicate_wrong_subject_type', '任务分解'
        rows = [x for x in raw.get('constraints', []) if x[3] in (4,5) and x[2] != 10001]
        evidence['offending_raw_constraint_rows'] = rows
        result = 'held/lifted 的主体字段填写了200xx物体角色代码，而协议要求10001执行者。它不是单纯选错场景物体ID，而是谓词主体类型不合法。'
        status = '原始模型矩阵违反类型契约，程序在解码时正确拒绝；若同一行expected=-1，还同时存在状态值错误。'
        expected = '持有/抬升状态用 robot(10001) 作主体，物体角色作宾语，expected用0或1；不擅自补不明初态。'
    elif 'State predicates require' in error:
        kind, stage = 'state_predicate_invalid_expected', '任务分解'
        evidence['offending_raw_constraint_rows'] = [x for x in raw.get('constraints', []) if x[3] in (4,5) and x[5] not in (0,1)]
        result = '把空间关系使用的expected=-1填到了held/lifted状态谓词。即使required=false，这个字段也必须是0或1。'
        status = '原始模型矩阵不符合输出契约，程序正确拒绝；这不是把未知状态合法编码成-1。'
        expected = '明确的状态条件使用0/1；不知道的初始状态应省略断言，而不是填-1。'
    elif 'Exactly one decision and reason' in error:
        kind, stage = 'binding_role_coverage_missing', '物体绑定'
        evidence['raw_binding_tables'] = wire
        result = '本次绑定输出的decisions/reasons/matches为空，没有为输入角色逐一给出决策和理由；程序因角色覆盖不完整而拒绝。'
        status = '绑定输出表缺项，不是证明场景里没有对象。'
        expected = '每个输入角色恰好一个decision和reason；无法绑定也应按协议明确missing或ambiguous。'
    elif 'Repeated binding reason' in error:
        kind, stage = 'duplicate_reason_rows_for_same_role', '物体绑定'
        evidence['raw_binding_tables'] = wire
        result = '每个角色先有一行对象匹配理由，又追加一行“No candidate ambiguity...”理由，同一角色出现两次reason记录。不是仅仅不同角色说了相似的话。'
        status = '原始绑定JSON重复定义同一角色的理由，程序的唯一性规则正确拦截；对象ID本身可能仍然正确。'
        expected = '每个角色只保留一行理由，在该行中完整表达匹配依据。'
    elif rid == 'scene200_00000126':
        kind, stage = 'duplicate_roles_for_same_scene_object', '任务分解→绑定拒绝'
        evidence['raw_roles'] = raw.get('roles')
        evidence['binding'] = binding
        result = '同一工作台165被拆成“目的工作台”和“来源工作台”两个角色，同一展台110也被拆成来源/目的两个角色；绑定协议禁止不同角色重复占用同一ID，因此其中两个角色被判missing。'
        status = '对象实际存在，直接原因是分解没有遵守同一对象只定义一个角色并复用，不能归为技能库或场景缺物体。'
        expected = '165和110各定义一个角色，并在不同任务中复用。'
    elif rid == 'd3_619_00000047':
        kind, stage = 'task_target_part_not_supported_by_graph', '任务输入/图粒度→绑定拒绝'
        evidence['binding'] = binding
        evidence['node_98'] = next(n for n in read(root/'cases'/scene/'graph/scene_graph.json') if n['id'] == 98)
        result = '任务要求把靠垫放到98号办公椅的座面，但98号节点caption描述的是椅背与支架，图中没有明确独立座面及其关联。绑定器拒绝把椅背节点直接当成座面。'
        status = '当前任务目标与可用图的部件粒度/证据不匹配；不能据此认定LLM出错，也不能断言真实椅子没有座面。'
        expected = '人工确认98号节点覆盖的实体范围，提供可用座面/目标支撑证据，再决定是否修改任务或图；不能盲目强行绑定。'
    elif 'OBJECT_BINDING_AMBIGUOUS' in error or 'OBJECT_RELATION_UNVERIFIED' in error:
        kind, stage = 'unsupported_required_initial_relation', '任务分解→绑定/关系校验拒绝'
        decoded = optional(run / 'llm_decomposition_decoded_attempt_1.json') or {}
        evidence['initial_scene_constraints'] = decoded.get('initial_scene_constraints')
        evidence['binding'] = binding
        result = '分解把无图证据或与已有支撑冲突的关系写成required初始条件。即使显式ID可找到对象，这个必须成立的初始关系仍不能确认，后续因此停止。'
        if rid == 'scene200_00000120':
            result += '该例绑定理由还声称near关系有 supplied edge 支持，但源图没有相应near边；程序最终拒绝了这个理由。'
        status = '首先是分解初态与输入不一致；ambiguous/missing是下游表现，不能直接解释为对象不存在。'
        expected = '依据实际来源角色与图边设置初态，不把目的地写成当前位置；未知关系不设为必需输入事实。'
    else:
        raise ValueError(f'Unreviewed failure type: {rid}')
    return {'category': kind, 'failure_stage': stage, 'direct_mechanism': result, 'attribution': status,
            'expected_behavior': expected, 'evidence': evidence, 'internal_model_cause': '未由单次实验确定'}


def build(root):
    qc = root / 'reports/quality_check_v1'
    dest = qc / 'cause_analysis'
    source_errors = read_jsonl(qc / 'errors.jsonl')
    missing = read_jsonl(qc / 'not_generated.jsonl')
    require_code_unchanged(root)
    graphs = {s: {n['id']:n for n in read(root/'cases'/s/'graph/scene_graph.json')} for s in SCENES}
    analyses, blocked = [], []
    for row in source_errors:
        scene, rid = row['scene_id'], row['record_id']
        run = root/'cases'/scene/'tasks'/rid/'model_run'
        assert sha(run/'engineering_plan.json') == row['original_engineering_sha256']
        task = read(run.parent/'task.json')
        decoded = optional(run/'llm_decomposition_decoded_attempt_1.json') or {}
        parts = []
        for issue in row['issues']:
            key = category(issue)
            if key not in CAUSES:
                raise ValueError(f'No reviewed explanation for {issue["code"]}: {key}')
            title, mechanism, gate, correct = CAUSES[key]
            if key == 'unsupported_initial_assertion' and any(
                    c.get('subject_role') == 'robot'
                    for c in decoded.get('initial_scene_constraints', [])):
                mechanism = '输入未给出机器人是否持有该对象，却将 held=false 写为初始事实。未知持有状态不能自行改成未持有。'
                gate = '机器人主体的状态条件被分流到运行时验证，不由场景图关系校验判真。本例即使 required=true，也作为 pending_executor_validation 保留；本轮只采集、不执行，因此该无依据初态没有在此处被拒绝。这不是 optional 关系放行。'
                correct = '输入持有状态未知时不新增 held=false 初始断言；在执行前通过真实状态观测确认所需前置条件。'
            propagation = issue.get('origin') == 'propagated'
            parts.append({'stage': issue['stage'], 'severity': issue['severity'], 'category': key,
                          'title': title, 'case_specific_observation': issue['reason'],
                          'direct_mechanism': ('前序问题保留在最终工程JSON中；这不是第四次LLM独立生成的新错误。' if propagation else mechanism),
                          'why_collection_did_not_block': gate, 'expected_behavior': correct,
                          'origin': issue.get('origin'), 'evidence': issue['evidence'],
                          'internal_model_cause': '单次输出不能区分模型能力、提示词表达或上下文干扰；需对照实验验证。'})
        original = model_content(run/'llm_decomposition_attempt_1.json') or {}
        impact = ('最终工程JSON仍含关键问题，不能仅凭生成成功判为合格。' if row['fidelity_scores']['E'] == 0 else
                  '最终项未发现关键问题；前序缺项可能被后续动作补足，或只是文本/组织偏离。不得据此宣称真实执行成功。')
        record = {'record_id': rid, 'scene_id': scene, 'instruction_original': task['instruction'],
                  'task_explanation_zh': task_explanation(task, graphs[scene]), 'total_score': row['total_score'],
                  'scores': {'completion': row['reliability_scores'], 'fidelity': row['fidelity_scores']},
                  'critical_error': row['critical_error'], 'impact_on_final_plan': impact,
                  'initial_state_evidence': task['initial_state'], 'required_goal': task['goal'],
                  'required_constraints': task['constraints'], 'errors': parts, 'actual_action_sequence': compact_sequence(run),
                  'raw_decomposition_tables': {k:original.get(k) for k in ('roles','tasks','constraints','process_constraints')},
                  'raw_skill_matrix': optional(run/'llm_matrix_proposal.json'),
                  'evidence_sources': row['evidence_sources'], 'review_notes': row['notes'],
                  'original_engineering_sha256': row['original_engineering_sha256'],
                  'score_revision': 'unchanged_from_quality_check_v1', 'physical_execution_verified': False}
        analyses.append(record)
        write(dest/scene/(rid+'.analysis.json'), record)
    for row in missing:
        task = read(root/'cases'/row['scene_id']/'tasks'/row['record_id']/'task.json')
        record = {'record_id': row['record_id'], 'scene_id': row['scene_id'], 'instruction_original': task['instruction'],
                  'task_explanation_zh': task_explanation(task, graphs[row['scene_id']]), 'total_score': None,
                  'reason_no_full_score': '没有工程JSON，不并入562份完整文件分数。', **blocked_analysis(row, root)}
        blocked.append(record)
        write(dest/row['scene_id']/(row['record_id']+'.collection_failure.json'), record)
    jsonl(dest/'per_task_analysis.jsonl', analyses)
    jsonl(dest/'collection_failures.jsonl', blocked)
    primary = defaultdict(set)
    for r in analyses:
        for issue in r['errors']:
            if issue['origin'] != 'propagated':
                primary[issue['category']].add(r['record_id'])
    summary = {'version':'per_task_error_cause_analysis_v1', 'generated_at_utc':datetime.now(timezone.utc).isoformat(),
               'scored_tasks_with_deductions':len(analyses), 'critical':sum(r['critical_error'] for r in analyses),
               'minor_only':sum(not r['critical_error'] for r in analyses), 'unscored_collection_failures':len(blocked),
               'primary_issue_record_counts':{k:len(v) for k,v in primary.items()},
               'collection_failure_category_counts':dict(Counter(r['category'] for r in blocked)),
               'limits':['Observed output/validation mechanisms only; no claim of neural internal causation.',
                         'Input graphs are source hypotheses, not physically verified truth.',
                         'No revised scores, source outputs, pipeline parameters or API calls.'],
               'code_evidence':[
                   {'file':'embodied_pipeline/orchestration/task_collection_worker.py','line':300,'finding':'Generates ungraded envelope without correctness/reference review.'},
                   {'file':'outputs/object_alignment_v1/structural_matrix.py','line':156,'finding':'Requires at least one required goal, not complete intent entailment.'},
                   {'file':'outputs/object_alignment_v1/llm_graph_binding.py','line':284,'finding':'Only unsupported required relations populate blocking errors.'},
                   {'file':'outputs/object_alignment_v1/llm_graph_binding.py','line':19,'finding':'Robot state constraints are partitioned into runtime-deferred checks, not scene-edge validation.'},
                   {'file':'outputs/skill_matching_v1/match_skills.py','line':335,'finding':'Dynamic preconditions are unknown, not chain-validated in this collector.'},
                   {'file':'outputs/skill_matching_v1/match_skills.py','line':391,'finding':'Copies initial constraints and goals to deferred.'},
                   {'file':'outputs/object_alignment_v1/structural_matrix.py','line':225,'finding':'Repeated reason means repeated role key, not merely repeated prose.'}]}
    write(dest/'summary.json', summary)
    shutil.copy2(__file__, dest/Path(__file__).name)
    render(dest/'all_tasks.html', analyses, blocked, summary)
    for scene in SCENES:
        selected = [r for r in analyses if r['scene_id'] == scene]
        failures = [r for r in blocked if r['scene_id'] == scene]
        render(dest/(scene+'.html'), selected, failures, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('code_evidence','limits')},ensure_ascii=False))


def render(path, analyses, blocked, summary):
    h = lambda value: escape(str(value))
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐任务错误原因分析</title>',
             '<style>body{max-width:1250px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui;color:#1e3042}h1,h2,h3{color:#183d62}details{border:1px solid #ccd6df;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;word-break:break-word;background:#f1f5f8;padding:12px;font-size:13px}li{margin:8px 0}.warn{color:#a1322a}.note{color:#546978}a{color:#145ca4}</style>',
             '<h1>逐任务错误原因分析</h1><p>按场景和完整任务编号逐一列出任务、实际错误、直接机制、工程放行/拦截原因、正确要求和最终影响。分数沿用上一版，不修改原始模型输出。</p>',
             '<p class="warn">可确认的是输出及校验机制；不能凭一次输出证明模型内部为什么犯错，也不能直接断定是模型固有能力还是prompt造成。这里不是独立人工审核或真机验证。</p>',
             '<p>本页包含 '+str(len(analyses))+' 个已评分扣分任务，以及 '+str(len(blocked))+' 个未生成工程JSON的失败任务。后者不补造12分评分。</p>',
             '<p>用 Ctrl+F 搜索任务编号，点开该条查看；“证据与实际动作”包含原始字段及模型矩阵。</p><h2>一、已生成JSON但有扣分的任务</h2>']
    for r in analyses:
        parts.append(f'<details id="{h(r["record_id"])}"><summary>{h(r["record_id"])} · {r["total_score"]}/12 · {"关键问题" if r["critical_error"] else "仅轻微偏差"}</summary>')
        parts.append('<h3>任务要求</h3><p>'+h(r['task_explanation_zh'])+'</p><p class="note">英文原指令：'+h(r['instruction_original'])+'</p>')
        parts.append('<p>忠实性 D/B/S/E：'+h(r['scores']['fidelity'])+'；流程完成度4/4。</p>')
        for i, e in enumerate(r['errors'],1):
            parts.append(f'<h3>{i}. {h(e["title"])}（{h(STAGES[e["stage"]])}；{h(e["origin"])}）</h3>')
            for label,key in [('本任务具体错误','case_specific_observation'),('直接机制','direct_mechanism'),('为什么采集时仍能输出','why_collection_did_not_block'),('正确要求','expected_behavior')]:
                parts.append('<p><b>'+label+'：</b>'+h(e[key])+'</p>')
            parts.append('<details><summary>该问题的字段证据</summary><pre>'+h(json.dumps(e['evidence'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<p class="warn"><b>最终影响：</b>'+h(r['impact_on_final_plan'])+'</p>')
        for note in r['review_notes']:
            parts.append('<p class="note">'+h(note)+'</p>')
        detail={k:r[k] for k in ['initial_state_evidence','required_goal','actual_action_sequence','raw_decomposition_tables','raw_skill_matrix','evidence_sources']}
        parts.append('<details><summary>原始证据与实际动作</summary><pre>'+h(json.dumps(detail,ensure_ascii=False,indent=2))+'</pre></details></details>')
    parts.append('<h2>二、未生成工程JSON的38条批次失败（本页显示所属场景部分）</h2>')
    for r in blocked:
        parts.append(f'<details id="{h(r["record_id"])}"><summary>{h(r["record_id"])} · 未生成工程JSON · {h(r["failure_stage"])}</summary>')
        parts.append('<p><b>任务：</b>'+h(r['task_explanation_zh'])+'</p><p class="note">英文原文：'+h(r['instruction_original'])+'</p>')
        for label,key in [('直接原因','direct_mechanism'),('归因边界','attribution'),('正确处理','expected_behavior')]:
            parts.append('<p><b>'+label+'：</b>'+h(r[key])+'</p>')
        parts.append('<details><summary>原始记录</summary><pre>'+h(json.dumps(r['evidence'],ensure_ascii=False,indent=2))+'</pre></details></details>')
    parts.append('<h2>工程层面的共同机制</h2><p>本次采集按要求先保存模型答案、不进行正确性比对。结构/引用校验通过不代表完整语义正确；错误能被保存不等于系统声明它可执行。</p><pre>'+h(json.dumps(summary['code_evidence'],ensure_ascii=False,indent=2))+'</pre></html>')
    path.write_text('\n'.join(parts),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    build(parser.parse_args().bundle)
