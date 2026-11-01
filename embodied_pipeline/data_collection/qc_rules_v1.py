"""Offline evidence checks for the 12-point rubric; no robot/API evaluation.

Findings are review candidates, not unreviewed automatic semantic scores.
Uses original task specification; reference step count is not an oracle.
"""
from collections import Counter
from pathlib import Path
import argparse
import re

from .batch import read, read_jsonl, write, jsonl, sha
from .plan_artifact import has_complete

SCENES = ('scene200', 'scene202', 'd3_619')


def flatten(plan):
    return [s for t in plan.get('subtask_plans', []) for s in t.get('steps', [])]


def object_arg(step, name):
    value = step.get('arguments', {}).get(name, {})
    return value.get('element_id') if isinstance(value, dict) else None


def audit_record(root, scene, rid):
    base = root / 'cases' / scene / 'tasks' / rid
    run = base / 'model_run'
    task, d, b, e = [read(p) for p in (base / 'task.json', run / 'task_decomposition.json',
                                     run / 'object_alignment.json', run / 'engineering_plan.json')]
    graph = read(root / 'cases' / scene / 'graph/scene_graph.json')
    edges = read(root / 'cases' / scene / 'graph/scene_graph_edges.json')
    graph_ids = {n['id'] for n in graph}
    edge_set = {(x['source'], x['relation'].removeprefix('a ').removesuffix(' b'), x['target']) for x in edges}
    bindings = {r: v.get('element_ids', []) for r, v in b['context_bindings'].items()}
    role_ids = {}
    issues = []

    def issue(stage, code, detail, file, pointer, observed=None, expected=None, origin='new'):
        issues.append({'stage': stage, 'code': code, 'reason': detail, 'origin': origin,
                       'evidence': [{'file': (run / file).relative_to(root).as_posix(), 'pointer': pointer,
                                     'observed': observed, 'expected': expected}]})

    for role in d.get('task_roles', []):
        role_id = role['role_id']
        explicit = set(map(int, re.findall(r'\bID\s*[:#]?\s*(\d+)\b', role.get('semantic_query_en', ''), re.I)))
        ids = bindings.get(role_id, [])
        role_ids[role_id] = next(iter(explicit)) if len(explicit) == 1 else (ids[0] if len(ids) == 1 else None)
        # Queries may mention the source/support's ID too; do not mistake that
        # relation description for a request to bind multiple payloads.
        if len(explicit) == 1 and set(ids) != explicit:
            issue('B', 'explicit_id_binding_mismatch', '绑定对象与角色明确指定ID不一致。', 'object_alignment.json',
                  f'/context_bindings/{role_id}/element_ids', ids, sorted(explicit))
        if len(ids) != role.get('quantity', 1) or len(set(ids)) != len(ids) or any(i not in graph_ids for i in ids):
            issue('B', 'binding_quantity_or_existence', '绑定数量、唯一性或对象存在性不符。', 'object_alignment.json', f'/context_bindings/{role_id}', ids)
    all_ids = [i for ids in bindings.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        issue('B', 'duplicate_cross_role_binding', '不同角色重复绑定同一对象。', 'object_alignment.json', '/context_bindings', all_ids)

    def ground(c):
        subject = c.get('subject_role')
        return (role_ids.get(subject, subject), c.get('relation'), role_ids.get(c.get('object_role')),
                c.get('expected_value'), c.get('required'))

    goal_rows = [ground(c) for c in d.get('goal_state', [])]
    expected_near = {g['object_id'] for g in task['goal']['final_predicates'] if g['predicate'] == 'near'}
    if task.get('task_family') == 'ordered_visit':
        extra = [c for c in d.get('goal_state', []) if c.get('relation') == 'near'
                 and c.get('subject_role') == 'robot' and role_ids.get(c.get('object_role')) not in expected_near]
        if extra:
            issue('D', 'intermediate_visits_encoded_as_final', '把中间访问点写成最终同时成立的near条件，混淆里程碑与终态。',
                  'task_decomposition.json', '/goal_state', extra, sorted(expected_near))
            if any(c in e['plan'].get('deferred', {}).get('goal_state', []) for c in extra):
                issue('E', 'inherited_intermediate_final_goals', '最终JSON继续保留所有中途访问点的同时near终态；动作顺序正确不能消除错误目标。',
                      'engineering_plan.json', '/plan/deferred/goal_state', extra, sorted(expected_near), 'propagated')
    for expected in task['goal']['final_predicates']:
        p, obj = expected['predicate'], expected['object_id']
        if p in ('held', 'lifted'):
            found = any(s == 'robot' and rel == p and o == obj and v == expected['expected'] and req
                        for s, rel, o, v, req in goal_rows)
        elif p == 'on':
            found = any(s == obj and rel == 'on' and o == expected['support_id'] and req for s, rel, o, v, req in goal_rows)
        elif p == 'near':
            found = any(s == 'robot' and rel == 'near' and o == obj and req for s, rel, o, v, req in goal_rows)
            # A held payload near the destination expresses the carry endpoint
            # equivalently; role-subject near is explicitly legal in the prompt.
            held_payloads = {o for s, rel, o, v, req in goal_rows if s == 'robot' and rel == 'held' and v is True and req}
            found = found or any(s in held_payloads and rel == 'near' and o == obj and req for s, rel, o, v, req in goal_rows)
        else:
            continue
        if not found:
            issue('D', 'missing_required_final_' + p, '分解目标矩阵缺少任务要求的最终条件（须结合文本与协议复核）。',
                  'task_decomposition.json', '/goal_state', d.get('goal_state'), expected)
    for index, c in enumerate(d.get('initial_scene_constraints', [])):
        s, rel, o, value, required = ground(c)
        unsupported = ((s != 'robot' and rel in ('on', 'in', 'near') and (s, rel, o) not in edge_set)
                       or (s == 'robot' and rel in ('held', 'lifted', 'near')))
        if unsupported:
            issue('D', 'unsupported_initial_assertion', '分解新增无输入证据或与输入冲突的初始条件；required=false不表示条件为假。',
                  'task_decomposition.json', f'/initial_scene_constraints/{index}', c, 'Supported input fact only')
            if c in e['plan'].get('deferred', {}).get('initial_scene_constraints', []):
                issue('E', 'inherited_unsupported_initial_assertion', '错误初始条件仍保留在最终工程JSON，属于前序传播。',
                      'engineering_plan.json', '/plan/deferred/initial_scene_constraints', c, 'No unsupported initial assertions', 'propagated')
    protected = set(task.get('constraints', {}).get('preserve_objects', []))
    protected_d = {role_ids.get(c.get('role_id')) for c in d.get('process_constraints', []) if c.get('restriction') == 'do_not_manipulate'}
    if protected - protected_d:
        issue('D', 'missing_process_protection', '明确保护对象未进入过程限制表。', 'task_decomposition.json', '/process_constraints', sorted(protected_d), sorted(protected))

    steps = flatten(e['plan'])
    lib = read(run / 'execution_skill_library.json')
    allowed = {s['skill_id']: s for s in lib['skills']}
    held, lifted, near = None, False, None
    grasp_prepared, placement, visited, events = set(), {}, [], []
    moved = set()
    for index, step in enumerate(steps):
        skill = step['skill_id']
        target = object_arg(step, 'target')
        obj, surface = object_arg(step, 'object'), object_arg(step, 'surface')
        pointer = f'/plan/action_sequence/{index}'
        if skill not in allowed:
            issue('S', 'unsupported_skill', '采用不在可用技能库内的技能。', 'engineering_plan.json', pointer, skill)
        for arg, val in step.get('arguments', {}).items():
            if isinstance(val, dict) and 'element_id' in val and val['element_id'] not in bindings.get(val.get('role_id'), []):
                issue('S', 'action_binding_mismatch', '动作参数与绑定结果不一致。', 'engineering_plan.json', pointer, step)
        if skill == 'navigate_to_object':
            if held is not None and not lifted:
                issue('S', 'carry_without_lift', '已抓取对象但未抬离支撑便执行携带导航。', 'engineering_plan.json', pointer, step, 'lift_held_object before carrying')
            near = target
            visited.append(target)
            events.append(('visited_near', target, None))
            grasp_prepared.clear()
        elif skill == 'position_arm_for_object':
            if near is None:
                issue('S', 'preparation_without_approach', '初始接近未知且此前没有导航，直接进行手臂准备。', 'engineering_plan.json', pointer, step)
            grasp_prepared.add(target)
        elif skill == 'grasp_object':
            if near is None or target not in grasp_prepared:
                issue('S', 'grasp_prerequisites_missing', '抓取前缺少接近或针对对象的手臂准备，输入未提供可继承就绪状态。', 'engineering_plan.json', pointer, step)
            if held is not None:
                issue('S', 'grasp_while_holding', '尚未放下前一对象便再次抓取，违反单物体持有前置。', 'engineering_plan.json', pointer, step)
            held, lifted = target, False
            moved.add(target)
            placement.pop(target, None)
        elif skill == 'lift_held_object':
            if held != obj:
                issue('S', 'lift_unheld_object', '抬升对象与先前抓取对象不一致或没有抓取。', 'engineering_plan.json', pointer, step)
            lifted = True
            moved.add(obj)
        elif skill == 'place_object_on_surface':
            if held != obj or near != surface or surface not in grasp_prepared:
                issue('S', 'placement_prerequisites_missing', '放置前持有、目的接近或支撑面准备不完整（需语义复核等效近邻）。', 'engineering_plan.json', pointer, step,
                      {'held': held, 'near': near, 'prepared': sorted(grasp_prepared)})
            placement[obj] = surface
            moved.add(obj)
            held, lifted = None, False
            events.append(('delivered_and_released', obj, surface))
    if moved & protected:
        issue('E', 'protected_object_manipulated', '实际计划操作了明确要求保护的对象。', 'engineering_plan.json', '/plan/subtask_plans', sorted(moved & protected))
    if task.get('constraints', {}).get('no_object_manipulation') and moved:
        issue('E', 'navigation_task_manipulates', '只导航任务包含操纵动作。', 'engineering_plan.json', '/plan/subtask_plans', sorted(moved))
    for goal in task['goal']['final_predicates']:
        p, obj = goal['predicate'], goal['object_id']
        ok = (placement.get(obj) == goal.get('support_id') if p == 'on' else
              ((held == obj) == goal.get('expected') if p == 'held' else near == obj if p == 'near' else True))
        if not ok:
            issue('E', 'final_goal_not_realized_' + p, '动作序列的条件性符号终态未覆盖原始任务目标。', 'engineering_plan.json', '/plan/subtask_plans',
                  {'held': held, 'near': near, 'placements': placement}, goal)
    cursor = 0
    for milestone in task['goal'].get('ordered_milestones', []):
        desired = (milestone['predicate'], milestone['object_id'], milestone.get('destination_id'))
        if desired[0] not in ('visited_near', 'delivered_and_released'):
            continue
        match = next((i for i in range(cursor, len(events)) if events[i] == desired), None)
        if match is None:
            issue('E', 'ordered_milestone_not_realized', '请求的中间目标缺失或顺序未满足。', 'engineering_plan.json', '/plan/subtask_plans', events, desired)
        else:
            cursor = match + 1
    return {'record_id': rid, 'scene_id': scene, 'instruction': task['instruction'],
            'source_engineering_sha256': sha(run / 'engineering_plan.json'), 'findings_for_review': issues,
            'abstract_outcome_is_conditional_not_executed': True}


def run_audit(root):
    output, missing = [], []
    for scene in SCENES:
        for task in read_jsonl(root / 'cases' / scene / 'tasks.jsonl'):
            rid = task['record_id']
            run = root / 'cases' / scene / 'tasks' / rid / 'model_run'
            # Scoring covers complete plans only; incomplete envelopes stay listed here, never dropped.
            if not has_complete(run / 'engineering_plan.json'):
                summary = read(run / 'planning_summary.json') if (run / 'planning_summary.json').exists() else {}
                missing.append({'record_id': rid, 'scene_id': scene, 'instruction': task['instruction'],
                                'engineering_plan_file_present': (run / 'engineering_plan.json').is_file(),
                                'status': summary.get('status'), 'stage': summary.get('stage'),
                                'error_type': summary.get('error_type'), 'error': summary.get('error'),
                                'failed_stage': summary.get('failed_stage'),
                                'errors': summary.get('errors'), 'summary_path': (run / 'planning_summary.json').relative_to(root).as_posix(),
                                'total_score': None, 'reason': 'No complete engineering plan (file missing or incomplete envelope); outside the complete-file score denominator.'})
            else:
                output.append(audit_record(root, scene, rid))
    dest = root / 'reports/quality_check_v1'
    dest.mkdir(parents=True, exist_ok=True)
    jsonl(dest / 'automated_checks.jsonl', output)
    jsonl(dest / 'not_generated.jsonl', missing)
    counts = Counter(f['code'] for r in output for f in r['findings_for_review'])
    write(dest / 'automated_check_summary.json', {'records': len(output), 'no_engineering_json': len(missing),
          'findings': dict(counts), 'scored': False, 'physical_execution': False})
    print(dict(counts))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    run_audit(p.parse_args().bundle)
