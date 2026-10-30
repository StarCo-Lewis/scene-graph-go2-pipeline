"""Versioned discrete action codes; strict decoding into existing proposal schema.

No guessed parameters, ID conversions, action insertion or empty-row deletion.
Object codes denote (role_id, element_id) bindings, not arbitrary scene indices.
"""
import copy
import hashlib
import json
import math
import os
import urllib.request

from match_skills import (PlanError, compact_skills, validate_proposal,
                          request_json_with_retry, LOG)
from response_audit import ResponseAudit
from matrix_type_contract import response_format

OBJECT_TYPES = {'scene_element', 'location', 'surface', 'container', 'person', 'control'}
MATRIX_SYSTEM = '''You propose ordered robot actions, never claim execution success.
Return ONLY a valid JSON object: {"matrix":[[subtask_code,skill_code,arg1_code,arg2_code],...]}.
Each row has exactly FOUR integers. Codes are provided by the program: never invent,
renumber, translate or return strings, floats, booleans, nulls, empty rows or objects.
0 is permitted ONLY in unused/optional parameter slots; required slots must be nonzero.
Use object codes for object slots. Optional numeric slots may use 0 to omit them or
the provided preset value code; NEVER write a literal number as a parameter value.
IMPORTANT: scene element_id and matrix object code are DIFFERENT namespaces.
Numbers in the instruction, goal_state, role_id, or runtime state identify scene
objects, NOT matrix codes. Resolve the bound role/element_id in object_codes and
copy ONLY that entry's code into arg1_code/arg2_code. Never copy element_id.
Use the current request's codebook; codes may change between planning segments.
Do not infer an object code from the subtask code or skill code, even if equal.
Use the selected skill's slot meanings exactly. Every supplied subtask must have one
or more actions, contiguous and in supplied subtask order. Keep required actions,
visits, deliberate repetitions, inherited held state and active-segment boundaries.
The FIRST column is the owning subtask code, NOT an action row number.
One subtask may require MULTIPLE rows: repeat its subtask code on all those rows.
N subtasks do NOT mean N actions; more than N rows are valid. Never create a new
subtask code for a prerequisite, or delete a required action to make row count N.
Preparation and placement can belong to the SAME placement subtask.
Use only supplied skills and bound objects; apply prerequisites and current state.
Similarity candidates are suggestions, not actions to execute or success probabilities.
Do not pick up/lift an already held/lifted object unless explicitly requested.
Do not navigate to the held object; transport uses the destination as navigation target.
Do not turn keep-holding into another grasp/lift or perform placement in a carry-only segment.
Only matrix is an output key. No prose, intent, steps, arguments or other wrappers.
If no valid plan can be proposed, return {"matrix":[]} and let the caller reject it.'''


def make_codebook(alignment, library):
    context=alignment.get('task_decomposition',{}).get('identifier_context')
    if context is not None:
        from typed_matrix_ids import action_book,without_context
        return action_book(make_codebook(without_context(alignment),library),context)
    if alignment.get('status') != 'success':
        raise PlanError('Matrix requires successful object alignment')
    tasks = alignment['task_decomposition']['subtasks']
    task_ids = [t['subtask_id'] for t in tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise PlanError('Matrix requires distinct subtask IDs')
    objects = []
    for role, binding in sorted(alignment['context_bindings'].items()):
        ids = binding.get('element_ids', [])
        if not role or not ids or any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
            raise PlanError('Matrix requires integer, distinct grounded IDs per role')
        for element in sorted(ids):
            objects.append({'code': len(objects)+1, 'role_id': role, 'element_id': element})
    skills, values = [], []
    for skill in library['skills']:
        parameters = skill['parameters']
        if len(parameters) > 2:
            raise PlanError('Matrix v1 supports at most two parameter slots per skill')
        for parameter in parameters:
            if parameter['type'] in OBJECT_TYPES:
                continue
            default = parameter.get('default')
            if (parameter['type'] not in ('number', 'height_m') or parameter.get('required', False)
                    or type(default) not in (int, float) or not math.isfinite(default)):
                raise PlanError('Matrix v1 nonobject slots require an optional finite numeric default')
            values.append({'code': len(objects)+len(values)+1, 'skill_id': skill['skill_id'],
                           'parameter': parameter['name'], 'value': default, 'type': parameter['type']})
        skills.append({'code': len(skills)+1, 'skill_id': skill['skill_id'],
                       'slots': [dict(p) for p in parameters]})
    if not skills or len({s['skill_id'] for s in skills}) != len(skills):
        raise PlanError('Matrix requires unique skill IDs')
    book = {'schema_version': 'action_matrix_v1',
            'columns': ['subtask_code', 'skill_code', 'arg1_code', 'arg2_code'],
            'padding_code': 0, 'objects': objects, 'values': values, 'skills': skills,
            'subtasks': [{'code': i, 'subtask_id': t} for i, t in enumerate(task_ids, 1)]}
    book['sha256'] = hashlib.sha256(json.dumps(book, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return book


def decode_matrix(value, codebook, alignment, library):
    if codebook.get('schema_version') in ('action_matrix_typed9_v1','action_matrix_typed5_v1'):
        from typed_matrix_ids import decode_action
        return decode_action(value,codebook,alignment,library)
    if codebook != make_codebook(alignment, library):
        raise PlanError('Matrix codebook differs from current bindings/library')
    if not isinstance(value, dict) or set(value) != {'matrix'}:
        raise PlanError('Matrix schema requires exactly the root key matrix')
    rows = value['matrix']
    if not isinstance(rows, list) or not rows or len(rows) > 256:
        raise PlanError('Matrix requires 1..256 nonempty action rows')
    skills = {s['code']: s for s in codebook['skills']}
    objects = {o['code']: o for o in codebook['objects']}
    values = {v['code']: v for v in codebook['values']}
    tasks = alignment['task_decomposition']['subtasks']
    plans = [{'subtask_id': t['subtask_id'], 'intent': t['description'], 'steps': []} for t in tasks]
    order = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4 or any(type(v) is not int for v in row):
            raise PlanError(f'Matrix row {index} must have exactly four integers')
        task, skill, *args = row
        if not 1 <= task <= len(tasks):
            raise PlanError(
                f'Matrix row {index} has unknown subtask code {task}; '
                f'allowed subtask codes={list(range(1, len(tasks)+1))}. '
                'The first column is the owning subtask code, NOT the action row number. '
                'Multiple actions/prerequisites for one subtask must repeat that same code '
                'in contiguous rows. Do not invent another subtask or delete necessary actions '
                'to match the number of subtasks; assign each action to its actual supplied subtask.')
        if skill not in skills:
            raise PlanError(f'Matrix row {index} has unknown skill code {skill}; '
                            f'allowed skill codes={sorted(skills)}; use the supplied skill code lookup')
        order.append(task)
        spec, decoded = skills[skill], {}
        for slot, object_code in enumerate(args):
            if slot >= len(spec['slots']):
                if object_code != 0:
                    raise PlanError(f'Matrix row {index} unused parameter slot must be 0')
                continue
            parameter = spec['slots'][slot]
            if object_code == 0:
                if parameter.get('required', False):
                    raise PlanError(f'Matrix row {index} required parameter {parameter["name"]} cannot be 0')
                continue
            if parameter['type'] in OBJECT_TYPES:
                if object_code not in objects:
                    matches = [o for o in objects.values() if o['element_id'] == object_code]
                    hint = ''
                    if matches:
                        hint = (f'; received {object_code} is a scene element_id, NOT a matrix object code'
                                '; current bound role/code lookup=' + json.dumps(matches, ensure_ascii=True))
                        if len(matches) > 1:
                            hint += '; multiple roles share this scene ID: select the intended bound role explicitly'
                    raise PlanError(
                        f'Matrix row {index} has unknown object code {object_code} '
                        f'in arg{slot+1}_code ({spec["skill_id"]}.{parameter["name"]}); '
                        f'allowed object codes={sorted(objects)}{hint}; '
                        'regenerate using object_codes[].code, not element_id; no automatic ID conversion')
                obj = objects[object_code]
                decoded[parameter['name']] = {'role_id': obj['role_id'], 'element_id': obj['element_id']}
            else:
                preset = values.get(object_code)
                if not preset or (preset['skill_id'], preset['parameter']) != (spec['skill_id'], parameter['name']):
                    raise PlanError(f'Matrix row {index} has invalid numeric preset code')
                decoded[parameter['name']] = preset['value']
        plans[task-1]['steps'].append({'skill_id': spec['skill_id'], 'arguments': decoded})
    if order != sorted(order) or set(order) != set(range(1, len(tasks)+1)):
        raise PlanError('Matrix must cover every subtask in contiguous original order; '
                        'repeat the same subtask code for its multiple actions/prerequisites, '
                        'do not drop actions or subtasks to force one row per subtask')
    proposal = {'schema_version': '1.0', 'subtask_plans': plans}
    validate_proposal(proposal, alignment, library)
    return proposal


def matrix_payload(alignment, library, codebook, model, retrieval, widened, validation_error=None,
                   matrix_contract='legacy_json'):
    typed=codebook.get('schema_version') in ('action_matrix_typed9_v1','action_matrix_typed5_v1')
    if typed != (matrix_contract in ('typed9_schema','typed5_schema')):
        raise PlanError('Matrix identifier protocol and selected generation mode differ; no downgrade')
    if typed:
        from typed_matrix_ids import width
        if width(codebook)!=(5 if matrix_contract=='typed5_schema' else 9):
            raise PlanError('Matrix identifier width differs from selected generation mode; no conversion')
        from typed_matrix_ids import action_payload
        return action_payload(alignment,library,codebook,model,retrieval,widened,validation_error)
    if codebook != make_codebook(alignment, library):
        raise PlanError('Matrix input codebook mismatch')
    skills = {s['skill_id']: s['code'] for s in codebook['skills']}
    tasks = alignment['task_decomposition']['subtasks']
    per_task = {s['subtask_id']: s['candidate_skill_ids'] for s in retrieval['subtasks']}
    if set(per_task) != {t['subtask_id'] for t in tasks} or any(
            not ids or not set(ids) <= set(skills) for ids in per_task.values()):
        raise PlanError('Matrix candidate inventory mismatch')
    allowed = set(skills) if widened else set().union(*(set(ids) for ids in per_task.values()))
    contract = copy.deepcopy(library.get('execution_contract', {}))
    # Keep the same state/scope/goal rules but replace incompatible output-format rules.
    strict = contract.get('strict_prompt_v2')
    if isinstance(strict, str):
        start, end = strict.find('SKILL JSON:'), strict.find('BEFORE RETURNING')
        if start >= 0 and end > start:
            strict = strict[:start] + 'SKILL OUTPUT: use the four-column matrix contract for this request.\n\n' + strict[end:]
        strict = strict.replace('Return only the schema requested by', 'Return only the matrix schema requested by')
        contract['strict_prompt_v2'] = strict
    compact = {s['skill_id']: s for s in compact_skills(library)}
    serial_rules = None
    if contract.get('adapter_mode') in ('json_symbolic_test_only', 'serial_scene201_proxy'):
        # Describe the existing SerialExecutor state machine, not a physical guarantee.
        serial_rules = {
            'scope': 'Serial proxy preconditions only; not physical affordance or execution evidence',
            'navigate_to_object': 'Successful navigation clears prepared_target_id, even while carrying.',
            'position_arm_for_object': 'Requires proximity; sets prepared_target_id to its target. Does not grasp or release.',
            'grasp_object': 'Requires empty gripper and prepared_target_id equal to target.',
            'lift_held_object': 'Requires the specified object already held.',
            'place_object_on_surface': (
                'Requires the specified object held and lifted, and prepared_target_id equal to surface. '
                'If surface preparation is not currently satisfied, propose position_arm_for_object '
                'with target=the bound surface BEFORE placement, after the last navigation. '
                'These two actions can share the SAME placement subtask code. '
                'Preparation for the held object does NOT prepare the destination surface. '
                'If the surface is already prepared and no intervening navigation occurs, '
                'do not add redundant preparation.')}
    prompt = {'instruction': alignment['task_decomposition']['instruction'],
        'goal_state': alignment['task_decomposition'].get('goal_state', []),
        'execution_contract': contract, 'object_codes': codebook['objects'],
        'optional_numeric_preset_codes': codebook['values'],
        'columns': codebook['columns'], 'codebook_sha256': codebook['sha256'],
        'subtasks': [dict(subtask_id=t['subtask_id'], description=t['description'],
            dependencies=t.get('dependencies', []), code=i, candidate_skill_codes=[skills[s] for s in
            (list(skills) if widened else per_task[t['subtask_id']])]) for i, t in enumerate(tasks, 1)],
        'skills': [dict(compact[s['skill_id']], code=s['code'],
            slot_order=[p['name'] for p in s['slots']]) for s in codebook['skills'] if s['skill_id'] in allowed],
        'zero_rule': '0 only for unused or optional parameter slots; never an action or object',
        'widened_inventory': widened}
    if validation_error:
        prompt['previous_validation_error'] = validation_error
        prompt['correction_requirement'] = (
            'Regenerate the whole matrix from unchanged inputs; no guessed codes or dropped subtasks. '
            'For an unknown object code, look up the intended bound role/scene element_id in '
            'object_codes and copy its code, NEVER the scene element_id. This is not permission '
            'to change the target or bindings. For a subtask code error, preserve all necessary '
            'actions and group them under their owning supplied subtask codes; the first column '
            'is NOT a row counter. Do not remove preparation, grasp, lift, visits or placement '
            'merely to make the number of rows equal the number of subtasks.')
        if serial_rules is not None and 'SURFACE_PROXY_PREPARATION_REQUIRED' in validation_error:
            prompt['missing_precondition_correction'] = (
                'The placement action has no valid preparation of its bound destination surface. '
                'Any preceding navigation clears preparation. Propose position_arm_for_object targeting that '
                'same bound surface after the last navigation and before place_object_on_surface; '
                'reuse the owning placement subtask code for both actions. Preserve the requested '
                'placement and all other required actions; do not change the target or claim success.')
    # Repeat the small wire contract AFTER the potentially large runtime context.
    # This is a lookup instruction, not an example action sequence or an ID repair.
    prompt['matrix_encoding_contract'] = {
        'version': 'action_matrix_namespaces_and_grouping_v2',
        'columns': codebook['columns'],
        'allowed_subtask_codes': [s['code'] for s in codebook['subtasks']],
        'action_grouping_rule': (
            'First column = owning subtask code, NOT row index. Multiple contiguous rows '
            'may share one subtask code; row count may exceed subtask count. Keep every '
            'required action and prerequisite; never invent a subtask code or delete a '
            'necessary action to obtain one row per subtask.'),
        'serial_proxy_preconditions': serial_rules,
        'allowed_object_slot_codes': [o['code'] for o in codebook['objects']],
        'bound_object_lookup': [
            f'role_id={o["role_id"]}, scene element_id={o["element_id"]} -> '
            f'matrix object code={o["code"]}; copy {o["code"]} into the object slot, NOT element_id'
            for o in codebook['objects']],
        'final_check': (
            'Resolve scene IDs to the intended bound roles first. Every object slot must use '
            'that role entry\'s code from this request. Zero is only unused/optional padding. '
            'Do not copy scene IDs, role-name numeric suffixes, or codes from another segment. '
            'The program decodes codes into role_id/element_id; you do not output those IDs.')}
    return {'model': model, 'temperature': 0,
            'response_format': response_format('skill_matching_matrix',matrix_contract,codebook), 'enable_thinking': False,
            'messages': [{'role': 'system', 'content': MATRIX_SYSTEM},
                         {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=True)}]}


def request_matrix(alignment, library, codebook, model, base_url, api_key_env,
                   retrieval, widen_inventory=False, validation_error=None,
                   timeout=180, attempts=2, diagnostics_dir=None, audit_context=None,
                   matrix_contract='legacy_json'):
    key = os.environ.get(api_key_env, '').strip()
    if not key:
        raise PlanError(f'{api_key_env} is not configured')
    payload = matrix_payload(alignment, library, codebook, model, retrieval,
                             widen_inventory, validation_error, matrix_contract)
    req = urllib.request.Request(base_url.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, method='POST')
    audit = ResponseAudit(diagnostics_dir, 'skill_matching_matrix', audit_context, req)
    LOG.info('Requesting four-column action matrix from %s', model)
    content = None
    try:
        response = request_json_with_retry(req, timeout, attempts, audit=audit)
        content = response['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise PlanError('Matrix response content must be a string')
        value = json.loads(content)
        audit.parsed(content)
        return value
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        audit.parsed(content, 'rejected', exc)
        raise
