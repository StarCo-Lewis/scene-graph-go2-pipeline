"""Text-only graph binding. No CLIP, image, fabricated coordinates or scores."""
import json
import copy
import os
import logging
from pathlib import Path
import urllib.request
from llm_transport import request_json
from response_audit import ResponseAudit, validation as record_validation, save as save_audit_json
from structural_matrix import BINDING_PROMPT, make_binding_codebook, from_binding_matrix
from matrix_type_contract import check_mode, response_format, enforce_and_record


class BindingError(ValueError):
    pass


def partition_initial_constraints(constraints):
    """Route robot predicates to runtime; never drop them from decomposition."""
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        raise BindingError('initial_scene_constraints must be an array')
    graph, runtime = [], []
    for constraint in constraints:
        if not isinstance(constraint, dict) or not isinstance(constraint.get('subject_role'), str):
            raise BindingError('Invalid initial constraint')
        destination = runtime if constraint['subject_role'] == 'robot' else graph
        destination.append(copy.deepcopy(constraint))
    return graph, runtime


def request_bindings(roles, nodes, model, base_url, api_key_env, instruction=None, edges=None, initial_constraints=None,
                     request_timeout=180, request_attempts=1, diagnostics_dir=None,
                     validation_error=None, validation_attempt=1, conversation_context=None,
                     inference_policy=None, scene_capabilities=None, factual_boundary=None,
                     output_format='nested_json', matrix_contract='legacy_json', identifier_context=None):
    check_mode(matrix_contract)
    if (matrix_contract in ('typed9_schema','typed5_schema')) != (identifier_context is not None):
        raise BindingError('Binding identifier context and generation mode differ; no downgrade')
    key = os.environ.get(api_key_env)
    if not key:
        raise BindingError(api_key_env + ' required')
    graph_constraints, _ = partition_initial_constraints(initial_constraints)
    prompt = '''Select only existing active node IDs matching each task role's category,
attributes and quantity. Descriptions are supplied metadata, not visual proof.
Never invent objects, coordinates, scores or spatial relations. If multiple
instances cannot be distinguished return ambiguous. If no adequate candidates
exist return missing. Return JSON {"bindings":[{"role_id":"...",
"status":"matched|missing|ambiguous", "element_ids":[], "reason":"English concise reason"}]}.
Honor explicit instance IDs in the original instruction, but never override
category/attribute mismatches or bind nonexistent IDs. An explicit ID can resolve
otherwise indistinguishable instances. For matched return exactly the requested quantity of distinct IDs. Preserve role IDs.
element_ids is an array of JSON integer numbers, never strings, floats or booleans.
For numeric syntax, [67] is valid and ["67"] is invalid; this is NOT a suggested
binding. Copy integer values only from existing active nodes. Distinctness is
checked within each role and matched IDs cannot be reused across roles.
Use supplied edges to distinguish instances only for initial scene requirements,
never for desired goal relations. a is source and b is target in edge labels.
Missing edges mean unknown, not false; do not infer an on/in relation from bbox
proximity. If a required relation is unsupported, return ambiguous instead of
guessing. This applies to supplied OBJECT-TO-OBJECT initial constraints only.
robot is a reserved executor, NOT a scene node. Robot proximity and held/lifted
state are checked separately by the runtime executor. Never demand robot nodes,
robot-near-object edges or holding edges to bind an otherwise matching object.
Do not claim these robot states are satisfied or invent evidence for them.
The words held or carry in the instruction do not themselves make the object's
category ambiguous. Actual instance/category ambiguity must still be reported.
No free-form reasoning trace. Optional roles must also be reported.'''
    data = {'instruction': instruction, 'roles': roles, 'nodes': nodes,
            'edges': edges or [], 'initial_scene_constraints': graph_constraints,
            'constraint_scope': 'object_to_object_scene_relations_only',
            'robot_state_validation': 'deferred_to_runtime_not_object_binding'}
    if conversation_context is not None:
        data['conversation_context'] = conversation_context
    if scene_capabilities is not None:
        data['scene_capabilities'] = scene_capabilities
    if factual_boundary is not None:
        data['factual_boundary'] = factual_boundary
    if inference_policy is not None:
        if inference_policy.get('mode') != 'bounded_inference':
            raise BindingError('Unsupported inference policy')
        # Replace the conflicting unconditional rule; do not silently force an ID in code.
        prompt = prompt.replace('If multiple\ninstances cannot be distinguished return ambiguous.',
            'If multiple instances cannot be distinguished, apply the supplied bounded inference policy. '
            'You may propose a provisional compatible choice with its basis and uncertainty in reason. '
            'If no supported interpretation exists, return ambiguous.')
        data['inference_policy'] = inference_policy
    if validation_error:
        data.update(previous_validation_error=validation_error,
            correction_requirement=('Recreate the complete bindings object from the unchanged inputs. '
                'Correct the named schema error, with integer element_ids. Do not copy malformed '
                'output, invent IDs, change roles, or turn missing/ambiguous into a forced match.'))
    if output_format == 'matrix':
        # Replace only legacy wire instructions; all selection/uncertainty rules remain.
        start = prompt.index('Return JSON {"bindings"')
        end = prompt.index('Honor explicit instance IDs')
        prompt = prompt[:start] + prompt[end:]
        prompt = prompt.replace('Preserve role IDs.', 'Preserve supplied role codes via the codebook.')
        start = prompt.index('element_ids is an array')
        end = prompt.index('Use supplied edges')
        prompt = prompt[:start] + ('Node ID cells in matches/candidates must be exact JSON integers copied '
            'from active nodes. Matched IDs cannot be reused across roles.\n') + prompt[end:]
        prompt += '\n' + BINDING_PROMPT
        data['object_binding_codebook'] = make_binding_codebook(roles, nodes, identifier_context)
        if identifier_context is not None:
            typed_version='object_binding_typed5_v1' if matrix_contract=='typed5_schema' else 'object_binding_typed9_v1'
            prompt=prompt.replace('object_binding_matrix_v1',typed_version)
            prompt=prompt.replace('active_scene_element_id','scene_node_code')
            prompt=prompt.replace('Node ID cells in matches/candidates must be exact JSON integers copied from active nodes.',
                'Node code cells must copy type-3 codes from scene_codes, NOT raw node IDs.')
            prompt+='\nUse type-2 role codes and type-3 scene codes from the CURRENT hash-scoped codebook only. '
            prompt+='The scene_codes lookup maps each issued scene code to a raw element_id; never output raw element_id.'
        data.pop('roles')  # Role definitions occur once, in the numbered codebook.
        if validation_error:
            data['correction_requirement'] = ('Recreate complete object_binding_matrix_v1 tables from unchanged '
                'inputs; correct only named protocol errors. No ID coercion, invented roles or forced matches.')
            if identifier_context is not None:
                data['correction_requirement']=data['correction_requirement'].replace(
                    'object_binding_matrix_v1',typed_version)
    elif output_format != 'nested_json':
        raise BindingError('Unsupported object binding output format')
    payload = {'model': model, 'temperature': 0,
               'response_format': (response_format('object_binding',matrix_contract,data['object_binding_codebook'])
                                   if output_format=='matrix' else {'type':'json_object'}),
               'messages': [{'role': 'system', 'content': prompt},
                            {'role': 'user', 'content': json.dumps(data)}]}
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    audit = ResponseAudit(diagnostics_dir, 'object_binding',
                          context={'validation_attempt': validation_attempt}, request=request)
    content = None
    try:
        result = (request_json(request, timeout=request_timeout, attempts=request_attempts, audit=audit)
                  if diagnostics_dir is not None else request_json(request, timeout=request_timeout, attempts=request_attempts))
        content = result['choices'][0]['message']['content']
        if output_format == 'matrix' and result['choices'][0].get('finish_reason') == 'length':
            raise BindingError('Truncated object binding matrix rejected')
        proposal = json.loads(content)
        audit.parsed(content)
        return proposal
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        audit.parsed(content, 'rejected', exc)
        raise


def request_validated_bindings(roles, nodes, model, base_url, api_key_env,
                               max_validation_attempts=3, **kwargs):
    """Bounded schema correction only; semantic missing/ambiguity returns unchanged.

    Transport/parse failures propagate under the existing transport policy.
    No ID coercion, fabricated replacement or hand-written binding fallback.
    The caller records the final decision after its relation check.
    """
    if type(max_validation_attempts) is not int or not 1 <= max_validation_attempts <= 3:
        raise BindingError('Binding validation attempts must be in [1, 3]')
    directory = kwargs.get('diagnostics_dir')
    matrix = kwargs.get('output_format', 'nested_json') == 'matrix'
    book = make_binding_codebook(roles, nodes, kwargs.get('identifier_context')) if matrix else None
    if matrix and directory is not None:
        save_audit_json(Path(directory) / 'object_binding_codebook.json', book)
    feedback = None
    for attempt in range(1, max_validation_attempts + 1):
        logging.info('LLM object binding schema attempt=%d/%d', attempt, max_validation_attempts)
        proposal = request_bindings(roles, nodes, model, base_url, api_key_env,
            validation_error=feedback, validation_attempt=attempt, **kwargs)
        evidence = f'llm_binding_proposal_attempt_{attempt}.json'
        if matrix:
            evidence = f'llm_binding_matrix_proposal_attempt_{attempt}.json'
        if directory is not None:
            save_audit_json(Path(directory) / evidence, proposal)
            if not matrix:
                save_audit_json(Path(directory) / 'llm_binding_proposal.json', proposal)
        try:
            if matrix:
                try:
                    proposal, _ = enforce_and_record(proposal,'object_binding',
                        kwargs.get('matrix_contract','legacy_json'),directory,
                        f'llm_binding_matrix_proposal_attempt_{attempt}',book)
                    proposal = from_binding_matrix(proposal, roles, nodes, book)
                except ValueError as exc:
                    raise BindingError(str(exc)) from exc
                if directory is not None:
                    save_audit_json(Path(directory) / f'llm_binding_decoded_attempt_{attempt}.json', proposal)
                    save_audit_json(Path(directory) / 'llm_binding_proposal.json', proposal)
            bindings, errors = validate_bindings(proposal, roles, nodes)
            return proposal, bindings, errors, attempt
        except BindingError as exc:
            feedback = str(exc)
            record_validation(directory, 'object_binding', attempt, 'rejected', exc,
                context={'proposal_origin': 'llm', 'correction_scope': 'binding_validation'}, evidence=evidence)
            logging.warning('Rejected object binding attempt %d: %s', attempt, exc)
            if attempt == max_validation_attempts:
                raise BindingError(f'LLM object binding remained invalid after {attempt} attempts: {exc}') from exc


def validate_bindings(proposal, roles, nodes):
    inventory = {n['id']: n for n in nodes if n.get('active', True)}
    specs = {r['role_id']: r for r in roles}
    rows = proposal.get('bindings') if isinstance(proposal, dict) else None
    if not isinstance(rows, list):
        raise BindingError('bindings must be an array')
    seen, used, bindings, errors = set(), set(), {}, []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get('role_id') not in specs or row['role_id'] in seen:
            raise BindingError('Invalid or duplicate role')
        role_id = row['role_id']; seen.add(role_id)
        ids, status = row.get('element_ids'), row.get('status')
        path = f'bindings[{row_index}].element_ids'
        if not isinstance(ids, list):
            raise BindingError(f'{path} for role {role_id!r} must be an array; received {type(ids).__name__}')
        for id_index, element_id in enumerate(ids):
            if type(element_id) is not int:
                raise BindingError(f'{path}[{id_index}] for role {role_id!r} must be a JSON integer; '
                    f'received {element_id!r} ({type(element_id).__name__}); '
                    'numeric strings, floats and booleans are invalid; no automatic conversion')
        if len(set(ids)) != len(ids):
            raise BindingError(f'{path} for role {role_id!r} contains duplicate IDs: {ids!r}; '
                               'IDs must be distinct within this role')
        if status not in ('matched', 'missing', 'ambiguous') or not isinstance(row.get('reason'), str):
            raise BindingError('Invalid decision status or reason')
        if status != 'matched':
            if status == 'missing' and ids:
                raise BindingError('Unresolved decision cannot bind IDs')
            if any(i not in inventory for i in ids):
                raise BindingError('Unresolved candidates contain unknown IDs')
            if specs[role_id]['required']:
                errors.append({'role_id': role_id, 'reason': status, 'candidate_ids': ids,
                               'candidate_ids_are_bindings': False})
            continue
        if len(ids) != specs[role_id]['quantity'] or any(i not in inventory or i in used for i in ids):
            raise BindingError('Unknown/reused IDs or wrong quantity')
        used.update(ids)
        bindings[role_id] = {'element_ids': ids, 'alternatives': [], 'selected': [
            {'element_id': i, 'object_tag': inventory[i]['object_tag'],
             'evidence': {'caption': inventory[i]['caption'], 'binding_reason': row['reason'],
                          'source': 'supplied_graph_metadata', 'visual_verified': False}}
            for i in ids]}
    if seen != set(specs):
        raise BindingError('Missing role decisions')
    return bindings, errors


def validate_edges(edges, nodes):
    if not isinstance(edges, list):
        raise BindingError('edges must be an array')
    ids = {n['id'] for n in nodes}
    for edge in edges:
        if not isinstance(edge, dict) or any(type(edge.get(k)) is not int or edge[k] not in ids for k in ('source', 'target')):
            raise BindingError('Edge endpoints must reference active node IDs')
        if not isinstance(edge.get('relation'), str) or not edge['relation'].strip():
            raise BindingError('Edge relation must be a nonempty string')
    return edges


def check_binding_relations(bindings, constraints, edges):
    """Validate initial relations, never goal relations. Unknown is not absence."""
    mapping = {'a on b': ('on', False), 'b on a': ('on', True),
               'a in b': ('in', False), 'b in a': ('in', True)}
    facts = set()
    for edge in edges:
        key = edge['relation'].strip().lower()
        if key in mapping:
            relation, reverse = mapping[key]
            a, b = edge['source'], edge['target']
            facts.add((b, relation, a) if reverse else (a, relation, b))
    checks, errors = [], []
    graph_constraints, _ = partition_initial_constraints(constraints)
    for constraint in graph_constraints:
        subjects = bindings.get(constraint['subject_role'], {}).get('element_ids', [])
        objects = bindings.get(constraint['object_role'], {}).get('element_ids', [])
        relation = constraint['relation']
        if not subjects or not objects:
            status = 'unknown'
        elif all(any((s, relation, o) in facts for o in objects) for s in subjects):
            status = 'observed_true'
        elif any(s == a and relation == r and b not in objects for s in subjects for a, r, b in facts):
            status = 'observed_conflict'
        else:
            status = 'unknown'
        row = {'constraint_id': constraint['constraint_id'], 'status': status,
               'relation': relation, 'source': 'supplied_scene_graph_edges'}
        checks.append(row)
        if constraint.get('required', True) and status != 'observed_true' and subjects and objects:
            errors.append({**row, 'reason': 'relation_conflict' if status == 'observed_conflict' else 'relation_unverified'})
    return checks, errors
