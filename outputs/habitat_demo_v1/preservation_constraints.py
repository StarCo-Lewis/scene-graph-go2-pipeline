"""Explicit-ID preservation protocol; symbolic state, not perception truth.

Only complete Leave/Keep-the-object-(ID)-on-the-surface-(ID) sentences qualify.
Unrecognized language never grants a no-action exemption or guessed bindings.
"""
import copy
import re
from runtime_constraints import valid_position

VERSION = 'explicit_id_preservation_v1'
_CLAUSE = re.compile(
    r'\s*(?:Leave|Keep)\s+the\s+[^().!?;\n]+?\(ID\s+(?P<object>\d+)\)'
    r'\s+on\s+the\s+[^().!?;\n]+?\(ID\s+(?P<surface>\d+)\)\s*[.!]?\s*', re.I)


def extract(instruction):
    if not isinstance(instruction, str):
        raise ValueError('Preservation source must be text')
    constraints = []
    for sentence in re.finditer(r'[^.!?\n]+[.!?]?', instruction):
        match = _CLAUSE.fullmatch(sentence.group())
        if match:
            subject, surface = int(match['object']), int(match['surface'])
            if subject == surface:
                raise ValueError('PRESERVATION_SELF_SUPPORT_REJECTED')
            constraints.append({'version': VERSION, 'object_id': subject, 'surface_id': surface,
                'source_text': sentence.group(), 'source_span': list(sentence.span()),
                'kind': 'preserve_support_and_pose_without_manipulation'})
    return constraints


def constraint_only(instruction):
    constraints = extract(instruction)
    return bool(constraints) and ''.join(c['source_text'] for c in constraints).strip() == instruction.strip()


def for_review(instruction, policy):
    sources = [instruction]
    original = (policy or {}).get('hierarchy_context', {}).get('original_instruction')
    if original is not None and original != instruction:
        sources.append(original)
    result, seen = [], set()
    for source in sources:
        for item in extract(source):
            key = item['object_id'], item['surface_id']
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def check_state(constraints, snapshot, nodes, reference=None):
    active = {n['id'] for n in nodes if n.get('active', True)}
    objects = snapshot.get('backend', {}).get('objects', {})
    reports = []
    for item in constraints:
        subject, surface = item['object_id'], item['surface_id']
        if subject not in active or surface not in active:
            raise ValueError('PRESERVATION_BINDING_UNAVAILABLE')
        if snapshot.get('held_object_id') in (subject, surface):
            raise ValueError('PRESERVATION_OBJECT_OR_SURFACE_IS_HELD')
        obj, support = objects.get(str(subject)), objects.get(str(surface))
        if not isinstance(obj, dict) or not isinstance(support, dict) or obj.get('support_id') != surface:
            raise ValueError(f'PRESERVATION_CONSTRAINT_NOT_SATISFIED: object={subject} surface={surface}')
        for identity in (subject, surface):
            if not valid_position(objects[str(identity)].get('center')):
                raise ValueError('PRESERVATION_POSE_UNKNOWN')
            if reference is not None and objects[str(identity)] != reference['backend']['objects'][str(identity)]:
                raise ValueError(f'PRESERVATION_STATE_CHANGED: object={identity}')
        reports.append(dict(item, status='satisfied_under_symbolic_hypothesis', physical_verification=False))
    return reports


def check_action(constraints, step, snapshot):
    protected = {i for c in constraints for i in (c['object_id'], c['surface_id'])}
    skill, args = step['skill_id'], step['arguments']
    manipulated = None
    if skill == 'grasp_object':
        manipulated = args['target']['element_id']
    elif skill in ('lift_held_object', 'place_object_on_surface'):
        manipulated = args['object']['element_id']
    elif skill == 'navigate_to_object':
        manipulated = snapshot.get('held_object_id')
    if manipulated is not None and manipulated in protected:
        raise ValueError(f'PRESERVATION_MANIPULATION_FORBIDDEN: object={manipulated} skill={skill}')


def no_action_leaf(instruction, state, nodes):
    if not constraint_only(instruction):
        raise ValueError('Only explicitly recognized pure constraints permit no actions')
    checks = check_state(extract(instruction), state, nodes)
    alignment = {'status': 'success', 'context_bindings': {},
        'task_decomposition': {'instruction': instruction, 'subtasks': [], 'task_roles': [],
                              'initial_scene_constraints': [], 'goal_state': []}, 'method': VERSION}
    # Typed constraints, NOT invented empty atomic skill rows.
    return {'kind': 'preservation_constraint', 'instruction': instruction,
            'alignment': alignment, 'plan': {'subtask_plans': []},
            'constraint_report': {'version': VERSION, 'checks': checks, 'action_count': 0,
                'fresh_llm_calls': False, 'state_unchanged': True,
                'simulation_executed': False, 'training_eligible': False},
            'begin_state': copy.deepcopy(state), 'end_state': copy.deepcopy(state)}
