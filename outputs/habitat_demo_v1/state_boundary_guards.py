"""Opt-in, conservative symbolic boundary guards; never observed robot state."""
import copy
import hashlib
import json
import math
import re


def clean_empty_annotations(plan):
    """Delete only empty non-executable comment/notes fields, never action slots.

    This is not a schema repair. Required blanks still reach the normal validator.
    Null state, false, zero, empty edges/dependencies and all arguments are untouched.
    """
    candidate = copy.deepcopy(plan)
    removed = []
    tasks = candidate.get('subtask_plans', [])
    if not isinstance(tasks, list):
        return candidate, removed
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        steps = task.get('steps', [])
        entries = [(f'subtask_plans[{i}]', task)]
        if isinstance(steps, list):
            entries += [(f'subtask_plans[{i}].steps[{j}]', step) for j,step in enumerate(steps)
                        if isinstance(step, dict)]
        for path, value in entries:
            for key in ('comment', 'notes'):
                if key in value and (value[key] is None or
                                     isinstance(value[key], str) and not value[key].strip()):
                    removed.append(f'{path}.{key}')
                    del value[key]
    return candidate, removed


def holding_constraint(text):
    # Deliberately narrow grammar. Unknown language/conditions are NOT no-ops.
    return bool(re.fullmatch(r'\s*(?:do not place it[.;,]\s*)?keep holding it[.;]?\s*',
                             text, re.IGNORECASE))


def attach_constraints(parts):
    """Attach known trailing invariants without dropping/reordering a single byte."""
    merged = []
    events = []
    for part in parts:
        if holding_constraint(part):
            if not merged:
                raise ValueError('CONSTRAINT_WITHOUT_PRECEDING_INTENT')
            events.append({'constraint': part, 'attached_to': merged[-1]})
            merged[-1] += part
        else:
            merged.append(part)
    if ''.join(merged) != ''.join(parts):
        raise ValueError('CONSTRAINT_TEXT_CHANGED')
    return merged, events


def state_facts(state, reach=1.5):
    if state.get('state_source') != 'symbolic_hypothesis_not_observation':
        raise ValueError('GUARD_REQUIRES_SYMBOLIC_STATE')
    backend = state['backend']
    position = backend['robot_position']
    def valid_vector(v):
        return (isinstance(v, list) and len(v) == 3
                and all(type(x) in (float, int) and math.isfinite(x) for x in v))
    if not valid_vector(position) or not math.isfinite(reach) or reach <= 0:
        raise ValueError('INVALID_ROBOT_POSITION_OR_REACH')
    near = []
    for key, obj in backend['objects'].items():
        if not valid_vector(obj['center']):
            raise ValueError('INVALID_OBJECT_POSITION')
        if math.hypot(position[0]-obj['center'][0], position[2]-obj['center'][2]) <= reach:
            near.append(int(key))
    return {'robot_position': copy.deepcopy(position), 'near_object_ids': sorted(near),
            'held_object_id': state['held_object_id'], 'lifted': state['lifted'],
            'memory_version': backend['memory_version'],
            'state_sha256': hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            'coordinate_frame': 'habitat_world_y_up', 'reach_m': reach,
            'source': 'program_computed_symbolic_not_observed',
            'rule': 'Proximity is a symbolic precondition, NOT navigability or physical success. '
                    'Read this snapshot; an API request/retry does not change it.'}


def check_holding_invariant(text, before, review):
    """Recognized keep-holding suffix is checked, not merely mentioned in a prompt."""
    if re.search(r'\bkeep holding it[.;]?\s*$', text, re.IGNORECASE) and before['held_object_id'] is not None:
        after = review['predicted_end_state']
        if (after['held_object_id'] != before['held_object_id']
                or after['lifted'] != before['lifted']
                or any(x['skill_id'] in ('place_object_on_surface', 'grasp_object')
                       for x in review['symbolic_trace'])):
            raise ValueError('KEEP_HOLDING_INVARIANT_VIOLATED')


def repair_placement_approach(descriptor, alignment, plan, library, nodes, state, reach=1.5):
    """Remove one proven redundant implicit approach from a placement-only leaf.

    Both original and candidate must pass the SAME reviewer and end identically.
    Never remove an explicit visit, an entire subtask, or an arbitrary empty plan.
    """
    from run_json_long_horizon import review_symbolic
    from match_skills import build_result
    original = review_symbolic(dict(descriptor, alignment=alignment, plan=plan),
                               library, nodes, state, reach)
    report = {'changed': False, 'rule': 'placement_only_equivalent_state',
              'state_facts': state_facts(state, reach)}
    text = descriptor['instruction']
    placement_only = re.fullmatch(
        r'\s*(?:and\s+)?(?:place|set)\s+(?:it|the (?:held |carried )?kettle(?:\s*\(?(?:ID )?\d+\)?)?)'
        r'\s+(?:there|on (?:that|the) (?:cabinet|sideboard))(?:\s+and release it)?[.]?\s*',
        text, re.IGNORECASE)
    if not placement_only or not state['lifted'] or state['held_object_id'] is None:
        return plan, original, report
    tasks = plan['subtask_plans']
    if len(tasks) != 1 or len(tasks[0]['steps']) != 3:
        return plan, original, report
    nav, prepare, place = tasks[0]['steps']
    if [x['skill_id'] for x in (nav, prepare, place)] != [
            'navigate_to_object', 'position_arm_for_object', 'place_object_on_surface']:
        return plan, original, report
    surface = place['arguments']['surface']['element_id']
    if (nav['arguments']['target']['element_id'] != surface
            or prepare['arguments']['target']['element_id'] != surface
            or place['arguments']['object']['element_id'] != state['held_object_id']
            or surface not in report['state_facts']['near_object_ids']):
        return plan, original, report
    proposal = {'subtask_plans': copy.deepcopy(tasks)}
    proposal['subtask_plans'][0]['steps'] = [
        {'skill_id': x['skill_id'], 'arguments': copy.deepcopy(x['arguments'])}
        for x in (prepare, place)]
    candidate = build_result(alignment, library, proposal)
    checked = review_symbolic(dict(descriptor, alignment=alignment, plan=candidate),
                              library, nodes, state, reach)
    if checked['predicted_end_state'] != original['predicted_end_state']:
        report['reason'] = 'candidate_changes_end_state; retained_original'
        return plan, original, report
    report.update(changed=True, removed_step=copy.deepcopy(nav),
                  original_navigation=original['navigation_targets'],
                  candidate_navigation=checked['navigation_targets'],
                  end_state_identical=True)
    return candidate, checked, report
