"""Explicit goal-to-executor-state adapter. No inferred actions or observations."""
import copy

STATE_RELATIONS = frozenset({'held', 'lifted'})
VERSION = 'proxy_state_goals_v1'
INITIAL_VERSION = 'proxy_initial_state_assertions_v1'
CONTRACT = {
    'version': VERSION,
    'state_goal_relations': ['held', 'lifted'],
    'subject_role': 'robot', 'object_role': 'one uniquely bound active scene object',
    'expected_value': 'JSON boolean; omitted means true; false negates only this object predicate',
    'mapping': {'held': 'held_object_id == bound_object_id',
                'lifted': 'held_object_id == bound_object_id AND lifted is true'},
    'initial_state_source': 'current_runtime_state, never scene edges or model claims',
    'initial_state_assertion_protocol': INITIAL_VERSION,
    'initial_state_relations': ['held', 'lifted'],
    'initial_state_assertion_rule': 'Optional initial_scene_constraints declarations are checked against the unchanged input executor snapshot AFTER grounding; never treated as graph facts or state updates.',
    'completion_source': 'executor snapshot after successful actions, not expected_end',
    'physical_grasp_verified': False,
}
PROMPT = '''Goal protocol proxy_state_goals_v1: keep on/in/near spatial goals. For
pickup-and-hold, emit robot held OBJECT_ROLE; for lift-and-hold also emit robot
lifted OBJECT_ROLE. Each uses constraint_id, subject_role="robot", relation,
object_role and required=true. Optional expected_value is a JSON boolean and
defaults to true. For completed placement use object on surface AND robot held
object with expected_value=false. False means THIS object is not held/lifted,
not an empty gripper or proof of placement. Never use near as grasp evidence.
held/lifted goals refer to the END state. Initial held/lifted conditions may also
be declared in initial_scene_constraints as robot held/lifted OBJECT_ROLE using
the same boolean expected_value schema. They are assertions to CHECK against the
supplied current_runtime_state after object binding, NOT state assignments or graph
edges. A false, missing or inconsistent required initial condition must be rejected;
do not add grasp/lift actions to make an initial condition true retroactively.
Initial state still comes only from current_runtime_state. No holding,
held_by, room or other invented goal relation. Keep-holding is not a new action.
State goals require exactly one grounded object; never fabricate bindings or
execution evidence. expected_end is an independent test assertion, not a substitute
for expressing the requested completion goals. These states are proxy bookkeeping,
not physical grasp/arm verification.'''


def normalize_state_goal(goal):
    if goal.get('relation') not in STATE_RELATIONS or goal.get('subject_role') != 'robot':
        raise ValueError('State goals require robot subject and held/lifted relation')
    if type(goal.get('expected_value', True)) is not bool:
        raise ValueError('State goal expected_value must be a JSON boolean')
    if type(goal.get('required', True)) is not bool:
        raise ValueError('State goal required must be a JSON boolean')
    if not isinstance(goal.get('object_role'), str) or not goal['object_role'].strip():
        raise ValueError('State goal requires a nonempty object_role')
    if set(goal) - {'constraint_id', 'subject_role', 'relation', 'object_role', 'required', 'expected_value'}:
        raise ValueError('Unknown state goal fields; no model-supplied execution evidence')
    return dict(goal, expected_value=goal.get('expected_value', True))


def bound_state_id(goal, bindings, active_ids=None):
    normalize_state_goal(goal)
    binding = bindings.get(goal.get('object_role'))
    ids = binding.get('element_ids') if isinstance(binding, dict) else None
    if (not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int
            or (active_ids is not None and ids[0] not in active_ids)):
        raise ValueError('State goal requires one uniquely bound active integer object ID')
    return ids[0]


def validate_state_goal_set(goals, bindings=None, active_ids=None):
    """Reject incompatible required states, including aliases after grounding."""
    expected = {}
    held_targets = set()
    for goal in goals:
        if goal.get('relation') not in STATE_RELATIONS:
            continue
        goal = normalize_state_goal(goal)
        target = (bound_state_id(goal, bindings, active_ids) if bindings is not None
                  else goal['object_role'])
        if not goal.get('required', True):
            continue
        relation, value = goal['relation'], goal['expected_value']
        key = (target, relation)
        if key in expected and expected[key] != value:
            raise ValueError('Conflicting required state goals')
        expected[key] = value
        if value:
            held_targets.add(target)
    if len(held_targets) > 1:
        raise ValueError('Single-object executor cannot hold/lift multiple required targets')
    for (target, relation), value in expected.items():
        if relation == 'lifted' and value and expected.get((target, 'held')) is False:
            raise ValueError('A lifted object must also be held; required goals conflict')


def check_state_goal(goal, bindings, snapshot, active_ids, source):
    """Read state explicitly; unknown never satisfies even a negative goal."""
    success = {'symbolic_hypothesis_not_observation': 'satisfied_under_symbolic_hypothesis',
               'read_only_runtime_prediction': 'satisfied_under_runtime_prediction',
               'proxy_executor_readback': 'observed_in_proxy',
               'current_runtime_snapshot': 'confirmed_in_runtime_snapshot'}
    if source not in success:
        raise ValueError('Unknown goal evidence source')
    check = {k: copy.deepcopy(goal.get(k)) for k in
             ('constraint_id', 'subject_role', 'relation', 'object_role')}
    check.update(required=goal.get('required', True), expected_value=goal.get('expected_value', True),
                 status='unknown', source=source, policy=VERSION,
                 physical_grasp_verified=False, physical_arm_motion_verified=False)
    try:
        target = bound_state_id(goal, bindings, active_ids)
    except ValueError as exc:
        return dict(check, reason=str(exc))
    if not isinstance(snapshot, dict) or 'held_object_id' not in snapshot:
        return dict(check, reason='missing_executor_held_state')
    held = snapshot['held_object_id']
    if held is not None and (type(held) is not int or held not in active_ids):
        return dict(check, reason='invalid_executor_held_state')
    lifted = snapshot.get('lifted')
    if ('lifted' in snapshot and type(lifted) is not bool) or (held is None and lifted is True):
        return dict(check, reason='invalid_or_inconsistent_executor_lifted_state')
    if goal['relation'] == 'lifted' and type(lifted) is not bool:
        return dict(check, reason='missing_executor_lifted_state')
    actual = held == target if goal['relation'] == 'held' else held == target and lifted is True
    check.update(bound_object_id=target, actual_value=actual,
                 executor_state={'held_object_id': held, 'lifted': lifted},
                 status=success[source] if actual == check['expected_value'] else 'not_satisfied',
                 reason='executor_state_matches' if actual == check['expected_value'] else 'executor_state_mismatch')
    return check
