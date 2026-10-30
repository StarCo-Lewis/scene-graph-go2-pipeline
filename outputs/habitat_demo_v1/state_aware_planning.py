"""Opt-in runtime-object closure and error routing; never insert robot actions."""
import copy

VERSION = 'runtime_object_closure_routed_replan_v1'
RULES = (
    'Use the supplied current state when decomposing the current request. A current held object '
    'may be needed as an auxiliary role even if the user did not name it again. The program may '
    'expose its exact runtime ID as an auxiliary binding; this is availability, not permission '
    'to release or a prescribed action. Account for one-gripper capacity and any ordinary '
    'prerequisites of the requested task. Do not assume the gripper is empty. Decide any auxiliary '
    'actions and their surfaces from the current request, capabilities and constraints, not a '
    'hard-coded destination. Preserve explicit keep-holding/protection constraints. If no legal '
    'plan exists, remain unsuccessful rather than silently violating a constraint. Do not invent '
    'objects, observations, contents or execution evidence.')


def enabled(library):
    policy = library.get('execution_contract',{}).get('state_aware_planning')
    if policy is None:
        return False
    if (not isinstance(policy,dict) or policy.get('version')!=VERSION or
            policy.get('enabled') is not True or type(policy.get('max_runtime_redecompositions')) is not int
            or policy.get('max_runtime_redecompositions') not in (0,1)):
        raise ValueError('Unsupported state-aware planning policy')
    return True


def close_runtime_objects(alignment,library,snapshot,nodes):
    """Bind only the exact active held instance. Do not change LLM tasks/goals."""
    value = copy.deepcopy(alignment)
    record = dict(version=VERSION,enabled=enabled(library),added=[],
        source='runtime_snapshot_not_llm_binding',actions_inserted=False,
        scope='Parameter availability only, not semantic intent or release authorization')
    if not record['enabled']:
        return value,record
    held = snapshot.get('held_object_id')
    if held is None:
        return value,dict(record,status='no_held_object')
    active = {n['id']:n for n in nodes if n.get('active',True)}
    if type(held) is not int or held not in active or str(held) not in snapshot['backend']['objects']:
        raise ValueError('Held runtime object is missing or inactive; no auxiliary binding fabricated')
    declared = library.get('execution_contract',{}).get('current_runtime_state')
    if declared is not None and declared!=snapshot:
        raise ValueError('Runtime state differs from execution contract; refusing object closure')
    bindings = value.setdefault('context_bindings',{})
    existing = [role for role,b in bindings.items() if held in b.get('element_ids',[])]
    if existing:
        return value,dict(record,status='already_bound',held_object_id=held,existing_roles=existing)
    role = 'runtime_held_object'
    suffix = 1
    while role in bindings:
        role = f'runtime_held_object_{suffix}'
        suffix += 1
    bindings[role] = dict(element_ids=[held],source='runtime_snapshot_auxiliary_binding',
        reason='Exact current held_object_id, not a semantic guess or permission to release.',
        caption=active[held].get('caption',''))
    record.update(status='auxiliary_binding_added',held_object_id=held,
                  added=[dict(role_id=role,element_id=held)])
    return value,record


class RuntimeReplanRequired(ValueError):
    def __init__(self,details):
        self.details = details
        super().__init__(details['code']+': bounded state-aware re-decomposition required')


def route_runtime_error(error,library,snapshot,alignment,evidence):
    # Only a concrete, independently rejected capacity precondition is routed.
    # Schema, network, unsupported targets and semantic ambiguity are not silently reclassified.
    if (not enabled(library) or library['execution_contract']['state_aware_planning']['max_runtime_redecompositions']==0
            or str(error)!='GRIPPER_NOT_EMPTY'):
        return None
    return dict(kind='runtime_precondition_redecomposition',code=str(error),
        source='serial_executor_rejection_not_user_intent',evidence=evidence,
        current_held_object_id=snapshot.get('held_object_id'),
        current_lifted=snapshot.get('lifted'),
        available_bound_ids=sorted({i for b in alignment['context_bindings'].values() for i in b['element_ids']}),
        correction=(
            'The rejected proposal tried to grasp while another object was held. Re-decompose '
            'the SAME current instruction from the SAME original runtime snapshot; the rejected '
            'actions were not executed. Include any genuinely necessary auxiliary object roles '
            'and prerequisite subtasks. Choose no automatic placement target. Do not release an '
            'object contrary to an explicit holding/protection constraint; if the request is '
            'infeasible under the available skills, do not fabricate a substitute success. '
            'Return a complete new decomposition and bindings; the action codebook will be rebuilt.'))
