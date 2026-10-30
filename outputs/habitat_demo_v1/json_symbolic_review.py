"""Closed symbolic test world; NEVER an observation or simulator backend."""
import copy
import math
from sequential_proxy_core import SerialExecutor, ExecutionRejected, PARAMETERS, serial_state_contract
from runtime_constraints import check_runtime_initial_constraints, valid_position
from match_skills import validate_proposal
from action_sequence import validate_sequence
from semantic_redundancy import check_lift, lift_repetition_policy
from runtime_goal_protocol import STATE_RELATIONS, validate_state_goal_set, check_state_goal
from preservation_constraints import (for_review as preservation_for_review,
    check_state as check_preservation_state, check_action as check_preservation_action,
    constraint_only)


SYMBOLIC_GOAL_POLICY = 'symbolic_goal_near_horizontal_all_pairs_v2'


class SymbolicGoalError(ValueError):
    def __init__(self, check):
        self.details = copy.deepcopy(check)
        super().__init__('Required goal is not covered under symbolic effects: '
            f"{check.get('constraint_id')} {check.get('subject_role')} "
            f"{check.get('relation')} {check.get('object_role')}; "
            f"status={check['status']} reason={check.get('reason', 'not_satisfied')} "
            f"distances_m={check.get('distances_m', [])}")


def check_symbolic_near(goal, bindings, backend):
    """Closed-world, horizontal proximity; NEVER geometry or observation proof.

    Object roles require every bound subject to be near every bound target.
    Unknown/invalid coordinates, IDs or threshold never establish a goal.
    Robot goals retain the existing single-destination restriction.
    """
    check = {'constraint_id': goal.get('constraint_id'), 'relation': 'near',
        'subject_role': goal.get('subject_role'), 'object_role': goal.get('object_role'),
        'required': goal.get('required', True), 'status': 'unknown',
        'source': 'symbolic_hypothesis_not_observation', 'policy': SYMBOLIC_GOAL_POLICY,
        'coordinate_frame': 'habitat_world_y_up', 'distance_metric': 'horizontal_xz',
        'quantifier': 'all_subject_target_pairs', 'reach_m': backend.reach,
        'distances_m': [], 'pair_checks': [], 'geometry_or_navigation_verified': False}
    if type(backend.reach) not in (int, float) or not math.isfinite(backend.reach) or backend.reach <= 0:
        return dict(check, reason='invalid_positive_finite_reach')

    def ids(role):
        binding = bindings.get(role)
        values = binding.get('element_ids') if isinstance(binding, dict) else None
        if (not isinstance(values, list) or not values
                or any(type(i) is not int or i not in backend.active_ids for i in values)
                or len(set(values)) != len(values)):
            return None
        return values

    targets = ids(goal.get('object_role'))
    robot_subject = goal.get('subject_role') == 'robot'
    subjects = ['robot'] if robot_subject else ids(goal.get('subject_role'))
    if targets is None or subjects is None:
        return dict(check, reason='unresolved_or_invalid_binding')
    if robot_subject and len(targets) != 1:
        return dict(check, reason='robot_near_requires_single_destination')
    if not robot_subject and set(subjects).intersection(targets):
        return dict(check, reason='object_near_self_reference')

    def position(identity):
        if identity == 'robot':
            return backend.state.get('robot_position')
        value = backend.state.get('objects', {}).get(str(identity))
        return value.get('center') if isinstance(value, dict) else None

    positions = {identity: position(identity) for identity in subjects + targets}
    if not all(valid_position(p) for p in positions.values()):
        return dict(check, reason='missing_or_invalid_position')
    for subject in subjects:
        for target in targets:
            a, b = positions[subject], positions[target]
            distance = math.hypot(a[0]-b[0], a[2]-b[2])
            check['distances_m'].append(distance)
            check['pair_checks'].append({'subject_id': subject, 'object_id': target,
                'horizontal_distance_m': distance, 'within_reach': distance <= backend.reach})
    check['status'] = ('satisfied_under_symbolic_hypothesis' if all(
        p['within_reach'] for p in check['pair_checks']) else 'not_satisfied')
    check['reason'] = 'within_horizontal_reach' if check['status'].startswith('satisfied_') else 'outside_horizontal_reach'
    return check


class SymbolicBackend:
    def __init__(self, snapshot, nodes, reach, policy=None):
        if snapshot.get('state_source') != 'symbolic_hypothesis_not_observation':
            raise ValueError('Symbolic test state must be explicitly marked hypothetical')
        self.state = copy.deepcopy(snapshot['backend'])
        self.nodes = {n['id']: n for n in nodes}
        self.active_ids = self.mapped_ids = {n['id'] for n in nodes if n.get('active',True)}
        self.reach = reach
        # Legacy Scene201 restrictions remain the default. New scene fixtures
        # must opt in with explicit, hypothetical manipulation capabilities.
        configured = (policy or {}).get('symbolic_scene_capabilities')
        self.graspable_ids, self.placeable_pairs = {67}, {(67,13),(67,62)}
        self.configured_capabilities = configured is not None
        if configured is not None:
            if not isinstance(configured,dict):
                raise ValueError('Symbolic scene capabilities must be an explicit object')
            ids, pairs = configured.get('graspable_ids'), configured.get('placeable_pairs')
            if (not isinstance(ids,list) or not ids
                    or any(type(i) is not int or i not in self.active_ids for i in ids)
                    or len(set(ids)) != len(ids)
                    or not isinstance(pairs,list) or not pairs
                    or any(not isinstance(p,list) or len(p)!=2 or any(type(i) is not int for i in p)
                           or p[0] not in ids or p[1] not in self.active_ids or p[0]==p[1] for p in pairs)):
                raise ValueError('Invalid explicit symbolic scene capabilities')
            self.graspable_ids, self.placeable_pairs = set(ids), {tuple(p) for p in pairs}
        self.support = {int(i): obj['support_id'] for i, obj in self.state['objects'].items()
                        if obj.get('support_id') is not None}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def require_near(self, target):
        a, b = self.state['robot_position'], self.state['objects'][str(target)]['center']
        if math.hypot(a[0]-b[0], a[2]-b[2]) > self.reach:
            raise ExecutionRejected('SYMBOLIC_PROXIMITY_PRECONDITION: navigate first')

    def require_graspable_instance(self, target):
        if target not in self.graspable_ids:
            raise ExecutionRejected('UNSUPPORTED_SYMBOLIC_GRASP_ID' if self.configured_capabilities
                                    else 'SYMBOLIC_TEST_ONLY_KETTLE_CAN_BE_HELD')

    def require_placeable_pair(self, target, surface):
        if (target,surface) not in self.placeable_pairs:
            raise ExecutionRejected('UNSUPPORTED_SYMBOLIC_PLACEMENT_PAIR')

    def _mark_updated(self, target):
        if 'symbolically_updated_object_ids' in self.state:
            self.state['symbolically_updated_object_ids'] = sorted(set(
                self.state['symbolically_updated_object_ids']+[target]))

    def _carry(self, target):
        self._mark_updated(target)
        p = self.state['robot_position']
        self.state['objects'][str(target)].update(center=[p[0], 1.35, p[2]], support_id=None)
        self.state['objects'][str(target)]['room_hypothesis'] = self.state.get('robot_room_hypothesis')
        self.support.pop(target, None)

    def navigate(self, target, held):
        if target == held:
            raise ExecutionRejected('NAVIGATION_TO_HELD_OBJECT_UNDEFINED')
        center = self.state['objects'][str(target)]['center']
        # Assumed effect only: not a path, navmesh projection or physical pose command.
        self.state['robot_position'] = [center[0], .1, center[2]]
        self.state['robot_room_hypothesis'] = self.state['objects'][str(target)].get('room_hypothesis')
        if held is not None:
            self._carry(held)
        return {'status': 'arrived', 'effect_source': 'symbolic_hypothesis',
                'navigation_verified': False}

    def attach(self, target):
        self._carry(target)

    def lift(self, target):
        self._carry(target)

    def place(self, target, surface):
        self.require_placeable_pair(target, surface)
        self._mark_updated(target)
        center = list(self.state['objects'][str(surface)]['center'])
        center[1] += self.nodes[surface]['bbox_extent'][2]/2 + self.nodes[target]['bbox_extent'][2]/2
        self.state['objects'][str(target)].update(center=center, support_id=surface)
        self.state['objects'][str(target)]['room_hypothesis'] = self.state['objects'][str(surface)].get('room_hypothesis')
        self.support[target] = surface
        # Internal dispatcher sentinel only. The exported trace never calls this verified.
        return {'status': 'proxy_instance_support_passed', 'effect_source': 'symbolic_hypothesis',
                'support_verified': False}


def review_symbolic(row, library, nodes, snapshot, reach):
    alignment, plan = row['alignment'], row['plan']
    if (alignment.get('status') != 'success' or plan.get('instruction') != row['instruction']
            or alignment['task_decomposition']['instruction'] != row['instruction']):
        raise ValueError('Unresolved binding or changed original instruction')
    validate_proposal(plan, alignment, library)
    validate_sequence(plan)
    preservation = preservation_for_review(row['instruction'], library.get('execution_contract'))
    preservation_checks = check_preservation_state(preservation, snapshot, nodes)
    constraint_only_plan = (not plan['subtask_plans'] and constraint_only(row['instruction'])
                           and plan.get('proposal_origin') == 'program_validated_preservation_constraints')
    if not plan['subtask_plans'] and not constraint_only_plan:
        raise ValueError('EMPTY_PLAN_WITHOUT_VALIDATED_PRESERVATION_CONSTRAINTS')
    initial = check_runtime_initial_constraints(alignment, snapshot, reach)
    for predicate in initial:
        predicate['source'] = 'symbolic_hypothesis_not_observation'
        if predicate['status'] == 'confirmed_in_runtime_snapshot':
            predicate['status'] = 'satisfied_under_symbolic_hypothesis'
    backend = SymbolicBackend(snapshot, nodes, reach, library.get('execution_contract'))
    validate_state_goal_set(alignment['task_decomposition']['goal_state'],
                            alignment['context_bindings'], backend.active_ids)
    executor = SerialExecutor(backend)
    for k in ('held_object_id', 'lifted', 'prepared_target_id'):
        setattr(executor, k, snapshot[k])
    trace, visits = [], []
    policy = lift_repetition_policy(row['instruction'])
    cycle_lifts, total_lifts = 0, 0
    for task in plan['subtask_plans']:
        for step in task['steps']:
            state_before = {k: getattr(executor, k) for k in
                            ('held_object_id', 'lifted', 'prepared_target_id')}
            robot_before = copy.deepcopy(backend.state.get('robot_position'))
            room_before = backend.state.get('robot_room_hypothesis')
            held_before = executor.held_object_id
            if step['skill_id'] == 'lift_held_object':
                check_lift(step, executor, cycle_lifts, policy)
            check_preservation_action(preservation, step, executor.snapshot())
            executor.execute(step, alignment)
            check_preservation_state(preservation, executor.snapshot(), nodes, reference=snapshot)
            if step['skill_id'] in ('grasp_object', 'place_object_on_surface'):
                cycle_lifts = 0
            elif step['skill_id'] == 'lift_held_object':
                cycle_lifts += 1
                total_lifts += 1
            if step['skill_id'] == 'navigate_to_object':
                visits.append(step['arguments']['target']['element_id'])
            trace.append({'step_id': step['step_id'], 'skill_id': step['skill_id'],
                          'status': 'symbolic_preconditions_satisfied', 'execution_status': 'not_executed',
                          'executor_state_before': state_before,
                          'executor_state_after': {k: getattr(executor, k) for k in state_before}})
            if 'action_id' in step:
                trace[-1].update(action_id=step['action_id'],action_ordinal=step['action_ordinal'])
            if backend.configured_capabilities:
                trace[-1].update(robot_position_before=robot_before,
                    robot_position_after=copy.deepcopy(backend.state.get('robot_position')),
                    robot_room_before=room_before,robot_room_after=backend.state.get('robot_room_hypothesis'),
                    held_object_before=held_before,held_object_after=executor.held_object_id,
                    room_transition_source='author_assigned_regions_under_assumed_navigation_effect')
    if total_lifts < policy['min_lifts_this_round']:
        raise ValueError('Explicit repeated lift request was dropped')
    if any(getattr(executor, k) != v for k, v in row.get('expected_end', {}).items()):
        raise ValueError('Symbolic held/lifted outcome mismatch')
    if 'expected_navigation_targets' in row and visits != row['expected_navigation_targets']:
        raise ValueError('Explicit visit order was not preserved')
    if 'expected_support_id' in row and backend.support.get(67) != row['expected_support_id']:
        raise ValueError('Wrong requested destination support in symbolic state')
    for identity,support in row.get('expected_support_by_object',{}).items():
        if backend.support.get(int(identity)) != support:
            raise ValueError(f'Wrong requested symbolic support for object {identity}')
    expected_room = row.get('expected_robot_room')
    if expected_room is not None and backend.state.get('robot_room_hypothesis') != expected_room:
        raise ValueError('Wrong requested robot room under author-assigned symbolic regions')
    goals = alignment['task_decomposition']['goal_state']
    # Pure typed invariants have independently checked preservation predicates,
    # not invented action-completion goals. All action-bearing plans still need goals.
    if (not goals or not any(g.get('required', True) for g in goals)) and not (
            constraint_only_plan and preservation_checks):
        raise ValueError('Required completion goal missing')
    goal_checks = copy.deepcopy(preservation_checks) if constraint_only_plan else []
    for goal in goals:
        bindings = alignment['context_bindings']
        passed = False
        check = {'constraint_id': goal.get('constraint_id'),
            'subject_role': goal['subject_role'], 'object_role': goal['object_role'],
            'relation': goal['relation'], 'required': goal.get('required', True),
            'source': 'symbolic_hypothesis_not_observation', 'policy': SYMBOLIC_GOAL_POLICY,
            'status': 'unknown', 'reason': 'unsupported_goal_relation',
            'geometry_or_navigation_verified': False}
        if goal['relation'] in STATE_RELATIONS:
            check = check_state_goal(goal, bindings, executor.snapshot(), backend.active_ids,
                                     'symbolic_hypothesis_not_observation')
            passed = check['status'] == 'satisfied_under_symbolic_hypothesis'
        elif goal['relation'] == 'on':
            objects = bindings[goal['object_role']]['element_ids']
            subjects = bindings[goal['subject_role']]['element_ids']
            passed = bool(subjects) and all(backend.support.get(i) in objects for i in subjects)
            check.update(status='satisfied_under_symbolic_hypothesis' if passed else 'not_satisfied',
                         reason='symbolic_support_matches' if passed else 'symbolic_support_mismatch')
        elif goal['relation'] == 'near':
            check = check_symbolic_near(goal, bindings, backend)
            passed = check['status'] == 'satisfied_under_symbolic_hypothesis'
        goal_checks.append(check)
        if goal.get('required', True) and not passed:
            raise SymbolicGoalError(check)
    predicted = executor.snapshot()
    predicted['state_source'] = 'symbolic_hypothesis_not_observation'
    return {'status': 'symbolically_valid_not_executed', 'symbolic_trace': trace,
            'executor_state_contract': serial_state_contract(),
            'symbolic_goal_policy': SYMBOLIC_GOAL_POLICY, 'goal_checks': goal_checks,
            'initial_runtime_constraint_checks': initial, 'predicted_end_state': predicted,
            'navigation_targets': visits, 'simulation_executed': False,
            'preservation_constraint_checks': preservation_checks,
            'preservation_checked_after_each_action': bool(preservation),
            'geometry_verified': False, 'physical_grasp_verified': False,
            'training_eligible': False,
            'scope': 'Assumed skill effects; no proof of routes, room transitions, collision or support'}
