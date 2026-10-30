"""Read-only runtime-state rehearsal for five Scene201 proxy skills, NOT physics."""
import copy
import math
from sequential_proxy_core import SerialExecutor, ExecutionRejected
from run_continuous_sequence import validate_episode
from semantic_redundancy import lift_repetition_policy, check_lift, SemanticRedundancyError
from runtime_constraints import check_runtime_initial_constraints
from runtime_goal_protocol import STATE_RELATIONS, check_state_goal


class PredictedBackend:
    active_ids = mapped_ids = {67, 13}

    def __init__(self, snapshot, reach):
        self.state = copy.deepcopy(snapshot['backend'])
        self.reach = reach
        self.support = {int(i): o['support_id'] for i, o in self.state['objects'].items()
                        if o['support_id'] is not None}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def require_near(self, target):
        a, b = self.state['robot_position'], self.state['objects'][str(target)]['center']
        if math.hypot(a[0] - b[0], a[2] - b[2]) > self.reach:
            raise ExecutionRejected('PREDICTED_PROXIMITY_REJECTED: navigate first')

    def require_graspable_instance(self, target):
        if target != 67:
            raise ExecutionRejected('INSTANCE_NOT_SUPPORTED_FOR_PROXY_GRASP')

    def require_placeable_pair(self, target, surface):
        if (target, surface) != (67, 13):
            raise ExecutionRejected('UNSUPPORTED_OBJECT_SURFACE_PAIR')

    def navigate(self, target, held):
        if target == held:
            raise ExecutionRejected('NAVIGATION_TO_HELD_OBJECT_UNDEFINED')
        center = self.state['objects'][str(target)]['center']
        self.state['robot_position'] = [center[0], .1, center[2]]
        if held is not None:
            self._carry(held)
        return {'status': 'arrived', 'target_id': target, 'prediction_only': True}

    def _carry(self, target):
        p = self.state['robot_position']
        self.state['objects'][str(target)]['center'] = [p[0], 1.43619, p[2]]
        self.state['objects'][str(target)]['support_id'] = None
        self.support.pop(target, None)

    def attach(self, target):
        self._carry(target)

    def lift(self, target):
        self._carry(target)

    def place(self, target, surface):
        self.require_placeable_pair(target, surface)
        self.support[target] = surface
        self.state['objects'][str(target)]['support_id'] = surface
        self.state['objects'][str(target)]['center'] = list(self.state['objects'][str(surface)]['center'])
        return {'status': 'proxy_instance_support_passed', 'prediction_only': True}


def review_live_plan(row, library, nodes, snapshot, reach):
    validate_episode({'rounds': [row]}, library, nodes)
    initial_runtime_checks = check_runtime_initial_constraints(row['alignment'], snapshot, reach)
    backend = PredictedBackend(snapshot, reach)
    executor = SerialExecutor(backend)
    for name in ('held_object_id', 'lifted', 'prepared_target_id'):
        setattr(executor, name, snapshot[name])
    trace, nav_targets = [], []
    policy = lift_repetition_policy(row['alignment']['task_decomposition']['instruction'])
    cycle_lifts, total_lifts, explicit_repeats = 0, 0, []
    for task in row['plan']['subtask_plans']:
        for step in task['steps']:
            if step['skill_id'] == 'lift_held_object':
                repeat = check_lift(step, executor, cycle_lifts, policy)
                if repeat:
                    explicit_repeats.append(repeat)
            executor.execute(step, row['alignment'])
            if step['skill_id'] in ('grasp_object', 'place_object_on_surface'):
                cycle_lifts = 0
            elif step['skill_id'] == 'lift_held_object':
                cycle_lifts += 1
                total_lifts += 1
            if step['skill_id'] == 'navigate_to_object':
                nav_targets.append(step['arguments']['target']['element_id'])
            trace.append({'step_id': step['step_id'], 'skill_id': step['skill_id'],
                          'status': 'conditionally_admissible_prediction', 'execution_status': 'not_executed'})
            if 'action_id' in step:
                trace[-1].update(action_id=step['action_id'],action_ordinal=step['action_ordinal'])
    if total_lifts < policy['min_lifts_this_round']:
        raise ValueError('Explicit repeated-lift request not covered; do not silently remove requested repetition')
    if any(getattr(executor, k) != v for k, v in row.get('expected_end', {}).items()):
        raise ValueError('Predicted held/lifted end state fails the instruction test assertion')
    if 'expected_navigation_targets' in row and nav_targets != row['expected_navigation_targets']:
        raise ValueError('Explicit visit order was not preserved')
    bindings = row['alignment']['context_bindings']
    goal_checks = []
    for goal in row['alignment']['task_decomposition']['goal_state']:
        if goal['relation'] in STATE_RELATIONS:
            check = check_state_goal(goal, bindings, executor.snapshot(), backend.active_ids,
                                     'read_only_runtime_prediction')
            goal_checks.append(check)
            if goal.get('required', True) and check['status'] != 'satisfied_under_runtime_prediction':
                raise ExecutionRejected('REQUIRED_STATE_GOAL_NOT_SATISFIED', check)
            continue
        objects = bindings[goal['object_role']]['element_ids']
        if goal['relation'] == 'on':
            subjects = bindings[goal['subject_role']]['element_ids']
            passed = bool(subjects) and all(backend.support.get(s) in objects for s in subjects)
        else:
            passed = False
            if len(objects) == 1:
                try:
                    backend.require_near(objects[0])
                    passed = True
                except ExecutionRejected:
                    pass
        goal_checks.append({'constraint_id': goal['constraint_id'], 'required': goal.get('required', True),
                            'status': 'satisfied_under_runtime_prediction' if passed else 'not_satisfied',
                            'source': 'read_only_runtime_prediction', 'physical_grasp_verified': False})
        if goal.get('required', True) and not passed:
            raise ValueError('Required completion goal is not covered by the predicted sequence')
    return {'status': 'conditionally_valid_proxy_sequence', 'symbolic_trace': trace,
            'goal_checks': goal_checks,
            'initial_runtime_constraint_checks': initial_runtime_checks,
            'semantic_redundancy': {'policy': policy, 'total_lift_calls': total_lifts,
                'explicit_proxy_repeat_calls': explicit_repeats,
                'scope': 'Fixed-height proxy lift only; not a general redundant-manipulation detector'},
            'initial_runtime_state': copy.deepcopy(snapshot), 'predicted_end_state': executor.snapshot(),
            'claim': 'Read-only rehearsal. Navigation and support are hypothetical until live execution.',
            'geometry_verified': False, 'physical_grasp_verified': False, 'training_eligible': False}
