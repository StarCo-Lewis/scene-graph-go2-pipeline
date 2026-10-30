"""Check initial robot near/held/lifted assertions, never assign model state."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'object_alignment_v1'))
from runtime_goal_protocol import STATE_RELATIONS, check_state_goal, validate_state_goal_set, INITIAL_VERSION


class RuntimeConstraintError(ValueError):
    def __init__(self, details):
        super().__init__('RUNTIME_INITIAL_CONSTRAINT_NOT_SATISFIED: ' + details['constraint_id'])
        self.details = details


def valid_position(position):
    return isinstance(position, (list, tuple)) and len(position) == 3 and all(
        type(v) in (int, float) and math.isfinite(v) for v in position)


def check_runtime_initial_constraints(alignment, snapshot, reach):
    checks = []
    constraints = alignment['task_decomposition'].get('initial_scene_constraints', [])
    bindings = alignment.get('context_bindings', {})
    backend = snapshot.get('backend', {})
    objects = backend.get('objects', {})
    active_ids = {int(k) for k in objects if str(k).isdigit()} if isinstance(objects,dict) else set()
    validate_state_goal_set(constraints)
    for constraint in constraints:
        if constraint['subject_role'] != 'robot':
            continue  # Object-to-object constraints remain in graph binding.
        if constraint['relation'] in STATE_RELATIONS:
            row = check_state_goal(constraint,bindings,snapshot,active_ids,'current_runtime_snapshot')
            row.update(time_scope='round_initial_state',initial_state_protocol=INITIAL_VERSION,
                       state_assignment_performed=False)
            checks.append(row)
            if row['required'] and row['status']!='confirmed_in_runtime_snapshot':
                raise RuntimeConstraintError({'code':'RUNTIME_INITIAL_CONSTRAINT_NOT_SATISFIED',
                                              **row,'checks':list(checks)})
            continue
        row = {'constraint_id': constraint['constraint_id'], 'relation': constraint['relation'],
            'object_role': constraint['object_role'], 'required': constraint.get('required', True),
            'status': 'unknown', 'source': 'current_runtime_snapshot',
            'coordinate_frame': 'habitat_world_y_up', 'reach_m': reach,
            'geometry_or_navigation_verified': False, 'distances_m': []}
        targets = bindings.get(constraint['object_role'], {}).get('element_ids', [])
        robot = backend.get('robot_position')
        centers = [backend.get('objects', {}).get(str(i), {}).get('center') for i in targets]
        if (constraint['relation'] == 'near' and targets and all(type(i) is int for i in targets)
                and valid_position(robot) and all(valid_position(p) for p in centers)
                and type(reach) in (int, float) and math.isfinite(reach) and reach > 0):
            row['distances_m'] = [math.hypot(robot[0] - p[0], robot[2] - p[2]) for p in centers]
            row['status'] = 'confirmed_in_runtime_snapshot' if all(d <= reach for d in row['distances_m']) else 'not_satisfied'
        checks.append(row)
        if row['required'] and row['status'] != 'confirmed_in_runtime_snapshot':
            raise RuntimeConstraintError({'code': 'RUNTIME_INITIAL_CONSTRAINT_NOT_SATISFIED',
                                          **row, 'checks': list(checks)})
    return checks
