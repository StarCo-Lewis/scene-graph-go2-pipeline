"""Upstream plan -> validated motion templates -> feedback-driven JSON state.

No transport, path planner, coordinate registration or physical manipulation is
implemented here. Goal poses must come from an independently validated adapter.
"""
import argparse
import copy
import json
import math
from pathlib import Path


def pose(value):
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError('Pose must contain six values')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Pose values must be finite numbers')
    return list(value)


def compile_upstream(plan, nodes, goals, frame_id):
    """Preserve upstream intent; reject capabilities not covered by motion API."""
    if not frame_id or plan.get('query_achievable') is not True:
        raise ValueError('Explicit frame and achievable upstream plan required')
    active = {n['id'] for n in nodes if n.get('active', True)}
    refs = plan.get('final_relevant_objects')
    if not isinstance(refs, list) or any(x not in active for x in refs):
        raise ValueError('NO object can be used in this task')
    actions = plan.get('actions')
    if not isinstance(actions, list) or not actions:
        raise ValueError('Nonempty upstream actions required')
    templates = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError('Action must be an object')
        kind = action.get('action_type')
        params = action.get('params', {})
        if not isinstance(params, dict):
            raise ValueError('Action params must be an object')
        if kind not in ('move_to', 'approach_and_lock'):
            raise ValueError('Unsupported upstream action: ' + str(kind))
        if kind == 'approach_and_lock':
            if params.get('target_id') not in active:
                raise ValueError('NO object can be used in this task')
            # Never silently discard upstream crouch/lock requirements.
            raise ValueError('UNSUPPORTED_CAPABILITY: approach_and_lock needs posture/lock feedback beyond motion API')
        goal = goals.get(str(index))
        if not isinstance(goal, dict) or goal.get('frame_id') != frame_id or goal.get('validated') is not True:
            raise ValueError('Each action needs a validated goal in the execution frame')
        templates.append({'command_id': 'move_' + str(index), 'upstream_action_index': index,
                          'goal_pose': pose(goal.get('goal_pose')), 'frame_id': frame_id,
                          'pos_tol': 0.20, 'yaw_tol': 0.10,
                          'start_pose_source': 'latest_measured_robot_pose_at_dispatch'})
    return {'schema_version': '1.0', 'status': 'ready', 'upstream_plan': copy.deepcopy(plan),
            'motion_templates': templates, 'scene_update_policy': 'on_exception_or_scene_change',
            'physical_execution_verified': False}


def dispatch(template, state):
    if state.get('scene_update_required'):
        raise ValueError('Resolve exception before dispatch')
    robot = state.get('robot', {})
    if robot.get('frame_id') != template['frame_id'] or robot.get('localization_valid') is not True:
        raise ValueError('Fresh localization in the execution frame required')
    if robot.get('pose_fresh') is not True:
        raise ValueError('Robot pose is stale')
    start, goal = pose(robot.get('pose')), pose(template['goal_pose'])
    return {**template, 'start_pose': start, 'parameters': start + goal,
            'position_unit': 'm', 'angle_unit': 'rad'}


def apply_feedback(state, request, feedback):
    """Pure update. Transport must correlate the single outstanding command."""
    if feedback.get('command_id') != request['command_id'] or feedback.get('frame_id') != request['frame_id']:
        raise ValueError('Feedback command/frame mismatch')
    result = copy.deepcopy(state)
    seen = result.setdefault('feedback_by_command', {})
    if request['command_id'] in seen:
        if seen[request['command_id']] != feedback:
            raise ValueError('Conflicting feedback replay')
        return result
    actual = pose(feedback.get('actual_pose')) if feedback.get('actual_pose') is not None else None
    goal = request['goal_pose']
    position_error = math.dist(actual[:3], goal[:3]) if actual else None
    yaw_error = abs(math.atan2(math.sin(actual[5]-goal[5]), math.cos(actual[5]-goal[5]))) if actual else None
    reached = (feedback.get('status') == 'succeeded' and actual is not None
               and position_error <= request['pos_tol'] and yaw_error <= request['yaw_tol'])
    changed = feedback.get('scene_changed', False)
    result['scene_update_required'] = bool(result.get('scene_update_required') or not reached or changed)
    if actual:
        result['robot'] = {'pose': actual, 'frame_id': request['frame_id'],
                           'localization_valid': True, 'pose_fresh': False}
    else:
        result.setdefault('robot', {})['pose_fresh'] = False
    result.setdefault('action_results', []).append({'command_id': request['command_id'],
        'status': 'succeeded' if reached else 'failed', 'position_error_m': position_error,
        'yaw_error_rad': yaw_error, 'scene_update_required': result['scene_update_required']})
    seen[request['command_id']] = copy.deepcopy(feedback)
    # Navigation never moves/deletes objects. Verified manipulation events use
    # the existing scene_memory_v1/update_memory.py, not inferred goal effects.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream-plan', 'scene-graph', 'validated-goals', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frame-id', required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))
    result = compile_upstream(read(args.upstream_plan), read(args.scene_graph),
                              read(args.validated_goals), args.frame_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Output already exists; preserve previous result')
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')


if __name__ == '__main__':
    main()
