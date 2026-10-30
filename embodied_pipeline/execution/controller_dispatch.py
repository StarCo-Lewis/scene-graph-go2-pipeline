"""Offline controller-dispatch framework; importing it never opens a transport.

Symbolic engineering plans remain the input. Group payloads and physical results
are supplied by future adapters, never inferred from model goals or bbox centres.
The only runnable dispatcher in this version uses explicitly simulated adapters.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Protocol
import uuid

from .._legacy import ROOT, load

VERSION = 'controller_dispatch_v1'
DEFAULT_LIBRARY = ROOT / 'outputs/quadruped_atomic_skill_library.v0.1.json'
DEFAULT_ROUTES = Path(__file__).with_name('controller_routes.json')
OBJECT_TYPES = {'scene_element', 'location', 'surface', 'container', 'person', 'control'}
TERMINAL = {'succeeded', 'failed', 'rejected', 'cancelled', 'unknown'}
STATUSES = TERMINAL | {'accepted', 'running', 'cancel_requested'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be a nonempty string')
    return value


def check_arguments(arguments, spec):
    if not isinstance(arguments, dict):
        raise ValueError('Skill arguments must be an object')
    parameters = {p['name']: p for p in spec['parameters']}
    if set(arguments) - set(parameters):
        raise ValueError('Unknown arguments for ' + spec['skill_id'])
    for name, parameter in parameters.items():
        if name not in arguments:
            if parameter.get('required'):
                raise ValueError('Missing required argument: ' + name)
            continue  # Preserve omission; do not insert controller defaults.
        value = arguments[name]
        if parameter['type'] in OBJECT_TYPES:
            if (not isinstance(value, dict) or set(value) != {'role_id', 'element_id'}
                    or type(value['element_id']) is not int):
                raise ValueError('Invalid object binding: ' + name)
            text(value['role_id'], name + '.role_id')
        elif parameter['type'] in {'number', 'height_m'}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid numeric argument: ' + name)
        else:
            text(value, name)


def compile_requests(path, library_path=DEFAULT_LIBRARY, routes_path=DEFAULT_ROUTES):
    """Translate our existing envelope into internal dispatch requests, offline."""
    source = Path(path).resolve()
    raw = source.read_bytes()
    envelope = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(envelope, dict) or envelope.get('schema_version') != 'real_robot_proposal_envelope_v1':
        raise ValueError('Expected an existing real_robot_proposal_envelope_v1 engineering JSON')
    source_hash = digest(raw)
    library, routes = read(library_path), read(routes_path)
    if routes.get('schema_version') != 'controller_routes_v1':
        raise ValueError('Unsupported controller route schema')
    skills = {s['skill_id']: s for s in library['skills']}
    if len(skills) != len(library['skills']):
        raise ValueError('Duplicate catalogue skill ID')
    plan = envelope.get('plan')
    if not isinstance(plan, dict) or not isinstance(plan.get('subtask_plans'), list) or not plan['subtask_plans']:
        raise ValueError('Nonempty subtask_plans required')
    if plan.get('unplanned_subtasks'):
        raise ValueError('Unplanned subtasks remain; no partial dispatch compilation')
    sequence = load('action_sequence')
    sequence.validate_subtask_dependencies(plan['subtask_plans'])
    sequence.validate_sequence(plan)
    requests, steps, actions = [], set(), set()
    for task in plan['subtask_plans']:
        if not isinstance(task.get('steps'), list) or not task['steps']:
            raise ValueError('Every subtask must have at least one action')
        for step in task['steps']:
            sid = text(step.get('step_id'), 'step_id')
            if sid in steps:
                raise ValueError('Duplicate step_id: ' + sid)
            steps.add(sid)
            aid = step.get('action_id')
            if aid is not None:
                text(aid, 'action_id')
                if aid in actions:
                    raise ValueError('Duplicate action_id: ' + aid)
                actions.add(aid)
            skill_id = text(step.get('skill_id'), 'skill_id')
            if skill_id not in skills:
                raise ValueError('Unknown catalogue skill: ' + skill_id)
            check_arguments(step.get('arguments'), skills[skill_id])
            module_id = routes['skills'].get(skill_id)
            if module_id not in routes['modules']:
                raise ValueError('No registered controller route for ' + skill_id)
            command_id = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + ':' + source_hash + ':' + sid))
            requests.append({
                'schema_version': 'controller_command_v1', 'command_id': command_id,
                'source_plan_sha256': source_hash, 'source_action_id': aid,
                'source_step_id': sid, 'subtask_id': task['subtask_id'],
                'task_id': envelope.get('task_id'), 'scene_id': envelope.get('scene_id'),
                'ordinal': len(requests) + 1,
                'depends_on': [requests[-1]['command_id']] if requests else [],
                'subtask_dependencies': copy.deepcopy(task.get('dependencies', [])),
                'module_id': module_id, 'skill_id': skill_id,
                'skill_arguments': copy.deepcopy(step['arguments']),
                'scene_context': {k: copy.deepcopy(envelope.get(k)) for k in
                    ('coordinate_frame', 'unit', 'source_graph_sha256', 'source_edges_sha256')},
                'source_precondition_checks': copy.deepcopy(step.get('precondition_checks', [])),
                'catalogue_required_capabilities': copy.deepcopy(skills[skill_id].get('required_robot_capabilities', [])),
                'controller_payload': None, 'execution_authorized': False,
            })
    return {
        'schema_version': VERSION, 'status': 'dispatch_requests_compiled_not_executed',
        'source_plan': str(source), 'source_plan_sha256': source_hash,
        'source_status': envelope.get('status'),
        'source_required_before_dispatch': copy.deepcopy(envelope.get('required_before_dispatch', [])),
        'library_version': library['version'], 'library_sha256': digest(Path(library_path).read_bytes()),
        'routes_sha256': digest(Path(routes_path).read_bytes()),
        'modules': copy.deepcopy(routes['modules']), 'commands': requests,
        'validation_scope': 'Structure, declared arguments and source order only; no scene feasibility or physical validation.',
        'robot_executed': False, 'memory_committed': False,
    }


class ControllerAdapter(Protocol):
    """Group integration seam. These dictionaries are OUR contract, not a group API.

    prepare: bind and validate a group payload without sending it.
    submit: send once and return acknowledgement/result with command_id.
    poll: obtain correlated feedback; None means no new observation.
    cancel: request cancellation; acknowledgement alone is not stopped/completed.
    Real adapters must own durable reservation, transport timeout and raw evidence.
    """
    module_id: str
    mode: str

    def prepare(self, command: Dict[str, Any], binding: Dict[str, Any]) -> Dict[str, Any]: ...
    def submit(self, command: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]: ...
    def poll(self, command_id: str) -> Optional[Dict[str, Any]]: ...
    def cancel(self, command_id: str) -> Dict[str, Any]: ...


class PendingGroupAdapter:
    mode = 'interface_pending'

    def __init__(self, module_id, module_spec):
        self.module_id, self.spec = module_id, copy.deepcopy(module_spec)

    def prepare(self, command, binding):
        return {'command_id': command['command_id'], 'module_id': self.module_id,
                'status': 'interface_pending', 'payload': None,
                'required_from_group': self.spec['required_from_group'],
                'known_interface': copy.deepcopy(self.spec.get('known_interface')),
                'runtime_binding_supplied': bool(binding),
                'reason': 'Group payload mapping and feedback semantics are not connected to this dispatcher.'}

    def submit(self, command, prepared):
        raise NotImplementedError(self.module_id + ': controller interface pending; no command sent')

    def poll(self, command_id):
        raise NotImplementedError(self.module_id + ': no controller feedback connection')

    def cancel(self, command_id):
        raise NotImplementedError(self.module_id + ': no controller cancellation connection')


class NavigationAdapterSlot(PendingGroupAdapter):
    """Reuse the established single-navigation converter when explicitly supplied.

    This slot NEVER imports ROS2/TCP clients, opens a connection or sends a goal.
    Multi-action plans await a group-approved per-action runtime binding contract.
    """
    def prepare(self, command, binding):
        result = super().prepare(command, binding)
        config = binding.get('navigation_replay_config')
        if config is not None:
            from .navigation_contract import prepare
            request = prepare(copy.deepcopy(config))
            if (request['source_plan_sha256'] != command['source_plan_sha256'] or
                    request['source_action_id'] != command['source_action_id']):
                raise ValueError('Navigation runtime binding belongs to a different plan/action')
            result['existing_navigation_preview'] = request
            result['payload'] = copy.deepcopy(request['goal'])
        return result


def preview(batch, bindings=None):
    bindings = {} if bindings is None else bindings
    known = {c['command_id'] for c in batch['commands']}
    if not isinstance(bindings, dict) or set(bindings) - known:
        raise ValueError('Runtime bindings must be keyed by current command_id')
    prepared = []
    for command in batch['commands']:
        cls = NavigationAdapterSlot if command['module_id'] == 'navigation' else PendingGroupAdapter
        adapter = cls(command['module_id'], batch['modules'][command['module_id']])
        binding = bindings.get(command['command_id'], {})
        if not isinstance(binding, dict):
            raise ValueError('Each runtime binding must be an object')
        prepared.append(adapter.prepare(copy.deepcopy(command), copy.deepcopy(binding)))
    return {**copy.deepcopy(batch), 'preparations': prepared,
            'status': 'interfaces_pending_not_executed', 'ready_for_robot_execution': False}


class SimulationState:
    """Single-process, serial state machine for testing the dispatch contract only.

    Acknowledgement is not completion. Failures/unknown outcomes stop the prefix.
    There is no live mode, persistence, automatic retry, resume or memory commit.
    """
    def __init__(self, batch):
        self.commands = copy.deepcopy(batch['commands'])
        self.index, self.active, self.halted = 0, None, False
        self.state, self.events = {}, []

    def reserve_next(self):
        if self.active is not None or self.halted or self.index == len(self.commands):
            return None
        command = self.commands[self.index]
        if any(self.state.get(dep) != 'succeeded' for dep in command['depends_on']):
            raise ValueError('A dependency has not completed successfully')
        self.active = command['command_id']
        self.state[self.active] = 'submitted'
        return copy.deepcopy(command)

    def record(self, event):
        if (not isinstance(event, dict) or self.active is None or event.get('command_id') != self.active
                or event.get('mode') != 'simulation' or event.get('status') not in STATUSES
                or event.get('module_id') != self.commands[self.index]['module_id']):
            raise ValueError('Uncorrelated, invalid or non-simulation controller feedback')
        status, previous = event['status'], self.state[self.active]
        allowed = {
            'submitted': STATUSES,
            'accepted': STATUSES - {'accepted'},
            'running': STATUSES - {'accepted'},
            'cancel_requested': TERMINAL | {'cancel_requested'},
        }
        if status not in allowed.get(previous, set()):
            raise ValueError('Invalid feedback state transition')
        self.state[self.active] = status
        self.events.append({**copy.deepcopy(event), 'recorded_at_utc': stamp(),
                            'robot_executed': False})
        if status in TERMINAL:
            self.halted = status != 'succeeded'
            self.index += 1
            self.active = None

    def report(self):
        completed = not self.halted and self.index == len(self.commands)
        return {'schema_version': 'controller_feedback_v1', 'mode': 'simulation',
                'status': 'simulation_completed' if completed else 'simulation_halted' if self.halted else 'simulation_in_progress',
                'events': copy.deepcopy(self.events), 'command_states': dict(self.state),
                'unstarted_command_ids': [c['command_id'] for c in self.commands if c['command_id'] not in self.state],
                'robot_executed': False, 'memory_committed': False, 'observed_robot_state': None}


class SimulatedGroupAdapter:
    """Synthetic feedback, not a robot, navigation simulation or success evidence."""
    mode = 'simulation'

    def __init__(self, module_id):
        self.module_id, self.pending = module_id, {}

    def event(self, command_id, status):
        return {'command_id': command_id, 'module_id': self.module_id, 'mode': self.mode,
                'status': status, 'evidence': {'source': 'synthetic_framework_fixture'}}

    def prepare(self, command, binding):
        return {'status': 'simulation_ready', 'command_id': command['command_id'],
                'payload': {'symbolic_arguments': copy.deepcopy(command['skill_arguments']), 'simulation_only': True}}

    def submit(self, command, prepared):
        cid = command['command_id']
        if cid in self.pending:
            raise ValueError('Duplicate simulated submission')
        self.pending[cid] = ['running', 'succeeded']
        return self.event(cid, 'accepted')

    def poll(self, command_id):
        states = self.pending[command_id]
        return self.event(command_id, states.pop(0)) if states else None

    def cancel(self, command_id):
        self.pending[command_id] = ['cancelled']
        return self.event(command_id, 'cancel_requested')


def simulate(batch, adapters=None, max_polls=10):
    if type(max_polls) is not int or max_polls < 1:
        raise ValueError('max_polls must be a positive integer')
    adapters = adapters if adapters is not None else {
        mid: SimulatedGroupAdapter(mid) for mid in {c['module_id'] for c in batch['commands']}}
    # Preflight the entire inventory before any adapter method is called.
    for command in batch['commands']:
        adapter = adapters.get(command['module_id'])
        if adapter is None or adapter.mode != 'simulation' or adapter.module_id != command['module_id']:
            raise ValueError('This runner only accepts matching, explicit simulation adapters')
    state = SimulationState(batch)
    while True:
        command = state.reserve_next()
        if command is None:
            break
        adapter = adapters[command['module_id']]
        try:
            prepared = adapter.prepare(copy.deepcopy(command), {})
            if prepared.get('status') != 'simulation_ready':
                state.record({'command_id': command['command_id'], 'module_id': command['module_id'],
                              'mode': 'simulation', 'status': 'rejected', 'reason': 'Adapter preparation was not ready.'})
                continue
            state.record(adapter.submit(copy.deepcopy(command), prepared))
            for _ in range(max_polls):
                if state.active is None:
                    break
                feedback = adapter.poll(command['command_id'])
                if feedback is not None:
                    state.record(feedback)
            if state.active is not None:
                state.record({'command_id': command['command_id'], 'module_id': command['module_id'],
                              'mode': 'simulation', 'status': 'unknown', 'reason': 'Simulation polling budget exhausted; no retry.'})
        except Exception as exc:
            if state.active is None:
                raise
            state.record({'command_id': command['command_id'], 'module_id': command['module_id'],
                          'mode': 'simulation', 'status': 'unknown', 'reason': str(exc)})
    return {**state.report(), 'source_plan_sha256': batch['source_plan_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Preview controller dispatch; optionally exercise synthetic feedback. No robot connection.')
    parser.add_argument('--engineering-plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are not overwritten')
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--routes', type=Path, default=DEFAULT_ROUTES)
    parser.add_argument('--runtime-bindings', type=Path)
    parser.add_argument('--simulate', action='store_true', help='Synthetic state-machine demonstration only')
    args = parser.parse_args(argv)
    batch = compile_requests(args.engineering_plan, args.library, args.routes)
    result = preview(batch, read(args.runtime_bindings) if args.runtime_bindings else None)
    trace = simulate(batch) if args.simulate else None
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write('dispatch_preview.json', result)
    if trace is not None:
        write('simulation_trace.json', trace)
    print(json.dumps({'status': result['status'], 'commands': len(batch['commands']),
                      'simulation_status': trace['status'] if trace else None,
                      'robot_executed': False, 'output': str(args.out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
