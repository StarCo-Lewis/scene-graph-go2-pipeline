"""Experimental bounded recursive planning; symbolic hypotheses, never observations.

Only this opt-in module changes the planning strategy. Legacy leaf planner,
skill checks and original full-round test assertions remain unchanged.
"""
import copy
import json
import logging
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from run_json_long_horizon import prepare_round
from run_navigation_demo import read_json, write_json
from json_symbolic_review import review_symbolic
from plan_live_round import plan as plan_leaf
from match_skills import request_json_with_retry, build_result
from response_audit import ResponseAudit, validation
from build_report import build as build_audit
from preservation_constraints import (extract as extract_preservation, constraint_only,
    check_state as check_preservation_state, no_action_leaf)
from sequential_proxy_core import serial_state_contract


SPLIT_SYSTEM = '''Partition the CURRENT text into ordered smaller executable intents,
or mark it a leaf if no meaningful subdivision is needed. This is a JSON-only
symbolic planning experiment, not robot execution. Return ONLY either
{"decision":"leaf","reason":"..."} or
{"decision":"split","reason":"...","parts":["exact source substring", "remaining exact substring"]}.
For split, concatenating parts MUST reproduce current_text EXACTLY, including spaces
and punctuation. Do not rewrite, summarize, omit, duplicate or reorder any text.
Use 2 to 8 nonblank parts, each strictly smaller than the parent. Do not split words,
object names, IDs, negations, argument phrases, or 'do not place; keep holding' off
from the intent they constrain. Keep pickup-and-lift together as one handling intent.
Keep place-and-release together. A leaf is a small coherent executable intent:
one visit/carry destination, pickup-and-lift, or placement, with its state constraints.
Do not manufacture standalone no-action leaves for keep-holding or no-placement.
Several sequential visits or handling followed by carry/place should be subdivided.
Repeated visits requested by the user MUST survive. Parts may be grammatical fragments;
the original instruction and preceding validated state will be available later.
Only navigate_to_object, position_arm_for_object, grasp_object, lift_held_object,
place_object_on_surface are supported. Unsupported instructions must not be replaced.
Leave/Keep an object on its CURRENT support is a preservation constraint, NOT a
new pickup/place instruction. Keep such a sentence attached to the action it
constrains where possible. Explicit-ID standalone preservation clauses are checked
by the program without atomic actions; do not invent actions for them.
Use state only to interpret references; do not drop an explicitly requested action
because its outcome appears already satisfied. Never predict an endpoint yourself.'''

REVIEW_SYSTEM = '''Audit an entire original instruction against the actual proposed
leaf instructions and ordered atomic actions, with their symbolic begin/end states.
Do not propose repairs. All state is hypothetical. Check the full instruction,
including explicit IDs, negations, repeated visits, order, payload, destination,
holding/release and missing/extra actions. Source text coverage alone is NOT semantic
coverage. Necessary approach/preparation actions implied by skill preconditions are
allowed; unrelated extra tasks are not. Judge only this JSON planning experiment;
no claims of physical success.
Use executor_state_contract to interpret state fields. held_object_id is attachment;
lifted is the held object's lift status. prepared_target_id is ONLY a temporary
preparation target for a subsequent grasp/place, NOT a holding flag. Navigation
clears preparation but preserves held_object_id and lifted. A carried object may
legitimately end with held_object_id=H, lifted=true, prepared_target_id=null.
Do not call this release, infer broken holding, or require an extra preparation
action just to maintain holding. Use the actual actions and state transitions;
still reject real releases, wrong objects, missing/extra actions and constraints.
Program symbolic validation is supporting evidence, NOT independent semantic or
physical truth. Audit the original instruction yourself using this state contract.
Return {"complete":true/false,"order_preserved":true/false,
"constraints_preserved":true/false,"no_unrequested_actions":true/false,
"reason":"nonempty explanation"}. Return false when uncertain.'''


def validate_partition(value, text):
    if not isinstance(value, dict) or not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('Partition requires a nonempty reason')
    if value.get('decision') == 'leaf' and set(value) == {'decision', 'reason'}:
        return []
    if value.get('decision') != 'split' or set(value) != {'decision', 'reason', 'parts'}:
        raise ValueError('Invalid partition decision/schema')
    parts = value['parts']
    if (not isinstance(parts, list) or not 2 <= len(parts) <= 8
            or any(not isinstance(p, str) or not p.strip() or len(p) >= len(text) for p in parts)
            or ''.join(parts) != text):
        raise ValueError('Split must be an exact ordered nonempty partition of its source')
    return parts


def validate_coverage(value):
    keys = ('complete', 'order_preserved', 'constraints_preserved', 'no_unrequested_actions')
    if (not isinstance(value, dict) or any(type(value.get(k)) is not bool for k in keys)
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()):
        raise ValueError('Invalid whole-instruction semantic audit schema')
    if not all(value[k] for k in keys):
        raise ValueError('WHOLE_INSTRUCTION_SEMANTIC_REJECTED: ' + value['reason'])
    return value


def request_decision(args, folder, stage, system, payload, check):
    """Bounded schema correction, raw responses retained by the same audit transport."""
    key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not key:
        raise ValueError('DASHSCOPE_API_KEY required')
    previous = None
    for attempt in range(1, 4):
        body = dict(payload)
        if previous:
            body['previous_error'] = previous
        wire = {'model': args.model, 'temperature': 0, 'response_format': {'type': 'json_object'},
                'enable_thinking': False,
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': json.dumps(body, ensure_ascii=True)}]}
        req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(wire).encode(), headers={'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'}, method='POST')
        audit = ResponseAudit(folder, stage, {'validation_attempt': attempt}, req)
        # Transport exhaustion propagates; do not disguise it as schema correction.
        response = request_json_with_retry(req, timeout=180, attempts=2, audit=audit)
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        try:
            if response['choices'][0].get('finish_reason') == 'length':
                raise ValueError('Truncated structured response rejected')
            value = json.loads(content)
            audit.parsed(content)
            write_json(folder / f'{stage}_attempt_{attempt}.json', value)
            check(value)
            validation(folder, stage, attempt, 'accepted')
            return value
        except (ValueError, TypeError, KeyError) as exc:
            audit.parsed(content, 'rejected', exc)
            validation(folder, stage, attempt, 'rejected', exc)
            if str(exc).startswith('WHOLE_INSTRUCTION_SEMANTIC_REJECTED'):
                raise  # No repeat-until-the-judge-agrees.
            previous = str(exc)
            if attempt == 3:
                raise


def namespace(value, prefix):
    """Rename identifiers only; do not infer, correct, delete or add actions/goals."""
    if isinstance(value, list):
        return [namespace(v, prefix) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for k, v in value.items():
        if k in ('role_id', 'subject_role', 'object_role', 'subtask_id', 'constraint_id', 'step_id'):
            result[k] = v if k == 'subject_role' and v == 'robot' else prefix + v
        elif k in ('dependencies', 'required_role_ids'):
            result[k] = [prefix + x for x in v]
        else:
            result[k] = namespace(v, prefix)
    return result


def combine_leaves(instruction, leaves, library):
    """Leaf goals are checked at leaf boundaries, NOT all imposed on final state."""
    if not leaves:
        raise ValueError('No validated leaves; no empty success')
    decomposition = {'instruction': instruction, 'subtasks': [], 'initial_scene_constraints': [],
                     'goal_state': [], 'task_roles': []}
    alignment = {'status': 'success', 'context_bindings': {}, 'task_decomposition': decomposition,
                 'method': 'namespaced_validated_leaf_composition', 'milestone_scopes': []}
    proposals = []
    previous = None
    for index, leaf in enumerate(leaves, 1):
        if leaf.get('kind') == 'preservation_constraint':
            if (not constraint_only(leaf['instruction']) or
                    leaf['begin_state'] != leaf['end_state'] or leaf['plan']['subtask_plans']):
                raise ValueError('INVALID_PRESERVATION_LEAF')
            alignment['milestone_scopes'].append({'leaf': index, 'instruction': leaf['instruction'],
                'kind': 'preservation_constraint', 'constraint_report': leaf['constraint_report'],
                'checked_at': 'round_initial_state_and_every_action_boundary'})
            continue
        prefix = f'leaf_{index:03d}_'
        source = leaf['alignment']
        d = namespace(source['task_decomposition'], prefix)
        tasks = d['subtasks']
        if previous and tasks:
            tasks[0]['dependencies'] = list(dict.fromkeys(tasks[0].get('dependencies', []) + [previous]))
        previous = tasks[-1]['subtask_id'] if tasks else previous
        decomposition['subtasks'].extend(tasks)
        decomposition['task_roles'].extend(d.get('task_roles', []))
        if not proposals:
            decomposition['initial_scene_constraints'] = d.get('initial_scene_constraints', [])
        decomposition['goal_state'] = d['goal_state']
        alignment['context_bindings'].update({prefix+k: namespace(v, prefix)
                                             for k, v in source['context_bindings'].items()})
        alignment['milestone_scopes'].append({'leaf': index, 'instruction': leaf['instruction'],
            'initial_scene_constraints': d.get('initial_scene_constraints', []),
            'goal_state': d['goal_state'], 'checked_at': 'corresponding_leaf_boundary'})
        proposals.extend(namespace(leaf['plan']['subtask_plans'], prefix))
    plan = build_result(alignment, library, {'subtask_plans': proposals})
    if not proposals:
        if not constraint_only(instruction):
            raise ValueError('No executable leaves for an action-bearing instruction')
        plan['proposal_origin'] = 'program_validated_preservation_constraints'
    plan['preservation_segments'] = [leaf['constraint_report'] for leaf in leaves
                                     if leaf.get('kind') == 'preservation_constraint']
    return alignment, plan


def plan_hierarchical(args, folder, limits=None, decide=None, leaf_planner=None, state_guards=False, prepare=None):
    """Write the legacy round-result interface for explicit batch compatibility."""
    limits = limits or {'max_depth': 3, 'max_nodes': 23, 'max_leaves': 12}
    if any(type(limits.get(k)) is not int or limits[k] < 1 for k in ('max_depth','max_nodes','max_leaves')):
        raise ValueError('Positive integer hierarchy limits required')
    decide, leaf_planner = decide or request_decision, leaf_planner or plan_leaf
    prepare = prepare or prepare_round
    descriptor = read_json(folder/'round_input.json')
    initial = read_json(folder/'runtime_state.json')
    nodes = read_json(folder/'input_memory/scene/scene_graph.json')
    library = read_json(folder/'execution_skill_library.json')
    original = descriptor['instruction']
    state = copy.deepcopy(initial)
    leaves, tree = [], []
    summary = {'status': 'running', 'strategy': 'hierarchical', 'limits': limits,
        'input_memory_version': initial['backend']['memory_version'], 'nodes': tree,
        'validation_mode': 'json_symbolic_test', 'simulation_executed': False,
        'geometry_verified': False, 'training_eligible': False, 'fresh_llm_calls': True,
        'goal_scope_policy': 'leaf goals at leaf boundary; original round assertions at full-round end'}

    def visit(text, depth, ancestors, offset):
        nonlocal state
        if len(tree) >= limits['max_nodes']:
            raise ValueError('HIERARCHY_NODE_LIMIT')
        if len(leaves) >= limits['max_leaves']:
            raise ValueError('HIERARCHY_LEAF_LIMIT')
        index = len(tree)+1
        directory = folder/'hierarchy'/f'node_{index:03d}'
        row = {'node': index, 'depth': depth, 'source_span': [offset, offset+len(text)],
               'text': text, 'status': 'deciding', 'begin_state': copy.deepcopy(state)}
        tree.append(row)
        write_json(folder/'hierarchy_summary.json', summary)
        # A typed, fully grounded invariant is not an empty LLM action plan.
        # Do not call the leaf model or invent skills to make this segment nonempty.
        if constraint_only(text):
            leaf = no_action_leaf(text, state, nodes)
            leaves.append(leaf)
            row.update(status='constraint_validated_no_actions', leaf_number=len(leaves),
                       end_state=copy.deepcopy(state), constraint_report=leaf['constraint_report'])
            write_json(directory/'validated_leaf.json', leaf)
            write_json(folder/'hierarchy_summary.json', summary)
            return
        decision = decide(args, directory, 'hierarchy_split', SPLIT_SYSTEM,
            {'original_instruction': original, 'current_text': text, 'ancestor_texts': ancestors,
             'current_state': state, 'completed_segments': [x['instruction'] for x in leaves],
             'depth': depth, 'limits': limits}, lambda v: validate_partition(v, text))
        parts = validate_partition(decision, text)
        row['decision'] = decision
        if state_guards and parts:
            from state_boundary_guards import attach_constraints
            parts, repairs = attach_constraints(parts)
            row['constraint_attachment'] = repairs
            row['effective_parts'] = parts
            # A split into action + invariant can collapse back to its original leaf.
            if len(parts) == 1:
                if parts[0] != text:
                    raise ValueError('GUARDED_PARTITION_CHANGED_SOURCE')
                parts = []
        if parts:
            if depth >= limits['max_depth']:
                raise ValueError('HIERARCHY_DEPTH_LIMIT; no forced unsplit fallback')
            start = offset
            for part in parts:
                visit(part, depth+1, ancestors+[text], start)
                start += len(part)
            row.update(status='children_validated', end_state=copy.deepcopy(state))
        else:
            leaf_dir = directory/'planning'
            # No full-round answer is supplied as a leaf end-state requirement.
            child = {'round': descriptor['round'], 'instruction': text, 'expected_end': {}}
            prepare(leaf_dir, child, state, nodes, args.model, args.base_url)
            policy = read_json(leaf_dir/'execution_contract.json')
            policy['preservation_constraints'] = extract_preservation(original)
            policy['preservation_rule'] = (
                'These explicit-ID Leave/Keep-on constraints apply throughout this round. '
                'They preserve existing support and pose; do NOT navigate, prepare, grasp, lift '
                'or place a protected object to re-establish an already satisfied invariant. '
                'They are not executable subtasks. Plan only the requested actions on other '
                'objects. Every proposed action will be checked for violations, even when '
                'the final support relation would be restored.')
            if state_guards:
                from state_boundary_guards import state_facts
                facts = state_facts(state)
                facts['segment_revision'] = len(leaves)
                policy['computed_state_facts'] = facts
                write_json(leaf_dir/'robot_state_snapshot.json', facts)
            policy['hierarchy_context'] = {'original_instruction': original, 'source_span': row['source_span'],
                'active_segment': text, 'completed_segments': [x['instruction'] for x in leaves],
                'segment_revision': len(leaves),
                'rule': 'Execute ONLY the active segment; use the full instruction to resolve references and '
                        'inherited prohibitions, not to execute other segments. Preserve explicit visits. '
                        'Completion goals apply at this segment boundary; later segments have not run. '
                        'The runtime state is program-validated symbolic prediction, not observation.'}
            write_json(leaf_dir/'execution_contract.json', policy)
            leaf_library = read_json(leaf_dir/'execution_skill_library.json')
            leaf_library['execution_contract'] = policy
            write_json(leaf_dir/'execution_skill_library.json', leaf_library)
            with_log = logging.FileHandler(leaf_dir/'planning.console.log', encoding='utf-8')
            logging.getLogger().addHandler(with_log)
            try:
                code = leaf_planner(SimpleNamespace(output_dir=leaf_dir, round_input=leaf_dir/'round_input.json',
                    model=args.model, base_url=args.base_url, reach=1.5), reviewer=review_symbolic)
            finally:
                logging.getLogger().removeHandler(with_log)
                with_log.close()
            if code != 0:
                failure = read_json(leaf_dir/'planning_summary.json')
                raise ValueError(f'LEAF_PLANNING_FAILED node={index}: {failure.get("error")}')
            alignment = read_json(leaf_dir/'object_alignment.json')
            plan = read_json(leaf_dir/'skill_plan_final.json')
            if state_guards:
                from state_boundary_guards import clean_empty_annotations
                uncleaned = copy.deepcopy(plan)
                plan, removed_annotations = clean_empty_annotations(plan)
                write_json(leaf_dir/'empty_annotation_cleanup.json', {
                    'removed_paths': removed_annotations, 'required_fields_and_actions_untouched': True})
                if removed_annotations:
                    write_json(leaf_dir/'skill_plan_before_annotation_cleanup.json', uncleaned)
                    write_json(leaf_dir/'skill_plan_final.json', plan)
            review = review_symbolic(dict(child, alignment=alignment, plan=plan), leaf_library, nodes, state, 1.5)
            if state_guards:
                from state_boundary_guards import repair_placement_approach, check_holding_invariant
                original_plan = copy.deepcopy(plan)
                plan, review, guard_report = repair_placement_approach(
                    child, alignment, plan, leaf_library, nodes, state)
                check_holding_invariant(text, state, review)
                write_json(leaf_dir/'state_guard_report.json', guard_report)
                if guard_report['changed']:
                    write_json(leaf_dir/'skill_plan_before_state_guard.json', original_plan)
                    write_json(leaf_dir/'skill_plan_final.json', plan)
                    write_json(leaf_dir/'symbolic_review_after_state_guard.json', review)
            if not review['symbolic_trace']:
                raise ValueError('EMPTY_LEAF_PLAN')
            leaf = {'instruction': text, 'alignment': alignment, 'plan': plan,
                    'begin_state': copy.deepcopy(state), 'end_state': review['predicted_end_state']}
            leaves.append(leaf)
            state = copy.deepcopy(leaf['end_state'])
            row.update(status='leaf_validated', leaf_number=len(leaves), end_state=copy.deepcopy(state))
            write_json(directory/'validated_leaf.json', leaf)
        write_json(folder/'hierarchy_summary.json', summary)

    stage = 'hierarchical_partition_and_leaf_planning'
    try:
        constraints = extract_preservation(original)
        summary['preservation_constraints'] = check_preservation_state(constraints, initial, nodes)
        write_json(folder/'preservation_constraints.json', {
            'constraints': summary['preservation_constraints'],
            'scope': 'Explicit-ID Leave/Keep-on clauses; symbolic invariant, not observed truth'})
        visit(original, 0, [], 0)
        stage = 'hierarchical_whole_round_replay'
        alignment, plan = combine_leaves(original, leaves, library)
        write_json(folder/'object_alignment.json', alignment)
        write_json(folder/'skill_plan_final.json', plan)
        write_json(folder/'action_registry.json', {'identity':plan.get('action_identity'),
            'actions':plan.get('action_index',[]), 'execution_status':'not_executed'})
        write_json(folder/'task_decomposition.json', alignment['task_decomposition'])
        # Unchanged strict visits, hold/lift, support and repetition checks on original text.
        review = review_symbolic(dict(descriptor, alignment=alignment, plan=plan), library, nodes, initial, 1.5)
        if review['predicted_end_state'] != state:
            raise ValueError('COMPOSED_REPLAY_STATE_DIFFERS_FROM_LEAF_CHAIN')
        write_json(folder/'symbolic_review_final.json', review)
        stage = 'hierarchical_semantic_coverage'
        coverage_payload = {
            'executor_state_contract': serial_state_contract(),
            'symbolic_validation_evidence': {
                'status': review['status'], 'goal_checks': review['goal_checks'],
                'state_transitions': review['symbolic_trace'],
                'scope': 'Program replay under assumed effects; NOT an independent semantic verdict or physical observation'},
            'original_instruction': original, 'initial_state': initial, 'predicted_end_state': state,
            'segments': [{'instruction': x['instruction'], 'actions': x['plan']['subtask_plans'],
                          'kind': x.get('kind', 'executable'),
                          'constraint_report': x.get('constraint_report'),
                          'begin_state': x['begin_state'], 'end_state': x['end_state']} for x in leaves]}
        write_json(folder/'coverage_input.json', coverage_payload)
        decision = decide(args, folder/'coverage', 'hierarchy_coverage', REVIEW_SYSTEM,
            coverage_payload,
            validate_coverage)
        validate_coverage(decision)
        write_json(folder/'whole_instruction_review.json', decision)
        summary.update(status='planning_passed', leaf_count=len(leaves), node_count=len(tree))
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc), leaf_count=len(leaves), node_count=len(tree))
        logging.exception('Hierarchical round stopped; no fallback and no failed-leaf replay')
    finally:
        summary['constraint_segment_count'] = sum(x.get('kind') == 'preservation_constraint' for x in leaves)
        summary['executable_leaf_count'] = len(leaves) - summary['constraint_segment_count']
        write_json(folder/'hierarchy_summary.json', summary)
        write_json(folder/'planning_summary.json', summary)
        build_audit(folder, folder/'llm_audit_report')
    return 0 if summary['status'] == 'planning_passed' else 2
