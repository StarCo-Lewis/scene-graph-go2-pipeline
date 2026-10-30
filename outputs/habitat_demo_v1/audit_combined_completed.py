"""Read-only completed-batch audit. No LLM calls, repairs or execution."""
import copy
import hashlib
import json
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    batch = read(root/'batch_summary.json')
    if batch['status'] != 'completed':
        raise ValueError('Audit requires a terminal completed batch')
    frozen = root/'frozen_engine'
    manifest = read(frozen/'engine_manifest.json')
    checks = {'frozen_files_unchanged': all(sha(frozen/p) == h for p, h in manifest['files'].items()),
              'case_files_unchanged': all(sha(root/'cases'/f'{n}.json') == h for n, h in batch['case_input_hashes'].items())}
    for directory in ('object_alignment_v1', 'skill_matching_v1', 'scene_memory_v1', 'llm_audit_v1', 'habitat_demo_v1'):
        sys.path.insert(0, str(frozen/'outputs'/directory))
    from json_symbolic_review import review_symbolic
    from token_usage_audit import usage
    results = []
    for trial in batch['trials']:
        folder = root/'trials'/f"{trial['case']}_{trial['repeat']}_{trial['arm']}"
        case = read(root/'cases'/f"{trial['case']}.json")
        summary = read(folder/'case_summary.json')
        good = {'round_count_and_status': len(summary['rounds']) == len(case['rounds']) and
                all(r['status'] == 'json_symbolic_passed' for r in summary['rounds']),
                'original_instructions': True, 'round_state_chain': True, 'segment_state_chain': True,
                'robot_snapshots_match': True, 'whole_replays_match': True, 'text_partition_lossless': True,
                'all_leaf_outputs_matrix': True, 'all_whole_instruction_reviews_accept': True,
                'round_inputs_unchanged': True, 'no_simulation_or_geometry_claim': True}
        previous = None
        detail = []
        for descriptor, round_row in zip(case['rounds'], summary['rounds']):
            index = descriptor['round']
            planning = folder/f'round_{index}/planning'
            snapshot = read(planning/'runtime_state.json')
            good['original_instructions'] &= descriptor['instruction'] == round_row['instruction'] == read(planning/'round_input.json')['instruction']
            good['round_state_chain'] &= snapshot == round_row['begin_state'] and snapshot['backend']['memory_version'] == index-1
            if previous is not None:
                good['round_state_chain'] &= snapshot == previous
            hierarchy = read(planning/'hierarchy_summary.json')
            leaves = sorted([n for n in hierarchy['nodes'] if n['status'] == 'leaf_validated'], key=lambda n: n['leaf_number'])
            good['text_partition_lossless'] &= ''.join(n['text'] for n in leaves) == descriptor['instruction']
            state = copy.deepcopy(snapshot)
            for n in leaves:
                leaf_dir = planning/'hierarchy'/f"node_{n['node']:03d}"
                leaf = read(leaf_dir/'validated_leaf.json')
                leaf_snapshot = read(leaf_dir/'planning/runtime_state.json')
                good['segment_state_chain'] &= leaf['begin_state'] == leaf_snapshot == state == n['begin_state']
                facts = read(leaf_dir/'planning/robot_state_snapshot.json')
                good['robot_snapshots_match'] &= all(facts[k] == state[k] for k in ('held_object_id', 'lifted'))
                good['robot_snapshots_match'] &= facts['robot_position'] == state['backend']['robot_position'] and facts['memory_version'] == index-1
                leaf_summary = read(leaf_dir/'planning/planning_summary.json')
                good['all_leaf_outputs_matrix'] &= leaf_summary['skill_output_format'] == 'matrix'
                state = leaf['end_state']
            row = dict(descriptor, alignment=read(planning/'object_alignment.json'), plan=read(planning/'skill_plan_final.json'))
            reviewed = review_symbolic(row, read(planning/'execution_skill_library.json'),
                                       read(planning/'input_memory/scene/scene_graph.json'), snapshot, 1.5)
            good['whole_replays_match'] &= reviewed['predicted_end_state'] == state
            state = copy.deepcopy(state)
            state['backend']['memory_version'] = index
            good['whole_replays_match'] &= state == round_row['end_state'] == read(folder/f'round_{index}/symbolic_projection_after.json')
            coverage = read(planning/'whole_instruction_review.json')
            good['all_whole_instruction_reviews_accept'] &= all(coverage[k] is True for k in ('complete', 'order_preserved', 'constraints_preserved', 'no_unrequested_actions'))
            good['round_inputs_unchanged'] &= read(planning/'round_input_integrity.json')['unchanged'] is True
            good['no_simulation_or_geometry_claim'] &= hierarchy['simulation_executed'] is False and hierarchy['geometry_verified'] is False and hierarchy['training_eligible'] is False
            nav = reviewed.get('navigation_targets', [])
            detail.append({'round': index, 'leaf_count': len(leaves), 'actions': len(reviewed['symbolic_trace']),
                           'actual_visit_order': nav, 'expected_visit_order': descriptor.get('expected_navigation_targets'),
                           'goal_checks': reviewed.get('goal_checks', []),
                           'held_object_id': state['held_object_id'], 'lifted': state['lifted'],
                           'support_id': state['backend']['objects']['67']['support_id']})
            previous = state
        u = usage(folder)
        good['provider_usage_matches_batch'] = all(u[k] == trial['token_usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'responses_missing_usage', 'transport_failures_unknown_usage'))
        good['matched_case_summary'] = trial['completed_round_count'] == len(detail) == summary['completed_round_count']
        results.append({'case': trial['case'], 'checks': good, 'all_checks_passed': all(good.values()),
                        'action_count': sum(r['actions'] for r in detail), 'round_details': detail,
                        'token_usage_by_stage': u['by_stage']})
    return {'checks': checks, 'trials': results, 'all_checks_passed': all(checks.values()) and all(r['all_checks_passed'] for r in results),
            'scope': 'Read-only replay using the frozen symbolic validator plus state/text/integrity checks; not independent semantic truth or physical execution'}


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    if sys.platform == 'win32':
        root = Path('\\\\?\\'+str(root))
    print(json.dumps(audit(root), ensure_ascii=False, indent=2))
