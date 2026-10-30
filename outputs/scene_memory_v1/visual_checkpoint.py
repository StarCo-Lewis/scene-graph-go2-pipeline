"""Restore independent fixture truth across audited visual repairs, never from estimated poses."""
import copy
import math
from pathlib import Path
from update_memory import read, digest, path_key
from run_memory_followup_navigation import restore_contract


def fixture_contract(memory, previous=None, depth=0):
    if depth > 20:
        raise ValueError('Visual checkpoint chain too deep')
    state = read(memory / 'memory_state.json')
    if previous is None:
        latest = next(r['event'] for r in reversed(state['event_log'])
                      if r['event']['object_id'] == 67 and r['event']['operation'] == 'move')
        return restore_contract(state, read(memory / 'scene/scene_graph.json'),
            read(Path(state['base_scene']) / 'scene_graph.json'), read(Path(latest['evidence']['reference'])))
    evaluation = read(previous / 'regression_summary.json')
    if evaluation.get('status') != 'passed' or path_key(evaluation['repaired_memory']) != path_key(memory):
        raise ValueError('Previous visual run does not validate this memory')
    recorded = read(Path(evaluation['repaired_memory']) / 'memory_state.json')
    if digest(recorded) != digest(state):
        raise ValueError('Visual checkpoint state mismatch')
    from run_visual_repaired_navigation import validate_visual_repair
    old_memory = Path(evaluation['source_memory'])
    validate_visual_repair(read(old_memory / 'memory_state.json'), state,
                          read(previous / 'estimated_translation.json'), evaluation)
    if path_key(state['event_log'][-1]['event']['evidence']['reference']) != path_key(previous / 'estimated_translation.json'):
        raise ValueError('Visual checkpoint references a different estimate')
    ancestor = evaluation.get('previous_visual_run')
    contract = copy.deepcopy(fixture_contract(old_memory, Path(ancestor) if ancestor else None, depth+1))
    fixture = read(previous / 'after_capture/capture_evaluation_only.json')
    delta = fixture['test_translation_habitat']
    if fixture.get('status') != 'controlled_capture_ready' or [delta[0], -delta[2], delta[1]] != evaluation['evaluation_translation_cg']:
        raise ValueError('Checkpoint truth displacement mismatch')
    contract['kettle_com'] = [contract['kettle_com'][i]+delta[i] for i in range(3)]
    return contract


def actual_bbox_cg(contract):
    point = [contract['kettle_com'][i]+contract['bbox_minus_com_offset'][i] for i in range(3)]
    return [point[0], -point[2], point[1]]
