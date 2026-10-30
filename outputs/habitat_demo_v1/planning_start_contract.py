"""Public planning inputs are separate from private benchmark assertions."""
import copy


def public_round(descriptor):
    # Allowlist, not a blacklist: new test annotations must not silently leak.
    return {'round': descriptor['round'], 'instruction': descriptor['instruction']}


def conversation_context(cfg, nodes):
    context = cfg.get('conversation_context')
    if context is None:
        return None
    required = {'source', 'user_location', 'user_activity'}
    if not required <= set(context) or set(context)-required-{'dialogue_history','symbolic_dialogue_history','proxy_execution_history'}:
        raise ValueError('Unexpected conversation context fields')
    if context['source'] != 'scenario_assumption_not_observation':
        raise ValueError('User location must be explicitly marked as a scenario assumption')
    location = context['user_location']
    if set(location) != {'room_name', 'near_scene_element_id'}:
        raise ValueError('User location requires a room and an existing reference object')
    room = location['room_name']
    if room not in {r['name'] for r in cfg['rooms']}:
        raise ValueError('Unknown user room')
    identity = location['near_scene_element_id']
    if type(identity) is not int or not any(n['id'] == identity and n.get('active', True) for n in nodes):
        raise ValueError('Missing or inactive user location reference')
    if cfg['control_node_rooms'].get(str(identity)) != room:
        raise ValueError('User reference object is not assigned to the specified room')
    if not isinstance(context['user_activity'], str) or not context['user_activity'].strip():
        raise ValueError('Nonempty user activity required')
    if 'dialogue_history' in context and not isinstance(context['dialogue_history'],list):
        raise ValueError('Dialogue history must be an array')
    for turn in context.get('dialogue_history',[]):
        if (not isinstance(turn,dict) or set(turn) != {'speaker','text'} or turn['speaker'] != 'user'
                or not isinstance(turn['text'],str) or not turn['text'].strip()):
            raise ValueError('Invalid declared dialogue history')
    if 'symbolic_dialogue_history' in context:
        validate_symbolic_history(context['symbolic_dialogue_history'],nodes)
    if 'proxy_execution_history' in context:
        if 'symbolic_dialogue_history' in context:
            raise ValueError('Cannot mix accepted prediction and execution-feedback histories')
        validate_symbolic_history(context['proxy_execution_history'],nodes,execution_feedback=True)
    return copy.deepcopy(context)


def validate_symbolic_history(history,nodes,execution_feedback=False):
    """Typed accepted predictions, distinct from user speech and physical observations."""
    if not isinstance(history,list):
        raise ValueError('Symbolic dialogue history must be an array')
    active = {n['id'] for n in nodes if n.get('active',True)}
    state_keys = {'held_object_id','lifted','prepared_target_id','robot_room_hypothesis',
                  'robot_position','memory_version','objects'}
    previous = None
    for index,entry in enumerate(history,1):
        keys = {'round','instruction','source','begin_state','end_state','actions','binding_notes'}
        if execution_feedback:
            keys |= {'execution_status','error','action_results'}
        expected_source = ('proxy_execution_feedback_not_physical' if execution_feedback else
                           'accepted_symbolic_prediction_not_executed')
        if (not isinstance(entry,dict) or set(entry) != keys or type(entry['round']) is not int
                or entry['round'] != index or not isinstance(entry['instruction'],str)
                or not entry['instruction'].strip()
                or entry['source'] != expected_source):
            raise ValueError('Invalid symbolic dialogue history entry')
        if execution_feedback:
            if (entry['execution_status'] not in ('proxy_actions_completed','proxy_action_failed',
                    'proposal_unavailable') or not isinstance(entry['action_results'],list)
                    or (entry['error'] is not None and not isinstance(entry['error'],str))):
                raise ValueError('Invalid proxy execution history result')
        for label,version in [('begin_state',index-1),('end_state',index)]:
            state = entry[label]
            if (not isinstance(state,dict) or set(state) != state_keys
                    or type(state['memory_version']) is not int or state['memory_version'] != version
                    or type(state['lifted']) is not bool or not isinstance(state['objects'],dict)
                    or not isinstance(state['robot_position'],list) or len(state['robot_position']) != 3):
                raise ValueError('Invalid symbolic history state')
            for key in ('held_object_id','prepared_target_id'):
                if state[key] is not None and (type(state[key]) is not int or state[key] not in active):
                    raise ValueError('Invalid symbolic history object ID')
            for identity,obj in state['objects'].items():
                if (not isinstance(identity,str) or not identity.isdigit() or int(identity) not in active
                        or not isinstance(obj,dict) or set(obj) != {'center','support_id','room_hypothesis'}
                        or (obj['support_id'] is not None and (type(obj['support_id']) is not int
                                                               or obj['support_id'] not in active))):
                    raise ValueError('Invalid symbolic history object state')
        if previous is not None and entry['begin_state'] != previous:
            raise ValueError('Symbolic history continuity mismatch')
        previous = entry['end_state']
        if not isinstance(entry['actions'],list) or not isinstance(entry['binding_notes'],list):
            raise ValueError('Symbolic history actions/notes must be arrays')
        for action in entry['actions']:
            if (not isinstance(action,dict) or set(action) != {'skill_id','arguments'}
                    or not isinstance(action['skill_id'],str) or not isinstance(action['arguments'],dict)):
                raise ValueError('Invalid symbolic history action')
