"""Two eight-turn episodes. Reference checks never enter public planning inputs."""
import copy
import math

CASE_VERSION = 'dialogue_chain_v2_clear_destination_20261018'


FOLLOWUPS = [
    'Could you also bring me the black hardcover book from the living room and leave it on my desk?',
    'Move the book from my desk to the coffee table in the living room. After leaving the book there, return here, pick up the bottle, and keep holding it.',
    'Put it down on my desk first. Then go and fetch that book again, and leave it here too.',
    'Leave the book here untouched. Take the bottle back to the desk in the office it originally came from.',
    'Bring back the one you just took away, and leave it on my desk this time.',
    'Take both the book and the bottle to the living-room coffee table. Leave the book there first, then the bottle.',
    'Now bring the book back to my desk. Leave the bottle where it is.',
]


def episodes():
    return [dict(episode_id=key,scene='scene202',rounds=[dict(round=n,instruction=text)
        for n,text in enumerate([start]+FOLLOWUPS,1)]) for key,start in [
            ('natural_start','I am thirsty. Could you get me something to drink?'),
            ('explicit_start','Bring the black bottle with the silver cap from office_A to my desk and leave it there.')]]


def compact_state(state,object_ids=(86,121)):
    value = {k:copy.deepcopy(state[k]) for k in ('held_object_id','lifted','prepared_target_id')}
    value.update({k:copy.deepcopy(state['backend'][k]) for k in
                  ('robot_room_hypothesis','robot_position','memory_version')})
    value['objects'] = {str(i):copy.deepcopy(state['backend']['objects'][str(i)]) for i in object_ids}
    return value


def evaluate(turn,begin,end,plan,episode_id):
    """Private diagnostic observations; do not repair or gate a public plan."""
    steps = [s for sub in plan['subtask_plans'] for s in sub['steps']]
    events = [[s['arguments']['object']['element_id'],s['arguments']['surface']['element_id']]
              for s in steps if s['skill_id']=='place_object_on_surface']
    checks = {}
    objects = end['backend']['objects']
    expected = {2:{86:2,121:2},3:{86:69},4:{86:2,121:2},5:{86:2,121:101},
                6:{86:2,121:2},7:{86:69,121:69},8:{86:2,121:69}}
    for identity,support in expected.get(turn,{}).items():
        checks[f'final_support_{identity}_is_{support}'] = objects[str(identity)]['support_id']==support
    if turn == 1:
        if episode_id == 'explicit_start':
            checks['bottle_on_user_desk'] = objects['121']['support_id']==2
            checks['released'] = end['held_object_id'] is None
        else:
            p,q = objects['121']['center'],objects['2']['center']
            checks['bottle_candidate_delivered_near_desk'] = (
                math.hypot(p[0]-q[0],p[2]-q[2])<=1.5 and
                (objects['121']['support_id']==2 or (end['held_object_id']==121 and end['lifted'])))
        checks['book_unchanged'] = objects['86']==begin['backend']['objects']['86']
    elif turn == 3:
        checks['bottle_still_held_lifted'] = end['held_object_id']==121 and end['lifted']
        checks['robot_back_in_user_office'] = end['backend']['robot_room_hypothesis']=='office_B'
        checks['book_placed_before_bottle_pickup'] = next((i for i,s in enumerate(steps)
            if s['skill_id']=='place_object_on_surface' and s['arguments']['object']['element_id']==86),len(steps)) < next(
            (i for i,s in enumerate(steps) if s['skill_id']=='grasp_object'
             and s['arguments']['target']['element_id']==121),-1)
    else:
        checks['released_at_end'] = end['held_object_id'] is None and not end['lifted']
    if turn == 4:
        checks['put_held_bottle_down_first'] = bool(events) and events[0]==[121,2]
    for identity in ({5:86,6:86,8:121}.get(turn),):
        if identity is not None:
            checks[f'protected_{identity}_no_manipulation'] = not any(
                s['skill_id'] in ('grasp_object','lift_held_object','place_object_on_surface') and
                any(isinstance(v,dict) and v.get('element_id')==identity for v in s['arguments'].values()) for s in steps)
            checks[f'protected_{identity}_state_unchanged'] = objects[str(identity)]==begin['backend']['objects'][str(identity)]
    if turn == 7:
        checks['book_then_bottle_placement'] = events==[[86,69],[121,69]]
    return dict(reference_checks=checks,reference_checks_all_passed=all(checks.values()),
        ordered_placement_events=events,semantic_quality='manual_review_required',
        unique_user_drink_intent_known=False if episode_id=='natural_start' else None,
        contains_water='unknown_not_observed',thirst_satisfied='not_established',
        scope='Private diagnostic constraints, not physical truth, not used for commit or planner feedback')
