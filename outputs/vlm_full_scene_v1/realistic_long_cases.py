"""Ten independent six-turn tasks. Authored references are PRIVATE evaluation data.

No reference actions live here. Public planner input remains current instruction,
declared user context, actual proxy state and past execution feedback only.
"""
import copy

CASE_VERSION='realistic_long10_locations_v1_20261018'


def turn(text,placements=(),protect=(),held=None,visits=(),no_place=False):
    return dict(instruction=text,expected_placement_events=[list(p) for p in placements],
        expected_protected_ids=list(protect),expected_end={'held_object_id':held,'lifted':held is not None},
        expected_visits=[list(p) for p in visits],expected_no_placement=no_place)


def episode(key,title,scene,room,anchor,activity,focus,rounds):
    support=({'86':85,'121':101} if scene=='scene202' else {'124':110,'152':133,'230':165})
    held=None
    prepared=[]
    for n,row in enumerate(rounds,1):
        row=copy.deepcopy(row)
        row['round']=n
        row['expected_begin']={'held_object_id':held,'lifted':held is not None,
                               'support_by_object':copy.deepcopy(support)}
        for obj,surface in row['expected_placement_events']:
            support[str(obj)]=surface
        held=row['expected_end']['held_object_id']
        if held is not None:
            support[str(held)]=None
        row['expected_support_by_object']=copy.deepcopy(support)
        prepared.append(row)
    return dict(episode_id=key,title_en=title,scene=scene,
        tracked_object_ids=[86,121] if scene=='scene202' else [124,152,230],
        user_location={'room_name':room,'near_scene_element_id':anchor},user_activity=activity,
        diagnostic_focus=focus,rounds=prepared)


def episodes():
    return [
        episode('t01','Set up a working session','scene202','office_B',2,
            'The user is working at the writing desk in office_B.',
            ['multi_object_delivery','changed_source','historical_reference','protection'],[
            turn('I will be working here for a while. Bring the black hardcover book from the living room and the silver-capped black bottle from office_A to my desk. Put the book down first, then the bottle.',[(86,2),(121,2)]),
            turn('I need more room to read. Move the bottle to the living-room coffee table; leave the book here.',[(121,69)],[86]),
            turn('Bring the one you just moved back to my desk. Do not move the book.',[(121,2)],[86]),
            turn('Put the book on the coffee table for later, then come back here and pick up the bottle. Keep holding it.',[(86,69)],held=121),
            turn('Set it down on my desk first. Then bring the book back here and put it beside the bottle.',[(121,2),(86,2)]),
            turn('I am done with the bottle. Return it to the office_A desk it originally came from, then come back to my desk. Leave the book here.',[(121,101)],[86],visits=[(101,121),(2,None)]),
        ]),
        episode('t02','Prepare and clear a living-room meeting','scene202','living_room',69,
            'The user is sitting beside the living-room coffee table preparing for a meeting.',
            ['user_location_shift','delivery_order','interruption','selective_return'],[
            turn('Please bring the silver-capped black bottle from office_A to this coffee table, then put the black hardcover book from the living room here too.',[(121,69),(86,69)]),
            turn('The book is in the way. Put it on the writing desk in office_B, but keep the bottle here.',[(86,2)],[121]),
            turn('Actually, bring that book back here. Leave the bottle alone.',[(86,69)],[121]),
            turn('Take the bottle to the office_A desk and leave it there. Then return to this table without moving the book.',[(121,101)],[86],visits=[(101,121),(69,None)]),
            turn('Bring the bottle back from that desk. Put it on this table before taking the book to the office_B writing desk.',[(121,69),(86,2)]),
            turn('Leave the book in office_B. Put the bottle back on its original office_A desk, then wait by this coffee table.',[(121,101)],[86],visits=[(101,121),(69,None)]),
        ]),
        episode('t03','Borrow a book while working in office_A','scene202','office_A',101,
            'The user is working at the wooden desk in office_A.',
            ['different_user_office','holding_carry','original_source','delayed_reference'],[
            turn('Bring the black hardcover book from the living room to my desk. Leave the silver-capped black bottle already here untouched.',[(86,101)],[121]),
            turn('Move the bottle to the writing desk in office_B. Leave the book here, and return to my desk afterward.',[(121,2)],[86],visits=[(2,121),(101,None)]),
            turn('Take the book to the living-room coffee table and leave it there. Then get the bottle from office_B and bring it here, but keep holding it.',[(86,69)],held=121),
            turn('Put it down on my desk first, then fetch the book again and leave it here as well.',[(121,101),(86,101)]),
            turn('Put the book on the writing desk in office_B, then put the bottle on the coffee table. Do it in that order.',[(86,2),(121,69)]),
            turn('Bring the bottle back to the desk it came from at the beginning of our conversation. Then bring the book here too, leaving both on my desk.',[(121,101),(86,101)]),
        ]),
        episode('t04','Handle interruptions with an occupied gripper','scene202','office_B',2,
            'The user is seated at the writing desk in office_B and giving successive instructions.',
            ['occupied_gripper','auxiliary_placement','same_object_revisit','holding_state'],[
            turn('Bring me the silver-capped black bottle from office_A. Keep holding it beside my desk rather than putting it down.',held=121,visits=[(2,121)],no_place=True),
            turn('I also need the black hardcover book from the living room. Put it on my desk. You may set the bottle down on my desk to free your hand.',[(121,2),(86,2)]),
            turn('Pick up the bottle again and take it past the office_A desk and then the living-room coffee table. Keep holding it throughout; leave the book alone.',protect=[86],held=121,visits=[(101,121),(69,121)],no_place=True),
            turn('Bring it back here, put it on my desk, then take the book to the coffee table and leave it there.',[(121,2),(86,69)]),
            turn('Bring the book back and keep holding it by my desk. Do not move the bottle.',protect=[121],held=86,visits=[(2,86)],no_place=True),
            turn('Set the book down here first. Then return the bottle to the office_A desk and come back here.',[(86,2),(121,101)],visits=[(101,121),(2,None)]),
        ]),
        episode('t05','A thirsty reader with unknown bottle contents','scene202','living_room',69,
            'The user is reading beside the living-room coffee table. Bottle contents have not been observed.',
            ['unknown_contents','unsupported_inference','pronouns','negative_constraint'],[
            turn('I am thirsty. Bring the silver-capped black bottle from office_A to this table. I do not know whether anything is inside it.',[(121,69)],[86]),
            turn('Bring the black hardcover book over to this table too. Leave that bottle where it is.',[(86,69)],[121]),
            turn('Put the bottle on the writing desk in office_B for now; leave the book here.',[(121,2)],[86]),
            turn('Bring that same bottle back here and leave it on this table. Do not assume that carrying it has filled it.',[(121,69)],[86]),
            turn('Take the book to the office_A desk first, then take the bottle to the office_B writing desk.',[(86,101),(121,2)]),
            turn('Bring the book back to this table. Leave the bottle in office_B, and do not claim that I have had a drink.',[(86,69)],[121]),
        ]),
        episode('t06','Organize papers at the office desk','scene200','office',133,
            'The user is working with papers at the office table.',
            ['functional_reference','two_payloads','changed_support','protection'],[
            turn('I need something to stick these papers together. Bring the tape dispenser from the workshop workbench to my desk; leave my computer mouse here.',[(230,133)],[152,124]),
            turn('Move my mouse to the workshop workbench so I have more space for these papers. Keep the tape dispenser here.',[(152,165)],[230,124]),
            turn('Bring the one you just moved back here and put it on my desk. Do not move the tape dispenser.',[(152,133)],[230,124]),
            turn('Return the tape dispenser to its original workbench, then come back and pick up the mouse. Keep holding the mouse by my desk.',[(230,165)],[124],held=152),
            turn('Put it down on my desk first, then fetch the tape dispenser again and leave it here too.',[(152,133),(230,133)],[124]),
            turn('Return just the tape dispenser to the workshop workbench and come back here. Leave the mouse and the geometric ornament alone.',[(230,165)],[152,124],visits=[(165,230),(133,None)]),
        ]),
        episode('t07','Rearrange a display without losing the originals','scene200','display_room',110,
            'The user is standing beside the display pedestal arranging a small exhibition.',
            ['display_context','reversible_change','three_objects','original_locations'],[
            turn('Move the white geometric ornament from this pedestal to the office table for a moment, then bring the tape dispenser from the workshop workbench here and leave it on the pedestal.',[(124,133),(230,110)],[152]),
            turn('Put the tape dispenser back on its workbench. Then bring the ornament back to this pedestal.',[(230,165),(124,110)],[152]),
            turn('Bring the computer mouse from the office table to the workbench, then return here. Do not disturb the ornament or the tape dispenser.',[(152,165)],[124,230],visits=[(165,152),(110,None)]),
            turn('Bring that mouse here and leave it on the pedestal. Keep the ornament where it is.',[(152,110)],[124,230]),
            turn('Return the mouse to the office table it originally came from, then pick up the ornament and keep holding it by this pedestal.',[(152,133)],[230],held=124),
            turn('Put the ornament down here again. Then visit the office table, the workshop workbench, and this pedestal in that order without moving any more objects.',[(124,110)],[152,230],visits=[(133,None),(165,None),(110,None)]),
        ]),
        episode('t08','Borrow items at a workshop station','scene200','workshop',165,
            'The user is standing at the workshop workbench preparing a demonstration.',
            ['workshop_context','return_before_fetch','deferred_reference','holding'],[
            turn('Bring the computer mouse from the office to this workbench. Leave the tape dispenser already here alone.',[(152,165)],[230,124]),
            turn('The tape dispenser is in my way. Put it on the office table; keep the mouse here.',[(230,133)],[152,124]),
            turn('Bring the white geometric ornament from the display pedestal here and leave it on the workbench. Do not move the mouse.',[(124,165)],[152,230]),
            turn('Return the ornament to its display pedestal first, then bring the tape dispenser back here.',[(124,110),(230,165)],[152]),
            turn('Return the mouse to the office table. Then pick up the tape dispenser and keep holding it by this workbench.',[(152,133)],[124],held=230),
            turn('Set it down on the workbench first. Then fetch the mouse one last time, put it here, and leave the ornament untouched.',[(230,165),(152,165)],[124]),
        ]),
        episode('t09','Clear a desk and revise delivery priorities','scene200','office',133,
            'The user is seated at the office table making space for a demonstration.',
            ['three_object_priority','temporary_storage','repeated_roles','restoration'],[
            turn('Put my mouse on the workshop workbench. Then bring the white geometric ornament from the display pedestal to my desk, followed by the tape dispenser from the workbench.',[(152,165),(124,133),(230,133)]),
            turn('The ornament takes up too much room. Put it back on its display pedestal, then bring my mouse back here. Keep the tape dispenser here.',[(124,110),(152,133)],[230]),
            turn('Take the tape dispenser to the display pedestal and leave it there temporarily. Leave my mouse alone.',[(230,110)],[152,124]),
            turn('Bring the item you just took away back to my desk, then move the mouse to the workbench again.',[(230,133),(152,165)],[124]),
            turn('Return the tape dispenser to its original workbench first. Then bring the mouse back to my desk.',[(230,165),(152,133)],[124]),
            turn('Bring the ornament here for one final look, then return it to its original pedestal. Leave the mouse on my desk and the tape dispenser on its workbench.',[(124,133),(124,110)],[152,230]),
        ]),
        episode('t10','Gather and restore items after a demonstration','scene200','display_room',110,
            'The user is beside the display pedestal preparing and later clearing a demonstration.',
            ['gathering_order','cross_room_retrieval','carry_visit_order','source_memory'],[
            turn('Bring the mouse from the office table here first, then the tape dispenser from the workshop. Leave both on this pedestal without moving the ornament.',[(152,110),(230,110)],[124]),
            turn('Move the ornament to the office table for now. Then pick up the mouse and keep holding it here by the pedestal.',[(124,133)],[230],held=152,visits=[(110,152)]),
            turn('Put the mouse down here first. Take the tape dispenser back to the workbench, then bring the ornament back to this pedestal.',[(152,110),(230,165),(124,110)]),
            turn('Return the mouse to its original office table. Then fetch the tape dispenser again and leave it here.',[(152,133),(230,110)],[124]),
            turn('Pick up the tape dispenser. While holding it, visit the office table first and the workshop workbench second. Leave it on the workbench, then return here.',[(230,165)],[152,124],visits=[(133,230),(165,230),(110,None)]),
            turn('For a final check, visit the office table, the workshop workbench, and this pedestal in that order. Do not pick up or move anything.',protect=[124,152,230],visits=[(133,None),(165,None),(110,None)],no_place=True),
        ]),
    ]


def subsequence(required,actual):
    iterator=iter(actual)
    return all(any(event==want for event in iterator) for want in required)


def evaluate(turn_number,begin,end,plan,episode_id):
    """Post-rollout authored checks, never an independent semantic ground truth.

    Multiple correct action sequences are permitted; requested events must occur
    in order, but unrequested harmless intermediate placements are not disallowed.
    """
    ep=next(e for e in episodes() if e['episode_id']==episode_id)
    return evaluate_episode(turn_number,begin,end,plan,ep)


def evaluate_episode(turn_number,begin,end,plan,ep):
    """Same private diagnostic rules for an explicitly supplied episode specification."""
    row=ep['rounds'][turn_number-1]
    steps=[s for sub in plan['subtask_plans'] for s in sub['steps']]
    events=[[s['arguments']['object']['element_id'],s['arguments']['surface']['element_id']]
            for s in steps if s['skill_id']=='place_object_on_surface']
    checks={f'end_{key}':end[key]==value for key,value in row['expected_end'].items()}
    for obj,surface in row['expected_support_by_object'].items():
        checks['support_'+obj]=end['backend']['objects'][obj]['support_id']==surface
    checks['requested_placement_order']=subsequence(row['expected_placement_events'],events)
    if row['expected_no_placement']:
        checks['no_placement']=not events
    for obj in row['expected_protected_ids']:
        checks[f'protected_{obj}_state']=begin['backend']['objects'][str(obj)]==end['backend']['objects'][str(obj)]
        checks[f'protected_{obj}_no_manipulation']=not any(
            s['skill_id'] in ('grasp_object','lift_held_object','place_object_on_surface') and
            any(v.get('element_id')==obj for v in s['arguments'].values()) for s in steps)
    held=begin['held_object_id']
    visits=[]
    for step in steps:
        skill,args=step['skill_id'],step['arguments']
        if skill=='grasp_object': held=args['target']['element_id']
        elif skill=='place_object_on_surface': held=None
        elif skill=='navigate_to_object': visits.append([args['target']['element_id'],held])
    expected_begin=row['expected_begin']
    dependency=(all(begin[k]==expected_begin[k] for k in ('held_object_id','lifted')) and all(
        begin['backend']['objects'][obj]['support_id']==surface
        for obj,surface in expected_begin['support_by_object'].items()))
    return dict(reference_checks=checks,reference_checks_all_passed=all(checks.values()),
        dependency_reference_matched=dependency,ordered_placement_events=events,
        navigation_visits=visits,strict_id_visit_sequence_matched=subsequence(row['expected_visits'],visits),
        strict_visit_reference=row['expected_visits'],
        strict_visit_caveat='A co-located surface/object may satisfy geometry but not this strict ID reference; review manually.',
        semantic_quality='manual_review_required',contents_status='unknown_not_observed',
        scope='Private authored constraints conditional on earlier intended outcomes; not independent intent or physical truth')
