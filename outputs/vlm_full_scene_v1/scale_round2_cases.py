"""Pre-outcome ordinal pilot labels; not a validated ambiguity measurement scale."""
import copy


def cases():
    return [
        ('O0','scene202','object_reference', 'Move the black hardcover book (ID 86) onto the office_B desk (ID 2) and release it.'),
        ('O1','scene202','object_reference', 'Move the black hardcover book from the living room onto the office_B desk (ID 2) and release it.'),
        ('O2','scene202','object_reference', 'Move the black hardcover book onto the office_B desk (ID 2) and release it.'),
        ('O3','scene202','object_reference', 'Move that thing onto the office_B desk (ID 2) and release it.'),
        ('L0','scene200','task_location', 'Move the white geometric ornament (ID 124) onto the office table (ID 133) and release it.'),
        ('L1','scene200','task_location', 'Move the white geometric ornament (ID 124) onto my desk and release it.'),
        ('L2','scene200','task_location', 'Move the white geometric ornament (ID 124) onto the table I have in mind and release it.'),
        ('L3','scene200','task_location', 'Move the white geometric ornament (ID 124) back to where I kept it yesterday and release it.'),
        ('G0','scene202','task_goal', 'Bring the silver-capped black bottle (ID 121) to the office_B desk (ID 2), place it on that desk and release it.'),
        ('G1','scene202','task_goal', 'Bring the silver-capped black bottle (ID 121) to the office_B desk (ID 2) for me.'),
        ('G2','scene202','task_goal', 'Bring the silver-capped black bottle (ID 121) to the office_B desk (ID 2). I want one specific delivery option: either set it down there or keep holding it there, but I have not told you which option I want.'),
        ('G3','scene202','task_goal', 'The silver-capped black bottle (ID 121) and the office_B desk (ID 2): take care of this for me.'),
        ('C0','scene202','constraints_preferences', 'Bring the black book (ID 86) and the black bottle (ID 121) to the office_B desk (ID 2), book first and bottle second. Place each on that desk and release it.'),
        ('C1','scene202','constraints_preferences', 'Bring the black book (ID 86) and the black bottle (ID 121) to the office_B desk (ID 2), in the same order as the earlier delivery described in our conversation. Place each on that desk and release it.'),
        ('C2','scene202','constraints_preferences', 'Bring the black book (ID 86) and the black bottle (ID 121) to the office_B desk (ID 2), in the order I currently prefer. Place each on that desk and release it.'),
        ('C3','scene202','constraints_preferences', 'Bring the black book (ID 86) and the black bottle (ID 121) to the office_B desk (ID 2). Place each on that desk and release it, following my personal handling rules that I have never told you.'),
        ('S0','scene202','object_state_fact_decision', 'For this textual test, assume that the silver-capped black bottle (ID 121) in office_A currently contains potable water. Based only on the supplied premises, decide whether it contains water and is eligible to be brought to my desk as a drink. Do not perform any action.'),
        ('S1','scene202','object_state_fact_decision', 'For this textual test, assume that the silver-capped black bottle (ID 121) in office_A was empty, was then filled with potable water, and has had no content added, removed, spilled or otherwise changed since filling. Based only on the supplied premises, decide whether it contains water and is eligible to be brought to my desk as a drink. Do not perform any action.'),
        ('S2','scene202','object_state_fact_decision', 'For this textual test, assume that the silver-capped black bottle (ID 121) in office_A contains a liquid, but it may be water or a different liquid whose identity and drinkability are unknown. Based only on the supplied premises, decide whether it contains water and is eligible to be brought to my desk as a drink. Do not perform any action.'),
        ('S3','scene202','object_state_fact_decision', 'No information about the contents of the silver-capped black bottle (ID 121) in office_A is supplied. Based only on the supplied premises, decide whether it contains water and is eligible to be brought to my desk as a drink. Do not perform any action.'),
    ]


def fixture_transform(case_id,cfg):
    result = copy.deepcopy(cfg)
    if case_id.startswith('C'):
        result['conversation_context']['dialogue_history'] = [{'speaker':'user', 'text':
            'For this test, assume this earlier completed episode: you delivered the black book '
            '(ID 86) before the black bottle (ID 121). This describes one earlier episode, '
            'not a standing instruction or a statement of my current preference.'}]
    return result


def evaluation_notes(case_id):
    axis,level = case_id[0],int(case_id[1])
    dimension = {'O':'object_reference','L':'task_location','S':'object_state',
                 'G':'task_goal','C':'constraints_preferences'}[axis]
    vector = dict(object_reference=0,task_location=0,object_state=0,task_goal=0,constraints_preferences=0)
    vector[dimension] = level
    if axis=='S':
        vector = {key:(level if key==dimension else None) for key in vector}
    result = {'case_id':case_id,'base_task_id':{'O':'scene202_book_to_desk2','L':'scene200_ornament_to_office',
              'G':'scene202_bottle_delivery','C':'scene202_two_object_delivery','S':'scene202_bottle_contents_question'}[axis],
        'target_dimension':dimension,'target_level':level,'draft_vector':vector,
        'label_status':'proposed_pre_outcome_not_human_adjudicated','scale_version':'ambiguity_annotation_v1',
        'track':'state_fact_decision' if axis=='S' else 'symbolic_planning',
        'single_factor_intended_not_guaranteed':True,'physical_execution':False,
        'unique_user_intent_identifiable':level <= 1,
        'semantic_success_reference_available':level <= 1,
        'high_level_stop_or_unknown_is_not_automatically_failure':level >= 2,
        'do_not_send_to_planner':True}
    if axis=='O':
        result.update(reference_support={'86':2} if level<=1 else None,
            category_candidates_before_capability_gate=[86] if level<=1 else
                ([86,129,140,142,154] if level==2 else None),
            confounds=['capability_inventory_restricted','book86_has_multiple_source_on_hypotheses'])
    elif axis=='L':
        result.update(reference_support={'124':133} if level<=1 else None,
            confounds=['destination_candidates_and_operable_destinations_are_not_identical',
                       'L3_adds_missing_temporal_location_reference'])
    elif axis=='G':
        result.update(reference_support={'121':2} if level==0 else None,
            acceptable_endings=(['placed_on_desk2'] if level==0 else
                                ['placed_on_desk2','held_and_lifted_near_desk2'] if level==1 else None),
            confounds=['G2_unspecified_preference_borders_constraint_ambiguity',
                       'G3_has_no_known_unique_goal_reference'])
    elif axis=='C':
        result.update(reference_support={'86':2,'121':2},
            reference_placement_order=[[86,2],[121,2]] if level<=1 else None,
            confounds=['C3_broadens_rule_scope_beyond_order','history_is_author_declared_not_observed'])
    else:
        result.update(unique_user_intent_identifiable=None,semantic_success_reference_available=True,
            premise_entails_water_reference='yes' if level<=1 else 'unknown',
            premise_entails_drinkability_reference='yes' if level<=1 else 'unknown',
            decision_reference='eligible_under_supplied_premise' if level<=1 else 'insufficient_information',
            robot_verified_reference=False,
            confounds=['textual_premises_not_perception','S0_S1_water_claim_only_conditional_on_premise',
                       'state_decision_track_is_not_whole_pipeline_success'])
    return result
