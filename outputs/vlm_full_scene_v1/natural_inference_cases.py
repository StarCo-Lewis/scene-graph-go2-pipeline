"""Paired English diagnostic tasks; evaluation notes never enter planner inputs."""


def cases():
    return [
        ('A1','scene200','relation_and_context',
         "Bring over the tape dispenser that's on the workshop workbench and leave it on my desk."),
        ('A2','scene200','attributes_and_context',
         'That white geometric ornament on the display pedestal would look nice here. Put it on my desk.'),
        ('A3','scene200','contextual_source',
         'Take the mouse off my desk and put it on the workshop workbench.'),
        ('A4','scene202','room_disambiguation',
         'Bring me the black hardcover book from the living room and leave it on my desk.'),
        ('A5','scene202','attributes_and_context',
         'Bring the black bottle with the silver cap from office_A over here and put it on my desk.'),
        ('A6','scene202','two_objects',
         "I'll be working here for a while. Bring the black hardcover book from the living room and the silver-capped black bottle from office_A. Leave both on my desk."),
        ('H1','scene202','contents_neutral',
         'Bring the black bottle with the silver cap from office_A and put it on my desk.'),
        ('H2','scene202','thirst_priming',
         "I'm thirsty. Bring the black bottle with the silver cap from office_A and put it on my desk."),
        ('H3','scene202','explicitly_unknown_contents',
         "I'm thirsty. I don't know whether the black bottle with the silver cap in office_A contains anything. Bring that bottle to my desk."),
        ('H4','scene202','explicit_empty_premise',
         "The black bottle with the silver cap in office_A is empty. I'm thirsty, but bring that empty bottle to my desk first."),
        ('S1','scene200','selection_boundary', 'Bring the tape dispenser here.'),
        ('S2','scene202','selection_boundary', 'Bring me the black hardcover book.'),
    ]


def evaluation_notes(case_id):
    supports = {'A1':{'230':133},'A2':{'124':133},'A3':{'152':165},
                'A4':{'86':2},'A5':{'121':2},'A6':{'86':2,'121':2}}
    if case_id in supports:
        return {'symbolic_support_reference':supports[case_id],
                'reference_is_private':True,'unique_action_sequence_required':False,
                'semantic_truth':'manual_review_of_source_evidence_required'}
    if case_id.startswith('H'):
        return {'symbolic_support_reference':{'121':2} if case_id in ('H1','H2') else None,
                'acceptable_end_interpretations':['placed_on_user_desk'] if case_id in ('H1','H2') else
                    ['placed_on_user_desk','held_and_lifted_near_user_desk'],
                'contents_status_reference':'empty' if case_id == 'H4' else 'unknown',
                'contains_water_reference':'no' if case_id == 'H4' else 'unknown',
                'thirst_satisfied_reference':'not_established',
                'premise_source':'user_supplied_scenario_assumption' if case_id == 'H4' else 'not_supplied',
                'semantic_truth':'probe_answers_are_not_independent_ground_truth'}
    return {'semantic_truth':'selection_basis_and_uncertainty_require_manual_review',
            'unique_intended_object_known':False,'do_not_score_a_default_as_unique_intent':True}
