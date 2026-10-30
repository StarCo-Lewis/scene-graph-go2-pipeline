"""Author-designed task variants on Scene201, NOT independent physical maps."""
import copy
from run_json_long_horizon import instructions

DEFAULT_CASES = ('short_navigation_1', 'split_transfer_3', 'compound_transfer_2',
                 'composed_6', 'long_navigation_8', 'extended_mixed_18',
                 'missing_apple', 'ambiguous_sideboard', 'unknown_on_relation')


def catalog(nodes):
    base = instructions()
    paraphrases = [
        'First approach kettle ID 67, next visit open-office sideboard ID 13, and finally return to kettle ID 67.',
        'Take hold of kettle ID 67 and lift it. Keep it in your grasp until the next command.',
        'With kettle ID 67 already held and lifted, go to open-office sideboard ID 13 without setting the kettle down.',
        'Set the kettle ID 67 you are carrying onto open-office sideboard ID 13 and release it.',
        'Retrieve the relocated kettle ID 67, lift it, bring it to kitchen cabinet ID 62, then set it on that cabinet and release it.',
        'Go to open-office sideboard ID 13 first, then go back to relocated kettle ID 67 in the kitchen.',
        'Take hold of kettle ID 67 in the kitchen, lift it, and retain it for the following command.',
        'Bring the already held and lifted kettle ID 67 to open-office sideboard ID 13; then set it on the sideboard and release it.',
        'Retrieve kettle ID 67 and lift it, then bring it to kitchen cabinet ID 62. Retain your grasp; do not set it down.',
        'Set the kettle ID 67 currently in your grasp on kitchen cabinet ID 62 and release it.',
        'Retrieve kettle ID 67, lift it, bring it to open-office sideboard ID 13, and leave it resting on that sideboard after release.',
        'Visit relocated kettle ID 67 first, open-office sideboard ID 13 second, and relocated kettle ID 67 last.',
    ]
    rewritten = copy.deepcopy(base)
    for row, text in zip(rewritten, paraphrases):
        row['instruction'] = text

    def row(index, text, held=False, support=None, visits=None):
        value = {'round': index, 'instruction': text, 'expected_support_id': support,
                 'expected_end': {'held_object_id': 67 if held else None, 'lifted': held}}
        if visits is not None:
            value['expected_navigation_targets'] = visits
        return value
    composed = [
        row(1, 'Visit kitchen cabinet ID 62, then open-office sideboard ID 13, then kettle ID 67.', visits=[62,13,67]),
        row(2, 'Pick up and lift kettle ID 67, carry it to open-office sideboard ID 13, and place it on that sideboard.', support=13),
        row(3, 'Visit kitchen cabinet ID 62, then return to the moved kettle ID 67. Do not move any objects.', support=13, visits=[62,67]),
        row(4, 'Pick up and lift kettle ID 67, carry it to kitchen cabinet ID 62, and keep holding it. Do not place it.', held=True),
        row(5, 'While carrying the already held and lifted kettle ID 67, visit open-office sideboard ID 13, then kitchen cabinet ID 62. Keep holding it.', held=True, visits=[13,62]),
        row(6, 'At kitchen cabinet ID 62, place the held kettle ID 67 on that cabinet and release it. Then visit open-office sideboard ID 13 and return to the moved kettle ID 67.', support=62, visits=[13,67]),
    ]
    simple = [row(1, 'Visit open-office sideboard ID 13.', visits=[13])]
    split = [
        row(1, 'Visit kettle ID 67, pick it up and lift it; keep holding it.', held=True, visits=[67]),
        row(2, 'Carry the already held and lifted kettle ID 67 to open-office sideboard ID 13. Keep holding it.', held=True, visits=[13]),
        row(3, 'Place the held kettle ID 67 on open-office sideboard ID 13 and release it.', support=13),
    ]
    compound = [
        row(1, 'Visit kitchen cabinet ID 62, then visit kettle ID 67.', visits=[62,67]),
        row(2, 'First visit kettle ID 67, pick it up and lift it, carry it to open-office sideboard ID 13 and place it there. Then visit kitchen cabinet ID 62 and return to the moved kettle ID 67.', support=13, visits=[67,13,62,67]),
    ]
    routes = [[13],[62],[67],[13,62],[67,13],[62,67],[13,67,62],[13,67]]
    names = {13:'open-office sideboard ID 13', 62:'kitchen cabinet ID 62', 67:'kettle ID 67'}
    navigation = [row(i, 'Visit ' + ', then visit '.join(names[j] for j in route) + '.', visits=route)
                  for i, route in enumerate(routes,1)]
    extended = copy.deepcopy(base) + [
        row(13, 'Visit kettle ID 67, pick it up and lift it, carry it to kitchen cabinet ID 62 and place it there.', support=62, visits=[67,62]),
        row(14, 'Visit open-office sideboard ID 13, then the moved kettle ID 67, then open-office sideboard ID 13, then return to the moved kettle ID 67.', support=62, visits=[13,67,13,67]),
        row(15, 'Pick up and lift kettle ID 67. Keep holding it for the next command.', held=True),
        row(16, 'While holding the already lifted kettle ID 67, visit open-office sideboard ID 13 and then kitchen cabinet ID 62. Do not place it; keep holding it.', held=True, visits=[13,62]),
        row(17, 'Place the held kettle ID 67 on kitchen cabinet ID 62 and release it.', support=62),
        row(18, 'First visit kettle ID 67, pick it up and lift it, carry it to open-office sideboard ID 13 and place it there. Then visit kitchen cabinet ID 62 and return to the moved kettle ID 67.', support=13, visits=[67,13,62,67]),
    ]
    def positive(name, rows, difficulty, complexity):
        return {'name': name, 'kind': 'positive', 'map_id': 'scene201_three_node_subset',
                'difficulty_rank': difficulty, 'complexity': complexity,
                'input_variation': 'unmodified_selected_nodes', 'nodes': copy.deepcopy(nodes), 'rounds': rows}
    def negative(name, text, inventory, errors, variation):
        return {'name': name, 'kind': 'negative', 'map_id': 'scene201_three_node_subset',
                'input_variation': variation, 'nodes': inventory, 'edges': [],
                'instruction': text, 'expected_error_codes': errors,
                'negative_scope': 'LLM decomposition/object binding only; no actions or state projection'}
    duplicate = copy.deepcopy(nodes)
    twin = copy.deepcopy(next(n for n in nodes if n['id'] == 13))
    twin['id'] = 113
    twin['bbox_center'][0] += 2
    twin['json_test_context']['duplicate_source'] = 'synthetic_test_clone_of_13_not_an_observed_instance'
    duplicate.append(twin)
    return [positive('short_navigation_1',simple,1,'short_single_navigation'),
        positive('split_transfer_3',split,2,'short_split_manipulation_intents'),
        positive('compound_transfer_2',compound,3,'short_many_actions_per_round'),
        positive('composed_6',composed,4,'medium_mixed_intents_and_held_state'),
        positive('long_navigation_8',navigation,5,'long_navigation_only'),
        positive('extended_mixed_18',extended,6,'long_mixed_intents_many_actions_and_memory'),
        positive('baseline_12',copy.deepcopy(base),6,'optional_previous_baseline'),
        positive('paraphrase_12',rewritten,6,'optional_baseline_paraphrase'),
        negative('missing_apple', 'Go to an apple. Do not substitute another category.', copy.deepcopy(nodes),
                 ['NO_OBJECT_CAN_BE_USED_IN_THIS_TASK'], 'existing_nodes_without_apple'),
        negative('ambiguous_sideboard', 'Go to an open-office sideboard. There is no preference between identical instances.',
                 duplicate, ['OBJECT_BINDING_AMBIGUOUS'], 'synthetic_duplicate_13_to_113'),
        negative('unknown_on_relation', 'Go to the kettle (ID 67) that is currently on the open-office sideboard (ID 13). Use the supplied on relation to select it.',
                 copy.deepcopy(nodes), ['OBJECT_RELATION_UNVERIFIED', 'OBJECT_BINDING_AMBIGUOUS'], 'on_edges_withheld_no_inferred_geometry')]


def validate_cases(cases):
    names = set()
    for case in cases:
        if case['name'] in names or not case['name'].replace('_', '').isalnum():
            raise ValueError('Invalid/duplicate case name')
        names.add(case['name'])
        ids = [n['id'] for n in case['nodes']]
        if any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('Case node IDs must be distinct integers')
        if case['kind'] == 'positive':
            if set(ids) != {13,62,67} or not case['rounds']:
                raise ValueError('Positive cases require the current three-node symbolic adapter')
            for index, row in enumerate(case['rounds'], 1):
                if (row['round'] != index or 'plan' in row or 'steps' in row
                        or not row['instruction'].isascii() or not row['instruction'].strip()):
                    raise ValueError('Only ordered English instructions/assertions, no prewritten plans')
        elif case['kind'] != 'negative' or not case['expected_error_codes']:
            raise ValueError('Unknown case kind or missing refusal assertion')
    return cases
