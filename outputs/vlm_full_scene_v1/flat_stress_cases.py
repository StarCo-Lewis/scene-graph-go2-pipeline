"""New authored tasks; same full Scene202 graph, not new physical scenes.

No atomic reference plans here. Expected events stay in the test descriptor,
not the model's execution contract. Only common scene metadata is reused.
"""
import copy


def catalog(base):
    def round_(i,text,bottle,book,visits,held_visits,placements,held=None,room='office_B'):
        return {'round':i,'instruction':text,
                'expected_end':{'held_object_id':held,'lifted':held is not None},
                'expected_support_by_object':{'121':bottle,'86':book},
                'expected_navigation_targets':visits,'expected_robot_room':room,
                'expected_placement_count':len(placements),'expected_placement_events':placements,
                'expected_held_during_visits':held_visits}
    cases = [
        ('alternating_transfers_revisit_1',[
            round_(1,
                'Visit the water bottle (ID 121) in office_A, pick it up and lift it, carry it to the office_B desk (ID 2), and place it there and release it. '
                'Then visit the black hardcover book (ID 86) in the living_room, pick it up and lift it, carry it to the office_A desk (ID 101), and place it there and release it. '
                'Then visit the moved water bottle (ID 121) in office_B, pick it up and lift it, carry it to the living_room coffee table (ID 69), and place it there and release it. '
                'Finally, without picking up anything, visit the office_B desk (ID 2), then the office_A desk (ID 101), then the office_B desk (ID 2) again, in exactly that order.',
                69,101,[121,2,86,101,121,69,2,101,2],
                [None,121,None,86,None,121,None,None,None],[[121,2],[86,101],[121,69]])]),
        ('held_waypoint_four_transfers_1',[
            round_(1,
                'Visit the water bottle (ID 121) in office_A, pick it up and lift it. '
                'While continuously holding it, visit the living_room coffee table (ID 69) without placing or releasing it, then carry it to the office_B desk (ID 2), and place it there and release it. '
                'Next visit the black hardcover book (ID 86) in the living_room, pick it up and lift it, carry it to the office_A desk (ID 101), and place it there and release it. '
                'Then visit the moved water bottle (ID 121) in office_B, pick it up and lift it, carry it to the office_A desk (ID 101), and place it there and release it. '
                'Next visit the moved black hardcover book (ID 86) in office_A, pick it up and lift it, carry it to the living_room coffee table (ID 69), and place it there and release it. '
                'Finally, without picking up either object, visit the office_B desk (ID 2), then the water bottle (ID 121) in office_A, then the office_B desk (ID 2) again, in exactly that order.',
                101,69,[121,69,2,86,101,121,101,86,69,2,121,2],
                [None,121,121,None,86,None,121,None,86,None,None,None],
                [[121,2],[86,101],[121,101],[86,69]])]),
        ('held_relay_alternating_4',[
            round_(1,
                'Visit the water bottle (ID 121) in office_A, pick it up and lift it. '
                'While continuously holding it, visit the living_room coffee table (ID 69), then the office_B desk (ID 2), then the office_A desk (ID 101), in exactly that order. '
                'Do not place or release the bottle anywhere; finish still holding it and lifted. Do not pick up or move the black hardcover book (ID 86).',
                None,85,[121,69,2,101],[None,121,121,121],[],held=121,room='office_A'),
            round_(2,
                'Place the currently held water bottle (ID 121) on the office_A desk (ID 101) and release it. '
                'Then visit the black hardcover book (ID 86) in the living_room, pick it up and lift it, carry it to the office_B desk (ID 2), and place it there and release it. '
                'Finally, without picking up anything, visit the living_room coffee table (ID 69), then the office_A desk (ID 101), then the living_room coffee table (ID 69) again, in exactly that order.',
                101,2,[86,2,69,101,69],[None,86,None,None,None],[[121,101],[86,2]],room='living_room'),
            round_(3,
                'Visit the moved black hardcover book (ID 86) in office_B, pick it up and lift it, carry it to the office_A desk (ID 101), and place it there and release it. '
                'Only after releasing the book, visit the water bottle (ID 121) in office_A, pick it up and lift it, then carry it to the office_B desk (ID 2). '
                'Do not place or release the bottle; finish holding it and lifted. After placing the book, do not pick it up or move it again during this instruction.',
                None,101,[86,101,121,2],[None,86,None,121],[[86,101]],held=121),
            round_(4,
                'Place the currently held water bottle (ID 121) on the office_B desk (ID 2) and release it. '
                'Then visit the black hardcover book (ID 86) in office_A, pick it up and lift it, carry it to the living_room coffee table (ID 69), and place it there and release it. '
                'Next visit the moved water bottle (ID 121) in office_B, pick it up and lift it, carry it to the living_room coffee table (ID 69), and place it there and release it. '
                'Finally, without picking up either object, visit the office_A desk (ID 101), then the office_B desk (ID 2), then the moved black hardcover book (ID 86) in the living_room, in exactly that order.',
                69,69,[86,69,121,69,101,2,86],[None,86,None,121,None,None,None],
                [[121,2],[86,69],[121,69]],room='living_room')])]
    result = []
    for name,rounds in cases:
        cfg = copy.deepcopy(base)
        cfg.update(case_id='scene202_flat_stress_'+name, rounds=rounds)
        result.append((name,cfg))
    return result
