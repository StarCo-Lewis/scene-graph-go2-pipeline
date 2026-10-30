"""Eight new authored positive task combinations: Scene202 two, Scene200 six.

This file contains instructions and independent event assertions, no atomic plans.
"""
import copy


def row(i,text,supports,visits,payloads,placements,room,held=None):
    return {'round':i,'instruction':text,'expected_end':{'held_object_id':held,'lifted':held is not None},
        'expected_support_by_object':{str(k):v for k,v in supports.items()},
        'expected_navigation_targets':visits,'expected_held_during_visits':payloads,
        'expected_placement_events':placements,'expected_placement_count':len(placements),
        'expected_robot_room':room}


def scene202_cases(base):
    return package(base,'scene202',[
        ('five_transfers_waypoint_1',[
            row(1,
                'Visit the water bottle (ID 121) in office_A, pick it up and lift it. While holding it, visit the living_room coffee table (ID 69) without releasing or placing it, then carry it to the office_B desk (ID 2), place it there and release it. '
                'Next visit the black hardcover book (ID 86) in the living_room, pick it up and lift it, carry it to the office_A desk (ID 101), place it there and release it. '
                'Then visit the moved water bottle (ID 121) in office_B, pick it up and lift it, carry it to the living_room coffee table (ID 69), place it there and release it. '
                'Next visit the moved book (ID 86) in office_A, pick it up and lift it, carry it to the office_B desk (ID 2), place it there and release it. '
                'Then visit the moved water bottle (ID 121) in the living_room, pick it up and lift it, carry it to the office_A desk (ID 101), place it there and release it. '
                'Finally, without picking up anything, visit the office_B desk (ID 2), then the living_room coffee table (ID 69), then the moved water bottle (ID 121) in office_A, in exactly that order.',
                {121:101,86:2},[121,69,2,86,101,121,69,86,2,121,101,2,69,121],
                [None,121,121,None,86,None,121,None,86,None,121,None,None,None],
                [[121,2],[86,101],[121,69],[86,2],[121,101]],'office_A')]),
        ('alternating_hold_handoffs_5',[
            row(1,'Visit the black hardcover book (ID 86) in the living_room, pick it up and lift it, then carry it to the office_A desk (ID 101). Do not place or release the book; finish holding it and lifted. Do not pick up or move the water bottle (ID 121).',
                {121:101,86:None},[86,101],[None,86],[],'office_A',86),
            row(2,'Place the currently held book (ID 86) on the office_A desk (ID 101) and release it. Then visit the water bottle (ID 121) in office_A, pick it up and lift it. While continuously holding the bottle, visit the living_room coffee table (ID 69), then the office_B desk (ID 2), in exactly that order. Do not place or release the bottle; finish holding it and lifted.',
                {121:None,86:101},[121,69,2],[None,121,121],[[86,101]],'office_B',121),
            row(3,'Place the currently held water bottle (ID 121) on the office_B desk (ID 2) and release it. Next visit the moved book (ID 86) in office_A, pick it up and lift it, carry it to the living_room coffee table (ID 69), place it there and release it. Finally, without picking up anything, visit the office_B desk (ID 2), then the office_A desk (ID 101), then the living_room coffee table (ID 69), in exactly that order.',
                {121:2,86:69},[86,69,2,101,69],[None,86,None,None,None],[[121,2],[86,69]],'living_room'),
            row(4,'Visit the moved water bottle (ID 121) in office_B, pick it up and lift it, carry it to the office_A desk (ID 101), place it there and release it. Then visit the book (ID 86) in the living_room, pick it up and lift it, carry it to the office_B desk (ID 2), place it there and release it. Finally visit the moved water bottle (ID 121) in office_A, pick it up and lift it, then carry it to the living_room coffee table (ID 69). Do not place or release the bottle; finish holding it and lifted.',
                {121:None,86:2},[121,101,86,2,121,69],[None,121,None,86,None,121],[[121,101],[86,2]],'living_room',121),
            row(5,'Place the currently held water bottle (ID 121) on the living_room coffee table (ID 69) and release it. Then visit the moved book (ID 86) in office_B, pick it up and lift it, carry it to the office_A desk (ID 101), place it there and release it. Next visit the moved water bottle (ID 121) in the living_room, pick it up and lift it, carry it to the office_B desk (ID 2), place it there and release it. Finally, without picking up anything, visit the living_room coffee table (ID 69), then the office_A desk (ID 101), then the moved book (ID 86), in exactly that order.',
                {121:2,86:101},[86,101,121,2,69,101,86],[None,86,None,121,None,None,None],[[121,69],[86,101],[121,2]],'office_A')])])


def scene200_cases(base):
    initial={124:110,152:133,230:165}
    return package(base,'scene200',[
        ('single_mouse_transfer_1',[
            row(1,'Visit the computer mouse (ID 152) in the office, pick it up and lift it, carry it to the workshop workbench (ID 165), place it on that workbench and release it. Do not pick up or move the geometric ornament (ID 124) or the tape dispenser (ID 230).',
                {124:110,152:165,230:165},[152,165],[None,152],[[152,165]],'workshop')]),
        ('ordered_six_visits_1',[
            row(1,'Without picking up or moving any objects, visit the display_room pedestal (ID 110), then the office table (ID 133), then the workshop workbench (ID 165), then the office table (ID 133) again, then the display_room pedestal (ID 110) again, then the workshop workbench (ID 165) again, in exactly that order.',
                initial,[110,133,165,133,110,165],[None]*6,[],'workshop')]),
        ('ornament_hold_relay_2',[
            row(1,'Visit the geometric ornament (ID 124) in the display_room, pick it up and lift it. While continuously holding it, visit the workshop workbench (ID 165), then the office table (ID 133), in exactly that order. Do not place or release it; finish holding it and lifted. Do not pick up or move the mouse (ID 152) or the tape dispenser (ID 230).',
                {124:None,152:133,230:165},[124,165,133],[None,124,124],[],'office',124),
            row(2,'Place the currently held geometric ornament (ID 124) on the office table (ID 133) and release it. Then visit the computer mouse (ID 152) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Finally, without picking up anything, visit the workshop workbench (ID 165), then the office table (ID 133), in exactly that order.',
                {124:133,152:110,230:165},[152,110,165,133],[None,152,None,None],[[124,133],[152,110]],'office')]),
        ('three_object_rotation_1',[
            row(1,'Visit the computer mouse (ID 152) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Then visit the geometric ornament (ID 124) in the display_room, pick it up and lift it, carry it to the workshop workbench (ID 165), place it there and release it. Next visit the tape dispenser (ID 230) in the workshop, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Finally, without picking up anything, visit the display_room pedestal (ID 110), then the workshop workbench (ID 165), then the office table (ID 133), in exactly that order.',
                {124:165,152:110,230:133},[152,110,124,165,230,133,110,165,133],
                [None,152,None,124,None,230,None,None,None],[[152,110],[124,165],[230,133]],'office')]),
        ('protected_tape_exchange_2',[
            row(1,'Visit the computer mouse (ID 152) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Next visit the geometric ornament (ID 124) in the display_room, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Finally visit the tape dispenser (ID 230) in the workshop without picking it up or moving it. Leave the tape dispenser (ID 230) on the workbench (ID 165).',
                {124:133,152:110,230:165},[152,110,124,133,230],[None,152,None,124,None],[[152,110],[124,133]],'workshop'),
            row(2,'Visit the moved geometric ornament (ID 124) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Then visit the moved mouse (ID 152) in the display_room, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Finally visit the tape dispenser (ID 230) in the workshop without picking it up or moving it, then visit the office table (ID 133). Leave the tape dispenser (ID 230) on the workbench (ID 165).',
                initial,[124,110,152,133,230,133],[None,124,None,152,None,None],[[124,110],[152,133]],'office')]),
        ('three_object_hold_relay_6',[
            row(1,'Visit the tape dispenser (ID 230) in the workshop, pick it up and lift it. While continuously holding it, visit the display_room pedestal (ID 110), then the office table (ID 133), in exactly that order. Do not place or release it; finish holding it and lifted.',
                {124:110,152:133,230:None},[230,110,133],[None,230,230],[],'office',230),
            row(2,'Place the currently held tape dispenser (ID 230) on the office table (ID 133) and release it. Next visit the mouse (ID 152) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Then visit the geometric ornament (ID 124) in the display_room, pick it up and lift it, carry it to the workshop workbench (ID 165). Do not place or release the ornament; finish holding it and lifted.',
                {124:None,152:110,230:133},[152,110,124,165],[None,152,None,124],[[230,133],[152,110]],'workshop',124),
            row(3,'Place the currently held geometric ornament (ID 124) on the workshop workbench (ID 165) and release it. Next visit the moved mouse (ID 152) in the display_room, pick it up and lift it. While holding the mouse, visit the office table (ID 133) without placing or releasing it, then carry it to the workshop workbench (ID 165), place it there and release it. Finally visit the moved tape dispenser (ID 230) in the office, pick it up and lift it, carry it to the display_room pedestal (ID 110). Do not place or release the tape dispenser; finish holding it and lifted.',
                {124:165,152:165,230:None},[152,133,165,230,110],[None,152,152,None,230],[[124,165],[152,165]],'display_room',230),
            row(4,'Place the currently held tape dispenser (ID 230) on the display_room pedestal (ID 110) and release it. Next visit the moved geometric ornament (ID 124) in the workshop, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Then visit the moved mouse (ID 152) in the workshop, pick it up and lift it, carry it to the display_room pedestal (ID 110). Do not place or release the mouse; finish holding it and lifted.',
                {124:133,152:None,230:110},[124,133,152,110],[None,124,None,152],[[230,110],[124,133]],'display_room',152),
            row(5,'Place the currently held mouse (ID 152) on the display_room pedestal (ID 110) and release it. Next visit the moved geometric ornament (ID 124) in the office, pick it up and lift it, carry it to the workshop workbench (ID 165), place it there and release it. Then visit the moved tape dispenser (ID 230) in the display_room, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Finally visit the mouse (ID 152) in the display_room without picking it up.',
                {124:165,152:110,230:133},[124,165,230,133,152],[None,124,None,230,None],[[152,110],[124,165],[230,133]],'display_room'),
            row(6,'Visit the moved mouse (ID 152) in the display_room, pick it up and lift it, carry it to the office table (ID 133), place it there and release it. Then visit the moved geometric ornament (ID 124) in the workshop, pick it up and lift it, carry it to the display_room pedestal (ID 110), place it there and release it. Next visit the moved tape dispenser (ID 230) in the office, pick it up and lift it, carry it to the workshop workbench (ID 165), place it there and release it. Finally, without picking up anything, visit the office table (ID 133), then the display_room pedestal (ID 110), then the workshop workbench (ID 165), in exactly that order.',
                initial,[152,133,124,110,230,165,133,110,165],
                [None,152,None,124,None,230,None,None,None],[[152,133],[124,110],[230,165]],'workshop')])])


def package(base,scene,entries):
    result=[]
    for name,rounds in entries:
        cfg=copy.deepcopy(base)
        full_name=scene+'_'+name
        cfg.update(schema_version='symbolic_scene_case_v1',case_id=full_name,scene_id=scene,rounds=rounds)
        result.append((full_name,cfg))
    return result
