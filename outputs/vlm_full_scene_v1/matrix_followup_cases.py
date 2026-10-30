"""Four new eight-turn tasks plus exact whole-chain historical protocol reruns.

All references are private diagnostics, not planner inputs or repair actions.
"""
import copy
from realistic_long_cases import episode, turn, episodes as historical_episodes, evaluate_episode

CASE_VERSION='matrix_followup_new4_rerun7_v1_20261018'
RERUN_IDS=('t01','t02','t05','t06','t07','t08','t09')


def new_episodes():
    return [
        episode('n01','Read, carry and revise a desk setup','scene202','office_B',2,
            'The user is working at the writing desk in office_B.',
            ['eight_turn_history','carry_without_placement','three_surface_route','delivery_order'],[
            turn('Bring the black hardcover book from the living room to my desk, then bring the silver-capped black bottle from office_A here too. Put each one down.',[(86,2),(121,2)]),
            turn('Pick up the bottle you just brought, and keep holding it. Leave the book on my desk.',protect=[86],held=121),
            turn('Without putting it down, take it past the office_A desk and the living-room coffee table, in that order, then bring it back to my desk while still holding it.',protect=[86],held=121,visits=[(101,121),(69,121),(2,121)],no_place=True),
            turn('Set that bottle down here first. Then move the book from my desk to the coffee table.',[(121,2),(86,69)]),
            turn('Bring the book back here before returning the bottle to the office_A desk.',[(86,2),(121,101)]),
            turn('Fetch that same bottle from office_A again and bring it here, but keep holding it this time. Do not move the book or assume anything about the bottle contents.',protect=[86],held=121,visits=[(2,121)]),
            turn('Leave the bottle on the coffee table first. Then collect the book from my desk and wait by the coffee table while holding the book.',[(121,69)],held=86,visits=[(69,86)]),
            turn('Put the book down on that coffee table first, return the bottle to the office_A desk, and finally bring the book back to my desk and release it.',[(86,69),(121,101),(86,2)]),
        ]),
        episode('n02','Alternate reading and storage at office_A','scene202','office_A',101,
            'The user is seated at the wooden desk in office_A.',
            ['eight_turn_history','held_object_handoff','repeated_transport','restore_after_detour'],[
            turn('Bring the black hardcover book from the living room to my desk. Keep the silver-capped black bottle already here untouched.',[(86,101)],[121]),
            turn('Move the book to the office_B writing desk first. Then pick up the bottle from my desk, take it to the living-room coffee table, and keep holding it there.',[(86,2)],held=121,visits=[(69,121)]),
            turn('Put the bottle you are holding on the office_B writing desk. Then bring the book from that desk back to me and put it on my desk.',[(121,2),(86,101)]),
            turn('Take the book from my desk to the coffee table without putting it down. Keep holding the book there and leave the bottle in office_B.',protect=[121],held=86,visits=[(69,86)],no_place=True),
            turn('Set the book down on the coffee table first. Then fetch the bottle from office_B and put it on my desk.',[(86,69),(121,101)]),
            turn('Move the bottle from here to the coffee table, then move the book from that table to the office_B writing desk.',[(121,69),(86,2)]),
            turn('Bring the bottle back to my desk first and the book back to my desk second. Leave both here.',[(121,101),(86,101)]),
            turn('Take the book to the coffee table and return it to my desk. Only after that, take the bottle to the office_B writing desk and return it to my desk too. Put each item down at every requested stop.',[(86,69),(86,101),(121,2),(121,101)]),
        ]),
        episode('n03','Borrow and restore workshop demonstration objects','scene200','workshop',165,
            'The user is preparing a demonstration at the workshop workbench.',
            ['eight_turn_history','three_objects','protected_supports','carry_route_and_restore'],[
            turn('Bring the computer mouse from the office table to this workbench. Leave the tape dispenser and the geometric ornament alone.',[(152,165)],[230,124]),
            turn('Move the tape dispenser to the office table for now. Keep the mouse here and do not touch the ornament.',[(230,133)],[152,124]),
            turn('Bring the white geometric ornament from the display pedestal to this workbench. Do not move either of the other two items.',[(124,165)],[152,230]),
            turn('Pick up the mouse and keep holding it. Leave the ornament on the workbench and the tape dispenser in the office.',protect=[124,230],held=152),
            turn('Carry that mouse past the office table, then the display pedestal, then back to this workbench. Do not put it down at any of those stops or move the other items.',protect=[124,230],held=152,visits=[(133,152),(110,152),(165,152)],no_place=True),
            turn('Set the mouse down on this workbench first. Return the ornament to its display pedestal next, and finally bring the tape dispenser back from the office table.',[(152,165),(124,110),(230,165)]),
            turn('Take the tape dispenser to the office table first, then take the mouse to that table too. Keep the ornament on its pedestal.',[(230,133),(152,133)],[124]),
            turn('Bring the mouse back here briefly, then return it to its original office table. After that, return the tape dispenser to its original workbench. Leave the ornament untouched.',[(152,165),(152,133),(230,165)],[124]),
        ]),
        episode('n04','Stage a display with interruptions and a final inspection','scene200','display_room',110,
            'The user is standing beside the display pedestal organizing a demonstration.',
            ['eight_turn_history','interrupted_placement','payload_during_visits','navigation_only_final_turn'],[
            turn('Move the white geometric ornament from this pedestal to the office table, then bring the computer mouse from that table here. Do not move the tape dispenser.',[(124,133),(152,110)],[230]),
            turn('Pick up the mouse from this pedestal and carry it to the workshop workbench, but keep holding it there. Leave the other items alone.',protect=[124,230],held=152,visits=[(165,152)],no_place=True),
            turn('Put the mouse down on that workbench first. Then bring the ornament from the office table back to this pedestal.',[(152,165),(124,110)],[230]),
            turn('Bring the tape dispenser from the workbench to this pedestal first, then bring the mouse here as well. Leave the ornament where it is.',[(230,110),(152,110)],[124]),
            turn('Take the tape dispenser to the office table and leave it there. Then pick up the ornament and wait beside this pedestal while holding it. Do not move the mouse.',[(230,133)],[152],held=124,visits=[(110,124)]),
            turn('Without putting the ornament down, carry it past the workshop workbench and then the office table, and return here still holding it. Leave the mouse and the tape dispenser alone.',protect=[152,230],held=124,visits=[(165,124),(133,124),(110,124)],no_place=True),
            turn('Set the ornament down on this pedestal. Next, return the mouse to its original office table, then return the tape dispenser to its original workbench, and finally come back here empty-handed.',[(124,110),(152,133),(230,165)],visits=[(133,152),(165,230),(110,None)]),
            turn('For the final check, visit the office table, the workshop workbench, and this pedestal in that order. Do not pick up, move, or put down any object.',protect=[124,152,230],visits=[(133,None),(165,None),(110,None)],no_place=True),
        ]),
    ]


def episodes():
    old=[copy.deepcopy(e) for e in historical_episodes() if e['episode_id'] in RERUN_IDS]
    for ep in old:
        ep['experiment_group']='historical_protocol_whole_chain_rerun'
    fresh=new_episodes()
    for ep in fresh:
        ep['experiment_group']='new_eight_turn_task'
    return old+fresh


def evaluate(turn_number,begin,end,plan,episode_id):
    ep=next(e for e in episodes() if e['episode_id']==episode_id)
    return evaluate_episode(turn_number,begin,end,plan,ep)
