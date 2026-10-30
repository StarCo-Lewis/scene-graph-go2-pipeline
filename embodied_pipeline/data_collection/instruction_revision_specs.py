"""Human-readable, evidence-scoped revisions; never applied to an active run."""

VERSION = 'instruction_input_audit_v1'

# Rewrite only the source/destination attachment, preserving the intended task.
SYNTAX = {
 'scene200_10000002': 'Help me prepare for the next part of my work. First take the light-coloured paper storage box off the black plastic crate and put it on the black round side table. Next move the tape dispenser from the workbench to the pedestal with the white ornament. Finally move the white geometric ornament to the workshop bench.',
 'scene200_10000021': 'Take the flat document box from the narrow shelving unit and put it on the workshop bench, but leave the archive box with the finger hole exactly where it is.',
 'scene200_10000041': 'Three things need moving: take the light-coloured paper storage box (ID 184) off the black plastic crate and put it on the workbench in the workshop (ID 165); put the dark grey storage tub with corner clips on its lid (ID 182) on the long white table in the office (ID 133); and put the black computer mouse (ID 152) on the round black side table with three legs (ID 136). There is no priority between them.',
 'scene200_10000045': 'Help me prepare for the next part of my work. First put the dark grey lidded tub on the workshop bench. Then take the flat document box from the narrow shelving unit and put it on the white filing cabinet. Only after those are done, set the white geometric ornament on the black round side table.',
 'scene200_10000050': 'First take the light-coloured paper storage box (ID 184) off the black plastic crate and put it on the long white table in the office (ID 133). Next move the dark grey storage tub with corner clips on its lid (ID 182) to the round black side table with three legs (ID 136). Finally move the white geometric ornament on a black base (ID 124) to the white three-drawer filing cabinet (ID 1). Set each item down at its destination.',
 'scene200_10000054': 'Could you take the flat document box from the narrow shelving unit and put it on the workshop bench, and take the light-coloured paper storage box off the black plastic crate and put it on the white filing cabinet? I do not mind which you do first.',
 'scene200_10000057': 'Help me set things out: take the flat document box from the narrow shelving unit to the workshop bench; take the light-coloured paper storage box from the black plastic crate to the office table; and take the archive box with the finger hole to the white filing cabinet. Set each item down at its destination. Any order is fine.',
 'scene202_10000006': 'Help me prepare for the next part of my work. First put the mouse on the living-room coffee table, then put the silver-capped bottle on the cross-braced wooden table. Only after those are done, take the black notebook from the office A desk and put it on the two-drawer writing desk.',
 'scene202_10000048': 'First take the packing tape dispenser from the workbench that currently supports the white tape roll and put it on the living-room coffee table. Then take the black notebook with an elastic band from the office A desk and put it on the office B desk.',
 'scene202_10000050': 'Change of plan. First put the white tape roll on the office A desk. Then take the black notebook from that office A desk and put it on the two-drawer writing desk.',
 'scene202_10000057': 'Move the mouse from the office A desk to the two-drawer writing desk; take the white tape roll from the workbench that currently supports the black-and-red dispenser and put it on the office B desk; and take the black notebook with an elastic band from the office A desk and put it on the cross-braced wooden table. Set each item down at its destination. Work through them in whichever order you prefer.',
 'd3_619_10000051': 'First take the square pillow from the curved upholstered partition and put it on the curved white modular sofa. Then put the shipping-labelled cardboard box on the round table with the orange base. Finally take the white rectangular pillow off the cushion it is currently resting on and put it on the beige footstool.',
 'd3_619_10000084': 'Take the white rectangular pillow off the cushion it is currently resting on and put it on the round table with the orange base. Then wait beside the dark office machine for the next request.',
 'd3_619_10000106': 'Here is the arrangement I want: take the white rectangular pillow off the cushion it is currently resting on and put it on the round table with the orange base; take the light grey blanket from the sofa corner and put it on the beige footstool. Choose the order yourself.',
}

RACK = {f'scene200_{10000000+n}' for n in [32,35,69,71,74,78]}
TAPE = {f'scene202_{10000000+n}' for n in [11,16,26,38,43,65,72,76,77,87,93]}
PILLOW = {f'd3_619_{10000000+n}' for n in [10,16,76,90]}
CONTAINER = {f'd3_619_{10000000+n}' for n in [39,74,113,116]}

# Missing public room identities, not a claim that the raw scene is wrong.
# Room metadata was private author annotation, except the one user landmark.
ROOM_GAPS = {
 'scene202_10000006':['A'], 'scene202_10000010':['A'],
 'scene202_10000017':['B'], 'scene202_10000029':['B'],
 'scene202_10000036':['A','B'], 'scene202_10000048':['A','B'],
 'scene202_10000057':['A','B'], 'scene202_10000059':['B'],
 'scene202_10000069':['A'], 'scene202_10000076':['A'],
 'scene202_10000081':['B'], 'scene202_10000087':['A'],
 'scene202_10000090':['B'],
}

def specification(rid):
    if rid in SYNTAX:
        return dict(kind='source_destination_attachment', reason='连续位置修饰与目标短语黏连；明确from/to及动词作用范围，保留原任务目标、顺序和保护要求。', information_change='wording_only_same_authored_intent')
    if rid in RACK:
        return dict(kind='nonunique_referent', reason='163和164都符合四层灰色架；用初始时支撑深灰带盖箱的关系限定163。', information_change='added_author_clarification_from_graph', node_ids=[163,164,182], edge_index=55)
    if rid in TAPE:
        return dict(kind='nonunique_referent', reason='281和289都符合黑红胶带座；限定为当前与白胶带卷共处一张工作台的胶带座289。', information_change='added_author_clarification_from_graph', node_ids=[281,289,290,221], edge_indices=[127,134,135])
    if rid in PILLOW:
        return dict(kind='nonunique_referent', reason='176和211都符合棕色枕/靠垫；限定为初始时位于带竖缝沙发段上的枕176。', information_change='added_author_clarification_from_graph', node_ids=[176,211,170], edge_index=27)
    if rid in CONTAINER:
        return dict(kind='nonunique_referent', reason='9和32都符合半透明带筋容器；只在这4条增加目标32的ID，不对全批次改成ID指令。', information_change='added_explicit_author_instance_selection', node_ids=[9,32])
    return None

def revised_text(rid, text):
    if rid in SYNTAX:return SYNTAX[rid]
    if rid in RACK:
        return text.replace('grey metal rack with four shelves', 'grey four-shelf metal rack where the dark grey lidded tub is at the start of this request')
    if rid in TAPE:
        return text.replace('black-and-red tape dispenser', 'black-and-red tape dispenser on the workbench that currently supports the white tape roll')
    if rid in PILLOW:
        return text.replace('tan pillow', 'tan pillow currently resting on the light beige sofa section with a vertical seam')
    if rid in CONTAINER:
        return text.replace('ribbed translucent container', 'ribbed translucent container (ID 32)')
    return text
