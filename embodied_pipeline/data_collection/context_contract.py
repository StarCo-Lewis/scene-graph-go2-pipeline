"""Strict public-input projection for reviewed natural-language collections."""
import copy
import json
from .catalog import _rooms

VERSION='reviewed_user_context_v1'
CONTEXT_KEYS={'source','user_location','user_activity','dialogue_history'}
SPATIAL_KEYS={'location_source','user_exact_position','user_heading','robot_location','room_status','near_semantics','deixis_policy','history_scope'}
BOUNDARY=(
 ' The user location is an explicitly authored scenario assumption, not perceived human pose. '
 'User and robot are different entities. A user near a landmark is not at its bbox centre; '
 'do not infer exact coordinates, orientation, robot location, ownership, pointing, free surface '
 'space, room adjacency or handover capability. Use supplied user location for here/me, but '
 'preserve explicit referents from supplied dialogue history for there/that item. '
 'History is user speech, not executed robot actions. Preserve unresolved referents and '
 'uncertainty instead of inventing targets or facts; existing output and binding protocols apply.')

def public_context(task,nodes):
    """Reject unexpected fields rather than silently leaking reviewer metadata."""
    mi=task.get('model_input')
    if not isinstance(mi,dict) or mi.get('instruction')!=task.get('instruction'):
        raise ValueError('Reviewed model input instruction is missing or differs')
    context=mi.get('conversation_context');spatial=mi.get('public_spatial_context')
    if not isinstance(context,dict) or set(context)-CONTEXT_KEYS or not {'source','user_location','user_activity'}<=set(context):
        raise ValueError('Unexpected public conversation fields')
    if context.get('source')!='scenario_assumption_not_observation':
        raise ValueError('Speaker location must be marked as authored, not observed')
    if task.get('conversation_context')!=context or task.get('public_spatial_context')!=spatial:
        raise ValueError('Public context copies differ')
    location=context['user_location']
    if not isinstance(location,dict) or set(location)!={'room_name','near_scene_element_id'}:
        raise ValueError('Invalid user location fields')
    by={n['id']:n for n in nodes};anchor=location['near_scene_element_id']
    if type(anchor) is not int or anchor not in by or by[anchor].get('active',True) is False:
        raise ValueError('User location anchor absent or inactive')
    rooms,_=_rooms(task['scene_id'],by)
    if location['room_name'] is not None and rooms.get(anchor)!=location['room_name']:
        raise ValueError('User room has no existing annotation for this anchor')
    if task['scene_id']=='d3_619' and location['room_name'] is not None:
        raise ValueError('D3-619 room membership is unknown')
    if not isinstance(context['user_activity'],str) or not context['user_activity'].strip():
        raise ValueError('User activity must be nonempty text')
    history=context.get('dialogue_history',[])
    if not isinstance(history,list):raise ValueError('History must be a list')
    for turn in history:
        if not isinstance(turn,dict) or set(turn)!={'speaker','text'} or turn['speaker']!='user' or not isinstance(turn['text'],str) or not turn['text'].strip():
            raise ValueError('Invalid user history')
    messages=mi.get('context_messages',[])
    if not isinstance(messages,list) or any(not isinstance(m,dict) or set(m)!={'role','content'} or m['role']!='user' for m in messages):
        raise ValueError('Invalid original context messages')
    if history!=[{'speaker':m['role'],'text':m['content']} for m in messages]:
        raise ValueError('Dialogue projection changed or lost history')
    if not isinstance(spatial,dict) or set(spatial)!=SPATIAL_KEYS:
        raise ValueError('Unexpected public spatial fields')
    if spatial['user_exact_position'] is not None or spatial['user_heading'] is not None or spatial['robot_location']!={'status':'unknown','same_as_user':None}:
        raise ValueError('Unknown physical poses were invented')
    for key in SPATIAL_KEYS-{'user_exact_position','user_heading','robot_location'}:
        if not isinstance(spatial[key],str) or not spatial[key].strip():raise ValueError('Invalid spatial text field')
    return copy.deepcopy(context),copy.deepcopy(spatial)

def attach(task,nodes,contract):
    marker=task.get('collection_public_context_version')
    if marker is None:
        if 'conversation_context' in task or 'model_input' in task:
            raise ValueError('Context-bearing task requires the reviewed-context collection entry; refusing to silently discard context')
        return False
    if marker!=VERSION:raise ValueError('Unsupported public context version')
    context,spatial=public_context(task,nodes)
    contract['conversation_context']=context
    contract['public_spatial_context']=spatial
    contract['factual_boundary']+=BOUNDARY
    contract['public_context_version']=VERSION
    return True
