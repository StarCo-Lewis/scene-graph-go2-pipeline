"""Program-issued five/nine-digit wire IDs; scoped lookup, never semantic repair.

Canonical runtime IDs stay unchanged. New wire matrices decode losslessly through
the existing domain validators. A checksum scopes IDs; it is not authentication.
"""
import copy
import hashlib
import json
import re
import uuid

VERSION = 'typed9_ids_v1'
MODE = 'typed9_schema'
BASE = 100_000_000
KINDS = {1:'executor', 2:'role', 3:'scene_node', 4:'skill', 5:'subtask',
         6:'constraint', 7:'numeric_preset', 8:'bound_instance'}
ROBOT = 100000001
CAPACITY = {'role':32, 'subtask':64, 'constraint':128}


def protocol(digits):
    if type(digits) is not int or digits not in (5,9):
        raise ValueError('Identifier width must be 5 or 9')
    return f'typed{digits}_ids_v1'


def width(value):
    if not isinstance(value,dict): raise ValueError('Missing typed identifier context')
    if 'identifier_context' in value: return width(value['identifier_context'])
    version=value.get('version',value.get('schema_version'))
    for digits in (5,9):
        if version==protocol(digits): return digits
    raise ValueError('Unknown typed identifier protocol')


def code(kind, serial, digits=9):
    protocol(digits); base=10**(digits-1)
    if type(kind) is not int or kind not in KINDS or type(serial) is not int or not 1 <= serial < base:
        raise ValueError(f'Typed ID requires a known type and serial 1..{base-1}')
    return kind*base+serial


def require(value, kind, inventory=None, digits=9):
    protocol(digits); base=10**(digits-1)
    if type(value) is not int or value//base != kind or not 1 <= value%base < base or kind not in KINDS:
        raise ValueError(f'TYPED_ID_KIND_MISMATCH: expected {KINDS.get(kind)}, received {value!r}')
    if inventory is not None and value not in inventory:
        raise ValueError(f'TYPED_ID_NOT_ISSUED: {value} is absent from the current {KINDS[kind]} lookup')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,
        allow_nan=False,separators=(',',':')).encode()).hexdigest()


def seal(book):
    value=copy.deepcopy(book); value.pop('sha256',None)
    value['sha256']=digest(value)
    return value


def intact(book):
    if not isinstance(book,dict) or seal(book)!=book:
        raise ValueError('TYPED_CODEBOOK_MODIFIED')


def allocation(plan_id=None, digits=9):
    plan_id=uuid.uuid4().hex if plan_id is None else plan_id
    if not isinstance(plan_id,str) or not plan_id.strip():
        raise ValueError('A nonempty plan scope is required')
    entries=[dict(code=code(1,1,digits),kind='executor',canonical='robot')]
    for kind,key in ((2,'role'),(5,'subtask'),(6,'constraint')):
        entries.extend(dict(code=code(kind,i,digits),kind=key,slot=i) for i in range(1,CAPACITY[key]+1))
    return seal(dict(schema_version=protocol(digits),plan_id=plan_id,entries=entries,
        definition_policy='Use program-issued slots in order; fill text, never allocate IDs',
        capacity=CAPACITY))


def lookup(book):
    intact(book)
    result={entry['code']:entry for entry in book['entries']}
    if len(result)!=len(book['entries']):
        raise ValueError('Duplicate issued ID')
    return result


def pool(book,kind):
    return [n for n,e in lookup(book).items() if e['kind']==KINDS[kind]]


def validate_context(context):
    if (not isinstance(context,dict) or context.get('version') not in (protocol(5),protocol(9))
            or not isinstance(context.get('plan_id'),str) or not context['plan_id'].strip()
            or not isinstance(context.get('allocation_sha256'),str)
            or re.fullmatch('[0-9a-f]{64}',context['allocation_sha256']) is None):
        raise ValueError('Invalid typed identifier plan context')
    for key,kind in (('role_codes',2),('task_codes',5)):
        values=context.get(key)
        if (not isinstance(values,dict) or not values or
                any(not isinstance(k,str) or not k.strip() for k in values)):
            raise ValueError('Invalid typed identifier mapping: '+key)
        for value in values.values(): require(value,kind,digits=width(context))
        if len(set(values.values()))!=len(values): raise ValueError('Duplicate typed IDs in '+key)


def decomp_prompt(original, digits=9):
    protocol(digits)
    result=original.replace('structural_matrix_v1','structural_typed9_v1')
    result=result.replace('schema_version="structural_typed9_v1", instruction',
        'schema_version="structural_typed9_v1", codebook_sha256 (copy supplied allocation hash), instruction')
    start=result.index('Role/task/constraint codes start at 1')
    end=result.index('quantity is a positive integer;',start)
    result=result[:start]+'''All role, task and constraint IDs are issued by the supplied identifier_allocation.
Fill the first N slots of each kind in listed order, without gaps or renumbering.
Type 1 is executor; the robot is 100000001, never an entry in roles or labels.
Type 2 is object role, type 5 subtask, type 6 constraint. References must use the
same issued IDs as their definitions. These are not scene IDs. Preserve explicit
scene instance IDs in query text; never invent them. Copy codebook_sha256 exactly.
For held/lifted the subject MUST be 100000001, and the object a defined type-2 role.
For near the subject may be robot or a role; on/in subjects must be roles.
Do not define robot/agent/executor as an object role. Slot IDs are scoped to this
request hash, not global identity or row numbers. If capacity is insufficient,
do not invent more IDs or silently drop task content; let the caller reject.
''' +result[end:]
    return result if digits==9 else result.replace('typed9','typed5').replace(str(ROBOT),str(code(1,1,digits)))


def schema(stage,book=None,digits=None):
    from matrix_type_contract import integer as i, text as s, table as t, object_schema as o
    digits=(width(book) if book is not None else 9) if digits is None else digits
    protocol(digits)
    if book is not None and width(book)!=digits: raise ValueError('Identifier width differs from selected mode')
    robot=code(1,1,digits)
    if stage=='decomposition':
        if book is None: raise ValueError('Typed decomposition requires issued allocation')
        roles,tasks,cons=pool(book,2),pool(book,5),pool(book,6)
        props=dict(schema_version=s(f'structural_typed{digits}_v1'),codebook_sha256=s(book['sha256']),
            instruction=s(),task_name=s(),roles=t([i(roles),s(),s(),i(),i((0,1))],1,len(roles)),
            labels=t([i(roles),s()],1),tasks=t([i(tasks),s()],1,len(tasks)),
            task_roles=t([i(tasks),i(roles)]),dependencies=t([i(tasks),i(tasks)]),
            constraints=t([i(cons),i((1,2)),i([robot]+roles),i((1,2,3,4,5)),
                i(roles),i((-1,0,1)),i((0,1))],1,len(cons)))
    elif stage=='object_binding':
        if book is None: raise ValueError('Typed binding requires current codebook')
        props=dict(schema_version=s(f'object_binding_typed{digits}_v1'),codebook_sha256=s(book['codebook_sha256']),
            decisions=t([i([r['role_code'] for r in book['roles']]),i((1,2,3))]),
            matches=t([i([r['role_code'] for r in book['roles']]),i(book['active_scene_ids'])]),
            candidates=t([i([r['role_code'] for r in book['roles']]),i(book['active_scene_ids'])]),
            reasons=t([i([r['role_code'] for r in book['roles']]),s()]))
    elif stage=='skill_matching_matrix':
        cols=[i() for _ in range(4)]
        if book is not None:
            slots=[0]+[e['code'] for e in book['objects']+book['values']]
            cols=[i([e['code'] for e in book['subtasks']]),i([e['code'] for e in book['skills']]),i(slots),i(slots)]
        props=dict(schema_version=s(f'action_matrix_typed{digits}_v1'),
            codebook_sha256=s(None if book is None else book['sha256']),matrix=t(cols,0,256))
    else: raise ValueError('Unknown typed matrix stage')
    return o(props)


def check(raw,stage,book=None):
    from matrix_type_contract import validate_owned_schema
    validate_owned_schema(raw,schema(stage,book))


def decode_decomposition(raw,book):
    from structural_matrix import from_decomposition_matrix
    check(raw,'decomposition',book)
    digits=width(book); robot=code(1,1,digits)
    old=copy.deepcopy(raw); old.pop('codebook_sha256'); old['schema_version']='structural_matrix_v1'
    maps={}
    for key,kind in (('roles',2),('tasks',5),('constraints',6)):
        issued=pool(book,kind); refs={}
        for index,row in enumerate(old[key],1):
            if row[0]!=issued[index-1]:
                raise ValueError(f'{key}: definitions must fill issued slots in order')
            refs[row[0]]=index; row[0]=index
            if key=='roles' and row[1].strip().casefold() in ('robot','agent','executor'):
                raise ValueError('Reserved executor cannot be defined as an object role')
        maps[kind]=refs
    def ref(n,kind):
        require(n,kind,maps[kind],digits); return maps[kind][n]
    for row in old['labels']: row[0]=ref(row[0],2)
    for row in old['task_roles']: row[:]=[ref(row[0],5),ref(row[1],2)]
    for row in old['dependencies']: row[:]=[ref(row[0],5),ref(row[1],5)]
    for row in old['constraints']:
        subject,relation=row[2],row[3]
        if relation in (4,5):
            require(subject,1,{robot},digits)
            if row[5] not in (0,1): raise ValueError('State predicates require expected 0/1')
        elif subject==robot and relation!=3:
            raise ValueError('Executor spatial predicate only supports near')
        row[2]=0 if subject==robot else ref(subject,2)
        row[4]=ref(row[4],2)
    result=from_decomposition_matrix(old)
    result['identifier_context']=dict(version=protocol(digits),plan_id=book['plan_id'],allocation_sha256=book['sha256'],
        role_codes={f'role_{v:03d}':k for k,v in maps[2].items()},
        task_codes={f'st_{v:03d}':k for k,v in maps[5].items()})
    result['structural_matrix_schema']=f'structural_typed{digits}_v1'
    return result


def binding_book(base,context):
    validate_context(context)
    digits=width(context)
    result=copy.deepcopy(base)
    result['schema_version']=f'object_binding_typed{digits}_codebook_v1'
    result['identifier_context']=copy.deepcopy(context)
    for entry in result['roles']:
        entry['role_code']=require(context['role_codes'][entry['role_id']],2,digits=digits)
    result['scene_codes']=[dict(code=code(3,i,digits),element_id=n) for i,n in enumerate(base['active_scene_ids'],1)]
    result['active_scene_ids']=[e['code'] for e in result['scene_codes']]
    result.pop('codebook_sha256')
    result['codebook_sha256']=digest(result)
    return result


def decode_binding(raw,roles,nodes,book):
    from structural_matrix import make_binding_codebook,from_binding_matrix
    expected=make_binding_codebook(roles,nodes,book['identifier_context'])
    if book!=expected: raise ValueError('Stale or modified typed binding codebook')
    check(raw,'object_binding',book)
    digits=width(book)
    role_map={r['role_code']:i for i,r in enumerate(book['roles'],1)}
    node_map={r['code']:r['element_id'] for r in book['scene_codes']}
    old=copy.deepcopy(raw); old['schema_version']='object_binding_matrix_v1'
    legacy=make_binding_codebook(roles,nodes); old['codebook_sha256']=legacy['codebook_sha256']
    for key in ('decisions','matches','candidates','reasons'):
        for row in old[key]:
            row[0]=role_map[require(row[0],2,role_map,digits)]
            if key in ('matches','candidates'): row[1]=node_map[require(row[1],3,node_map,digits)]
    return from_binding_matrix(old,roles,nodes,legacy)


def without_context(alignment):
    result=copy.deepcopy(alignment)
    result['task_decomposition'].pop('identifier_context',None)
    return result


def action_book(base,context):
    validate_context(context)
    digits=width(context)
    result=copy.deepcopy(base); result.pop('sha256')
    result['schema_version']=f'action_matrix_typed{digits}_v1'
    result['identifier_context']=copy.deepcopy(context)
    for key,kind in (('objects',8),('values',7),('skills',4),('subtasks',5)):
        for index,entry in enumerate(result[key],1):
            entry['code']=code(kind,index,digits)
            if key=='subtasks': entry['code']=require(context['task_codes'][entry['subtask_id']],5,digits=digits)
    # Bound instance codes are separate from role codes because a role may have quantity > 1.
    roles=result['identifier_context']['role_codes']
    for entry in result['objects']:
        if entry['role_id'] not in roles:
            # Existing runtime closure may expose the exact held object without a model role.
            # Allocate an unused issued role slot; never add or change an action/binding.
            free=[code(2,i,digits) for i in range(1,CAPACITY['role']+1) if code(2,i,digits) not in roles.values()]
            if not free: raise ValueError('Typed role allocation capacity exhausted by runtime closure')
            roles[entry['role_id']]=free[0]
        entry['role_code']=roles[entry['role_id']]
    result['entries']=[dict(code=e['code'],kind=KINDS[kind],
        canonical=copy.deepcopy(e)) for key,kind in
        (('objects',8),('values',7),('skills',4),('subtasks',5)) for e in result[key]]
    return seal(result)


def decode_action(raw,book,alignment,library):
    from action_matrix import make_codebook,decode_matrix
    if book!=make_codebook(alignment,library): raise ValueError('Stale or modified typed action codebook')
    check(raw,'skill_matching_matrix',book)
    digits=width(book); base=10**(digits-1)
    legacy_alignment=without_context(alignment)
    legacy=make_codebook(legacy_alignment,library)
    maps={}
    for key,kind in (('objects',8),('values',7),('skills',4),('subtasks',5)):
        maps[kind]={new['code']:old['code'] for new,old in zip(book[key],legacy[key])}
    rows=[]
    for row in raw['matrix']:
        task=maps[5][require(row[0],5,maps[5],digits)]
        skill=maps[4][require(row[1],4,maps[4],digits)]
        args=[]
        for n in row[2:]:
            if n==0: args.append(0); continue
            kind=n//base
            if kind not in (7,8): raise ValueError('Action slot requires a bound instance or numeric preset')
            args.append(maps[kind][require(n,kind,maps[kind],digits)])
        rows.append([task,skill,*args])
    return decode_matrix({'matrix':rows},legacy,legacy_alignment,library)


def action_payload(alignment,library,book,model,retrieval,widened,validation_error):
    from action_matrix import make_codebook,matrix_payload,MATRIX_SYSTEM
    from matrix_type_contract import response_format
    if book!=make_codebook(alignment,library): raise ValueError('Modified typed action book')
    digits=width(book)
    old_alignment=without_context(alignment); old_book=make_codebook(old_alignment,library)
    payload=matrix_payload(old_alignment,library,old_book,model,retrieval,widened,validation_error)
    data=json.loads(payload['messages'][1]['content'])
    mapping={key:{a['code']:b['code'] for a,b in zip(old_book[key],book[key])}
             for key in ('skills','subtasks')}
    data['object_codes']=book['objects']; data['optional_numeric_preset_codes']=book['values']
    data['codebook_sha256']=book['sha256']; data['identifier_context']=book['identifier_context']
    for row in data['subtasks']:
        row['code']=mapping['subtasks'][row['code']]
        row['candidate_skill_codes']=[mapping['skills'][c] for c in row['candidate_skill_codes']]
    for row in data['skills']: row['code']=mapping['skills'][row['code']]
    data['matrix_encoding_contract'].update(version=protocol(digits),
        allowed_subtask_codes=[e['code'] for e in book['subtasks']],
        allowed_object_slot_codes=[e['code'] for e in book['objects']],
        bound_object_lookup=book['objects'],
        final_check='Use only current issued typed IDs and echo the current codebook hash. No scene IDs in slots.')
    system=MATRIX_SYSTEM.replace('{"matrix":[[subtask_code,skill_code,arg1_code,arg2_code],...]}',
        '{"schema_version":"action_matrix_typed'+str(digits)+'_v1","codebook_sha256":"COPY_CURRENT_HASH",'
        '"matrix":[[subtask_code,skill_code,arg1_code,arg2_code],...]}')
    system=system.replace('Only matrix is an output key.',
        'Only schema_version, codebook_sha256 and matrix are output keys.')
    system=system.replace('return {"matrix":[]}',
        'return the same schema_version and hash with matrix=[]')
    system+='\nType 5=subtask, 4=skill, 8=bound instance, 7=numeric preset; 0=optional padding only. '
    system+='Copy issued codes; IDs and hashes belong to the CURRENT plan, not earlier rounds.'
    payload['messages']=[dict(role='system',content=system),dict(role='user',content=json.dumps(data,ensure_ascii=True))]
    payload['response_format']=response_format('skill_matching_matrix',f'typed{digits}_schema',book)
    return payload
