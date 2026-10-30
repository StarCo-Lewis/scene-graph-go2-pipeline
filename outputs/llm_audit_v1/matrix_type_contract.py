"""Typed matrix generation schemas and explicit, audited compatibility decoding.

No provider fallback, semantic repair, guessed reference or action insertion.
Only the small schema keyword subset generated here is validated locally.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

VERSION = 'matrix_type_contract_v1'
MODES = ('typed5_schema', 'typed9_schema', 'strict_schema', 'typed_json', 'legacy_json')
INTEGER_TEXT = re.compile(r'(?:0|[1-9][0-9]*|-[1-9][0-9]*)\Z', re.ASCII)
COLUMNS = {
    'decomposition': {'roles': ('i','s','s','i','i'), 'labels': ('i','s'),
        'tasks': ('i','s'), 'task_roles': ('i','i'), 'dependencies': ('i','i'),
        'constraints': ('i','i','i','i','i','i','i')},
    'object_binding': {'decisions': ('i','i'), 'matches': ('i','i'),
        'candidates': ('i','i'), 'reasons': ('i','s')},
    'skill_matching_matrix': {'matrix': ('i','i','i','i')},
}


class MatrixTypeError(ValueError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def check_mode(mode):
    if mode not in MODES:
        raise ValueError('Unknown matrix contract mode: '+str(mode))
    return mode


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        allow_nan=True, separators=(',', ':')).encode()).hexdigest()


def integer(values=None):
    values=None if values is None else list(values)
    return dict(type='integer', **({'enum': values} if values else {}))


def text(const=None):
    return dict(type='string', **({'const': const} if const is not None else {}))


def table(columns, minimum=0, maximum=2048):
    return dict(type='array', minItems=minimum, maxItems=maximum,
        items=dict(type='array', prefixItems=columns, minItems=len(columns), maxItems=len(columns)))


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def decomposition_schema():
    return object_schema(dict(schema_version=text('structural_matrix_v1'), instruction=text(), task_name=text(),
        roles=table([integer(),text(),text(),integer(),integer((0,1))],1),
        labels=table([integer(),text()],1), tasks=table([integer(),text()],1),
        task_roles=table([integer(),integer()]), dependencies=table([integer(),integer()]),
        constraints=table([integer(),integer((1,2)),integer(),integer((1,2,3,4,5)),
                           integer(),integer((-1,0,1)),integer((0,1))],1)))


def binding_schema(book=None):
    roles=None if book is None else [r['role_code'] for r in book['roles']]
    nodes=None if book is None else book['active_scene_ids']
    return object_schema(dict(schema_version=text('object_binding_matrix_v1'),
        codebook_sha256=text(None if book is None else book['codebook_sha256']),
        decisions=table([integer(roles),integer((1,2,3))]),
        matches=table([integer(roles),integer(nodes)]),
        candidates=table([integer(roles),integer(nodes)]), reasons=table([integer(roles),text()])))


def action_schema(book=None):
    if book is None:
        columns=[integer() for _ in range(4)]
    else:
        slots=[0]+[o['code'] for o in book['objects']]+[v['code'] for v in book['values']]
        columns=[integer([t['code'] for t in book['subtasks']]),
                 integer([s['code'] for s in book['skills']]),integer(slots),integer(slots)]
    # Empty is an explicit no-plan proposal; the existing domain decoder rejects it.
    return object_schema(dict(matrix=table(columns,0,256)))


def schema_for(stage, book=None):
    if stage == 'decomposition': return decomposition_schema()
    if stage == 'object_binding': return binding_schema(book)
    if stage == 'skill_matching_matrix': return action_schema(book)
    raise ValueError('Unknown matrix stage: '+str(stage))


def response_format(stage, mode, book=None):
    check_mode(mode)
    if mode in ('typed5_schema','typed9_schema'):
        from typed_matrix_ids import schema
        digits=5 if mode=='typed5_schema' else 9
        return dict(type='json_schema',json_schema=dict(name=stage+f'_typed{digits}_v1',
            strict=True,schema=schema(stage,book,digits)))
    if mode != 'strict_schema':
        return {'type':'json_object'}
    return dict(type='json_schema',json_schema=dict(name=stage+'_typed_matrix_v1',
        strict=True,schema=schema_for(stage,book)))


def validate_owned_schema(value, schema, path='$'):
    """No external dependency; not a general-purpose JSON Schema implementation."""
    kind=schema['type']
    expected={'object':dict,'array':list,'string':str,'integer':int}[kind]
    if type(value) is not expected:
        raise ValueError(f'{path}: expected {kind}, got {type(value).__name__}; no implicit conversion')
    if 'const' in schema and value!=schema['const']:
        raise ValueError(f'{path}: value differs from the required constant')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: value is outside the supplied enum/code inventory')
    if kind=='object':
        missing=set(schema['required'])-set(value)
        extra=set(value)-set(schema['properties'])
        if missing or extra:
            raise ValueError(f'{path}: missing fields={sorted(missing)}; extra fields={sorted(extra)}')
        for key,prop in schema['properties'].items():
            validate_owned_schema(value[key],prop,path+'.'+key)
    elif kind=='array':
        if not schema['minItems']<=len(value)<=schema['maxItems']:
            raise ValueError(f'{path}: invalid row/table length {len(value)}')
        if 'prefixItems' in schema:
            for index,(cell,prop) in enumerate(zip(value,schema['prefixItems'])):
                validate_owned_schema(cell,prop,f'{path}[{index}]')
        else:
            for index,item in enumerate(value):
                validate_owned_schema(item,schema['items'],f'{path}[{index}]')


def enforce(raw, stage, mode, book=None):
    """Atomic normalization only in typed_json; strict_schema never repairs output."""
    check_mode(mode)
    before=fingerprint(raw)
    record=dict(version=VERSION,stage=stage,mode=mode,status='unchanged',changed=False,
        raw_sha256=before,normalized_sha256=before,changes=[],
        raw_schema_conformant=None,semantic_validity_checked=False,
        scope='Column types only; no reference/role/action/goal/state inference or provider fallback')
    if mode=='legacy_json':
        record['status']='legacy_unchecked'
        return copy.deepcopy(raw),record
    value=copy.deepcopy(raw)
    changes=[]
    try:
        if mode=='typed_json':
            if type(value) is not dict:
                raise ValueError('$: expected a matrix object, not an inferred wrapper')
            for key,types in COLUMNS[stage].items():
                rows=value.get(key)
                if type(rows) is not list or len(rows)>2048:
                    raise ValueError(f'$.{key}: missing or invalid table')
                for ri,row in enumerate(rows):
                    if type(row) is not list or len(row)!=len(types):
                        raise ValueError(f'$.{key}[{ri}]: invalid fixed-column row; no flattening')
                    for ci,kind in enumerate(types):
                        cell=row[ci]
                        if kind=='i' and type(cell) is str:
                            if len(cell)>32 or INTEGER_TEXT.fullmatch(cell) is None:
                                raise ValueError(f'$.{key}[{ri}][{ci}]: not a canonical decimal integer string')
                            row[ci]=int(cell)
                            changes.append(dict(rule='canonical_integer_string_by_column',
                                path=f'$.{key}[{ri}][{ci}]',before=cell,after=row[ci]))
        if mode in ('typed5_schema','typed9_schema'):
            from typed_matrix_ids import schema
            validate_owned_schema(value,schema(stage,book,5 if mode=='typed5_schema' else 9))
        else:
            validate_owned_schema(value,schema_for(stage,book))
    except ValueError as exc:
        record.update(status='rejected',normalized_sha256=None,raw_schema_conformant=False,
                      error=str(exc),changes=[],attempted_changes_discarded=changes)
        raise MatrixTypeError('MATRIX_TYPE_CONTRACT_REJECTED: '+str(exc),record) from exc
    record.update(status='normalized' if changes else 'unchanged',changed=bool(changes),
                  changes=changes,normalized_sha256=fingerprint(value),raw_schema_conformant=not bool(changes))
    return value,record


def enforce_and_record(raw,stage,mode,folder,stem,book=None):
    from response_audit import save
    try:
        value,record=enforce(raw,stage,mode,book)
    except MatrixTypeError as exc:
        if folder is not None:
            save(Path(folder)/(stem+'.type_contract.json'),exc.record)
        raise
    if folder is not None:
        record['raw_evidence']='Original model response is retained separately; this record never replaces it'
        if record['changed']:
            record['normalized_evidence']=stem+'.typed.json'
            save(Path(folder)/record['normalized_evidence'],value)
        save(Path(folder)/(stem+'.type_contract.json'),record)
    return value,record
