"""Lossless wire-format normalization; never infer an absent action or parameter."""
import copy
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm_audit_v1'))
from matrix_type_contract import enforce, check_mode, MatrixTypeError

VERSION = 'skill_output_format_v1'


class NormalizationError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def action_count(value, kind):
    if kind == 'matrix':
        rows = value if isinstance(value, list) else value.get('matrix') if isinstance(value, dict) else None
        return len(rows) if isinstance(rows, list) else None
    if not isinstance(value, dict) or not isinstance(value.get('subtask_plans'), list):
        return None
    tasks = value['subtask_plans']
    if any(not isinstance(t, dict) or not isinstance(t.get('steps'), list) for t in tasks):
        return None
    return sum(len(t['steps']) for t in tasks)


def normalize(raw, kind, library, matrix_contract='legacy_json'):
    if kind not in ('nested_json', 'matrix'):
        raise ValueError('Unknown skill output format')
    check_mode(matrix_contract)
    value = copy.deepcopy(raw)
    changes, ambiguities = [], []
    type_record = None
    if kind == 'matrix' and matrix_contract in ('strict_schema','typed9_schema','typed5_schema'):
        # A provider schema violation must not be hidden by envelope/type repair.
        value, type_record = enforce(raw,'skill_matching_matrix',matrix_contract)
    if kind == 'matrix' and isinstance(value, list):
        value = {'matrix': value}
        changes.append(dict(rule='wrap_bare_matrix', path='$',
                            detail='Added matrix envelope; all rows and codes unchanged'))
    elif kind == 'nested_json' and isinstance(value, dict) and isinstance(value.get('subtask_plans'), list):
        allowed = {s['skill_id'] for s in library['skills']}
        for ti, task in enumerate(value['subtask_plans']):
            if not isinstance(task, dict) or not isinstance(task.get('steps'), list):
                continue
            for si, step in enumerate(task['steps']):
                if not isinstance(step, dict) or 's' not in step:
                    continue
                path = f'$.subtask_plans[{ti}].steps[{si}]'
                if 'skill_id' in step:
                    ambiguities.append(dict(path=path, reason='Both s and skill_id supplied; no precedence guessing'))
                elif isinstance(step['s'], str) and step['s'] in allowed:
                    step['skill_id'] = step.pop('s')
                    changes.append(dict(rule='rename_s_to_skill_id', path=path,
                                        skill_id=step['skill_id']))
                # An s object, empty s or unknown skill is NOT a resolvable alias.
                # Preserve it so the existing validator rejects the missing skill.
    # If any alias is ambiguous, reject the whole proposal without partial rewrites.
    if ambiguities:
        value, changes = copy.deepcopy(raw), []
    if kind == 'matrix' and matrix_contract == 'typed_json':
        value, type_record = enforce(value,'skill_matching_matrix',matrix_contract)
        changes.extend(type_record['changes'])
    before, after = action_count(raw, kind), action_count(value, kind)
    if before != after:
        raise NormalizationError('Action count changed during format normalization')
    record = dict(version=VERSION, format=kind,
        status='rejected_ambiguous' if ambiguities else 'normalized' if changes else 'unchanged',
        changed=bool(changes), changes=changes, ambiguities=ambiguities,
        raw_sha256=digest(raw), normalized_sha256=digest(value),
        raw_action_count=before, normalized_action_count=after,
        scope='Formatting only; no action/parameter filling, validity or execution claim')
    if type_record is not None:
        record['matrix_type_contract'] = type_record
    return value, record


def normalize_and_record(raw, kind, library, folder, stem, raw_evidence, matrix_contract='legacy_json'):
    try:
        value, record = normalize(raw, kind, library, matrix_contract)
    except MatrixTypeError as exc:
        folder.mkdir(parents=True, exist_ok=True)
        record=dict(version=VERSION,format=kind,status='rejected_type_contract',changed=False,
            changes=[],raw_evidence=str(raw_evidence),raw_sha256=exc.record['raw_sha256'],normalized_sha256=None,
            matrix_type_contract=exc.record)
        (folder/(stem+'.normalization.json')).write_text(json.dumps(record,ensure_ascii=True,indent=2)+'\n',encoding='ascii')
        raise
    record['raw_evidence'] = str(raw_evidence)
    folder.mkdir(parents=True, exist_ok=True)
    if record['changed']:
        record['normalized_evidence'] = stem+'.normalized.json'
        (folder/record['normalized_evidence']).write_text(
            json.dumps(value, ensure_ascii=True, indent=2)+'\n', encoding='ascii')
    (folder/(stem+'.normalization.json')).write_text(
        json.dumps(record, ensure_ascii=True, indent=2)+'\n', encoding='ascii')
    if record['ambiguities']:
        raise NormalizationError('AMBIGUOUS_SKILL_FIELD: both s and skill_id supplied')
    return value, record
