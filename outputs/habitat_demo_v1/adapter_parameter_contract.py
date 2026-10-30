"""Project planning declarations onto the serial proxy's actual parameter API.

This changes the advertised library BEFORE any request; it never strips fields
from model output or relaxes the executor's checks. General skill library stays
unchanged and unsupported required parameters cause preflight failure.
"""
import copy
import hashlib
import json

from sequential_proxy_core import PARAMETERS, serial_state_contract

VERSION = 'serial_proxy_exact_parameters_v1'
MODES = {'json_symbolic_test_only', 'serial_scene201_proxy'}
OBJECT_TYPES = {'scene_element', 'location', 'surface', 'container', 'person', 'control'}


def signature():
    return {skill: sorted(names) for skill, names in sorted(PARAMETERS.items())}


def signature_hash():
    return hashlib.sha256(json.dumps(signature(), sort_keys=True).encode()).hexdigest()


def assert_contract(library):
    contract = library.get('execution_contract', {})
    if contract.get('adapter_mode') not in MODES:
        raise ValueError('Parameter projection requires a serial proxy execution contract')
    if (contract.get('parameter_contract_version') != VERSION or
            contract.get('allowed_parameters_by_skill') != signature() or
            contract.get('parameter_signature_sha256') != signature_hash()):
        raise ValueError('Advertised parameter contract differs from SerialExecutor.PARAMETERS')
    if ('state_semantics_contract' in contract and
            contract['state_semantics_contract'] != serial_state_contract()):
        raise ValueError('Advertised state semantics differ from SerialExecutor')
    skills = library.get('skills', [])
    ids = [s['skill_id'] for s in skills]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(PARAMETERS):
        raise ValueError('Unsupported or duplicate skill in projected library')
    for skill in skills:
        specs = skill.get('parameters', [])
        names = [p['name'] for p in specs]
        if len(set(names)) != len(names) or set(names) != PARAMETERS[skill['skill_id']]:
            raise ValueError('Advertised parameters do not match executor: '+skill['skill_id'])
        if any(p.get('required') is not True or p.get('type') not in OBJECT_TYPES for p in specs):
            raise ValueError('Proxy requires explicit required object-binding parameters')
    return True


def project_library(library):
    """Copy a scoped library and omit ONLY optional declarations not executable here."""
    result = copy.deepcopy(library)
    contract = result.get('execution_contract', {})
    if contract.get('adapter_mode') not in MODES:
        raise ValueError('Parameter projection requires a serial proxy execution contract')
    excluded = []
    for skill in result['skills']:
        sid = skill['skill_id']
        if sid not in PARAMETERS:
            raise ValueError('Cannot silently project an unsupported skill: '+sid)
        kept = []
        for spec in skill['parameters']:
            if spec['name'] in PARAMETERS[sid]:
                kept.append(spec)
            elif spec.get('required', False):
                raise ValueError('Unsupported required adapter parameter: '+sid+'.'+spec['name'])
            else:
                excluded.append(dict(skill_id=sid, parameter=spec['name'],
                    reason='not_implemented_by_serial_proxy', original_spec=spec))
        skill['parameters'] = kept
    contract.update(parameter_contract_version=VERSION, allowed_parameters_by_skill=signature(),
                    parameter_signature_sha256=signature_hash(),
                    state_semantics_contract=serial_state_contract())
    result['execution_contract'] = contract
    # Preserve provenance when an already-projected library is used again.
    previous = result.get('adapter_parameter_projection', {}).get('excluded_optional_declarations', [])
    result['adapter_parameter_projection'] = dict(version=VERSION,
        source='SerialExecutor.PARAMETERS', excluded_optional_declarations=excluded or previous,
        model_output_fields_removed=False, executor_checks_relaxed=False)
    assert_contract(result)
    return result
