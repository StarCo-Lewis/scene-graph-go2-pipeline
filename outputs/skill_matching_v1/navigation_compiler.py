"""Opt-in exact-ID navigation assembly; not a general language or motion planner.

Only an explicit, unconditional visit sequence with one bound role per subtask
qualifies. Other instructions retain the LLM route. All usual validators still
run afterwards, and compiled actions are never labelled LLM responses.
"""
import re

from match_skills import validate_proposal


_PREFIX = r'(?:(?:and\s+then|then|next|first|finally)\s*,?\s*)?'
_VERB = r'(?:visit|approach|(?:go|move|navigate|return)\s+(?:to|near))'
_NOUN = r'[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*)*'
_NAME = r'(?:' + _NOUN + r'\s+)?'
_CLAUSE = re.compile(_PREFIX + _VERB + r'\s+' + _NAME + r'(?:ID\s+(\d+)|\(\s*ID\s+(\d+)\s*\))', re.I)
_UNSAFE_NAME = re.compile(r'(?<![A-Za-z_-])(?:and|or|then|after|before|while|until|if|unless|not|without|with|'
    r'inspect|check|verify|measure|scan|photograph|grasp|pick|lift|place|deliver|carry|'
    r'take|hold|holding|release|remove|delete|open|close|cut|eat|turn|twice|again)(?![A-Za-z_-])', re.I)
_DESCRIPTION = re.compile(_VERB + r'\s+(?:ID\s+\d+|' + _NOUN + r'(?:\s+(?:\(\s*ID\s+\d+\s*\)|ID\s+\d+))?)'
    r'(?:\s+by\s+moving\s+near\s+it)?[.]?', re.I)


def explicit_visits(instruction):
    """Return ordered IDs for the narrow grammar, or None; never scan substrings."""
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    text = instruction.strip()
    visits = []
    while text:
        match = _CLAUSE.match(text)
        if match is None:
            return None
        # Sequence prefixes contain 'then'; exclude them from the noun guard.
        if _UNSAFE_NAME.search(re.sub('^'+_PREFIX, '', match.group(0), flags=re.I)):
            return None
        visits.append(int(match.group(1) or match.group(2)))
        text = text[match.end():].strip()
        if not text or text in ('.', ',', ';'):
            return visits
        separator = re.match(r'(?:[,;]\s*(?:and\s+)?|and\s+)', text, re.I)
        if separator is None:
            return None
        text = text[separator.end():].strip()
        if not text:
            return visits
    return visits or None


def compile_navigation(alignment, library, instruction, nodes, snapshot):
    """Return (proposal or None, audit). Eligibility is not execution success."""
    audit = {'schema_version': '1.0', 'route': 'llm_required',
             'compiler': 'exact_id_navigation_v1', 'source': 'validated_object_alignment',
             'memory_version': snapshot.get('backend', {}).get('memory_version'),
             'simulation_executed': False, 'training_eligible': False}

    def defer(reason):
        audit['reason'] = reason
        return None, audit

    visits = explicit_visits(instruction)
    if visits is None:
        return defer('Instruction outside the explicit unconditional ID-visit grammar')
    audit['ordered_requested_ids'] = visits
    decomposition = alignment.get('task_decomposition', {})
    if alignment.get('status') != 'success' or decomposition.get('instruction') != instruction:
        return defer('Unresolved alignment or changed instruction')
    skills = [s for s in library.get('skills', []) if s.get('skill_id') == 'navigate_to_object']
    contract = library.get('execution_contract', {})
    if len(skills) != 1 or ('supported_skill_ids' in contract and
                          'navigate_to_object' not in contract['supported_skill_ids']):
        return defer('Navigation not uniquely present in the supported inventory')
    subtasks = decomposition.get('subtasks', [])
    if len(subtasks) != len(visits):
        return defer('Requires exactly one navigation subtask per requested visit')
    active = {n['id'] for n in nodes if type(n.get('id')) is int}
    objects = snapshot.get('backend', {}).get('objects', {})
    bindings = alignment.get('context_bindings', {})
    plans, used_roles = [], set()
    for subtask, target in zip(subtasks, visits):
        description = subtask.get('description', '').strip()
        # A schema-valid mixed-action decomposition must not be silently replaced.
        if (not _DESCRIPTION.fullmatch(description) or _UNSAFE_NAME.search(description)):
            return defer('Subtask description is not a single pure navigation intent')
        described_ids = re.findall(r'\bID\s+(\d+)\b', description, re.I)
        if described_ids and list(map(int, described_ids)) != [target]:
            return defer('Subtask description and explicit visit disagree')
        roles = subtask.get('required_roles', [])
        if len(roles) != 1 or not isinstance(roles[0], dict):
            return defer('Requires exactly one grounded role per navigation subtask')
        role = roles[0]
        role_id = role.get('role_id')
        binding = bindings.get(role_id, {}) if isinstance(role_id, str) else {}
        ids = binding.get('element_ids', [])
        if (not role_id or type(role.get('quantity')) is not int or role.get('quantity') != 1
                or role.get('required', True) is not True
                or len(ids) != 1 or type(ids[0]) is not int or ids != [target]
                or binding.get('alternatives')):
            return defer('Binding not unique, certain and identical to requested ID')
        if target not in active or str(target) not in objects:
            return defer('Target absent from active graph or current state')
        if target == snapshot.get('held_object_id'):
            return defer('Cannot navigate to a held object')
        used_roles.add(role_id)
        plans.append({'subtask_id': subtask['subtask_id'], 'intent': description,
            'steps': [{'skill_id': 'navigate_to_object',
                       'arguments': {'target': {'role_id': role_id, 'element_id': target}}}]})
    if set(bindings) != used_roles:
        return defer('Unused bindings: preserve the original LLM validation route')
    declared = decomposition.get('task_roles')
    if declared is not None and {r.get('role_id') for r in declared} != used_roles:
        return defer('Unused or missing task roles')
    goals = [g for g in decomposition.get('goal_state', []) if g.get('required', True)]
    if not goals or any(g.get('subject_role') != 'robot' or g.get('relation') != 'near' or
                       bindings.get(g.get('object_role'), {}).get('element_ids') != [visits[-1]] for g in goals):
        return defer('End goals are not robot-near the last requested destination')
    proposal = {'schema_version': '1.0', 'subtask_plans': plans}
    validate_proposal(proposal, alignment, library)
    audit.update(route='compiled', reason='Explicit visits agree with unique validated bindings',
                 explicit_repetitions_preserved=True, required_downstream_review=True,
                 skill_proposal_llm_requests=0)
    return proposal, audit
