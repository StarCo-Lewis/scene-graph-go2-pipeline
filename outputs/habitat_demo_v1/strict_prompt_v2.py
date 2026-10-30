"""Opt-in prompt-only treatment. No validators, actions, budgets or state repairs."""
import hashlib

from run_navigation_demo import read_json, write_json


CONTRACT = '''STRICT PROMPT V2 -- mandatory output and semantic checklist.
These instructions refine the existing JSON-only execution contract, never authorize
unsupported actions or fabricated observations. Return only the schema requested by
the current stage, without commentary, wrappers, shorthand keys or empty actions.

TASK SCOPE AND STATE:
Use current_runtime_state as the authoritative symbolic input, NOT the model's
assumed initial state. Check robot position, held_object_id, lifted and preparation.
If hierarchy_context exists, execute ONLY active_segment. Use original_instruction
and completed_segments to resolve pronouns and inherited constraints, not to execute
future segments. A carry-only segment must NOT include placement belonging to the
next segment. Do not repeat a completed prerequisite merely to make this segment
look self-contained. A required explicit visit in the active instruction still must
be represented, even when already near; do NOT remove deliberate repeated visits.
For pickup or placement that does not explicitly request a visit, add approach only
when the supplied current position does not satisfy the specified proximity rule.
If proximity cannot be established, do not assume it. Never navigate to a held object.
If already held/lifted, do not grasp/lift again unless explicitly requested.

DECOMPOSITION AND GOALS:
goal_state is the conjunction of states at the END of the active instruction or
segment. Intermediate visits are ordered subtasks, NOT simultaneous final near goals.
For A then B, preserve both visits in order; final robot-near concerns the last
destination, not both A and B. For the supplied JSON symbolic backend, near completion
goals have subject_role=robot and one bound destination object_role. Object-object
near is NOT implemented as a completion predicate. Use on for a requested completed
placement, together with robot held OBJECT_ROLE expected_value=false. For pickup
and keep-holding use robot held OBJECT_ROLE; for lift-and-hold also robot lifted
OBJECT_ROLE. expected_value is a JSON boolean, default true. These are explicit
state goals mapped to FINAL executor fields, not scene edges. Initial robot held/lifted
may be asserted in initial_scene_constraints with the same boolean schema, but must
match the supplied INPUT runtime snapshot after grounding. Never assign state or
insert future actions to make an initial assertion true. Unknown must stay unknown.
Do not invent holding/held_by/room predicates or replace held/lifted goals with near.
Do not erase those task requirements: proposed actions and resulting state must fulfill them.
If a task truly needs unsupported capabilities, do not substitute a simpler task.
All role references must resolve; robot is the only implicit subject role. Reuse
stable role_id values, list only needed roles, and reference every defined object role.
Preserve explicit object IDs in the instruction/query and all negations and ordering.
For flat_roles output, semantic_labels and required_role_ids are FLAT arrays of
nonempty strings, not arrays of arrays; quantity is a positive integer and required
is a boolean, not strings. Do not generate atomic skill steps during decomposition.

SKILL JSON:
subtask_plans IDs and their array order must exactly match the supplied decomposition.
Each entry has subtask_id, intent and steps. Each steps entry DIRECTLY contains
skill_id and arguments. No solution/plan/steps/subtask wrappers inside an action,
no s1/s2 shortcuts, no empty objects, and no misspelled keys such as }skill_id.
Use only supplied skill IDs and their exact required parameter names. Object-valued
arguments contain role_id and integer element_id grounded in available_objects.
Do not omit arguments even when the target is obvious from natural language.
Preserve every explicit visit, payload, destination, release/keep-holding constraint.
Do not copy illustrative IDs or example actions as the actual solution.

BEFORE RETURNING (silently, no extra output): check JSON field types and nesting,
all references and subtask IDs/order, supported predicates, final-vs-intermediate
goals, active-segment boundary, and prerequisites against the supplied current state.
Never claim execution success; do not invent a replacement plan when infeasible.'''

SPLIT = '''STRICT PARTITION V2: Copy source substrings verbatim, including every
space after punctuation. Concatenation of parts must equal current_text byte-for-byte
for this JSON string. Before returning, mentally concatenate them and check spaces,
punctuation, IDs and negations. Never split off keep-holding/no-placement constraints
as an independent action. Never split inside a name, ID or word. Do not drop an
explicit repeated visit just because its destination was visited previously.
Use the supplied latest state, not predicted future state. Preserve coherent intent
groups where further subdivision would detach an argument or constraint. Do not
invent a leaf merely to evade the depth limit. Use only the requested decision schema.'''


def fingerprints():
    return {name: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for name, text in {'execution_contract_appendix': CONTRACT, 'split_system_appendix': SPLIT}.items()}


def apply_contract(folder):
    """Only add prompt text to two inputs already sent to the model."""
    policy = read_json(folder/'execution_contract.json')
    library = read_json(folder/'execution_skill_library.json')
    if 'strict_prompt_v2' in policy:
        if policy['strict_prompt_v2'] != CONTRACT:
            raise ValueError('Conflicting strict prompt treatment')
        if library['execution_contract'] != policy:
            raise ValueError('Prompt contract/library mismatch')
        return
    policy['strict_prompt_v2'] = CONTRACT
    library['execution_contract'] = policy
    write_json(folder/'execution_contract.json', policy)
    write_json(folder/'execution_skill_library.json', library)
    write_json(folder/'prompt_profile.json', {'profile': 'v2', 'sha256': fingerprints(),
        'treatment': 'prompt text only; no schema/state/validator/retry changes'})


def hierarchical_v2(args, folder, limits, state_guards=False):
    from hierarchical_json_planner import plan_hierarchical, request_decision, plan_leaf
    def decide(a, directory, stage, system, payload, check):
        if stage == 'hierarchy_split':
            system += '\n\n' + SPLIT
        return request_decision(a, directory, stage, system, payload, check)
    def leaf(a, reviewer):
        apply_contract(a.output_dir)
        a.compile_navigation = getattr(args, 'compile_navigation', False)
        return plan_leaf(a, reviewer=reviewer)
    write_json(folder/'prompt_profile.json', {'profile':'v2_state_guards' if state_guards else 'v2',
                                            'sha256':fingerprints()})
    if state_guards:
        return plan_hierarchical(args, folder, limits, decide=decide, leaf_planner=leaf, state_guards=True)
    return plan_hierarchical(args, folder, limits, decide=decide, leaf_planner=leaf)
