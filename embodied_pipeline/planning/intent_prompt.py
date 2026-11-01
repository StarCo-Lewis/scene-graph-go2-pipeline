"""Task-agnostic prompt treatment; no case answers or private references."""

# v2: rule 7, ordered visits keep intermediate destinations out of final goals.
VERSION = "complete_intent_relevant_constraints_v2"
ARMS = ("baseline_prompt", "intent_constraints_prompt")

ADDENDUM = """
COMPLETE INTENTS AND RELEVANT CONSTRAINTS
Apply the following organization rules within the existing matrix contract.
1. Each task row should describe a coherent user-level intention with a clear
completion condition. For an ordinary transport intention, describe the payload,
the supported source when known, the destination, and whether the object is to
be released or kept held. Let downstream skill planning choose navigation,
preparation, grasping, lifting and placement actions. Do not make each of those
atomic skills a separate task merely because it may be needed.
2. Split intentions where the instruction introduces an independently meaningful
ordered objective, an explicit intermediate visit, or a different completion
condition. Preserve every requested repetition and ordering dependency. Do not
merge across such a boundary simply to reduce the task count. There is no fixed
number of task rows to produce.
3. Include the relevant existing constraints directly in the affected English
task descriptions: objects that must not be moved, prohibited operations,
required ordering, final release or continued holding, and relevant factual
uncertainty. Carry through only requirements supported by the instruction and
applicable supplied context. Do not invent protections for unrelated objects,
new goals, observations, inspection actions, or unsupported source locations.
4. A protection such as 'do not move this object' applies throughout the task;
an unchanged final support alone is not equivalent. Preserve the process
restriction in the affected task text. Encode supported initial facts and final
conditions in the existing constraint table as appropriate, without inventing
a new relation or using an initial assertion to assign a desired state.
5. Define every participating payload, source/destination surface and protected
object that is needed for a task or constraint as its own role, then reuse that
role consistently. Link roles to the tasks in which they participate. A role
used only by a constraint remains valid and requires no artificial action.
6. The task text, role descriptions and constraint rows must express the same
requirements and agree with the current input state. Preserve unknown facts as
unknown. A successful transport goal does not establish any unobserved effect
such as contents, drinkability, or satisfaction of a human need.
7. For ordered navigation such as 'visit A, then B', keep each visit as its own
ordered task with a dependency on the preceding visit, but put only the robot
near the last destination in the final conditions of the constraint table.
Earlier visits are intermediate milestones completed by their own task; do not
also keep them as final robot-near conditions, because the robot cannot end the
task near several separate destinations. For a return such as A, then B, then
A, keep all three visits as ordered tasks and require only robot near A at the
end. Do not drop a requested visit or invent a visited relation; ask for
clarification when explicitly requested final conditions conflict.
"""


def compose(base, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown prompt arm: {arm}")
    return base if arm == ARMS[0] else base + "\n" + ADDENDUM
