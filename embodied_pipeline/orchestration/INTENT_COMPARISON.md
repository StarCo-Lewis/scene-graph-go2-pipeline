# Complete-intent prompt comparison

This opt-in experiment preserves the existing default entry points and legacy
engine sources. It reuses the 11 reviewed Oracle-study tasks and their original
scene, runtime, user-context and conversation-history snapshots.

## Design

- `baseline_prompt`: the existing decomposition prompt, freshly sampled.
- `intent_constraints_prompt`: the same prompt plus the English addendum in
  `planning/intent_prompt.py`.
- Both arms freshly generate decomposition, object binding and action matrices.
  No human decomposition is injected into either arm.
- Both arms use `typed5_schema` and prerequisite-complete skill candidates.
  Fresh identifier scopes are issued per branch.
- The treatment requests coherent intentions with relevant existing constraints,
  not a fixed number of subtasks or an exhaustive list of protected objects.
- Full factual inputs are retained. Reviewed answers are available only to the
  post-execution evaluator, not the model request.
- Branches start independently from the archived snapshots. This is not a new
  end-to-end dialogue rollout, and one branch's errors do not alter another's input.
- The experiment retains the existing feedback commits and transport retries.
  It does not add clarification, semantic repair, or action recovery.

Default: 11 cases × 2 prompt arms × 1 repeat = 22 branches. Arm order alternates.
The cases are C01, A01, A02, A04, A06, A07, A08, A09, A10, A11 and C00.
A09 and C00 were identity controls in the earlier Oracle study; here both arms
generate new decompositions, so they are prompt-comparison cases, not identity
interventions.

## Run in WSL

From the repository root, with the existing `DASHSCOPE_API_KEY` environment
variable and embedding environment available:

```bash
/home/lewis/miniforge3/envs/conceptgraph/bin/python -m embodied_pipeline intent-comparison
```

Offline input preparation only:

```bash
python -m embodied_pipeline intent-comparison --preflight-only
```

On Windows, use a short, new `--output-dir` if the default path exceeds the
platform's path-length limit. No key is needed for preflight.

Optional `--case-ids A04 A06` selects a subset; `--repeat 3` creates three paired
repeats. Output directories must be new. API transport failures are recorded per
branch and independent branches continue by default. `--stop-on-transport-error`
restores the previous stop policy. Other infrastructure or integrity failures
still stop the run; ordinary rejected proposals remain recorded.

Transport failures have `status=transport_error`, no semantic evaluation, and no
execution-state commit. Their request audits and known/unknown usage are retained.
A run finishing all selected branches with such failures reports
`completed_with_transport_errors`, not all-task success. The legacy Oracle entry
retains its default stop policy; `--continue-on-transport-error` enables the same
branch isolation there. Existing legacy source files are unchanged.

To fill unfinished branches after a stopped run:

```bash
python -m embodied_pipeline intent-comparison --resume-from outputs/vlm_full_scene_v1/runs/PREVIOUS_RUN/comparison_summary.json
```

This creates a new run containing only branches absent from the source summary's
completed rows. Completed model failures are retained and skipped just like
successes. The interrupted branch restarts from its archived input, including a
new decomposition; it does not reuse the earlier decomposition or continue an
in-flight binding request. The old timeout and any partial output remain in the
old directory. The new summary links the source path and hash for later combined
analysis, including the original transport failures and their unknown usage.
Repeat counts are inherited. With `--resume-from`, optional `--case-ids A11 C00`
selects only those unfinished cases; other unfinished branches are listed under
`deferred_branch_keys` and remain unresolved. Completed rows, including recorded
transport-error rows, are skipped rather than automatically retried.
Prompt hashes and source integrity must
match before execution. This selective retry does not change the prompts or add
semantic repair. Results collected at different times should retain their run
provenance when compared.

## Evidence and interpretation

### Process-constraint contract (2026-10-21)

New runs default to `complete_intent_relevant_constraints_process_v2`.
Both prompt arms now share an explicit `process_constraints_v1` extension to
typed5 decomposition: root `process_constraint_contract="process_constraints_v1"`
and `process_constraints=[[object_role_code,1],...]`. Rule 1 means
`do_not_manipulate` throughout this round. Empty arrays are valid. All model
inputs and contract prose remain English.

Only actionable intentions belong in `tasks`; every task still requires ordered,
contiguous action coverage. Protection belongs in the separate process table and
the affected intent's text. Initial assertions and final goals retain their old
scopes. Protected roles are grounded even if they are used only by a process
restriction. A temporary in-memory role-inventory bridge is removed before
binding/action generation; no task, action, binding or spatial fact is inferred.
Live passive-task detection/migration from English text is deliberately absent.
The model must use the explicit contract; no keyword-based task deletion occurs.

Grounded restrictions are forwarded in the action request's execution contract
and saved in `process_constraints_grounded.json`. Feedback contains
`process_constraint_evaluation`: whole-trajectory manipulation attempts,
pose/support/held-state changes and missing evidence are checked. Moving a
protected object and restoring it still fails. Using an unchanged protected
surface as the support of another payload is not itself moving that surface.
Unknown future skill effects produce an unknown audit, not a clean pass.
The current proxy exposes center/support/room and gripper state, not full physical
pose/contact. The audit runs after proxy effects and does not prevent or undo
execution or memory commits. It participates in the post-execution evaluation.

This is a shared engineering intervention in both arms, not merely a new
treatment prompt. Do not pool v1/v2 results as one unchanged experiment.
`--legacy-process-contract` selects the historical representation on a fresh
run. Resume automatically inherits the source profile; explicitly mixing
contracts during resume is rejected.

To retest the two regressions in a new run (run manually in WSL):

```bash
/home/lewis/miniforge3/envs/conceptgraph/bin/python -m embodied_pipeline intent-comparison --process-constraints --case-ids C01 A01
```

Only do-not-manipulate is machine-encoded by this extension. Other prohibitions
remain in intent text and are not claimed as newly verified predicates.
Passive-only requests are unsupported by this transport-planning profile.
Offline migration fixtures are explicitly authored, not fresh model successes;
historical rejected records remain untouched.

The run writes `comparison_summary.json`. Each branch retains its exact
decomposition system prompt, request/response audit, decomposition, binding,
candidate audit, action proposal, execution feedback, memory and evaluation.
Summary rows include decomposition validation, subtask count, object-binding
acceptance, skill-matrix acceptance, execution status and private-reference
checks. Duplicate binding-reason rejection is separately flagged; it is not fixed
or discarded by this experiment.

Protocol acceptance is not semantic correctness. Authored reference checks are
not independent semantic or physical ground truth; inspect raw outputs before
claiming success. Preflight is not a model trial. Missing provider usage is
unknown, not zero.

The treatment bundles intent granularity, explicit relevant constraints and
cross-field consistency instructions. A difference cannot isolate which of those
caused it. One repeat is exploratory; historical Oracle results are descriptive
context rather than a contemporaneous control. No Habitat or physical execution
claim follows from this symbolic proxy experiment.
