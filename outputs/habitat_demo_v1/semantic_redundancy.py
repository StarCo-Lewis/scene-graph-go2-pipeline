"""Narrow English repetition checks for fixed-height Scene201 proxy lift only."""
import re


class SemanticRedundancyError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def lift_repetition_policy(instruction):
    text = instruction.lower()
    # Only narrowly recognized positive imperative forms authorize extra calls.
    # 'again' alone NEVER authorizes two lifts in the same newly grasped cycle.
    patterns = [
        (r'\blift\b[^.;!?]{0,80}?\btwice\b', 2),
        (r'\blift\b[^.;!?]{0,80}?\b(?:two|2)\s+times\b', 2),
        (r'\blift\b[^.;!?]{0,80}?\b(?:and then|then|and)\s+lift\b[^.;!?]{0,80}?\bagain\b', 2),
    ]
    negation = re.compile(r"\b(?:not|never|without|avoid|don't|do not)\b")
    for pattern, count in patterns:
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start() - 30):match.end()]
            if not negation.search(context):
                return {'max_lifts_per_grasp_cycle': count, 'min_lifts_this_round': count,
                        'explicit_repeat_quote': instruction[match.start():match.end()],
                        'allow_initial_repeat_once': True, 'recognition': 'bounded_positive_English_form'}
    initial_repeat = False
    for match in re.finditer(r'\blift\b[^.;!?]{0,80}?\bagain\b', text):
        if not negation.search(text[max(0, match.start() - 30):match.end()]):
            initial_repeat = True
    return {'max_lifts_per_grasp_cycle': 1, 'min_lifts_this_round': 0,
            'explicit_repeat_quote': None, 'allow_initial_repeat_once': initial_repeat,
            'recognition': 'default_single_lift; again_does_not_mean_twice'}


def check_lift(step, executor, cycle_lifts, policy):
    target = step['arguments']['object']['element_id']
    if executor.held_object_id == target and executor.lifted:
        allowed = (cycle_lifts < policy['max_lifts_per_grasp_cycle'] and
                   (policy['explicit_repeat_quote'] is not None or
                    (cycle_lifts == 0 and policy['allow_initial_repeat_once'])))
        if not allowed:
            details = {'code': 'REDUNDANT_LIFT_ALREADY_LIFTED', 'step_id': step['step_id'],
                'target_id': target, 'lifts_in_current_grasp_cycle': cycle_lifts,
                'state': {'held_object_id': executor.held_object_id, 'lifted': executor.lifted},
                'policy': policy,
                'correction': 'Merge keep-holding into the pickup/lift intent and end-state assertion. '
                    'Do not create a second lift for historical again. Re-decompose if redundant intents require empty steps.'}
            raise SemanticRedundancyError('REDUNDANT_LIFT_ALREADY_LIFTED: ' + details['correction'], details)
        return {'step_id': step['step_id'], 'status': 'explicitly_requested_proxy_noop_repeat',
                'physical_incremental_lift_verified': False}
    return None
