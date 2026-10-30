"""Offline checks of advertised state effects against the real serial dispatcher."""
import unittest

from sequential_proxy_core import SerialExecutor, ExecutionRejected, serial_state_contract


class Backend:
    active_ids = mapped_ids = {7, 19}

    def snapshot(self): return {}
    def navigate(self, target, held): return {'status': 'arrived'}
    def require_near(self, target): pass
    def require_graspable_instance(self, target): pass
    def attach(self, target): pass
    def lift(self, target): pass
    def require_placeable_pair(self, target, surface): pass
    def place(self, target, surface): return {'status': 'proxy_instance_support_passed'}


class StateContractTests(unittest.TestCase):
    def setUp(self):
        self.executor = SerialExecutor(Backend())
        self.alignment = {'context_bindings': {'r7': {'element_ids': [7]}, 'r19': {'element_ids': [19]}}}

    def action(self, skill, **values):
        args = {k: {'role_id': 'r'+str(v), 'element_id': v} for k, v in values.items()}
        before = self.executor.snapshot()
        self.executor.execute({'step_id': 'unit', 'skill_id': skill, 'arguments': args}, self.alignment)
        after = self.executor.snapshot()
        for field, effect in serial_state_contract()['successful_action_effects'][skill].items():
            if effect == 'unchanged': expected = before[field]
            elif effect == 'null': expected = None
            elif effect == 'true': expected = True
            elif effect == 'false': expected = False
            else: expected = values[effect.split('.')[1]]
            self.assertEqual(after[field], expected, (skill, field))
        return after

    def test_all_five_effects_and_valid_null_preparation_during_carry(self):
        self.action('navigate_to_object', target=7)
        self.action('position_arm_for_object', target=7)
        self.action('grasp_object', target=7)
        self.action('lift_held_object', object=7)
        state = self.action('navigate_to_object', target=19)
        self.assertEqual((state['held_object_id'], state['lifted'], state['prepared_target_id']), (7, True, None))
        # Null preparation does not release, but still prevents placement.
        with self.assertRaisesRegex(ExecutionRejected, 'SURFACE_PROXY_PREPARATION_REQUIRED'):
            self.action('place_object_on_surface', object=7, surface=19)
        self.assertEqual(self.executor.snapshot(), state)
        self.action('position_arm_for_object', target=19)
        self.action('place_object_on_surface', object=7, surface=19)

    def test_navigation_before_lift_still_rejected(self):
        self.action('position_arm_for_object', target=7)
        self.action('grasp_object', target=7)
        state = self.executor.snapshot()
        with self.assertRaisesRegex(ExecutionRejected, 'HELD_OBJECT_MUST_BE_LIFTED'):
            self.action('navigate_to_object', target=19)
        self.assertEqual(self.executor.snapshot(), state)

    def test_failed_support_after_release_is_failure_not_fictitious_hold(self):
        self.action('position_arm_for_object', target=7)
        self.action('grasp_object', target=7)
        self.action('lift_held_object', object=7)
        self.action('position_arm_for_object', target=19)
        self.executor.backend.place = lambda *unused: {'status': 'failed'}
        with self.assertRaisesRegex(ExecutionRejected, 'PLACEMENT_SUPPORT_NOT_VERIFIED'):
            self.action('place_object_on_surface', object=7, surface=19)
        state = self.executor.snapshot()
        self.assertEqual((state['held_object_id'], state['lifted'], state['prepared_target_id']), (None, False, None))

    def test_request_mutation_does_not_change_contract(self):
        value = serial_state_contract()
        value['successful_action_effects']['navigate_to_object']['held_object_id'] = 'null'
        self.assertEqual(serial_state_contract()['successful_action_effects']['navigate_to_object']['held_object_id'], 'unchanged')


if __name__ == '__main__': unittest.main()
