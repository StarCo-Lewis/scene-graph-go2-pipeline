"""Occurrence IDs distinguish equal actions/states but never certify success."""
import copy
import unittest

from test_json_long_horizon import nodes, row
from run_json_long_horizon import initial_state
from action_identity import assign_action_identity,validate_action_identity
from action_sequence import refresh_sequence,validate_sequence
from json_symbolic_review import review_symbolic


class ActionIdentityTests(unittest.TestCase):
    def fixture(self):
        fixture,library=row([('navigate_to_object',13),('navigate_to_object',13)])
        scope={'case_id':'caseA','round':4,'run_id':'runA'}
        plan=refresh_sequence(assign_action_identity(fixture['plan'],scope))
        return fixture,library,plan

    def test_same_skill_arguments_and_state_still_have_two_distinct_ids(self):
        fixture,library,plan=self.fixture()
        steps=plan['subtask_plans'][0]['steps']
        self.assertEqual(steps[0]['arguments'],steps[1]['arguments'])
        self.assertNotEqual(steps[0]['action_id'],steps[1]['action_id'])
        self.assertEqual([s['action_ordinal'] for s in steps],[1,2])
        self.assertEqual([s['step_id'] for s in steps],
                         [s['step_id'] for s in fixture['plan']['subtask_plans'][0]['steps']])
        self.assertEqual(plan['action_count'],2)
        review=review_symbolic(dict(fixture,plan=plan),library,nodes(),initial_state(nodes()),1.5)
        self.assertEqual([t['action_id'] for t in review['symbolic_trace']],
                         [s['action_id'] for s in plan['action_index']])
        validate_sequence(plan)

    def test_case_round_run_and_content_revision_do_not_alias(self):
        _,_,plan=self.fixture()
        ids={plan['action_index'][0]['action_id']}
        for changes in ({'case_id':'caseB'},{'round':5},{'run_id':'runB'}):
            revised=assign_action_identity(plan,dict(plan['action_identity']['scope'],**changes))
            ids.add(revised['action_index'][0]['action_id'])
        changed=copy.deepcopy(plan)
        changed['subtask_plans'][0]['steps'][0]['arguments']['target']['element_id']=62
        revised=assign_action_identity(changed)
        ids.add(revised['action_index'][0]['action_id'])
        self.assertEqual(len(ids),5)

    def test_deterministic_idempotent_and_input_untouched(self):
        _,_,plan=self.fixture()
        before=copy.deepcopy(plan)
        self.assertEqual(assign_action_identity(plan),plan)
        self.assertEqual(plan,before)

    def test_tampered_ids_indexes_or_changed_actions_rejected_then_refreshable(self):
        _,_,plan=self.fixture()
        for mutation in ('id','index','arguments'):
            bad=copy.deepcopy(plan)
            if mutation=='id': bad['subtask_plans'][0]['steps'][0]['action_id']='fake'
            elif mutation=='index': bad['action_index'].reverse()
            else: bad['subtask_plans'][0]['steps'][0]['arguments']['target']['element_id']=62
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_action_identity(bad)
        bad=copy.deepcopy(plan)
        bad['subtask_plans'][0]['steps'].pop()
        with self.assertRaises(ValueError): validate_sequence(bad)
        fixed=refresh_sequence(bad)
        validate_sequence(fixed)
        self.assertEqual(fixed['action_count'],1)
        self.assertNotEqual(fixed['action_identity']['plan_revision_sha256'],plan['action_identity']['plan_revision_sha256'])


if __name__=='__main__': unittest.main()
